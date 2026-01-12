import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import math
import threading
from datetime import datetime, timedelta
import time
import requests
import xml.etree.ElementTree as ET
import os
import urllib3
import csv

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Charting Libraries ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# --- TRANSFORMERS SETUP ---
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ===================== 1. Switchable Transformer Engine =====================
class SentimentEngine:
    def __init__(self):
        self.current_model_name = "FinBERT"
        self.models = {
            "FinBERT": {
                "id": "ProsusAI/finbert",
                "dir": "my_finbert_model",
                "loaded": False,
                "tokenizer": None,
                "model": None
            }
        }
        self.status_msg = "Initializing..."

    def load_model(self, model_key):
        if not TRANSFORMERS_AVAILABLE:
            self.status_msg = "Error: 'transformers' library missing."
            return False

        target = self.models[model_key]
        if target["loaded"]:
            self.current_model_name = model_key
            self.status_msg = f"{model_key} Ready (Cached)."
            return True

        local_path = os.path.join(os.getcwd(), target["dir"])
        
        # Determine device (CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"[System] Loading {model_key} on {device}...")
            if os.path.exists(local_path) and os.listdir(local_path):
                target["tokenizer"] = AutoTokenizer.from_pretrained(local_path)
                target["model"] = AutoModelForSequenceClassification.from_pretrained(local_path).to(device)
            else:
                print(f"[System] Downloading {model_key} (First Run)...")
                target["tokenizer"] = AutoTokenizer.from_pretrained(target["id"])
                target["model"] = AutoModelForSequenceClassification.from_pretrained(target["id"]).to(device)
                
                print(f"[System] Saving {model_key} locally...")
                target["tokenizer"].save_pretrained(local_path)
                target["model"].save_pretrained(local_path)
            
            target["loaded"] = True
            self.current_model_name = model_key
            self.status_msg = f"{model_key} Loaded ({device.upper()})."
            return True

        except Exception as e:
            self.status_msg = f"Failed to load {model_key}: {e}"
            print(f"[Error] {self.status_msg}")
            return False

    def predict_batch(self, texts):
        target = self.models[self.current_model_name]
        
        # Use "Pending" instead of 0.5 for non-loaded models
        if not target["loaded"]:
            return ["Pending" for _ in texts]
            
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return ["Pending" for _ in texts]
            
        try:
            # Move inputs to the same device as the model
            device = next(target["model"].parameters()).device
            inputs = target["tokenizer"](clean_texts, return_tensors="pt", truncation=True,
                                          padding=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = target["model"](**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)
            
            pos = probs[:, 0]
            neg = probs[:, 1]
            # Convert back to CPU for list conversion
            scores_clean = (0.5 + (pos * 0.5) - (neg * 0.5)).cpu().tolist()
                
            full_scores = []
            idx = 0
            for t in texts:
                if t and t.strip():
                    full_scores.append(scores_clean[idx])
                    idx += 1
                else:
                    full_scores.append("Pending")
            return full_scores
        except Exception as e:
            print(f"[Model Error] {e}")
            return ["Pending" for _ in texts]

sentiment_engine = SentimentEngine()


# ===================== 2. Math Core (Log-Space Stability) =====================
class VegaChimpCore:
    @staticmethod
    def N(x): 
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    @staticmethod
    def bs_price(S, K, r, q, sig, T, kind):
        """Standard Black-Scholes (European) - Fallback."""
        if sig <= 1e-4 or T <= 1e-4: 
            return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
            
        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        disc = math.exp(-r * T); disc_q = math.exp(-q * T)
        
        if kind == "call": 
            return S * disc_q * VegaChimpCore.N(d1) - K * disc * VegaChimpCore.N(d2)
        return K * disc * VegaChimpCore.N(-d2) - S * disc_q * VegaChimpCore.N(-d1)

    @staticmethod
    def garch_forecast(log_returns, days=252):
        if len(log_returns) < 30: return 0.0
        try:
            alpha, beta, omega = 0.05, 0.94, 1e-6
            variance = np.var(log_returns)
            for r in log_returns:
                variance = omega + alpha * (r**2) + beta * variance
            return np.sqrt(variance * days)
        except:
            return 0.0

    @staticmethod
    def bjerksund_stensland(S, K, T, r, q, sigma, option_type='call'):
        """
        Approximation for American Options (2002).
        Uses Log-Space algebra to prevent overflow errors.
        """
        # 1. Sanity Checks
        if S <= 0 or K <= 0 or T <= 0: return 0.0
        
        # 2. Critical Volatility Clamp
        # Volatility < 1% causes exponents to approach Infinity.
        # We clamp it to 0.01 (1%) for stability.
        sigma = max(sigma, 0.01)

        if option_type == 'put':
            return VegaChimpCore.bjerksund_stensland(K, S, T, q, r, sigma, 'call')

        b = r - q 
        if b >= r:
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')

        try:
            # --- HELPER: Safe Exponential ---
            def safe_exp(val):
                if val > 700: return float('inf') # Prevent overflow
                if val < -700: return 0.0
                return math.exp(val)

            # --- HELPER: Phi Function (Rewritten in Log-Space) ---
            # Original: exp(lam) * (S^gamma) * ( N(d) - (I/S)^k * N(d2) )
            # New:      exp(lam + gamma*lnS) * N(d)  -  exp(lam + gamma*lnS + k*(lnI - lnS)) * N(d2)
            def phi(s, t, gamma, h, i):
                lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2) * t
                d_den = (sigma * math.sqrt(t))
                d_num = -(math.log(s / h) + (b + (gamma - 0.5) * sigma**2) * t)
                d = d_num / d_den
                
                k = 2 * b / (sigma**2) + (2 * gamma - 1)
                
                # Pre-calculate Logs
                ln_s = math.log(s)
                ln_i = math.log(i)
                
                # Term 1: exp(lam + gamma * ln_s) * N(d)
                power_term_1 = lam + (gamma * ln_s)
                val_1 = safe_exp(power_term_1) * VegaChimpCore.N(d)
                
                # Term 2: exp(power_term_1 + k * (ln_i - ln_s)) * N(d - ...)
                d2 = d - 2 * math.log(i/s) / d_den
                power_term_2 = power_term_1 + k * (ln_i - ln_s)
                val_2 = safe_exp(power_term_2) * VegaChimpCore.N(d2)

                return val_1 - val_2

            # Exercise Boundary
            beta = (0.5 - b/sigma**2) + math.sqrt((b/sigma**2 - 0.5)**2 + 2*r/sigma**2)
            if abs(beta - 1) < 1e-5: return S - K

            inf_boundary = K * beta / (beta - 1)
            h = -(b * T + 2 * sigma * math.sqrt(T)) * (K**2 / ((inf_boundary - K) * inf_boundary))
            
            # Safe calculation for I (exercise trigger)
            I = inf_boundary + (K - inf_boundary) * (1 - safe_exp(h))
            
            # Safe calculation for alpha
            # alpha = (I - K) * I^(-beta)  -->  (I-K) * exp(-beta * lnI)
            alpha = (I - K) * safe_exp(-beta * math.log(I))
            
            if S >= I:
                return S - K
            else:
                # Main Formula
                term1 = alpha * safe_exp(beta * math.log(S))
                term2 = alpha * phi(S, T, beta, I, I)
                term3 = phi(S, T, 1, I, I)
                term4 = phi(S, T, 1, K, I)
                term5 = K * phi(S, T, 0, I, I)
                term6 = K * phi(S, T, 0, K, I)
                
                return term1 - term2 + term3 - term4 - term5 + term6

        except (OverflowError, ValueError, ZeroDivisionError):
            # If math still breaks (e.g. extreme inputs), use Black-Scholes
            print("[Warning] Bjerksund-Stensland calculation failed, falling back to Black-Scholes.")
            return VegaChimpCore.bs_price(S, K, r, q, sigma, T, 'call')
# ===================== 3. Technicals Logic =====================
def calculate_technicals(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    return df

class Tooltip:
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 10
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = ttk.Label(self._tip, text=self.text, justify="left", relief="solid", borderwidth=1,
                        background="lightyellow", padding=5, wraplength=280)
        lbl.pack(ipadx=1)

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None

# ===================== 4. Main GUI =====================
class MarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel: Stock & Options Analyzer")
        self.root.geometry("1450x900")

        self.headline_limit = 1000
        self.ev_absolute = False
        self.data_cache = {}
        self.DATA_CACHE_DURATION = 60 
        self.sent_cache = {}
        self.SENT_CACHE_DURATION = 1800 
        
        self.ax = None 
        
        self.use_sentiment = False

        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "AMD")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data()) 
        
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")

        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        self.left_frame = ttk.Frame(self.paned, width=350)
        self.paned.add(self.left_frame, weight=1)

        self.lbl_price = ttk.Label(self.left_frame, text="---", font=("Arial", 28, "bold"))
        self.lbl_price.pack(anchor="center", pady=10)

        self.grid_frame = ttk.LabelFrame(self.left_frame, text="Technical Analysis", padding=15)
        self.grid_frame.pack(fill="x", pady=5)
        self.annual_growth_offset = 0
        
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0, "Relative Strength Index. Range 0-100.")
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI", 1, "Stochastic RSI. Range 0-1.")
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2, "Moving Average Convergence Divergence.")
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3, "20-day SMA +/- 2 STDs.")
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility)", 4, "Average True Range (Daily Move in $).")
        # Updated tooltip for GARCH
        self.lbl_vol = self.add_row(self.grid_frame, "Vol (HV vs GARCH)", 5, "HV: 30d Historical Volatility.\nGARCH: Forward-looking Vol Forecast.")
        if self.use_sentiment:
            self.lbl_sent = self.add_row(self.grid_frame, "AI Sentiment", 6, "Headline sentiment scored 0-1.")
            self.lbl_return = self.add_row(self.grid_frame, "Return (Period)", 7, "Total return over selected period.")
        else:
            self.lbl_return = self.add_row(self.grid_frame, "Return (Period)", 6, "Total return over selected period.")


        self.btn_opt = ttk.Button(self.left_frame, text="🔎 Options Explorer", command=self.open_options_window, state="disabled")
        self.btn_opt.pack(fill="x", padx=20, pady=20, ipady=10)

        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=3)
        
        ctrl_frame = ttk.Frame(self.right_frame)
        ctrl_frame.pack(fill="x", pady=5)
        
        self.btn_1d = ttk.Button(ctrl_frame, text="1D", command=lambda: self.load_chart("1d", "1m"), width=5)
        self.btn_1d.pack(side="left", padx=2)
        self.btn_5d = ttk.Button(ctrl_frame, text="5D", command=lambda: self.load_chart("5d", "5m"), width=5)
        self.btn_5d.pack(side="left", padx=2)
        self.btn_1m = ttk.Button(ctrl_frame, text="1M", command=lambda: self.load_chart("1mo", "30m"), width=5) 
        self.btn_1m.pack(side="left", padx=2)
        self.btn_3m = ttk.Button(ctrl_frame, text="3M", command=lambda: self.load_chart("3mo", "60m"), width=5)
        self.btn_3m.pack(side="left", padx=2)
        self.btn_1y = ttk.Button(ctrl_frame, text="1Y", command=lambda: self.load_chart("1y", "60m"), width=5)
        self.btn_1y.pack(side="left", padx=2)
        self.btn_5y = ttk.Button(ctrl_frame, text="5Y", command=lambda: self.load_chart("5y", "1d"), width=5)
        self.btn_5y.pack(side="left", padx=2)
        self.btn_10y = ttk.Button(ctrl_frame, text="10Y", command=lambda: self.load_chart("10y", "1wk"), width=5)
        self.btn_10y.pack(side="left", padx=2)
        self.btn_25y = ttk.Button(ctrl_frame, text="25Y", command=lambda: self.load_chart("25y", "1mo"), width=5)
        self.btn_25y.pack(side="left", padx=2)

        self.lbl_status = ttk.Label(ctrl_frame, text="", foreground="gray", font=("Arial", 8))
        self.lbl_status.pack(side="right", padx=10)

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111) 
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.hover_annot = None
        self.last_plot_df = None
        self.last_plot_x = None
        self.last_plot_times = None
        self.use_compressed_hover = False
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

# --- SYSTEM LOG & CONTROLS ---
        log_frame = ttk.LabelFrame(root, text="System Log & AI Controls", padding=5)
        log_frame.pack(fill="x", padx=10, pady=5)

        ctrl_panel = ttk.Frame(log_frame)
        ctrl_panel.pack(fill="x", pady=2)
        
        # [NEW] Toggle Button (Right Side)
        self.btn_log = ttk.Button(ctrl_panel, text="Show Log", command=self.toggle_log, width=10)
        self.btn_log.pack(side="right", padx=5)

        ttk.Label(ctrl_panel, text="Active Model:").pack(side="left")
        
        self.model_var = tk.StringVar(value="FinBERT")
        self.combo_model = ttk.Combobox(ctrl_panel, textvariable=self.model_var, 
                                        values=["DistilBERT", "FinBERT"], state="readonly", width=15)
        self.combo_model.pack(side="left", padx=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.change_model)

        self.lbl_model_status = ttk.Label(ctrl_panel, text="Status: Init...", foreground="blue")
        self.lbl_model_status.pack(side="left", padx=10)

        # [MODIFIED] Create widgets but DO NOT pack them yet (Hidden by default)
        self.log_box = tk.Text(log_frame, height=6, font=("Consolas", 9))
        self.log_scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box['yscrollcommand'] = self.log_scroll.set
        
        # Track visibility state
        self.log_visible = False
        

        self.current_ticker = None
        self.current_price = 0
        self.hv_30 = 0
        self.garch_vol = 0 # New variable for GARCH
        self.projected_earnings = [] 
        # Add this line where your other technical labels are initialized
        self.lbl_pe = self.add_row(self.grid_frame, "P/E Ratio", 8, "Price-to-Earnings Ratio (TTM vs Forward).")
        # Add this in your __init__ section
        self.lbl_pe_percentile = self.add_row(self.grid_frame, "P/E Percentile", 9, 
                                     "How expensive the current P/E is vs the last 5 years (0-100%).")
        # Only initialize the transformer if the toggle is True
        if self.use_sentiment:
            self.log("App Started. Defaulting to FinBERT.")
            threading.Thread(target=self.init_model_bg, args=("FinBERT",), daemon=True).start()
        else:
            self.log("AI Sentiment is currently disabled.")
            self.lbl_model_status.config(text="Status: Disabled", foreground="gray")
        threading.Thread(target=self.init_model_bg, args=("FinBERT",), daemon=True).start()
        
        self.root.after(500, self.load_data)

    def init_model_bg(self, model_name):
        self.root.after(0, lambda: self.lbl_model_status.config(text=f"Loading {model_name}..."))
        success = sentiment_engine.load_model(model_name)
        
        if success:
            msg = f"Ready ({model_name})"
            if self.current_ticker:
                self.sent_cache.pop(self.current_ticker, None) 
                self.root.after(500, self.load_data)
        else:
            msg = "Failed"
            
        self.root.after(0, lambda: self.lbl_model_status.config(text=msg))
        self.root.after(0, lambda: self.log(sentiment_engine.status_msg))

    def change_model(self, event):
        target = self.model_var.get()
        if target == sentiment_engine.current_model_name: return
        self.log(f"Switching AI Model to {target}...")
        threading.Thread(target=self.init_model_bg, args=(target,), daemon=True).start()

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}\n"
        def _append():
            self.log_box.insert("end", full_msg)
            self.log_box.see("end")
        self.root.after(0, _append)
        print(full_msg)
        
    def toggle_log(self):
        if self.log_visible:
            # Hide widgets
            self.log_box.pack_forget()
            self.log_scroll.pack_forget()
            self.btn_log.config(text="Show Log")
            self.log_visible = False
        else:
            # Show widgets
            self.log_box.pack(fill="x", side="left", expand=True)
            self.log_scroll.pack(side="right", fill="y")
            self.btn_log.config(text="Hide Log")
            self.log_visible = True

    def add_row(self, parent, text, row, tooltip_text=None):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(f, text=text, font=("Arial", 10, "bold")).pack(side="left")
        if tooltip_text:
            q = ttk.Label(f, text="?", foreground="blue", cursor="question_arrow", padding=(4, 0))
            q.pack(side="left")
            Tooltip(q, tooltip_text)
        lbl = ttk.Label(parent, text="---", font=("Arial", 10))
        lbl.grid(row=row, column=1, sticky="e", padx=10, pady=5)
        return lbl

    def load_data(self):
        """Refreshes chart data and updates fundamentals only on ticker change."""
        new_ticker = self.entry_ticker.get().upper().strip()
        if not new_ticker: return

        # 1. Handle Ticker Change (Heavy Logic)
        if new_ticker != self.current_ticker:
            self.current_ticker = new_ticker
            self.data_cache = {} 
            self.sent_cache = {}
            
            # Show a 'loading' state in the UI for fundamentals
            self.lbl_pe.config(text="Updating...", foreground="blue")
            
            # Start heavy fundamental fetch in the background
            threading.Thread(target=self.get_info, daemon=True).start()
            self.log(f"Ticker changed: {new_ticker}. Fetching fundamentals...")

        # 2. Always Refresh Chart (Light Logic)
        # Moving this outside the IF allows users to 'Refresh' the current ticker
        self.load_chart("5d", "5m")

    def load_chart(self, period, interval):
        ticker = self.entry_ticker.get().upper().strip()
        if not ticker: return
        self.current_ticker = ticker
        self.last_interval = interval
        self.root.after(0, lambda: self.lbl_status.config(text=f"Loading {ticker} ({period})..."))
        self.log(f"Requesting Chart: {ticker} ({period})")
        threading.Thread(target=self.fetch_and_plot, args=(ticker, period, interval), daemon=True).start()

    def fetch_history_with_retry(self, stock, period, interval, retries=2, delay=1.0):
        last_exc = None
        for attempt in range(retries + 1):
            try:
                df = stock.history(period=period, interval=interval)
                if not df.empty:
                    return df
                last_exc = RuntimeError("Empty data returned")
            except Exception as e:
                last_exc = e
                self.log(f"History fetch error try {attempt+1}/{retries+1}: {e}")
            time.sleep(delay)
        raise last_exc if last_exc else RuntimeError("Unknown history fetch failure")

    def get_cached_df(self, ticker, period, interval):
        key = (ticker, period, interval)
        if key in self.data_cache:
            data, ts = self.data_cache[key]
            if time.time() - ts < self.DATA_CACHE_DURATION:
                return data, True
        return None, False

    def save_df_cache(self, ticker, period, interval, df):
        key = (ticker, period, interval)
        self.data_cache[key] = (df, time.time())

    def get_google_news_rss(self, ticker):
        self.log(f"Fetching Google RSS for {ticker} (Multi-Query)...")
        all_titles = set()
        
        # Define search variations to broaden the 100-item limit
        queries = [
            f"{ticker}+stock",
            f"{ticker}+earnings+news",
            f"{ticker}+market+analysis",
            f"{ticker}+financial+news"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for q in queries:
            try:
                url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                # Using verify=False as per your original code to ignore SSL overhead
                resp = requests.get(url, headers=headers, timeout=5, verify=False) 
                
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    items = root.findall('.//item')
                    for item in items:
                        title = item.find('title').text
                        if title:
                            all_titles.add(title)
                    
                    # Safety break if we've already exceeded your headline_limit
                    if len(all_titles) >= self.headline_limit:
                        break
                        
            except Exception as e:
                 self.log(f"RSS Variation Error ({q}): {e}")
        
        # Convert set back to list and enforce the headline_limit
        final_titles = list(all_titles)[:self.headline_limit]
        self.log(f"Total Unique RSS Headlines Found: {len(final_titles)}")
        return final_titles



    def calculate_sentiment(self, ticker, stock_obj):
        if ticker in self.sent_cache:
            val, ts = self.sent_cache[ticker]
            if time.time() - ts < self.SENT_CACHE_DURATION:
                self.log("Using Cached Sentiment.")
                return val

        current_model = sentiment_engine.models[sentiment_engine.current_model_name]
        if not current_model["loaded"]:
            self.log("Model not ready. Waiting...")
            return 'Pending'

        headlines = []
        try:
            ynews = stock_obj.news
            if ynews:
                for n in ynews:
                    title = n.get('title') or n.get('headline') or ""
                    if title.strip(): headlines.append(title)
        except Exception as e:
            self.log(f"Yahoo news error: {e}")

        if not headlines:
            headlines = self.get_google_news_rss(ticker)

        if not headlines:
            self.log("No headlines found.")
            return None

        self.log(f"Analyzing {len(headlines)} headlines with {sentiment_engine.current_model_name}...")
        top_headlines = headlines[: self.headline_limit]
        scores = sentiment_engine.predict_batch(top_headlines)
        for i, (h, score) in enumerate(zip(top_headlines, scores)):
            self.log(f"[{i+1}] {score:.2f} | {h[:40]}...")

        if not scores: 
            return 'Pending'

        avg_score = sum(scores) / len(scores)
        self.log(f"FINAL SCORE ({sentiment_engine.current_model_name}): {avg_score:.4f}")
        
        self.sent_cache[ticker] = (avg_score, time.time())
        return avg_score

    def treeview_sort_column(self, tv, col, reverse):
        """Sorts the treeview contents when a column header is clicked."""
        # Get all data from the column
        l = [(tv.set(k, col), k) for k in tv.get_children('')]

        # Helper to convert values to floats for proper numerical sorting
        def sort_key(v):
            val = v[0]
            try:
                # Remove symbols that break float conversion
                clean_val = str(val).replace('%', '').replace('$', '').replace('+', '')
                return float(clean_val)
            except ValueError:
                return str(val).lower()

        # Sort the list
        l.sort(key=sort_key, reverse=reverse)

        # Rearrange items in sorted order
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        # Update the heading command to toggle the sort direction next time
        tv.heading(col, command=lambda _col=col: self.treeview_sort_column(tv, _col, not reverse))

    def fetch_and_plot(self, ticker, period, interval):
        try:
            # 1. Chart Data
            df, is_cached = self.get_cached_df(ticker, period, interval)
            stock = yf.Ticker(ticker)
            
            # --- Earnings Cycle Detection (Existing) ---
            try:
                self.projected_earnings = []
                cal = stock.calendar
                anchor_date = None

                if isinstance(cal, dict):
                    if 'Earnings Date' in cal:
                        dates = cal['Earnings Date']
                        if dates and len(dates) > 0:
                            anchor_date = pd.to_datetime(dates[0]).date()
                
                elif cal is not None and hasattr(cal, 'empty') and not cal.empty:
                    raw_date = cal.iloc[0].values[0]
                    anchor_date = pd.to_datetime(raw_date).date()

                if anchor_date:
                    self.projected_earnings.append(anchor_date)
                    current = anchor_date
                    for _ in range(3):
                        current = current + timedelta(days=91)
                        self.projected_earnings.append(current)
                    
                    self.log(f"Earnings Cycle Detected: {self.projected_earnings}")
                
            except Exception as e:
                self.log(f"Earnings fetch skipped: {e}")
                self.projected_earnings = []

            # --- Main Price Data Fetching ---
            if df is None:
                df = self.fetch_history_with_retry(stock, period, interval)
                if df.empty: 
                    self.log("No price data found.")
                    self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                    return
                
                # EMAs: Calculate unconditionally
                df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
                df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
                df['EMA_63'] = df['Close'].ewm(span=63, adjust=False).mean()
                df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
                
                self.save_df_cache(ticker, period, interval, df)
                status_msg = f"Live Data ({interval})"
            else:
                status_msg = f"Cached Data ({interval})"
            
            # Calculate Period Return for UI
            if not df.empty:
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                period_return = (end_price - start_price) / start_price
            else:
                period_return = 0.0

            self.root.after(0, lambda: self.lbl_status.config(text=status_msg))

            # --- 2. [NEW] 5-Year Growth Drift Calculation ---
            try:
                # Using 1mo interval for 5y history to keep it fast
                df_5y = self.fetch_history_with_retry(stock, "5y", "1mo")
                if not df_5y.empty and len(df_5y) > 12:
                    p_start = df_5y['Close'].iloc[0]
                    p_end = df_5y['Close'].iloc[-1]
                    
                    # CAGR Formula: (End/Start)^(1/5) - 1
                    cagr = ((p_end / p_start) ** (1/5)) - 1
                    
                    # Clamp between -5% and +15% to ground the model
                    self.annual_growth_offset = max(min(cagr*0.5, 0.15), -0.05) # QQQ was crazy the last 5 years, we halve the effect
                    self.log(f"5Y Historical Drift (CAGR): {self.annual_growth_offset:+.2%}")
                else:
                    self.annual_growth_offset = 0.05 # Default Fallback
            except Exception as e:
                self.log(f"Growth Offset Error: {e}")
                self.annual_growth_offset = 0.05

            # --- 3. Technical Data & GARCH (1y Daily) ---
            df_tech, tech_cached = self.get_cached_df(ticker, "1y", "1d")
            if df_tech is None:
                df_tech = self.fetch_history_with_retry(stock, "1y", "1d")
                df_tech = calculate_technicals(df_tech)
                df_tech['log_ret'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1))
                self.save_df_cache(ticker, "1y", "1d", df_tech)
            
            last = df_tech.iloc[-1]
            current_price = df['Close'].iloc[-1] if not df.empty else last['Close']

            hv_30 = df_tech['log_ret'].rolling(30).std().iloc[-1] * np.sqrt(252)
            
            # GARCH(1,1) Forecast
            returns = df_tech['log_ret'].dropna().values
            self.garch_vol = VegaChimpCore.garch_forecast(returns)

            self.current_price = current_price
            self.hv_30 = hv_30
            
            # --- 4. Sentiment Analysis ---
            if self.use_sentiment:
                sentiment_score = self.calculate_sentiment(ticker, stock)
            else:
                sentiment_score = None  # Skip analysis if False

            # --- 5. UI Updates ---
            last_copy = last.copy()
            last_copy['Close'] = current_price 
            
            self.root.after(0, lambda: self.update_technicals(
                last_copy, hv_30, self.garch_vol, sentiment_score, period_return
            ))
            self.root.after(0, self.update_chart, df, ticker, period)

        except Exception as e:
            self.log(f"CRITICAL ERROR in fetch_and_plot: {e}")
            self.root.after(0, lambda: self.lbl_status.config(text="Error"))
    def update_chart(self, df, ticker, period):
        if not hasattr(self, 'ax') or self.ax is None: return

        try:
            if df is None or df.empty:
                self.log("Chart skipped: no data")
                self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                return
            plot_df = df.copy()
            interval = getattr(self, "last_interval", None)
            intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

            if interval in intraday:
                idx = plot_df.index
                if getattr(idx, "tz", None):
                    idx_eastern = idx.tz_convert("America/New_York")
                else:
                    idx_eastern = idx.tz_localize("UTC").tz_convert("America/New_York")
                minutes = idx_eastern.hour * 60 + idx_eastern.minute
                mask = (minutes >= 570) & (minutes <= 960) & (idx_eastern.dayofweek < 5)
                filtered = plot_df[mask]
                if not filtered.empty:
                    plot_df = filtered

            use_compressed = interval in intraday
            if use_compressed:
                times_for_labels = plot_df.index
                if getattr(times_for_labels, "tz", None):
                    times_for_labels = times_for_labels.tz_convert("America/New_York")
                else:
                    times_for_labels = times_for_labels.tz_localize("UTC").tz_convert("America/New_York")
                x_vals = np.arange(len(plot_df))
            else:
                times_for_labels = plot_df.index
                x_vals = mdates.date2num(times_for_labels.to_pydatetime())

            self.ax.clear()
            self.hover_annot = None 

            self.ax.plot(x_vals, plot_df['Close'], label='Price', color='black', linewidth=1.5)
            if 'EMA_5' in plot_df.columns: self.ax.plot(x_vals, plot_df['EMA_5'], label='EMA 5', color='blue', linewidth=1)
            if 'EMA_21' in plot_df.columns: self.ax.plot(x_vals, plot_df['EMA_21'], label='EMA 21', color='orange', linewidth=1)
            if 'EMA_63' in plot_df.columns: self.ax.plot(x_vals, plot_df['EMA_63'], label='EMA 63', color='purple', linewidth=1)
            
            if 'EMA_200' in plot_df.columns:
                if plot_df['EMA_200'].notna().sum() > 0:
                    self.ax.plot(x_vals, plot_df['EMA_200'], label='EMA 200', color='red', linewidth=1.5)
            self.ax.set_xlim(left=x_vals[0], right=x_vals[-1])
            self.ax.set_title(f"{ticker} Price Action ({period})")
            self.ax.legend(loc='upper right', fontsize='small')
            self.ax.grid(True, alpha=0.3)
            
            if use_compressed:
                if len(times_for_labels) > 0:
                    tick_count = min(6, len(times_for_labels))
                    tick_idx = np.linspace(0, len(times_for_labels) - 1, tick_count, dtype=int)
                    if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
                        tick_labels = [times_for_labels[i].strftime("%m-%d %H:%M") for i in tick_idx]
                    elif interval in {"1d", "1wk"}:
                        tick_labels = [times_for_labels[i].strftime("%Y-%m-%d") for i in tick_idx]
                    elif interval == "1mo":
                        tick_labels = [times_for_labels[i].strftime("%Y-%b") for i in tick_idx]
                    else:
                        tick_labels = [str(times_for_labels[i]) for i in tick_idx]
                    self.ax.set_xticks(tick_idx)
                    self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            else:
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                self.figure.autofmt_xdate()
            
            self.canvas.draw()
            self.last_plot_df = plot_df
            self.last_plot_x = x_vals
            self.last_plot_times = times_for_labels
            self.use_compressed_hover = use_compressed
        except Exception as e:
            self.log(f"Chart Render Error: {e}")

# --- UPDATED UI METHOD TO SHOW GARCH & PENDING SENTIMENT ---
    def update_technicals(self, data, hv, garch, sentiment, period_return):
        self.lbl_price.config(text=f"${data['Close']:.2f}")
        
        rsi_val = data['RSI']
        rsi_c = "green" if rsi_val < 30 else "red" if rsi_val > 70 else "black"
        self.lbl_rsi.config(text=f"{rsi_val:.2f}", foreground=rsi_c)
        
        stoch_val = data['StochRSI']
        stoch_c = "green" if stoch_val < 0.2 else "red" if stoch_val > 0.8 else "black"
        self.lbl_stoch.config(text=f"{stoch_val:.2f}", foreground=stoch_c)
        
        macd_val = data['MACD']
        macd_c = "green" if macd_val > 0 else "red"
        self.lbl_macd.config(text=f"{macd_val:.2f}", foreground=macd_c)
        
        try:
            # Format the text using already-stored fundamental data (No new network call)
            pe_ttm_str = f"{self.pe_ttm:.2f}" if isinstance(self.pe_ttm, (int, float)) else "N/A"
            pe_fwd_str = f"{self.pe_fwd:.2f}" if isinstance(self.pe_fwd, (int, float)) else "N/A"

            # Update the display label
            self.lbl_pe.config(text=f"TTM: {pe_ttm_str} | Fwd: {pe_fwd_str}")
        
            # Inside your update_technicals method:
            if hasattr(self, 'pe_percentile'):
                p_val = self.pe_percentile
                # Color: Red if > 80% (Expensive), Green if < 20% (Cheap)
                p_color = "red" if p_val > 80 else "green" if p_val < 20 else "black"
                self.lbl_pe_percentile.config(text=f"{p_val:.1f}%", foreground=p_color)
        except Exception as e:
            self.log(f"P/E Fetch Error: {e}")
            self.lbl_pe.config(text="N/A", foreground="gray")
        
        bb_pos = "Inside"
        bb_c = "black"
        if data['Close'] > data['BB_Upper']: 
            bb_pos = "Overbought"; bb_c = "red"
        elif data['Close'] < data['BB_Lower']: 
            bb_pos = "Oversold"; bb_c = "green"
        self.lbl_bb.config(text=f"{bb_pos}\n[{data['BB_Lower']:.2f}-{data['BB_Upper']:.2f}]", foreground=bb_c)
        
        self.lbl_atr.config(text=f"${data['ATR']:.2f}")
        
        # --- GARCH & HV DISPLAY ---
        self.lbl_vol.config(text=f"HV: {hv:.1%} | GARCH: {garch:.1%}")
        
        # --- UPDATED SENTIMENT BLOCK (Handles "Pending" string) ---
        if self.use_sentiment:
            if sentiment == "Pending":
                self.lbl_sent.config(text="Pending", foreground="gray")
            elif sentiment is not None:
                try:
                # Ensure it's treated as a float for comparison
                    val = float(sentiment)
                    sent_c = "red" if val < 0.4 else "green" if val > 0.6 else "black"
                    self.lbl_sent.config(text=f"{val:.2f}", foreground=sent_c)
                except (ValueError, TypeError):
                    self.lbl_sent.config(text="N/A", foreground="gray")
            else:
                self.lbl_sent.config(text="N/A", foreground="gray")
        
        ret_c = "green" if period_return > 0 else "red" if period_return < 0 else "black"
        self.lbl_return.config(text=f"{period_return:+.2%}", foreground=ret_c)

        self.btn_opt.config(state="normal", text=f"🔎 Open {self.current_ticker} Option Scanner")

    def on_hover(self, event):
        if event.inaxes != self.ax or self.last_plot_df is None or self.last_plot_df.empty:
            if self.hover_annot:
                self.hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return

        try:
            # --- FIND THE DATA POINT ---
            if self.use_compressed_hover:
                xdata = self.last_plot_x
                if xdata is None or len(xdata) == 0: return
                idx = int(round(event.xdata))
                idx = np.clip(idx, 0, len(xdata) - 1)
                xval = xdata[idx]
            else:
                xdata = mdates.date2num(self.last_plot_times.to_pydatetime()) if self.last_plot_times is not None else []
                if len(xdata) == 0: return
                idx = np.searchsorted(xdata, event.xdata)
                idx = np.clip(idx, 0, len(xdata) - 1)
                xval = xdata[idx]

            row = self.last_plot_df.iloc[idx]
            yval = row['Close']
            
            # --- FORMAT DATE ---
            # The index of the row is the timestamp
            curr_time = self.last_plot_df.index[idx]
            # Use specific format based on interval (Time for intraday, Date for daily)
            if self.last_interval in ["1d", "5d", "1wk", "1mo"]:
                date_str = curr_time.strftime("%Y-%m-%d")
            else:
                date_str = curr_time.strftime("%Y-%m-%d %H:%M")

            # --- BUILD TOOLTIP TEXT ---
            parts = [f"Date: {date_str}", f"Price: ${yval:.2f}"] # Added Date here
            
            if 'EMA_5' in row and not np.isnan(row['EMA_5']):
                parts.append(f"EMA5: ${row['EMA_5']:.2f}")
            if 'EMA_21' in row and not np.isnan(row['EMA_21']):
                parts.append(f"EMA21: ${row['EMA_21']:.2f}")
            if 'EMA_63' in row and not np.isnan(row['EMA_63']):
                parts.append(f"EMA63: ${row['EMA_63']:.2f}")
            if 'EMA_200' in row and not np.isnan(row['EMA_200']):
                parts.append(f"EMA200: ${row['EMA_200']:.2f}")
            
            text = "\n".join(parts)

            if not self.hover_annot:
                self.hover_annot = self.ax.annotate(
                    text, xy=(xval, yval), xytext=(15, 15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="0.5")
                )
            else:
                self.hover_annot.set_text(text)
                self.hover_annot.xy = (xval, yval)
                self.hover_annot.set_visible(True)

            self.canvas.draw_idle()
        except Exception as e:
            self.log(f"Hover error: {e}")
            
    def scan_all_undervalued(self):
        # Clear current results
        for i in self.tree.get_children(): self.tree.delete(i)
        
        # Search ALL dates, but enable filtering for "Under" only
        if hasattr(self, 'all_exps'):
            self.log(f"Scanning {len(self.all_exps)} chains for value...")
            threading.Thread(target=self.fetch_options_batch, args=(self.all_exps, True), daemon=True).start()
    
    
    def get_info(self):
        """Consolidated fundamental fetch called when ticker changes."""
        ticker_str = self.entry_ticker.get().upper().strip()
        try:
            stock = yf.Ticker(ticker_str)
            info = stock.info
            
            # Store these for display and for the Z-Score math
            self.pe_fwd = info.get('forwardPE')
            self.pe_ttm = info.get('trailingPE')
            self.eps = info.get('trailingEps')
            self.dividend_yield = info.get('dividendYield', 0.0)
            
            # Pre-calculate the valuation multiplier right now
            self.val_multiplier = self.calculate_valuation_multiplier(stock)
            self.log(f"Valuation Multiplier: {self.val_multiplier:.2f}x (Z: {self.current_z_score:.2f})")
            
        except Exception as e:
            self.log(f"get_info error: {e}")
            self.pe_fwd = self.pe_ttm = self.eps = None
            self.val_multiplier = 1.0
            
    def calculate_valuation_multiplier(self, ticker_obj):
        try:
            if not self.pe_fwd or not self.eps or self.eps <= 0:
                self.current_z_score = 0
                self.pe_percentile = 50 # Neutral
                return 1.0
                
            hist = ticker_obj.history(period="5y")
            if hist.empty: return 1.0
            
            pe_series = hist['Close'] / self.eps
            
            # --- NEW PERCENTILE LOGIC ---
            # Count how many historical P/E days were lower than the current Forward P/E
            count_lower = (pe_series < self.pe_fwd).sum()
            self.pe_percentile = (count_lower / len(pe_series)) * 100
            
            # Existing Z-Score Logic
            mean_pe = pe_series.mean()
            std_pe = pe_series.std()
            z_score = (self.pe_fwd - mean_pe) / std_pe
            self.current_z_score = z_score
            
            multiplier = 1.0 + (z_score * 0.25)
            return max(0.1, multiplier)
        except:
            return 1.0     

    def get_smart_dividend(self, stock_obj):
        try:
            # 1. Try 'dividendYield' from info
            info = stock_obj.info
            div = info.get('dividendYield')
            
            # --- FIX: SANITY CHECK ---
            if div is not None and isinstance(div, (int, float)):
                # If yield is > 0.5 (50%), it's almost certainly a percentage (e.g. 2.94)
                # We need it to be a decimal (0.0294)
                div = div / 100
                
                print(f"[DEBUG] Found Dividend (Info): {div:.4%}") 
                return div
            
            # 2. Try 'trailingAnnualDividendYield'
            div = info.get('trailingAnnualDividendYield')
            if div is not None and isinstance(div, (int, float)):
                div = div / 100 # Apply fix here too
                print(f"[DEBUG] Found Dividend (Trailing): {div:.4%}")
                return div

            # 3. Fallback: Calculation
            hist = stock_obj.dividends
            if not hist.empty:
                recent_total = hist.iloc[-4:].sum() 
                if self.current_price > 0:
                    yield_calc = recent_total / self.current_price
                    print(f"[DEBUG] Calculated Dividend (History): {yield_calc:.4%}")
                    return yield_calc
            
            # 4. NVO Specific Fallback
            if self.current_ticker == "NVO":
                print("[DEBUG] Forcing NVO Default: 1.5000%")
                return 0.015

            return 0.0
            
        except Exception as e:
            print(f"[DEBUG] Div fetch error: {e}")
            return 0.0

    def open_options_window(self):
        if not self.current_ticker: return
        win = Toplevel(self.root)
        win.title(f"Options Explorer: {self.current_ticker}")
        win.geometry("1200x800")

        left_panel = ttk.Frame(win, width=200)
        left_panel.pack(side="left", fill="y", padx=5, pady=5)
        right_panel = ttk.Frame(win)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        ttk.Label(left_panel, text="Target Date (YYYY-MM-DD):").pack(fill="x")
        self.entry_date = ttk.Entry(left_panel)
        self.entry_date.pack(fill="x", pady=2)
        self.entry_date.insert(0, (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"))
        
        ttk.Button(left_panel, text="Select Prev 7 Expirations", command=self.filter_expirations).pack(fill="x", pady=5)
        ttk.Button(left_panel, text="⚡ Scan ALL Undervalued", command=self.scan_all_undervalued).pack(fill="x", pady=20)
        ttk.Button(left_panel, text="💾 Export Results to CSV", command=self.export_to_csv).pack(fill="x", pady=5)
          
        self.exp_list = tk.Listbox(left_panel, selectmode="extended", height=25)
        self.exp_list.pack(fill="both", expand=True)
        self.exp_list.bind('<<ListboxSelect>>', self.on_exp_select)
        
        # --- FIXED COLUMNS (Removed extra 'Fair' column) ---
        # 10 Columns Total
        # In open_options_window()
        cols = ("Date", "Type", "Strike", "Vol", "Price", "Breakeven", "Imp Vol", "Fair", "EV", "Verdict")
        # I changed "Last" to "Price" so you know it's the effective price used for calc, it used bid+ask/2
        self.tree = ttk.Treeview(right_panel, columns=cols, show="headings")
        
        for c in cols: 
            # Added command= to trigger sorting on click
            self.tree.heading(c, text=c, command=lambda _c=c: self.treeview_sort_column(self.tree, _c, False))
            
            # Keep your existing width logic
            w = 80 if c == "Breakeven" else 65
            if c == "Date": w = 90
            self.tree.column(c, width=w, anchor="center")
        
        scr = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scr.set); self.tree.pack(side="left", fill="both", expand=True); scr.pack(side="right", fill="y")
        
        self.tree.tag_configure("green", background="#d4f8d4")
        self.tree.tag_configure("red", background="#f8d4d4")
        self.tree.tag_configure("blue", background="#d4eef8")
        
        threading.Thread(target=self.load_expirations, daemon=True).start()

    def export_to_csv(self):
        # 1. Ask user where to save
        filename = filedialog.asksaveasfilename(
            initialfile=f"{self.current_ticker}_options_scan.csv",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filename: return

        try:
            # 2. Collect Data from Treeview
            rows = self.tree.get_children()
            if not rows:
                messagebox.showinfo("Export", "No data to export!")
                return

            # 3. Write to CSV
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write Headers
                cols = ("Date", "Type", "Strike", "Vol", "Price", "Breakeven", "Imp Vol", "Fair", "EV", "Verdict")
                writer.writerow(cols)
                
                # Write Rows
                count = 0
                for row_id in rows:
                    row_data = self.tree.item(row_id)['values']
                    writer.writerow(row_data)
                    count += 1
            
            messagebox.showinfo("Success", f"Successfully exported {count} rows to:\n{filename}")
            self.log(f"Exported {count} rows to CSV.")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}")
            self.log(f"Export Error: {e}")

    def load_expirations(self):
        stock = yf.Ticker(self.current_ticker)
        self.all_exps = stock.options
        self.root.after(0, lambda: self.update_exp_list(self.all_exps))

    def update_exp_list(self, exp_list):
        self.exp_list.delete(0, "end")
        for e in exp_list: self.exp_list.insert("end", e)

    def filter_expirations(self):
        target_str = self.entry_date.get()
        try:
            target_dt = datetime.strptime(target_str, "%Y-%m-%d")
            valid = [e for e in self.all_exps if datetime.strptime(e, "%Y-%m-%d") <= target_dt]
            final = valid[-7:] if len(valid) >= 7 else valid
            self.exp_list.selection_clear(0, "end")
            for i, e in enumerate(self.all_exps):
                if e in final: self.exp_list.selection_set(i)
            self.on_exp_select(None)
        except: messagebox.showerror("Error", "Invalid Date")

    def on_exp_select(self, event):
        sel = self.exp_list.curselection()
        dates = [self.exp_list.get(i) for i in sel]
        if not dates: return
        for i in self.tree.get_children(): self.tree.delete(i)
        threading.Thread(target=self.fetch_options_batch, args=(dates,), daemon=True).start()

    def get_risk_free_rate(self):
        """Fetches the 13-week Treasury Bill yield (^IRX) as the risk-free rate."""
        try:
            tnx = yf.Ticker("^IRX")
            # ^IRX is quoted in percentage (e.g., 4.50), we need decimal (0.045)
            rate = tnx.fast_info['last_price'] / 100
            if rate <= 0: return 0.045 # Fallback
            return rate
        except Exception:
            return 0.045 # Fallback to a standard rate if fetch fails

    def fetch_options_batch(self, dates, filter_under_only=False):
        stock = yf.Ticker(self.current_ticker)
        
        # --- DYNAMIC INPUTS ---
        DIV_YIELD = self.get_smart_dividend(stock) 
        RFR = self.get_risk_free_rate() # <--- NEW DYNAMIC RATE
        
        print(f"[DEBUG] Market Environment | Ticker: {self.current_ticker}")
        print(f"[DEBUG] Risk-Free Rate: {RFR:.4%}")
        print(f"[DEBUG] Dividend Yield: {DIV_YIELD:.4%}")

        earnings_contracts = set()
        if self.projected_earnings and hasattr(self, 'all_exps') and self.all_exps:
            for p_date in self.projected_earnings:
                p_str = p_date.strftime("%Y-%m-%d")
                valid_exps = [e for e in self.all_exps if e >= p_str]
                if valid_exps: earnings_contracts.add(min(valid_exps))

        for date in dates:
            try:
                T = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days / 365.0
                if T <= 0: T=0.001
                chain = stock.option_chain(date)
                calls = chain.calls.assign(Type="CALL"); puts = chain.puts.assign(Type="PUT")
                
                top = pd.concat([calls, puts]).sort_values('volume', ascending=False).head(20)
                
                for _, row in top.iterrows():
                    bid, ask, last = row.get('bid', 0), row.get('ask', 0), row['lastPrice']
                    
                    if bid > 0 and ask > 0:
                        market_price = (bid + ask) / 2
                        if (ask - bid) / market_price > 0.4: continue 
                    elif last > 0:
                        market_price = last
                    else:
                        continue
                        
                    iv = row['impliedVolatility']
                    if not iv or iv < 0.01: continue

                    # Blend GARCH with Market IV to prevent extreme outliers
                    market_iv = row.get('impliedVolatility', self.hv_30)
                    if self.garch_vol > 0:
                        # Give the market IV 60% weight and GARCH 40% weight
                        # This respects market consensus while keeping your "edge"
                        vol_input = (self.garch_vol * 0.4) + (market_iv * 0.6)
                    else:
                        vol_input = market_iv

                    # Sanity Check: Volatility Mean Reversion for LEAPS
                    # If time to expiry (T) is > 1 year, blend toward historical mean (approx 18%)
                    if T > 1.0:
                        vol_input = (vol_input + 0.18) / 2
                    
                    # --- MODEL CALL WITH DYNAMIC RFR ---
                    
                    # Indices like QQQ have a historical upward drift (approx 7-8% annually)
                    # You can add a 'Growth' parameter to offset the dividend drain
                    growth_offset = self.annual_growth_offset # 5% growth assumption

                    # Use this adjusted RFR to signal that the stock isn't just a falling rock
                    adjusted_rfr = RFR + growth_offset

                    fair = VegaChimpCore.bjerksund_stensland(
                        self.current_price, 
                        row['strike'], 
                        T, 
                        adjusted_rfr,  # Using RFR + Growth Offset
                        DIV_YIELD,  
                        vol_input, 
                        row['Type'].lower()
                    )
                    ev = fair - market_price
                    
                    if row['Type'] == "CALL":
                        breakeven = row['strike'] + market_price
                    else:
                        breakeven = row['strike'] - market_price

                    verdict = "Fair"
                    tag = ""
                    is_earnings = date in earnings_contracts
                    
                    if self.ev_absolute:
                        threshold = 0.25 if is_earnings else 0.15
                        if ev > threshold:
                            verdict = "Earnings Under" if is_earnings else "Under"
                            tag = "green"
                        elif ev < -threshold:
                            verdict = "Earnings Over" if is_earnings else "Over"
                            tag = "red"
                    else:
                        safe_price = max(market_price, 0.01)
                        edge_percent = (ev / safe_price) * 100
                        # --- Inside your fetch_options_batch loop ---
                        # (After calculating ev and edge_percent)

                        # Base requirement (e.g., 10% edge)
                        base_min_edge = 10.0 if is_earnings else 5.0

                        # Apply the Valuation Multiplier
                        # For TSLA (Z=2.0), bar becomes 15% (Strict)
                        # For a Value stock (Z=-2.0), bar becomes 5% (Aggressive Reward)
                        adjusted_bar = base_min_edge * self.val_multiplier

                        if edge_percent > adjusted_bar and ev > 0.05:
                            verdict = f"Under ({edge_percent:.0f}%)"
                            tag = "green"
                            if is_earnings: verdict = f"Earning Under ({edge_percent:.0f}%)"
                        elif edge_percent < -adjusted_bar and ev < -0.05:
                            verdict = "Over"
                            tag = "red"
                            if is_earnings: verdict = f"Earning Over ({edge_percent:.0f}%)"

                    if filter_under_only and "Under" not in verdict:
                        continue

                    vals = (date, row['Type'], row['strike'], int(row['volume'] or 0), 
                            f"{market_price:.2f}", f"{breakeven:.2f}", f"{iv:.1%}", 
                            f"{fair:.2f}", f"{ev:+.2f}", verdict)
                    
                    self.root.after(0, lambda v=vals, t=tag: self.tree.insert("", "end", values=v, tags=(t,)))
            
            except Exception as e:
                self.log(f"Options fetch error for {date}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()