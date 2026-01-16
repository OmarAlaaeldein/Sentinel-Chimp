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
import webbrowser
import csv
import re
import html
# --- ADD THESE IMPORTS ---
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors # Essential for correct Green/Red coloring

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
# Put this at the very top of your imports
try:
    import pyi_splash
except ImportError:
    pass

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Charting Libraries ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import matplotlib.style as mplstyle  # <--- ADD THIS
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
                        background="#333333", foreground="#ffffff", padding=2, wraplength=300)
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
        self.setup_dark_theme() # <--- CALL THEME HERE

        self.headline_limit = 1000
        self.ev_absolute = True
        self.data_cache = {}
        self.DATA_CACHE_DURATION = 60 
        self.sent_cache = {}
        self.SENT_CACHE_DURATION = 1800 
        
        self.ax = None 
        
        self.use_sentiment = True  # New toggle for sentiment analysis

        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "AMD")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data()) 
        
        
        
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")
        
        # --- [NEW] News Button ---
        self.btn_news = ttk.Button(input_frame, text="📰 News", command=self.open_news_window, state="disabled")
        self.btn_news.pack(side="left", padx=10)
        # -------------------------

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

        # --- MODIFIED: Create Figure with Dark Background ---
        # facecolor='#121212' matches a dark UI better than pure black
        self.figure = Figure(figsize=(5, 4), dpi=120, facecolor='#121212') 
        
        self.ax = self.figure.add_subplot(111) 
        
        # Make the chart background slightly lighter than the border for contrast
        self.ax.set_facecolor('#121212') 
        
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Make the Tkinter canvas widget black to match
        self.canvas.get_tk_widget().configure(bg='#121212')
        self.hover_annot = None
        self.last_plot_df = None
        self.last_plot_x = None
        self.last_plot_times = None
        self.use_compressed_hover = False
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

# --- SYSTEM LOG & CONTROLS ---
        if self.use_sentiment:
            log_frame = ttk.LabelFrame(root, text="System Log & AI Controls", padding=5)

        else:
            log_frame = ttk.LabelFrame(root, text="System Log", padding=5)
            
        log_frame.pack(fill="x", padx=10, pady=5)

        ctrl_panel = ttk.Frame(log_frame)
        ctrl_panel.pack(fill="x", pady=2)
        
        # [NEW] Toggle Button (Right Side)
        self.btn_log = ttk.Button(ctrl_panel, text="Show Log", command=self.toggle_log, width=10)
        self.btn_log.pack(side="right", padx=5)
        if self.use_sentiment:
            ttk.Label(ctrl_panel, text="Active Model:").pack(side="left")
        
            self.model_var = tk.StringVar(value="FinBERT")

            self.lbl_model_status = ttk.Label(ctrl_panel, text="Status: Init...", foreground="orange")
            self.lbl_model_status.pack(side="left", padx=10)

        # [MODIFIED] Create widgets but DO NOT pack them yet (Hidden by default)
        self.log_box = tk.Text(log_frame, height=6, font=("Consolas", 9), 
                               bg="#1e1e1e", fg="#00ff00", # Matrix Green Text
                               insertbackground="white") # Cursor color
        self.log_scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box['yscrollcommand'] = self.log_scroll.set
        
        # Track visibility state
        self.log_visible = False
        
        
        self.current_ticker = None
        self.stock = None  # <--- NEW: Store the object here
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
            #self.lbl_model_status.config(text="Status: Disabled", foreground="gray")
        self.scan_data = []
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(500, self.load_data)
    def on_close(self):
        """Force kills the application and all background threads."""
        print("Shutting down Sentinel...")
        try:
            self.root.destroy()
        except Exception:
            pass
        
        # Hard exit to kill any lingering threads (like the AI or Scanner)
        import os
        os._exit(0)

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
            q = ttk.Label(f, text="?", foreground="orange", cursor="question_arrow", padding=(4, 0))
            q.pack(side="left")
            Tooltip(q, tooltip_text)
        lbl = ttk.Label(parent, text="---", font=("Arial", 10))
        lbl.grid(row=row, column=1, sticky="e", padx=10, pady=5)
        return lbl

    def load_data(self):
        """Refreshes chart data and updates fundamentals only on ticker change."""
        new_ticker = self.entry_ticker.get().upper().strip()
        if not new_ticker: return

        # Only reload the Heavy Ticker Object if the symbol changed
        if new_ticker != self.current_ticker:
            self.current_ticker = new_ticker
            
            # Use the shared session for the ticker
            self.stock = yf.Ticker(new_ticker)
            
            # Clear caches and reset drift for the new stock
            self.data_cache = {} 
            self.sent_cache = {}
            self.annual_growth_offset = None 
            self.projected_earnings = []
            
            self.lbl_pe.config(text="Updating...", foreground="orange")
            
            # Start background fundamental fetch
            threading.Thread(target=self.get_info, daemon=True).start()
            self.log(f"Ticker changed: {new_ticker}. Session reused.")

        # Always refresh the chart (light logic)
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
        self.log(f"Fetching Google RSS for {ticker} (Rich Data)...")
        news_items = []
        seen_titles = set()
        
        queries = [
            f"{ticker}+stock",
            f"{ticker}+financial+news"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for q in queries:
            try:
                url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                resp = requests.get(url, headers=headers, timeout=5, verify=False) 
                
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    items = root.findall('.//item')
                    for item in items:
                        title = item.find('title').text
                        link = item.find('link').text
                        pub_date_str = item.find('pubDate').text
                        
                        # --- HTML CLEANUP (The Fix) ---
                        raw_desc = item.find('description').text or ""
                        
                        # 1. Remove all HTML tags (<a href...>, </a>, <font...>)
                        clean_desc = re.sub(r'<[^>]+>', '', raw_desc)
                        
                        # 2. Fix weird symbols (&nbsp; -> space, &amp; -> &)
                        clean_desc = html.unescape(clean_desc)
                        
                        # 3. Clean up whitespace
                        clean_desc = " ".join(clean_desc.split())
                        # ------------------------------

                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            try:
                                dt = pd.to_datetime(pub_date_str)
                            except:
                                dt = datetime.now()

                            news_items.append({
                                'title': title,
                                'link': link,
                                'published': dt,
                                'summary': clean_desc,
                                'source': 'Google RSS'
                            })
                    
                    if len(news_items) >= self.headline_limit:
                        break
                        
            except Exception as e:
                 self.log(f"RSS Variation Error ({q}): {e}")
        
        return news_items

    def calculate_sentiment(self, ticker, stock_obj):
        # 1. Check Cache
        if ticker in self.sent_cache:
            cache_data = self.sent_cache[ticker]
            if len(cache_data) == 3:
                val, news_items, ts = cache_data
                if time.time() - ts < self.SENT_CACHE_DURATION:
                    self.log("Using Cached News.")
                    return val, news_items

        # 2. Gather Headlines
        all_news = []
        
        # A. Yahoo News (Best Quality)
        try:
            ynews = stock_obj.news
            if ynews:
                for n in ynews:
                    title = n.get('title') or n.get('headline') or ""
                    if not title.strip(): continue
                    
                    ts = n.get('providerPublishTime', time.time())
                    dt = datetime.fromtimestamp(ts)
                    summary = n.get('summary') or f"Source: {n.get('publisher', 'Yahoo')}"
                    
                    all_news.append({
                        'title': title,
                        'link': n.get('link', ''),
                        'published': dt,
                        'summary': summary,
                        'source': 'Yahoo'
                    })
        except Exception as e:
            self.log(f"Yahoo news error: {e}")

        # B. Google RSS (Backup)
        if len(all_news) < 5:
            google_news = self.get_google_news_rss(ticker)
            all_news.extend(google_news)

        if not all_news:
            return None, []

        # 3. Sort: Newest First
        all_news.sort(key=lambda x: x['published'], reverse=True)
        all_news = all_news[:self.headline_limit]

        # 4. AI Analysis (Uses Titles)
        avg_score = None
        headlines_for_ai = [x['title'] for x in all_news]
        
        if self.use_sentiment:
            current_model = sentiment_engine.models[sentiment_engine.current_model_name]
            if current_model["loaded"]:
                self.log(f"AI Analyzing {len(headlines_for_ai)} headlines...")
                scores = sentiment_engine.predict_batch(headlines_for_ai)
                if scores:
                    valid_scores = [s for s in scores if isinstance(s, (int, float))]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                        self.log(f"FINAL AI SCORE: {avg_score:.4f}")

        self.sent_cache[ticker] = (avg_score, all_news, time.time())
        return avg_score, all_news

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
    
    def open_news_window(self):
        if not self.current_ticker: return
        
        # Retrieve rich data from cache
        if self.current_ticker not in self.sent_cache:
            messagebox.showinfo("News", "No news loaded yet for this ticker.")
            return

        cache_data = self.sent_cache[self.current_ticker]
        if len(cache_data) == 3:
            _, news_items, _ = cache_data
        else:
            messagebox.showinfo("News", "Old cache format. Please reload data.")
            return
        
        if not news_items:
            messagebox.showinfo("News", "No headlines found.")
            return

        win = Toplevel(self.root)
        win.title(f"News Feed: {self.current_ticker}")
        win.geometry("900x600")
        win.configure(bg="#1e1e1e")

        # Header
        header = ttk.Frame(win)
        header.pack(fill="x", padx=10, pady=10)
        ttk.Label(header, text=f"Latest News ({len(news_items)})", 
                  font=("Arial", 16, "bold"), background="#1e1e1e", foreground="white").pack(side="left")
        ttk.Label(header, text="(Double-click to read)", 
                  font=("Arial", 10), background="#1e1e1e", foreground="gray").pack(side="left", padx=10, pady=(5,0))

        # Treeview
        columns = ("Date", "Source", "Headline")
        tree = ttk.Treeview(win, columns=columns, show="headings", height=20)
        
        tree.heading("Date", text="Date")
        tree.heading("Source", text="Source")
        tree.heading("Headline", text="Headline")
        
        tree.column("Date", width=120, anchor="center")
        tree.column("Source", width=100, anchor="center")
        tree.column("Headline", width=600, anchor="w")
        
        scr = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scr.set)
        
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scr.pack(side="right", fill="y", pady=10)

        # Styles
        tree.tag_configure('odd', background='#252526', foreground='white')
        tree.tag_configure('even', background='#333333', foreground='white')

        # Insert Data (Already sorted)
        for i, item in enumerate(news_items):
            tag = 'even' if i % 2 == 0 else 'odd'
            # Format date for display
            date_str = item['published'].strftime("%Y-%m-%d %H:%M")
            tree.insert("", "end", iid=i, values=(date_str, item['source'], item['title']), tags=(tag,))

        # Bind Double Click
        def on_double_click(event):
            item_id = tree.selection()[0]
            news_obj = news_items[int(item_id)]
            self.view_news_content(news_obj)
            
        tree.bind("<Double-1>", on_double_click)

    def view_news_content(self, news_item):
        """Opens a pane to read the selected news item."""
        reader = Toplevel(self.root)
        reader.title("News Reader")
        reader.geometry("600x450")
        reader.configure(bg="#1e1e1e")

        # Headline
        tk.Label(reader, text=news_item['title'], font=("Arial", 14, "bold"), 
                 bg="#1e1e1e", fg="white", wraplength=550, justify="left").pack(pady=15, padx=15, anchor="w")

        # Metadata
        meta = tk.Frame(reader, bg="#1e1e1e")
        meta.pack(fill="x", padx=15)
        tk.Label(meta, text=f"{news_item['source']}  •  {news_item['published']}", 
                 bg="#1e1e1e", fg="#00e6ff", font=("Arial", 9)).pack(side="left")

        # Summary Box
        tk.Label(reader, text="Snippet:", bg="#1e1e1e", fg="gray", anchor="w").pack(fill="x", padx=15, pady=(20, 5))
        
        text_box = tk.Text(reader, height=10, bg="#252526", fg="#dddddd", 
                           font=("Segoe UI", 11), wrap="word", relief="flat", padx=10, pady=10)
        
        # If the summary is just the title repeated (Google behavior), show a helpful message
        display_text = news_item.get('summary', '')
        if len(display_text) < 10 or display_text == news_item['title']:
            display_text = "No detailed summary available. Please read the full article below."
            
        text_box.insert("1.0", display_text)
        text_box.config(state="disabled") # Read-only
        text_box.pack(fill="both", expand=True, padx=15, pady=5)

        # Button
        btn_frame = tk.Frame(reader, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=20, padx=15)
        
        def open_link():
            if news_item['link']:
                webbrowser.open(news_item['link'])
            else:
                messagebox.showerror("Error", "No link found.")

        # Large, clear button
        btn = tk.Button(btn_frame, text="🌐  Open Full Article in Browser", command=open_link,
                        bg="#007acc", fg="white", font=("Arial", 11, "bold"), 
                        relief="flat", pady=8, cursor="hand2")
        btn.pack(fill="x")
    def fetch_and_plot(self, ticker, period, interval):
        try:
            # 1. Use the shared stock object instead of recreating it
            stock = self.stock 
            
            # Chart Data - check cache first
            df, is_cached = self.get_cached_df(ticker, period, interval)
            
            # --- Earnings Cycle Detection (Optimized) ---
            # We only need to fetch the calendar once per ticker change.
            # If self.projected_earnings is already populated, we skip this.
            if not self.projected_earnings:
                try:
                    cal = stock.calendar
                    anchor_date = None
                    if isinstance(cal, dict) and 'Earnings Date' in cal:
                        dates = cal['Earnings Date']
                        if dates: anchor_date = pd.to_datetime(dates[0]).date()
                    elif cal is not None and not cal.empty:
                        anchor_date = pd.to_datetime(cal.iloc[0].values[0]).date()

                    if anchor_date:
                        self.projected_earnings = [anchor_date]
                        for i in range(1, 4):
                            self.projected_earnings.append(anchor_date + timedelta(days=91*i))
                        self.log(f"Earnings Cycle Detected: {self.projected_earnings}")
                except Exception as e:
                    self.log(f"Earnings fetch skipped: {e}")

            # --- Main Price Data Fetching ---
            if df is None:
                df = self.fetch_history_with_retry(stock, period, interval)
                if df.empty: 
                    self.log("No price data found.")
                    self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                    return
                
                # Vectorized EMA calculations
                for span in [5, 21, 63, 200]:
                    df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
                
                self.save_df_cache(ticker, period, interval, df)
                status_msg = f"Live Data ({interval})"
            else:
                status_msg = f"Cached Data ({interval})"
            
            # Calculate Period Return
            period_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] if not df.empty else 0.0
            self.root.after(0, lambda: self.lbl_status.config(text=status_msg))

            # --- 2. [REVISED] Earnings-Based Growth Drift ---
            if self.annual_growth_offset is None:
                try:
                    # Try to get fundamental growth (EPS growth)
                    info = stock.info
                    
                    # FIX: Check if info exists and is a dictionary before calling .get()
                    f_growth = None
                    if isinstance(info, dict):
                        f_growth = info.get('earningsGrowth')
                    
                    if f_growth is not None:
                        # Clamp the earnings growth to realistic drift expectations (e.g., -5% to +15%)
                        self.annual_growth_offset = max(min(f_growth * 0.5, 0.15), -0.05)
                        self.log(f"Fundamental EPS Growth Drift: {self.annual_growth_offset:+.2%}")
                    else:
                        # Fallback to your original Price CAGR logic if info is None or missing growth
                        df_5y = self.fetch_history_with_retry(stock, "5y", "1mo")
                        if df_5y is not None and not df_5y.empty and len(df_5y) > 12:
                            p_start, p_end = df_5y['Close'].iloc[0], df_5y['Close'].iloc[-1]
                            cagr = ((p_end / p_start) ** (1/5)) - 1
                            self.annual_growth_offset = max(min(cagr * 0.5, 0.15), -0.05)
                            self.log(f"Fallback 5Y Price CAGR Drift: {self.annual_growth_offset:+.2%}")
                        else:
                            self.annual_growth_offset = 0.05
                            
                except Exception as e:
                    self.log(f"Growth calculation error: {e}")
                    self.annual_growth_offset = 0.05

            # --- 3. Technical Data & GARCH (1y Daily) ---
            df_tech, tech_cached = self.get_cached_df(ticker, "1y", "1d")
            if df_tech is None:
                df_tech = self.fetch_history_with_retry(stock, "1y", "1d")
                df_tech = calculate_technicals(df_tech)
                df_tech['log_ret'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1))
                self.save_df_cache(ticker, "1y", "1d", df_tech)
            
            # --- NEW CODE: Prioritize Real-Time Tick Price ---
            last = df_tech.iloc[-1]
            
            # 1. Try to get the absolute latest tick from fast_info
            real_time_price = None
            try:
                # fast_info is lighter/faster than .info and usually has the latest metadata
                real_time_price = stock.fast_info['last_price']
            except Exception:
                pass

            # 2. Assign Current Price (Priority: Fast Info -> Intraday Chart -> Daily Cache)
            if real_time_price and not math.isnan(real_time_price):
                current_price = real_time_price
            elif not df.empty:
                current_price = df['Close'].iloc[-1]
            else:
                current_price = last['Close']

            # Update the class variable so the Option Scanner sees the real price
            self.current_price = current_price

            # Volatility Logic
            hv_30 = df_tech['log_ret'].rolling(30).std().iloc[-1] * np.sqrt(252)
            self.hv_30 = hv_30
            self.garch_vol = VegaChimpCore.garch_forecast(df_tech['log_ret'].dropna().values)
            
            # --- 4. Sentiment Analysis ---
            sentiment_result = self.calculate_sentiment(ticker, stock)
            
            sentiment_score = None
            headlines = []
            
            if sentiment_result:
                sentiment_score, headlines = sentiment_result

            # Enable News Button if we have headlines
            if headlines:
                self.root.after(0, lambda: self.btn_news.config(state="normal"))
            else:
                self.root.after(0, lambda: self.btn_news.config(state="disabled"))
            # --- 5. UI Updates ---
            last_copy = last.copy()
            last_copy['Close'] = current_price 
            
            self.root.after(0, lambda: self.update_technicals(last_copy, hv_30, self.garch_vol, sentiment_score, period_return))
            self.root.after(0, self.update_chart, df, ticker, period)

        except Exception as e:
            self.log(f"CRITICAL ERROR in fetch_and_plot: {e}")
            self.root.after(0, lambda: self.lbl_status.config(text="Error"))
            
    def setup_dark_theme(self):
        # 1. Main Window & Common Backgrounds
        dark_bg = "#1e1e1e"
        dark_fg = "#ffffff"
        entry_bg = "#2d2d2d"
        
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam' is easiest to recolor
        
        # 2. Configure General Widgets
        self.style.configure(".", background=dark_bg, foreground=dark_fg)
        self.style.configure("TLabel", background=dark_bg, foreground=dark_fg)
        self.style.configure("TButton", background="#333333", foreground=dark_fg, borderwidth=1)
        self.style.map("TButton", background=[("active", "#ff8c00")]) # Orange highlight on hover
        
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=dark_fg)
        self.style.configure("TFrame", background=dark_bg)
        self.style.configure("TLabelframe", background=dark_bg, foreground=dark_fg)
        self.style.configure("TLabelframe.Label", background=dark_bg, foreground=dark_fg)
        
        # 3. Configure Treeview (The Scanner)
        self.style.configure("Treeview", 
                             background="#252526", 
                             foreground=dark_fg, 
                             fieldbackground="#252526",
                             rowheight=25)
        self.style.map("Treeview", background=[("selected", "#007acc")])
        
        # Header (Column Titles)
        self.style.configure("Treeview.Heading", 
                             background="#333333", 
                             foreground=dark_fg, 
                             relief="flat")
        self.style.map("Treeview.Heading", background=[("active", "#4d4d4d")])

        # 4. Standard Tkinter Widgets (Text, Listbox need manual config)
        self.root.configure(bg=dark_bg)
        # We also need to configure the specific widgets created in __init__
        # (See Step 2 below for where to apply this)        
            
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

            # --- Intraday Market Hours Filtering ---
            if interval in intraday:
                idx = plot_df.index
                if getattr(idx, "tz", None):
                    idx_eastern = idx.tz_convert("America/New_York")
                else:
                    idx_eastern = idx.tz_localize("UTC").tz_convert("America/New_York")
                minutes = idx_eastern.hour * 60 + idx_eastern.minute
                # Filter 9:30 AM (570) to 4:00 PM (960)
                mask = (minutes >= 570) & (minutes <= 960) & (idx_eastern.dayofweek < 5)
                filtered = plot_df[mask]
                if not filtered.empty:
                    plot_df = filtered

            # --- X-Axis Logic ---
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

            # --- PLOTTING & STYLING ---
            self.ax.clear()
            self.hover_annot = None 

            # 1. Neon Colors
            # Price: Neon Cyan
            self.ax.plot(x_vals, plot_df['Close'], label='Price', color='#00e6ff', linewidth=1.5)
            
            # EMAs: Neon Pink, Yellow, Red
            if 'EMA_5' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_5'], label='EMA 5', color='#ff00ff', linewidth=1, alpha=0.8)
            if 'EMA_21' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_21'], label='EMA 21', color='#ffe100', linewidth=1, alpha=0.8)
            if 'EMA_63' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_63'], label='EMA 63', color='#9900ff', linewidth=1, alpha=0.8) # Purple
            
            if 'EMA_200' in plot_df.columns:
                if plot_df['EMA_200'].notna().sum() > 0:
                    self.ax.plot(x_vals, plot_df['EMA_200'], label='EMA 200', color='#ff3333', linewidth=1.5)

            # 2. Limits & Title
            self.ax.set_xlim(left=x_vals[0], right=x_vals[-1])
            self.ax.set_title(f"{ticker} Price Action ({period})", color="white", fontweight="bold")
            
            # 3. Clean Legend (No box, white text)
            self.ax.legend(loc='upper right', fontsize='small', frameon=False, labelcolor='white')
            
            # 4. Subtle Grid & Minimal Spines
            self.ax.grid(True, color='#2a2a2a', linestyle='-', linewidth=0.5)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['bottom'].set_color('#444444')
            self.ax.spines['left'].set_color('#444444')
            self.ax.tick_params(axis='x', colors='gray')
            self.ax.tick_params(axis='y', colors='gray')

            # --- Ticks Formatting ---
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
            
            # --- Finalize ---
            # Force background color (Fix for clear() resetting it)
            self.ax.set_facecolor('#121212')
            self.figure.patch.set_facecolor('#121212')

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
        rsi_c = "green" if rsi_val < 30 else "red" if rsi_val > 70 else "white"
        self.lbl_rsi.config(text=f"{rsi_val:.2f}", foreground=rsi_c)
        
        stoch_val = data['StochRSI']
        stoch_c = "green" if stoch_val < 0.2 else "red" if stoch_val > 0.8 else "white"
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
                p_color = "red" if p_val > 80 else "green" if p_val < 20 else "white"
                self.lbl_pe_percentile.config(text=f"{p_val:.1f}%", foreground=p_color)
        except Exception as e:
            self.log(f"P/E Fetch Error: {e}")
            self.lbl_pe.config(text="N/A", foreground="gray")
        
        bb_pos = "Inside"
        bb_c = "white"
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
                    sent_c = "red" if val < 0.4 else "green" if val > 0.6 else "white"
                    self.lbl_sent.config(text=f"{val:.2f}", foreground=sent_c)
                except (ValueError, TypeError):
                    self.lbl_sent.config(text="N/A", foreground="gray")
            else:
                self.lbl_sent.config(text="N/A", foreground="gray")
        
        ret_c = "green" if period_return > 0 else "red" if period_return < 0 else "white"
        self.lbl_return.config(text=f"{period_return:+.2%}", foreground=ret_c)

        self.btn_opt.config(state="normal", text=f"🔎 Open {self.current_ticker} Option Scanner")
    # ================= 3D VISUALIZATION METHODS =================
    def visualize_3d(self, option_type):
        """Generates a rotatable 3D Scatter plot of Date vs Strike vs EV."""
        if not hasattr(self, 'scan_data') or not self.scan_data:
            messagebox.showinfo("3D Plot", "No data to plot. Please run a Scan first.")
            return

        # 1. Filter Data
        filtered = [row for row in self.scan_data if row['type'] == option_type]
        
        if not filtered:
            messagebox.showinfo("3D Plot", f"No {option_type} data found.")
            return

        # 2. Extract Axis Data
        dates_x = []       # Numeric (Days to expiry) for plotting
        date_labels = []   # String ("2025-01-01") for Tooltip
        strikes = []
        evs = []
        
        today = datetime.now()
        
        for row in filtered:
            try:
                # X: Days to Expiry (Float)
                dt = datetime.strptime(row['date'], "%Y-%m-%d")
                days_diff = (dt - today).days
                
                # Y: Strike
                strike = float(row['strike'])
                
                # Z: EV
                ev = float(row['ev'])
                
                dates_x.append(days_diff)
                date_labels.append(row['date']) # <--- Store the text date
                strikes.append(strike)
                evs.append(ev)
            except ValueError:
                continue

        if not dates_x: return

        # 3. Create Window
        vis_win = Toplevel(self.root)
        vis_win.title(f"3D Surface: {self.current_ticker} {option_type}s")
        vis_win.geometry("900x700")
        vis_win.configure(bg="#1e1e1e")

        # 4. Create Matplotlib 3D Plot
        fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="#1e1e1e")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#1e1e1e")
        
        # Color Map
        try:
            divnorm = mcolors.TwoSlopeNorm(vmin=min(evs), vcenter=0., vmax=max(evs))
        except ValueError:
            divnorm = plt.Normalize(vmin=min(evs), vmax=max(evs))
        
        sc = ax.scatter(dates_x, strikes, evs, c=evs, cmap='RdYlGn', norm=divnorm, marker='o', s=40, edgecolors='black', linewidth=0.5)

        # Labels
        ax.set_xlabel('Days to Expiry', color='white', labelpad=10)
        ax.set_ylabel('Strike Price', color='white', labelpad=10)
        ax.set_zlabel('Expected Value (EV)', color='white', labelpad=10)
        ax.set_title(f"{self.current_ticker} {option_type} EV Landscape", color='white', fontsize=14)

        # Axis Colors
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        
        # Transparent grid
        ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('EV Profitability', color='white')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=vis_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # 5. Save Button (Interactive HTML)
        btn_frame = ttk.Frame(vis_win)
        btn_frame.pack(fill="x", pady=10)
        
        # Pass the date_labels to the save function
        def save_html_action():
            self.save_3d_html(option_type, dates_x, strikes, evs, date_labels)
            
        ttk.Button(btn_frame, text="沈 Save Interactive 3D HTML (With Hover Tooltips)", command=save_html_action).pack(fill="x", padx=50)

        # Instructions
        ttk.Label(btn_frame, text="* Matplotlib (Window): Rotate/Zoom Only", foreground="gray").pack(pady=0)
        ttk.Label(btn_frame, text="* HTML Export: Hover to see Strike/Date/EV", foreground="#00e6ff").pack(pady=(0,5))

    def save_3d_html(self, option_type, dates, strikes, evs, date_labels):
        """Saves the current data as an interactive HTML using Plotly."""
        
        # Detailed error if library is missing
        if not PLOTLY_AVAILABLE:
            messagebox.showerror("Missing Library", "Plotly is not installed.\n\nTo enable HTML export with tooltips, run:\npip install plotly")
            return

        filename = filedialog.asksaveasfilename(
            initialfile=f"{self.current_ticker}_{option_type}_3D_Analysis.html",
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html")]
        )
        if not filename: return

        try:
            # Create the custom hover text list
            hover_texts = []
            for d_str, stk, val in zip(date_labels, strikes, evs):
                # HTML formatting for the tooltip
                txt = (f"<b>Date:</b> {d_str}<br>"
                       f"<b>Strike:</b> ${stk}<br>"
                       f"<b>EV:</b> {val:+.2f}")
                hover_texts.append(txt)

            fig = go.Figure(data=[go.Scatter3d(
                x=dates,
                y=strikes,
                z=evs,
                mode='markers',
                marker=dict(
                    size=5,
                    color=evs,                
                    colorscale='RdYlGn', 
                    opacity=0.9,
                    showscale=True,
                    colorbar=dict(title="EV Profit")
                ),
                text=hover_texts, # <--- Attach the custom text here
                hoverinfo="text"  # <--- Tell Plotly to use our text
            )])

            fig.update_layout(
                title=f"{self.current_ticker} {option_type} Option Surface (EV)",
                scene=dict(
                    xaxis_title='Days to Expiry',
                    yaxis_title='Strike Price',
                    zaxis_title='Expected Value (EV)',
                    bgcolor='#1e1e1e',
                    xaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                    yaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                    zaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                ),
                paper_bgcolor='#1e1e1e',
                font=dict(color="white"),
                margin=dict(l=0, r=0, b=0, t=40)
            )

            plot(fig, filename=filename, auto_open=True)
            self.log(f"Saved HTML to {filename}")

        except Exception as e:
            # Catch errors (like permission denied, or data issues)
            self.log(f"HTML Export Error: {e}")
            messagebox.showerror("Export Error", f"Failed to generate HTML.\n\nError: {e}")
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
                    text, 
                    xy=(xval, yval), 
                    xytext=(10, 10),      # Reduced offset (closer to cursor)
                    textcoords="offset points",
                    color="white",        # Keep your Dark Mode text color
                    fontsize=8,           # <--- SMALLER FONT (Default is ~10)
                    fontweight="bold",
                    bbox=dict(
                        # <--- SMALLER PADDING (pad=0.3 makes the box tighter)
                        boxstyle="round,pad=0.3", 
                        fc="#252526", 
                        ec="#00e6ff", 
                        alpha=0.9
                    ),
                    arrowprops=dict(arrowstyle="->", color="#00e6ff")
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
        try:
            stock = self.stock
            info = stock.info
            
            # 1. Basic Fundamental Extraction
            self.pe_fwd = info.get('forwardPE')
            self.pe_ttm = info.get('trailingPE')
            self.eps = info.get('trailingEps')
            self.dividend_yield = info.get('dividendYield', 0.0)
            
            # 2. TRIGGER PERCENTILE CALCULATION (The Missing Link)
            # This calls the logic that compares current P/E to 5Y history
            self.calculate_valuation_multiplier(stock)
            
            # 3. Force UI update now that data is ready
            self.root.after(0, self.update_pe_display)
            
        except Exception as e:
            self.log(f"Fundamental fetch error: {e}")
            self.root.after(0, self.update_pe_display)
            
    def calculate_valuation_multiplier(self, ticker_obj):
        try:
            # 1. Safety Checks
            if not self.pe_fwd or not self.eps or self.eps <= 0:
                self.current_z_score = 0
                return 1.0
            
            # 2. Get 5y History
            hist = ticker_obj.history(period="5y")
            if hist.empty: return 1.0
            
            # 3. Calculate P/E Series (Filter valid only)
            pe_series = (hist['Close'] / self.eps)
            pe_series = pe_series[pe_series > 0]
            if pe_series.empty: return 1.0
            
            # Calculate percentage of historical P/Es that are strictly lower than current Forward P/E
            if self.pe_fwd:
                self.pe_percentile = (pe_series < self.pe_fwd).mean() * 100
                print(f"[DEBUG] P/E Percentile: {self.pe_percentile:.1f}%")
            else:
                self.pe_percentile = 'N/A'
        except Exception as e:
            self.log(f"P/E Percentile error: {e}")
            return 1.0

    def refresh_valuation(self):
        """Called when the Median/Mean toggle is clicked."""
        if not self.current_ticker: return
        self.log(f"Switching Valuation Mode... (Median={self.use_median.get()})")
        
        # Re-trigger get_info to recalculate Z-score and update labels
        threading.Thread(target=self.get_info, daemon=True).start()
        
    def get_smart_dividend(self, stock_obj):
        """
        Retrieves the dividend yield using a high-performance priority queue:
        1. fast_info (Fastest, lightest)
        2. info (Standard, slower)
        3. Manual calculation from history (Fallback)
        """
        try:
            # 1. High-Performance Priority: fast_info (Milliseconds)
            # This is significantly faster as it avoids scraping the full JSON blob.
            fast_div = stock_obj.fast_info.get('dividend_yield')
            if fast_div is not None and isinstance(fast_div, (int, float)):
                # fast_info usually returns decimal (e.g., 0.0294)
                print(f"[DEBUG] Found Dividend (fast_info): {fast_div:.4%}")
                return fast_div

            # 2. Standard Ticker Info (Second Priority)
            info = stock_obj.info
            div = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
            div = div / 100
            print(f"[DEBUG] Found Dividend (info): {div:.4%}") 
            return div
            
        except Exception as e:
            print(f"[DEBUG] Div fetch error: {e}")
            return 0.0

    def open_options_window(self):
        if not self.current_ticker: return
        win = Toplevel(self.root)
        win.title(f"Options Explorer: {self.current_ticker}")
        win.geometry("1200x800")
        win.configure(bg="#1e1e1e") # <--- Dark Background for Pop-up

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
        
        # --- INSERT THIS 3D BLOCK HERE ---
        viz_frame = ttk.LabelFrame(left_panel, text="3D Visualizer", padding=5)
        viz_frame.pack(fill="x", pady=20)
        
        ttk.Button(viz_frame, text="3D Plot (CALLS)", command=lambda: self.visualize_3d("CALL")).pack(fill="x", pady=2)
        ttk.Button(viz_frame, text="3D Plot (PUTS)", command=lambda: self.visualize_3d("PUT")).pack(fill="x", pady=2)
        # ---------------------------------
        
        ttk.Button(left_panel, text="💾 Export Results to CSV", command=self.export_to_csv).pack(fill="x", pady=5)
          
        self.exp_list = tk.Listbox(left_panel, selectmode="extended", height=25,
                                   bg="#252526", fg="white", highlightthickness=0)
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
        
        # Green (Undervalued): "Sage Green" 
        # Old: #d4f8d4 (Too bright)
        self.tree.tag_configure("green", background="#8fbc8f", foreground="black")
        
        # Arbitrage breakevev < price
        self.tree.tag_configure("gold", background="#ffd700", foreground="black")
    
        # Red (Overvalued): "Muted Salmon"
        # Old: #f8d4d4 (Too bright)
        self.tree.tag_configure("red", background="#e57373", foreground="black")   
    
        # Blue (Info/Neutral): "Steel Blue"
        # Old: #d4eef8 (Too bright)
        self.tree.tag_configure("blue", background="#90caf9", foreground="black")
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
        stock = self.stock
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
        
    def update_pe_display(self):
        """Updates only the P/E labels. Called when get_info finishes."""
        try:
            # Update TTM/Fwd Label
            pe_ttm_str = f"{self.pe_ttm:.2f}" if isinstance(self.pe_ttm, (int, float)) else "N/A"
            pe_fwd_str = f"{self.pe_fwd:.2f}" if isinstance(self.pe_fwd, (int, float)) else "N/A"
            self.lbl_pe.config(text=f"TTM: {pe_ttm_str} | Fwd: {pe_fwd_str}", foreground="white")

            # Update Percentile Label
            if hasattr(self, 'pe_percentile'):
                p_val = self.pe_percentile
                p_color = "red" if p_val > 80 else "green" if p_val < 20 else "white"
                self.lbl_pe_percentile.config(text=f"{p_val:.1f}%", foreground=p_color)
            else:
                self.lbl_pe_percentile.config(text="Loading...", foreground="orange")
                
        except Exception as e:
            self.log(f"UI Update Error: {e}")

    def fetch_options_batch(self, dates, filter_under_only=False):
        # Clear scan data if this is a fresh batch (optional logic, but safe)
        if hasattr(self, 'scan_data') and len(dates) == 1: 
             # Only clear if checking specific dates (manual selection), 
             # for "Scan All" we usually clear before calling this.
             pass 
            
        stock = self.stock
        
        # 1. PURE INPUTS
        DIV_YIELD = self.get_smart_dividend(stock) 
        RFR = self.get_risk_free_rate() 
        
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
                
                # Get ALL options (no .head limit)
                all_options = pd.concat([calls, puts]).sort_values('volume', ascending=False)
                
                for _, row in all_options.iterrows():
                    # Get volume safely
                    vol = row.get('volume', 0)
                    
                    # Check if it is NaN (empty) or Zero
                    if pd.isna(vol) or vol == 0: 
                        continue
                    bid, ask, last = row.get('bid', 0), row.get('ask', 0), row['lastPrice']
                    if bid > 0 and ask > 0:
                        market_price = (bid + ask) / 2
                    elif last > 0:
                        market_price = last
                    else:
                        continue 
                        
                    # --- VOLATILITY SANITY CHECK ---
                    iv = row['impliedVolatility']
                    if not iv or math.isnan(iv) or iv < 0.01: continue

                    # Fallback to HV if Yahoo IV is broken (< 20%)
                    if iv < 0.20:
                        vol_input = self.hv_30
                    else:
                        vol_input = iv 

                    adjusted_rfr = RFR 

                    fair = VegaChimpCore.bjerksund_stensland(
                        self.current_price, 
                        row['strike'], 
                        T, 
                        adjusted_rfr,
                        DIV_YIELD,  
                        vol_input, 
                        row['Type'].lower()
                    )
                    ev = fair - market_price
                    
                    # --- SAVE DATA FOR 3D PLOTTER ---
                    if not hasattr(self, 'scan_data'): self.scan_data = []
                    self.scan_data.append({
                        'date': date,
                        'type': row['Type'],
                        'strike': row['strike'],
                        'ev': ev,
                        'price': market_price,
                        'vol': row['volume']
                    })
                    # -------------------------------

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
                        base_min_edge = 10.0 if is_earnings else 5.0

                        if edge_percent > base_min_edge and ev > 0.05:
                            verdict = f"Under ({edge_percent:.0f}%)"
                            tag = "green"
                        elif edge_percent < -base_min_edge and ev < -0.05:
                            verdict = "Over"
                            tag = "red"
                    
                    if filter_under_only and "Under" not in verdict and "Arbitrage" not in verdict:
                        continue

                    vals = (date, row['Type'], row['strike'], int(row['volume'] or 0), 
                            f"{market_price:.2f}", f"{breakeven:.2f}", f"{iv:.1%}", 
                            f"{fair:.2f}", f"{ev:+.2f}", verdict)
                    
                    self.root.after(0, lambda v=vals, t=tag: self.tree.insert("", "end", values=v, tags=(t,)))
            
            except Exception as e:
                self.log(f"Options fetch error for {date}: {e}")

if __name__ == "__main__":
    try:
        import pyi_splash
        if pyi_splash.is_alive():
            pyi_splash.close()
    except ImportError:
        pass
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()