import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Charting Libraries ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
            "DistilBERT": {
                "id": "distilbert-base-uncased-finetuned-sst-2-english",
                "dir": "my_distilbert_model",
                "loaded": False,
                "tokenizer": None,
                "model": None
            },
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
        
        try:
            print(f"[System] Loading {model_key}...")
            if os.path.exists(local_path) and os.listdir(local_path):
                target["tokenizer"] = AutoTokenizer.from_pretrained(local_path)
                target["model"] = AutoModelForSequenceClassification.from_pretrained(local_path)
            else:
                print(f"[System] Downloading {model_key} (First Run)...")
                target["tokenizer"] = AutoTokenizer.from_pretrained(target["id"])
                target["model"] = AutoModelForSequenceClassification.from_pretrained(target["id"])
                
                print(f"[System] Saving {model_key} locally...")
                target["tokenizer"].save_pretrained(local_path)
                target["model"].save_pretrained(local_path)
            
            target["loaded"] = True
            self.current_model_name = model_key
            self.status_msg = f"{model_key} Loaded."
            return True

        except Exception as e:
            self.status_msg = f"Failed to load {model_key}: {e}"
            print(f"[Error] {self.status_msg}")
            return False

    def predict(self, text):
        target = self.models[self.current_model_name]
        if not target["loaded"] or not text or not text.strip():
            return 0.5 

        try:
            inputs = target["tokenizer"](text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = target["model"](**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)
            
            if self.current_model_name == "DistilBERT":
                # Index 1 is Positive
                return probs[0][1].item()
            elif self.current_model_name == "FinBERT":
                # Index 0=Pos, 1=Neg, 2=Neu
                pos = probs[0][0].item()
                neg = probs[0][1].item()
                # Balance around 0.5
                return 0.5 + (pos * 0.5) - (neg * 0.5)
                
        except Exception as e:
            print(f"[Model Error] {e}")
            return 0.5

    def predict_batch(self, texts):
        """Batch inference for better throughput."""
        target = self.models[self.current_model_name]
        if not target["loaded"]:
            return [0.5 for _ in texts]
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return [0.5 for _ in texts]
        try:
            inputs = target["tokenizer"](clean_texts, return_tensors="pt", truncation=True,
                                          padding=True, max_length=128)
            with torch.no_grad():
                outputs = target["model"](**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            if self.current_model_name == "DistilBERT":
                scores_clean = probs[:, 1].tolist()
            elif self.current_model_name == "FinBERT":
                pos = probs[:, 0]
                neg = probs[:, 1]
                scores_clean = (0.5 + (pos * 0.5) - (neg * 0.5)).tolist()
            else:
                scores_clean = [0.5 for _ in clean_texts]
            full_scores = []
            idx = 0
            for t in texts:
                if t and t.strip():
                    full_scores.append(scores_clean[idx])
                    idx += 1
                else:
                    full_scores.append(0.5)
            return full_scores
        except Exception as e:
            print(f"[Model Error] {e}")
            return [0.5 for _ in texts]

sentiment_engine = SentimentEngine()

# ===================== 2. Math Core =====================
class VegaChimpCore:
    @staticmethod
    def N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    @staticmethod
    def bs_price(S, K, r, q, sig, T, kind):
        if sig <= 0 or T <= 0 or S <= 0 or K <= 0: return 0.0
        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        disc = math.exp(-r * T); disc_q = math.exp(-q * T)
        if kind == "call": return S * disc_q * VegaChimpCore.N(d1) - K * disc * VegaChimpCore.N(d2)
        return K * disc * VegaChimpCore.N(-d2) - S * disc_q * VegaChimpCore.N(-d1)

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


# Lightweight tooltip helper for hover explanations
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
        self.root.title("Technical Command Center V18 (Fix: Chart Crash)")
        self.root.geometry("1100x950")

        # Configurable cap on headlines scored for sentiment
        self.headline_limit = 20

        self.data_cache = {}
        self.DATA_CACHE_DURATION = 60 
        self.sent_cache = {}
        self.SENT_CACHE_DURATION = 1800 
        
        # Initialize Chart Variable EARLY to prevent AttributeError
        self.ax = None 

        # --- Input Frame ---
        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "AMD")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data()) 
        
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")

        # --- Split View ---
        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        self.left_frame = ttk.Frame(self.paned, width=350)
        self.paned.add(self.left_frame, weight=1)

        self.lbl_price = ttk.Label(self.left_frame, text="---", font=("Arial", 28, "bold"))
        self.lbl_price.pack(anchor="center", pady=10)

        # Tech Grid
        self.grid_frame = ttk.LabelFrame(self.left_frame, text="Technical Analysis", padding=15)
        self.grid_frame.pack(fill="x", pady=5)
        
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0,
                         "Relative Strength Index. Range 0-100. Under 30 often signals oversold; over 70 signals overbought. 50 is neutral.")
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI", 1,
                          "Stochastic RSI on a 14-period lookback. Range 0-1. Under 0.2 = oversold, over 0.8 = overbought.")
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2,
                         "MACD = EMA(12) - EMA(26) with a 9-period signal. Positive values imply bullish momentum; negative imply bearish.")
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3,
                        "Bands are 20-day SMA ± 2 standard deviations. Price above upper band suggests overbought; below lower suggests oversold.")
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility)", 4,
                        "Average True Range over 14 periods. Higher ATR = higher daily dollar swings; use for stop sizing.")
        self.lbl_vol = self.add_row(self.grid_frame, "Hist. Vol (30d)", 5,
                        "30-day historical (realized) volatility, annualized. Higher values = choppier price action; compare with implied vol.")
        self.lbl_sent = self.add_row(self.grid_frame, "AI Sentiment", 6,
                         "Headline sentiment scored 0-1 by the selected transformer. Under 0.4 bearish, over 0.6 bullish, mid-range neutral.")

        self.btn_opt = ttk.Button(self.left_frame, text="🔎 Options Explorer", command=self.open_options_window, state="disabled")
        self.btn_opt.pack(fill="x", padx=20, pady=20, ipady=10)

        # Chart Frame
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=3)
        
        ctrl_frame = ttk.Frame(self.right_frame)
        ctrl_frame.pack(fill="x", pady=5)
        
        self.btn_1d = ttk.Button(ctrl_frame, text="1D", command=lambda: self.load_chart("1d", "1m"), width=5)
        self.btn_1d.pack(side="left", padx=2)
        self.btn_5d = ttk.Button(ctrl_frame, text="5D", command=lambda: self.load_chart("5d", "5m"), width=5)
        self.btn_5d.pack(side="left", padx=2)
        self.btn_1m = ttk.Button(ctrl_frame, text="1M", command=lambda: self.load_chart("1mo", "1d"), width=5)
        self.btn_1m.pack(side="left", padx=2)
        self.btn_3m = ttk.Button(ctrl_frame, text="3M", command=lambda: self.load_chart("3mo", "1d"), width=5)
        self.btn_3m.pack(side="left", padx=2)
        self.btn_1y = ttk.Button(ctrl_frame, text="1Y", command=lambda: self.load_chart("1y", "1wk"), width=5)
        self.btn_1y.pack(side="left", padx=2)
        self.btn_5y = ttk.Button(ctrl_frame, text="5Y", command=lambda: self.load_chart("5y", "1mo"), width=5)
        self.btn_5y.pack(side="left", padx=2)

        self.lbl_status = ttk.Label(ctrl_frame, text="", foreground="gray", font=("Arial", 8))
        self.lbl_status.pack(side="right", padx=10)

        # Chart Initialization
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111) # <--- Created here
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- SYSTEM LOG & CONTROLS ---
        log_frame = ttk.LabelFrame(root, text="System Log & AI Controls", padding=5)
        log_frame.pack(fill="x", padx=10, pady=5)

        ctrl_panel = ttk.Frame(log_frame)
        ctrl_panel.pack(fill="x", pady=2)
        ttk.Label(ctrl_panel, text="Active Model:").pack(side="left")
        
        self.model_var = tk.StringVar(value="FinBERT")
        self.combo_model = ttk.Combobox(ctrl_panel, textvariable=self.model_var, 
                                        values=["DistilBERT", "FinBERT"], state="readonly", width=15)
        self.combo_model.pack(side="left", padx=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.change_model)

        self.lbl_model_status = ttk.Label(ctrl_panel, text="Status: Init...", foreground="blue")
        self.lbl_model_status.pack(side="left", padx=10)

        self.log_box = tk.Text(log_frame, height=6, font=("Consolas", 9))
        self.log_box.pack(fill="x", side="left", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_box['yscrollcommand'] = scrollbar.set

        self.current_ticker = None
        self.current_price = 0
        self.hv_30 = 0
        
        self.log("App Started. Defaulting to FinBERT.")
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
        new_ticker = self.entry_ticker.get().upper().strip()
        if new_ticker != self.current_ticker:
            self.data_cache = {} 
            self.sent_cache = {}
            self.log(f"Ticker changed to {new_ticker}. Cleared Cache.")
        self.load_chart("1mo", "1d")

    def load_chart(self, period, interval):
        ticker = self.entry_ticker.get().upper().strip()
        if not ticker: return
        self.current_ticker = ticker
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
        self.log(f"Fetching Google RSS for {ticker}...")
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=5, verify=False) 
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                titles = [item.find('title').text for item in root.findall('.//item')][:10]
                self.log(f"RSS Found {len(titles)} headlines.")
                return titles
        except Exception as e:
             self.log(f"RSS Error: {e}")
        return []

    def calculate_sentiment(self, ticker, stock_obj):
        if ticker in self.sent_cache:
            val, ts = self.sent_cache[ticker]
            if time.time() - ts < self.SENT_CACHE_DURATION:
                self.log("Using Cached Sentiment.")
                return val

        current_model = sentiment_engine.models[sentiment_engine.current_model_name]
        if not current_model["loaded"]:
            self.log("Model not ready. Waiting...")
            return 0.5

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
            return 0.5

        avg_score = sum(scores) / len(scores)
        self.log(f"FINAL SCORE ({sentiment_engine.current_model_name}): {avg_score:.4f}")
        
        self.sent_cache[ticker] = (avg_score, time.time())
        return avg_score

    def fetch_and_plot(self, ticker, period, interval):
        try:
            # 1. Chart Data
            df, is_cached = self.get_cached_df(ticker, period, interval)
            stock = yf.Ticker(ticker)

            if df is None:
                df = self.fetch_history_with_retry(stock, period, interval)
                if df.empty: 
                    self.log("No price data found.")
                    self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                    return
                if len(df) > 5: df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
                if len(df) > 21: df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
                if len(df) > 63: df['EMA_63'] = df['Close'].ewm(span=63, adjust=False).mean()
                self.save_df_cache(ticker, period, interval, df)
                status_msg = f"Live Data ({interval})"
            else:
                status_msg = f"Cached Data ({interval})"

            self.root.after(0, lambda: self.lbl_status.config(text=status_msg))

            # 2. Technical Data
            df_tech, tech_cached = self.get_cached_df(ticker, "1y", "1d")
            if df_tech is None:
                df_tech = self.fetch_history_with_retry(stock, "1y", "1d")
                df_tech = calculate_technicals(df_tech)
                df_tech['log_ret'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1))
                self.save_df_cache(ticker, "1y", "1d", df_tech)
            
            last = df_tech.iloc[-1]
            if not df.empty: current_price = df['Close'].iloc[-1]
            else: current_price = last['Close']

            hv_30 = df_tech['log_ret'].rolling(30).std().iloc[-1] * np.sqrt(252)
            self.current_price = current_price
            self.hv_30 = hv_30
            
            # 3. Sentiment
            sentiment_score = self.calculate_sentiment(ticker, stock)

            last_copy = last.copy()
            last_copy['Close'] = current_price 
            
            self.root.after(0, lambda: self.update_technicals(last_copy, hv_30, sentiment_score))
            self.root.after(0, self.update_chart, df, ticker, period)

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            self.root.after(0, lambda: self.lbl_status.config(text="Error"))

    def update_chart(self, df, ticker, period):
        # --- SAFETY FIX: Ensure 'ax' exists before using it ---
        if not hasattr(self, 'ax') or self.ax is None:
            return

        try:
            if df is None or df.empty:
                self.log("Chart skipped: no data")
                self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                return
            self.ax.clear()
            self.ax.plot(df.index, df['Close'], label='Price', color='black', linewidth=1.5)
            if 'EMA_5' in df.columns: self.ax.plot(df.index, df['EMA_5'], label='EMA 5', color='blue', linewidth=1, linestyle='--')
            if 'EMA_21' in df.columns: self.ax.plot(df.index, df['EMA_21'], label='EMA 21', color='orange', linewidth=1)
            if 'EMA_63' in df.columns: self.ax.plot(df.index, df['EMA_63'], label='EMA 63', color='purple', linewidth=1)
            self.ax.set_title(f"{ticker} Price Action ({period})")
            self.ax.legend(loc='upper left', fontsize='small')
            self.ax.grid(True, alpha=0.3)
            self.figure.autofmt_xdate()
            self.canvas.draw()
        except Exception as e:
            self.log(f"Chart Render Error: {e}")

    def update_technicals(self, data, hv, sentiment):
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
        
        bb_pos = "Inside"
        bb_c = "black"
        if data['Close'] > data['BB_Upper']: 
            bb_pos = "Overbought"; bb_c = "red"
        elif data['Close'] < data['BB_Lower']: 
            bb_pos = "Oversold"; bb_c = "green"
        self.lbl_bb.config(text=f"{bb_pos}\n[{data['BB_Lower']:.2f}-{data['BB_Upper']:.2f}]", foreground=bb_c)
        
        self.lbl_atr.config(text=f"${data['ATR']:.2f}")
        self.lbl_vol.config(text=f"{hv:.1%}")
        
        if sentiment is not None:
            sent_c = "red" if sentiment < 0.4 else "green" if sentiment > 0.6 else "black"
            self.lbl_sent.config(text=f"{sentiment:.2f}", foreground=sent_c)
        else:
            self.lbl_sent.config(text="N/A", foreground="gray")

        self.btn_opt.config(state="normal", text=f"🔎 Open {self.current_ticker} Option Scanner")

    # ===================== 5. Options Window (Same as Pro) =====================
    def open_options_window(self):
        if not self.current_ticker: return
        win = Toplevel(self.root)
        win.title(f"Options Explorer: {self.current_ticker}")
        win.geometry("1100x700")

        left_panel = ttk.Frame(win, width=200)
        left_panel.pack(side="left", fill="y", padx=5, pady=5)
        right_panel = ttk.Frame(win)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        ttk.Label(left_panel, text="Target Date (YYYY-MM-DD):").pack(fill="x")
        self.entry_date = ttk.Entry(left_panel)
        self.entry_date.pack(fill="x", pady=2)
        self.entry_date.insert(0, (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"))
        
        ttk.Button(left_panel, text="Select Prev 7 Expirations", command=self.filter_expirations).pack(fill="x", pady=5)
        
        self.exp_list = tk.Listbox(left_panel, selectmode="extended", height=25)
        self.exp_list.pack(fill="both", expand=True)
        self.exp_list.bind('<<ListboxSelect>>', self.on_exp_select)
        
        cols = ("Date", "Type", "Strike", "Vol", "Last", "Imp Vol", "Fair(HV)", "Fair", "EV", "Verdict")
        self.tree = ttk.Treeview(right_panel, columns=cols, show="headings")
        for c in cols: 
            self.tree.heading(c, text=c)
            self.tree.column(c, width=70, anchor="center")
        
        scr = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scr.set); self.tree.pack(side="left", fill="both", expand=True); scr.pack(side="right", fill="y")
        
        self.tree.tag_configure("green", background="#d4f8d4")
        self.tree.tag_configure("red", background="#f8d4d4")
        
        threading.Thread(target=self.load_expirations, daemon=True).start()

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

    def fetch_options_batch(self, dates):
        stock = yf.Ticker(self.current_ticker)
        for date in dates:
            try:
                T = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days / 365.0
                if T <= 0: T=0.001
                chain = stock.option_chain(date)
                calls = chain.calls.assign(Type="CALL"); puts = chain.puts.assign(Type="PUT")
                top = pd.concat([calls.sort_values('volume', ascending=False).head(5), 
                                 puts.sort_values('volume', ascending=False).head(5)])
                for _, row in top.iterrows():
                    iv = row['impliedVolatility']
                    if not iv: continue
                    fair = VegaChimpCore.bs_price(self.current_price, row['strike'], 0.045, 0.0, self.hv_30, T, row['Type'].lower())
                    ev = fair - row['lastPrice']
                    tag = "green" if ev > 0.1 else "red" if ev < -0.1 else ""
                    vals = (date, row['Type'], row['strike'], int(row['volume'] or 0), f"{row['lastPrice']:.2f}", 
                            f"{iv:.1%}", f"{self.hv_30:.1%}", f"{fair:.2f}", f"{ev:+.2f}", "Under" if ev>0.1 else "Over" if ev<-0.1 else "Fair")
                    self.root.after(0, lambda v=vals, t=tag: self.tree.insert("", "end", values=v, tags=(t,)))
            except Exception as e:
                self.log(f"Options fetch error for {date}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()