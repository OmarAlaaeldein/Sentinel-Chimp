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

# ===================== 1. Local Transformer Logic =====================
class LocalSentimentEngine:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", local_dir="my_local_model"):
        self.model_name = model_name
        self.local_path = os.path.join(os.getcwd(), local_dir)
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.status_msg = "Initializing..."

    def load_model(self):
        """Checks local folder. If empty, downloads and saves. If exists, loads local."""
        if not TRANSFORMERS_AVAILABLE:
            self.status_msg = "Error: 'transformers' or 'torch' library missing."
            return

        try:
            # Check if we already downloaded it
            if os.path.exists(self.local_path) and os.listdir(self.local_path):
                print(f"[System] Loading model from local folder: {self.local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.local_path)
            else:
                print(f"[System] Downloading model '{self.model_name}' (approx 260MB)...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                # Save locally for next time
                print(f"[System] Saving model to {self.local_path}...")
                self.tokenizer.save_pretrained(self.local_path)
                self.model.save_pretrained(self.local_path)
            
            self.is_ready = True
            self.status_msg = "Model Loaded Successfully."
            print("[System] Model Ready.")
            
        except Exception as e:
            self.status_msg = f"Model Load Failed: {e}"
            print(f"[System] {self.status_msg}")

    def predict(self, text):
        if not self.is_ready or not text:
            return 0.5 # Neutral fallback

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert logits to probabilities (Softmax)
            probs = F.softmax(outputs.logits, dim=-1)
            # The SST-2 model outputs [NEGATIVE, POSITIVE]
            # We want the probability of POSITIVE (index 1)
            score = probs[0][1].item()
            return score
        except Exception as e:
            print(f"[Model Error] {e}")
            return 0.5

# Initialize the engine (Global singleton to avoid reloading)
sentiment_engine = LocalSentimentEngine()

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

# ===================== 4. Main GUI =====================
class MarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Technical Command Center V14 (Local Transformer)")
        self.root.geometry("1100x900")

        self.data_cache = {}
        self.DATA_CACHE_DURATION = 60 
        self.sent_cache = {}
        self.SENT_CACHE_DURATION = 1800 

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
        
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0)
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI", 1)
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2)
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3)
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility)", 4)
        self.lbl_vol = self.add_row(self.grid_frame, "Hist. Vol (30d)", 5)
        self.lbl_sent = self.add_row(self.grid_frame, "AI Sentiment", 6)

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

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- SYSTEM LOG ---
        log_frame = ttk.LabelFrame(root, text="System Log (Model Status)", padding=5)
        log_frame.pack(fill="x", padx=10, pady=5)
        self.log_box = tk.Text(log_frame, height=8, font=("Consolas", 9))
        self.log_box.pack(fill="x", side="left", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_box['yscrollcommand'] = scrollbar.set

        self.current_ticker = None
        self.current_price = 0
        self.hv_30 = 0
        
        self.log("App Started. Initializing AI Model in background...")
        # Start model load in background thread to not freeze UI
        threading.Thread(target=self.init_model_bg, daemon=True).start()
        
        self.root.after(500, self.load_data)

    def init_model_bg(self):
        sentiment_engine.load_model()
        self.root.after(0, lambda: self.log(sentiment_engine.status_msg))

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}\n"
        self.log_box.insert("end", full_msg)
        self.log_box.see("end")
        print(full_msg)

    def add_row(self, parent, text, row):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(f, text=text, font=("Arial", 10, "bold")).pack(side="left")
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
        self.log(f"Requesting Chart: {ticker} ({period})")
        threading.Thread(target=self.fetch_and_plot, args=(ticker, period, interval), daemon=True).start()

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

    # --- GOOGLE NEWS FETCH ---
    def get_google_news_rss(self, ticker):
        self.log(f"Fetching News for {ticker}...")
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

        if not sentiment_engine.is_ready:
            self.log("Model not ready yet. Waiting...")
            return 0.5

        headlines = []
        # Try Yahoo then Google
        try:
            ynews = stock_obj.news
            if ynews: headlines = [n.get('title', '') for n in ynews]
        except: pass

        if not headlines: headlines = self.get_google_news_rss(ticker)

        if not headlines:
            self.log("No headlines found.")
            return None

        self.log(f"Analyzing {len(headlines)} headlines with DistilBERT...")
        scores = []
        for i, h in enumerate(headlines[:5]):
            score = sentiment_engine.predict(h)
            scores.append(score)
            self.log(f"[{i+1}] {score:.2f} | {h[:40]}...")

        if not scores: return 0.5
        
        avg_score = sum(scores) / len(scores)
        self.log(f"FINAL AI SCORE: {avg_score:.4f}")
        
        self.sent_cache[ticker] = (avg_score, time.time())
        return avg_score

    def fetch_and_plot(self, ticker, period, interval):
        try:
            # 1. Chart Data
            df, is_cached = self.get_cached_df(ticker, period, interval)
            stock = yf.Ticker(ticker)

            if df is None:
                df = stock.history(period=period, interval=interval)
                if df.empty: 
                    self.log("No price data found.")
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
                df_tech = stock.history(period="1y", interval="1d")
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

    def update_chart(self, df, ticker, period):
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
            except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()