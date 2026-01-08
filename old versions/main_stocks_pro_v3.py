import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import yfinance as yf
import pandas as pd
import numpy as np
import math
import threading
from textblob import TextBlob
from datetime import datetime, timedelta

# --- Charting Libraries ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== 1. ToolTip Helper Class =====================
class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.wait_time = 500     # milliseconds
        self.wrap_length = 250   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.wait_time, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       wraplength = self.wrap_length)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

# ===================== 2. Math Core (VegaChimp) =====================
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
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # Stoch RSI
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)

    # --- Charting EMAs ---
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_63'] = df['Close'].ewm(span=63, adjust=False).mean()

    return df

# ===================== 4. Main GUI =====================
class MarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Technical Command Center Pro")
        self.root.geometry("1100x850") # Slightly wider

        # --- Top: Input ---
        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "AMD")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data()) 
        
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")

        # --- Middle: Dashboard + Chart Split ---
        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Side: Numeric Data
        self.left_frame = ttk.Frame(self.paned, width=350)
        self.paned.add(self.left_frame, weight=1)

        self.lbl_price = ttk.Label(self.left_frame, text="---", font=("Arial", 28, "bold"))
        self.lbl_price.pack(anchor="center", pady=10)

        # Technical Grid
        self.grid_frame = ttk.LabelFrame(self.left_frame, text="Technical Analysis", padding=15)
        self.grid_frame.pack(fill="x", pady=5)
        
        # Add rows with Tooltips
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0, 
            "Relative Strength Index: Measures momentum.\n<30: Oversold (Bullish)\n>70: Overbought (Bearish)")
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI", 1, 
            "Stochastic RSI: More sensitive than RSI.\nGood for finding exact entry/exit points in volatile stocks.")
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2, 
            "Moving Average Convergence Divergence.\nPositive: Bullish Trend\nNegative: Bearish Trend")
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3, 
            "Volatility bands. Price touching Upper Band = Overbought.\nTouching Lower Band = Oversold.")
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility)", 4, 
            "Average True Range: The average $ amount the stock moves per day.\nUse this for setting stop losses.")
        self.lbl_vol = self.add_row(self.grid_frame, "Hist. Vol (30d)", 5, 
            "Historical Volatility: The actual speed of the stock over the last 30 days.\nUsed to calculate Fair Value of options.")
        
        # Added Sentiment Row
        self.lbl_sent = self.add_row(self.grid_frame, "News Sentiment", 6, 
            "AI Analysis of recent news headlines.\n0 = Extreme Fear (Bearish)\n1 = Extreme Greed (Bullish)\n0.5 = Neutral")

        # Options Button
        self.btn_opt = ttk.Button(self.left_frame, text="🔎 Options Explorer", command=self.open_options_window, state="disabled")
        self.btn_opt.pack(fill="x", padx=20, pady=20, ipady=10)

        # Right Side: Chart
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=3)
        
        # Chart Controls
        ctrl_frame = ttk.Frame(self.right_frame)
        ctrl_frame.pack(fill="x", pady=5)
        ttk.Label(ctrl_frame, text="Chart Period: ").pack(side="left")
        
        # Added 1D and 5D buttons
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

        # Plot Placeholder
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Data Cache
        self.current_ticker = None
        self.current_price = 0
        self.hv_30 = 0
        
        # Default Load
        self.root.after(500, self.load_data)

    def add_row(self, parent, text, row, tooltip_text):
        # Container for Label + (?)
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ttk.Label(f, text=text, font=("Arial", 10, "bold")).pack(side="left")
        
        # Question Mark Label
        qm = tk.Label(f, text="(?)", fg="blue", cursor="hand2", font=("Arial", 9))
        qm.pack(side="left", padx=5)
        CreateToolTip(qm, tooltip_text) # Attach Tooltip

        # Value Label
        lbl = ttk.Label(parent, text="---", font=("Arial", 10))
        lbl.grid(row=row, column=1, sticky="e", padx=10, pady=5)
        return lbl

    def load_data(self):
        # Triggers the default 1M chart load and the Technicals load
        self.load_chart("1mo", "1d")

    def load_chart(self, period, interval):
        ticker = self.entry_ticker.get().upper().strip()
        if not ticker: return
        self.current_ticker = ticker
        
        # Run in thread
        threading.Thread(target=self.fetch_and_plot, args=(ticker, period, interval), daemon=True).start()

    def calculate_sentiment(self, stock):
        try:
            news = stock.news
            if not news:
                return None
            
            polarities = []
            for n in news:
                title = n.get('title', '')
                if title:
                    blob = TextBlob(title)
                    polarities.append(blob.sentiment.polarity)
            
            if not polarities:
                return None
            
            # Average polarity (-1 to 1)
            avg_polarity = sum(polarities) / len(polarities)
            
            # Normalize to 0 (Bearish) - 1 (Bullish)
            # Original: -1 .. 0 .. +1
            # Add 1:     0 .. 1 .. +2
            # Div 2:     0 .. 0.5 .. 1
            normalized_score = (avg_polarity + 1) / 2
            return normalized_score

        except Exception as e:
            print(f"Sentiment Error: {e}")
            return None

    def fetch_and_plot(self, ticker, period, interval):
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch for Chart
            df = stock.history(period=period, interval=interval)
            if df.empty: raise ValueError("No Data")
            
            # Calculate Chart Indicators (EMAs)
            # EMAs require enough data points. On very short intervals (1d/1m) 
            # with early market hours, you might not have 63 points.
            if len(df) > 5:
                df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            if len(df) > 21:
                df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            if len(df) > 63:
                df['EMA_63'] = df['Close'].ewm(span=63, adjust=False).mean()
            
            # Update Left Panel Technicals (only if not a quick intraday refresh)
            # Or we can just update it every time to be safe.
            # We use 1y daily data for robust RSI/MACD calc.
            df_tech = stock.history(period="1y", interval="1d")
            df_tech = calculate_technicals(df_tech)
            
            # Volatility
            df_tech['log_ret'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1))
            hv_30 = df_tech['log_ret'].rolling(30).std().iloc[-1] * np.sqrt(252)
            
            last = df_tech.iloc[-1]
            self.current_price = last['Close']
            self.hv_30 = hv_30
            
            # Get Sentiment
            sentiment_score = self.calculate_sentiment(stock)

            self.root.after(0, lambda: self.update_technicals(last, hv_30, sentiment_score))

            # Update Chart
            self.root.after(0, self.update_chart, df, ticker, period)

        except Exception as e:
            print(f"Fetch Error: {e}")

    def update_chart(self, df, ticker, period):
        self.ax.clear()
        
        # Plot Price
        self.ax.plot(df.index, df['Close'], label='Price', color='black', linewidth=1.5)
        
        # Plot EMAs only if columns exist
        if 'EMA_5' in df.columns:
            self.ax.plot(df.index, df['EMA_5'], label='EMA 5', color='blue', linewidth=1, linestyle='--')
        if 'EMA_21' in df.columns:
            self.ax.plot(df.index, df['EMA_21'], label='EMA 21', color='orange', linewidth=1)
        if 'EMA_63' in df.columns:
            self.ax.plot(df.index, df['EMA_63'], label='EMA 63', color='purple', linewidth=1)
        
        self.ax.set_title(f"{ticker} Price Action ({period})")
        self.ax.legend(loc='upper left', fontsize='small')
        self.ax.grid(True, alpha=0.3)
        
        # Auto-rotate dates
        self.figure.autofmt_xdate()
        self.canvas.draw()

    def update_technicals(self, data, hv, sentiment):
        self.lbl_price.config(text=f"${data['Close']:.2f}")
        
        # Color Logic
        # RSI
        rsi_val = data['RSI']
        rsi_c = "green" if rsi_val < 30 else "red" if rsi_val > 70 else "black"
        self.lbl_rsi.config(text=f"{rsi_val:.2f}", foreground=rsi_c)
        
        # Stoch RSI
        stoch_val = data['StochRSI']
        stoch_c = "green" if stoch_val < 0.2 else "red" if stoch_val > 0.8 else "black"
        self.lbl_stoch.config(text=f"{stoch_val:.2f}", foreground=stoch_c)
        
        # MACD
        macd_val = data['MACD']
        macd_c = "green" if macd_val > 0 else "red"
        self.lbl_macd.config(text=f"{macd_val:.2f}", foreground=macd_c)
        
        # Bollinger
        bb_pos = "Inside"
        bb_c = "black"
        if data['Close'] > data['BB_Upper']: 
            bb_pos = "Overbought"
            bb_c = "red"
        elif data['Close'] < data['BB_Lower']: 
            bb_pos = "Oversold"
            bb_c = "green"
        self.lbl_bb.config(text=f"{bb_pos}\n[{data['BB_Lower']:.2f}-{data['BB_Upper']:.2f}]", foreground=bb_c)
        
        self.lbl_atr.config(text=f"${data['ATR']:.2f}")
        self.lbl_vol.config(text=f"{hv:.1%}")
        
        # Update Sentiment Label
        if sentiment is not None:
            # Color scale
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