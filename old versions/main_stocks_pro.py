import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import yfinance as yf
import pandas as pd
import numpy as np
import math
import threading
from textblob import TextBlob
from datetime import datetime, timedelta

# ===================== 1. Math Core (VegaChimp) =====================
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
        # Put formula correction
        return K * disc * VegaChimpCore.N(-d2) - S * disc_q * VegaChimpCore.N(-d1)

# ===================== 2. Enhanced Technicals =====================
def calculate_technicals(df):
    # Standard: RSI, MACD, Bollinger
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
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # --- NEW: Indicators for Leveraged/Volatile Assets ---
    
    # 1. ATR (Average True Range) - For Volatility Sizing
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 2. Stochastic RSI (Fast Stoch) - For rapid overbought/oversold
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)

    return df

# ===================== 3. Main GUI =====================
class MarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trader's Command Center Pro")
        self.root.geometry("700x750")

        # Top Input
        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "SOXL")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data())
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")

        # Dashboard
        self.dash_frame = ttk.Frame(root, padding=10)
        self.dash_frame.pack(fill="both", expand=True)
        
        self.lbl_price = ttk.Label(self.dash_frame, text="---", font=("Arial", 28, "bold"))
        self.lbl_price.pack(anchor="center", pady=10)

        # Technical Grid
        self.grid_frame = ttk.LabelFrame(self.dash_frame, text="Technical Analysis", padding=15)
        self.grid_frame.pack(fill="x", pady=5)
        
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0)
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI (Fast)", 1)
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2)
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3)
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility $)", 4)
        self.lbl_vol = self.add_row(self.grid_frame, "Hist. Vol (30d)", 5)

        self.btn_opt = ttk.Button(root, text="🔎 Options Explorer", command=self.open_options_window, state="disabled")
        self.btn_opt.pack(fill="x", padx=20, pady=20, ipady=10)

        # State
        self.current_ticker = None
        self.current_price = 0
        self.hv_30 = 0

        self.root.after(500, self.load_data)

    def add_row(self, parent, text, row):
        ttk.Label(parent, text=text, font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        lbl = ttk.Label(parent, text="---", font=("Arial", 10))
        lbl.grid(row=row, column=1, sticky="e", padx=10, pady=5)
        return lbl

    def load_data(self):
        ticker = self.entry_ticker.get().upper().strip()
        if not ticker: return
        self.current_ticker = ticker
        self.btn_opt.config(state="disabled", text="Loading...")
        threading.Thread(target=self.fetch_data, args=(ticker,), daemon=True).start()

    def fetch_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df.empty: raise ValueError("No Data")
            
            df = calculate_technicals(df)
            last = df.iloc[-1]
            
            # Volatility
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
            hv_30 = df['log_ret'].rolling(30).std().iloc[-1] * np.sqrt(252)
            
            self.current_price = last['Close']
            self.hv_30 = hv_30
            
            self.root.after(0, self.update_gui, last, hv_30)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def update_gui(self, data, hv):
        self.lbl_price.config(text=f"${data['Close']:.2f}")
        
        # Color Logic
        rsi_c = "red" if data['RSI'] > 70 else "green" if data['RSI'] < 30 else "black"
        stoch_c = "red" if data['StochRSI'] > 0.8 else "green" if data['StochRSI'] < 0.2 else "black"
        
        self.lbl_rsi.config(text=f"{data['RSI']:.2f}", foreground=rsi_c)
        self.lbl_stoch.config(text=f"{data['StochRSI']:.2f} (0-1)", foreground=stoch_c)
        self.lbl_macd.config(text=f"{data['MACD']:.2f}")
        
        bb_pos = "Inside"
        if data['Close'] > data['BB_Upper']: bb_pos = "Upper Break!"
        elif data['Close'] < data['BB_Lower']: bb_pos = "Lower Break!"
        self.lbl_bb.config(text=f"{bb_pos} [{data['BB_Lower']:.2f}-{data['BB_Upper']:.2f}]")
        
        self.lbl_atr.config(text=f"${data['ATR']:.2f}")
        self.lbl_vol.config(text=f"{hv:.1%}")
        
        self.btn_opt.config(state="normal", text=f"🔎 Open {self.current_ticker} Option Scanner")

    # ===================== 4. Advanced Option Window =====================
    def open_options_window(self):
        if not self.current_ticker: return
        win = Toplevel(self.root)
        win.title(f"Options Explorer: {self.current_ticker}")
        win.geometry("1100x700")

        # --- Layout ---
        # Left: Expiry List (25% width)
        # Right: Option Chain (75% width)
        
        left_panel = ttk.Frame(win, width=200)
        left_panel.pack(side="left", fill="y", padx=5, pady=5)
        
        right_panel = ttk.Frame(win)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # --- Left Panel: Filter & List ---
        ttk.Label(left_panel, text="Target Date (YYYY-MM-DD):").pack(fill="x")
        self.entry_date = ttk.Entry(left_panel)
        self.entry_date.pack(fill="x", pady=2)
        # Default date 6 months out
        default_date = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")
        self.entry_date.insert(0, default_date)
        
        ttk.Button(left_panel, text="Select Prev 7 Expirations", command=self.filter_expirations).pack(fill="x", pady=5)
        
        ttk.Label(left_panel, text="Available Expirations:").pack(fill="x", pady=(10,0))
        
        # Listbox for expirations
        self.exp_list = tk.Listbox(left_panel, selectmode="extended", height=25)
        self.exp_list.pack(fill="both", expand=True)
        self.exp_list.bind('<<ListboxSelect>>', self.on_exp_select)
        
        # --- Right Panel: Treeview ---
        ttk.Label(right_panel, text="Top 5 High Volume Puts & Calls per Selection", font=("Arial", 12, "bold")).pack(pady=5)
        
        cols = ("Date", "Type", "Strike", "Vol", "Last", "Imp Vol", "Fair(HV)", "Fair", "EV", "Verdict")
        self.tree = ttk.Treeview(right_panel, columns=cols, show="headings")
        col_widths = [80, 50, 60, 60, 60, 70, 70, 70, 70, 80]
        for c, w in zip(cols, col_widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
            
        scr = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scr.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")
        
        self.tree.tag_configure("green", background="#d4f8d4")
        self.tree.tag_configure("red", background="#f8d4d4")
        
        # Load Expirations
        threading.Thread(target=self.load_expirations, daemon=True).start()

    def load_expirations(self):
        stock = yf.Ticker(self.current_ticker)
        self.all_exps = stock.options
        self.root.after(0, lambda: self.update_exp_list(self.all_exps))

    def update_exp_list(self, exp_list):
        self.exp_list.delete(0, "end")
        for e in exp_list:
            self.exp_list.insert("end", e)

    def filter_expirations(self):
        target_str = self.entry_date.get()
        try:
            target_dt = datetime.strptime(target_str, "%Y-%m-%d")
            valid_exps = []
            
            # Find dates <= target
            for e in self.all_exps:
                e_dt = datetime.strptime(e, "%Y-%m-%d")
                if e_dt <= target_dt:
                    valid_exps.append(e)
            
            # Take last 7
            final_exps = valid_exps[-7:] if len(valid_exps) >= 7 else valid_exps
            
            # Select them in UI
            self.exp_list.selection_clear(0, "end")
            for i, e in enumerate(self.all_exps):
                if e in final_exps:
                    self.exp_list.selection_set(i)
            
            # Trigger load
            self.on_exp_select(None)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid Date Format. Use YYYY-MM-DD")

    def on_exp_select(self, event):
        # Get selected indices
        selection = self.exp_list.curselection()
        selected_dates = [self.exp_list.get(i) for i in selection]
        if not selected_dates: return
        
        # Clear Tree
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Thread search
        threading.Thread(target=self.fetch_options_batch, args=(selected_dates,), daemon=True).start()

    def fetch_options_batch(self, dates):
        stock = yf.Ticker(self.current_ticker)
        
        for date in dates:
            try:
                # Calculate T
                exp_dt = datetime.strptime(date, "%Y-%m-%d")
                T = (exp_dt - datetime.now()).days / 365.0
                if T <= 0: T = 0.001
                
                # Fetch Chain
                chain = stock.option_chain(date)
                calls = chain.calls.assign(Type="CALL")
                puts = chain.puts.assign(Type="PUT")
                
                # Filter Top 5 Calls & Top 5 Puts
                top_calls = calls.sort_values('volume', ascending=False).head(5)
                top_puts = puts.sort_values('volume', ascending=False).head(5)
                combined = pd.concat([top_calls, top_puts])
                
                for _, row in combined.iterrows():
                    iv = row['impliedVolatility']
                    if iv is None or iv == 0: continue
                    
                    # VegaChimp EV Logic
                    kind = row['Type'].lower()
                    fair_price = VegaChimpCore.bs_price(
                        self.current_price, row['strike'], 0.045, 0.0, self.hv_30, T, kind
                    )
                    ev = fair_price - row['lastPrice']
                    
                    # Tag
                    tag = "green" if ev > 0.1 else "red" if ev < -0.1 else ""
                    verdict = "Underpriced" if ev > 0.1 else "Overpriced" if ev < -0.1 else "Fair"
                    
                    vals = (
                        date, row['Type'], row['strike'], int(row['volume'] if not pd.isna(row['volume']) else 0),
                        f"{row['lastPrice']:.2f}", f"{iv:.1%}", f"{self.hv_30:.1%}",
                        f"{fair_price:.2f}", f"{ev:+.2f}", verdict
                    )
                    
                    self.root.after(0, lambda v=vals, t=tag: self.tree.insert("", "end", values=v, tags=(t,)))
                    
            except Exception as e:
                print(f"Error on {date}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MarketApp(root)
    root.mainloop()