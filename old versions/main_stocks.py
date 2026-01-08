import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import math
import threading
from textblob import TextBlob
from datetime import datetime

# ===================== 1. VegaChimp Core Logic (Ported) =====================
class VegaChimpCore:
    """Core math from VegaChimp.py for Black-Scholes and EV calculation."""
    
    @staticmethod
    def N(x):  # CDF
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def n(x):  # PDF
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    @staticmethod
    def d1(S, K, r, q, sig, T):
        if sig <= 0 or T <= 0: return 0
        return (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))

    @staticmethod
    def d2(S, K, r, q, sig, T):
        if sig <= 0 or T <= 0: return 0
        return VegaChimpCore.d1(S, K, r, q, sig, T) - sig * math.sqrt(T)

    @staticmethod
    def bs_price(S, K, r, q, sig, T, kind):
        """Calculate Black-Scholes price."""
        if sig <= 0 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        
        _d1 = VegaChimpCore.d1(S, K, r, q, sig, T)
        _d2 = VegaChimpCore.d2(S, K, r, q, sig, T)
        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)
        
        if kind == "call":
            return S * disc_q * VegaChimpCore.N(_d1) - K * disc_r * VegaChimpCore.N(_d2)
        else: # put
            return K * disc_r * VegaChimpCore.N(-_d2) - S * disc_q * VegaChimpCore.N(-_d1)

    @staticmethod
    def calculate_ev(S, K, T, r, q, market_price, fair_vol, kind):
        """
        Calculates Expected Value (EV).
        EV = (Fair Value using Historical Vol) - (Market Price using Implied Vol)
        """
        fair_price = VegaChimpCore.bs_price(S, K, r, q, fair_vol, T, kind)
        ev = fair_price - market_price
        return fair_price, ev

# ===================== 2. Main GUI Application =====================
class StockAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VegaChimp Analyzer Pro (Free Tier)")
        self.root.geometry("1100x800") # Wider for data columns

        # --- Input Section ---
        input_frame = ttk.LabelFrame(root, text="Ticker Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="Stock Symbol:").pack(side="left")
        self.ticker_entry = ttk.Entry(input_frame, width=15)
        self.ticker_entry.pack(side="left", padx=5)
        self.ticker_entry.insert(0, "AMD")
        self.ticker_entry.bind('<Return>', lambda event: self.run_analysis())
        
        ttk.Button(input_frame, text="Analyze & Calculate EV", command=self.run_analysis).pack(side="left", padx=5)

        # --- Dashboard ---
        self.dash_frame = ttk.Frame(root)
        self.dash_frame.pack(fill="x", padx=10, pady=5)
        
        # Tech / Volatility Panel
        self.tech_frame = ttk.LabelFrame(self.dash_frame, text="Volatility & Technicals", padding=10)
        self.tech_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.lbl_price = ttk.Label(self.tech_frame, text="Price: N/A", font=("Arial", 12, "bold"))
        self.lbl_price.pack(anchor="w")
        self.lbl_hv = ttk.Label(self.tech_frame, text="Hist Vol (30d): N/A", foreground="blue")
        self.lbl_hv.pack(anchor="w")
        self.lbl_ema = ttk.Label(self.tech_frame, text="EMA (20/50): N/A")
        self.lbl_ema.pack(anchor="w")
        self.lbl_emo = ttk.Label(self.tech_frame, text="Emo (Ease of Move): N/A")
        self.lbl_emo.pack(anchor="w")

        # Sentiment Panel
        self.sent_frame = ttk.LabelFrame(self.dash_frame, text="News Sentiment", padding=10)
        self.sent_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.lbl_sentiment = ttk.Label(self.sent_frame, text="Score: N/A")
        self.lbl_sentiment.pack(anchor="w")
        self.lbl_news = ttk.Label(self.sent_frame, text="Reading news...")
        self.lbl_news.pack(anchor="w")

        # --- Options Analysis Table ---
        self.opt_frame = ttk.LabelFrame(root, text="VegaChimp Option Scanner (Top 20 by Volume)", padding=10)
        self.opt_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Treeview with EV columns
        cols = ("Type", "Strike", "Exp", "Vol", "Last", "Implied Vol", "Fair Vol (HV)", "Fair Price", "EV", "Verdict")
        self.tree = ttk.Treeview(self.opt_frame, columns=cols, show="headings", height=15)
        
        # Define Columns
        col_widths = [50, 60, 80, 60, 60, 80, 85, 70, 70, 80]
        for col, width in zip(cols, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.opt_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Color tags
        self.tree.tag_configure("underpriced", background="#d4f8d4") # Green tint
        self.tree.tag_configure("overpriced", background="#f8d4d4")  # Red tint

    def run_analysis(self):
        ticker = self.ticker_entry.get().upper().strip()
        if not ticker: return
        
        # Clear UI
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_price.config(text="Loading...")
        
        # Threading
        threading.Thread(target=self.perform_tasks, args=(ticker,), daemon=True).start()

    def perform_tasks(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Fetch History & Calc Volatility
            hist = stock.history(period="6mo")
            if hist.empty: raise ValueError("No data")
            
            current_price = hist['Close'].iloc[-1]
            
            # --- Historical Volatility (HV) Calculation ---
            # Log returns
            hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
            # 30-day rolling std dev * sqrt(252) for annualized vol
            hist['HV_30'] = hist['log_ret'].rolling(window=30).std() * np.sqrt(252)
            hv_30 = hist['HV_30'].iloc[-1] # This is our "Fair Vol"
            if pd.isna(hv_30): hv_30 = 0.40 # Fallback if not enough data
            
            # --- EMA / EMO ---
            hist['EMA_20'] = hist['Close'].ewm(span=20).mean()
            hist['EMA_50'] = hist['Close'].ewm(span=50).mean()
            # EMO (Ease of Movement)
            distance = ((hist['High'] + hist['Low'])/2) - ((hist['High'].shift(1) + hist['Low'].shift(1))/2)
            box = (hist['Volume'] / 100000000) / (hist['High'] - hist['Low'])
            emo = (distance / box).rolling(14).mean().iloc[-1]

            # 2. Sentiment
            try:
                news = stock.news
                score = 0
                count = 0
                if news:
                    pol_sum = sum([TextBlob(n.get('title', '')).sentiment.polarity for n in news])
                    count = len(news)
                    if count > 0: score = pol_sum / count
            except: score, count = 0, 0

            # 3. Option Chain & VegaChimp Analysis
            exps = stock.options
            opt_data = []
            
            if exps:
                # Get nearest expiry
                chain = stock.option_chain(exps[0])
                
                # Parameters for BS
                # Calculate time to expiry in years
                exp_date = datetime.strptime(exps[0], "%Y-%m-%d")
                today = datetime.now()
                T = (exp_date - today).days / 365.0
                if T <= 0: T = 1/365.0 # Avoid div by zero for 0DTE
                
                r = 0.045 # Approx Risk Free Rate (4.5%)
                q = 0.0   # Dividend Yield (simplified)
                info = stock.info
                if 'dividendYield' in info and info['dividendYield'] is not None:
                    q = info['dividendYield']

                # Process Calls and Puts together
                calls = chain.calls.copy(); calls['Type'] = 'CALL'
                puts = chain.puts.copy(); puts['Type'] = 'PUT'
                df = pd.concat([calls, puts])
                
                # Filter for liquidity
                df = df.sort_values(by='volume', ascending=False).head(20)
                
                for _, row in df.iterrows():
                    kind = row['Type'].lower()
                    K = row['strike']
                    imp_vol = row['impliedVolatility']
                    last_price = row['lastPrice']
                    
                    # --- VegaChimp Logic Applied Here ---
                    # We use HV (hv_30) as the "Fair Vol" (rv_exp in original code)
                    fair_price, ev = VegaChimpCore.calculate_ev(
                        S=current_price, 
                        K=K, 
                        T=T, 
                        r=r, 
                        q=q, 
                        market_price=last_price, 
                        fair_vol=hv_30, 
                        kind=kind
                    )
                    
                    verdict = "FAIR"
                    tag = ""
                    if ev > 0.05: # Threshold
                        verdict = "UNDERPRICED"
                        tag = "underpriced"
                    elif ev < -0.05:
                        verdict = "OVERPRICED"
                        tag = "overpriced"
                        
                    opt_data.append((
                        row['Type'], K, exps[0], int(row['volume']),
                        last_price,
                        f"{imp_vol:.2%}",
                        f"{hv_30:.2%}",
                        f"{fair_price:.2f}",
                        f"{ev:+.2f}",
                        verdict,
                        tag
                    ))

            # Update GUI
            self.root.after(0, self.update_gui, current_price, hv_30, hist['EMA_20'].iloc[-1], 
                           hist['EMA_50'].iloc[-1], emo, score, count, opt_data)
                           
        except Exception as e:
            print(e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def update_gui(self, price, hv, ema20, ema50, emo, score, news_count, opt_data):
        self.lbl_price.config(text=f"Price: ${price:.2f}")
        self.lbl_hv.config(text=f"Hist Vol (30d): {hv:.2%} (Used as Fair Vol)")
        self.lbl_ema.config(text=f"EMA 20: {ema20:.2f} | EMA 50: {ema50:.2f}")
        self.lbl_emo.config(text=f"Emo: {emo:.4f}")
        
        sent_str = "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral"
        self.lbl_sentiment.config(text=f"Score: {score:.2f} ({sent_str})")
        self.lbl_news.config(text=f"Scanned {news_count} headlines")
        
        for item in opt_data:
            # item[:-1] excludes the tag from the values list
            self.tree.insert("", "end", values=item[:-1], tags=(item[-1],))

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzerApp(root)
    root.mainloop()