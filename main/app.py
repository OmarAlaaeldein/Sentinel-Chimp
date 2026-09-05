"""MarketApp GUI controller (Phase I MVC — UI still co-located here).

Follow-up: peel chart/options/news widgets into ui/ modules once the
core + data provider split has settled.
"""
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Charting Libraries ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.pricing import VegaChimpCore
from core.technicals import calculate_technicals
from core.sentiment import sentiment_engine
from core.data import YFinanceProvider
from core.vol_models import (
    garch11_vol_forecast, fit_quadratic_smile, smile_vol_arr,
)
from ui.tooltip import Tooltip
from ui.theme import setup_dark_theme as apply_dark_theme, DARK_BG


class MarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel: Stock & Options Analyzer")
        self.root.geometry("1450x900")
        self.style = apply_dark_theme(self.root)
        self.data_provider = YFinanceProvider(cache_duration=60)

        self.headline_limit = 1000
        self.data_cache = {}
        self.DATA_CACHE_DURATION = 60 
        self.sent_cache = {}
        self.SENT_CACHE_DURATION = 1800 
        self.valuation_cache = {}
        self.VALUATION_CACHE_DURATION = 3600
        self.pe_fwd = None
        self.pe_ttm = None
        self.peg_ratio = None
        self.pe_percentile = None
        self.earnings_growth = None
        self.valuation_status = {}
        
        self.use_sentiment = False
        # Vol / Greek experiments (EWMA path preserved unless blend flags are on)
        self.use_garch_blend = False   # blend GARCH(1,1) into historical vol for FV
        self.use_smile_vol = False     # replace raw IV with quadratic smile fit
        self.use_american_greeks = True  # FD Greeks on BS2002 (else European BS)
        self.garch_vol = 0.0
        self.garch_info = {}

        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Ticker:").pack(side="left")
        self.entry_ticker = ttk.Entry(input_frame, width=10)
        self.entry_ticker.pack(side="left", padx=5)
        self.entry_ticker.insert(0, "AMD")
        self.entry_ticker.bind('<Return>', lambda e: self.load_data()) 
        
        
        
        ttk.Button(input_frame, text="Load Data", command=self.load_data).pack(side="left")
        
        self.btn_news = ttk.Button(input_frame, text="📰 News", command=self.open_news_window, state="disabled")
        self.btn_news.pack(side="left", padx=10)

        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        self.left_frame = ttk.Frame(self.paned, width=350)
        self.paned.add(self.left_frame, weight=1)

        self.lbl_price = ttk.Label(self.left_frame, text="---", font=("Arial", 28, "bold"))
        self.lbl_price.pack(anchor="center", pady=10)

        self.grid_frame = ttk.LabelFrame(self.left_frame, text="Technical Analysis", padding=15)
        self.grid_frame.pack(fill="x", pady=5)
        
        self.lbl_rsi = self.add_row(self.grid_frame, "RSI (14d)", 0, "Relative Strength Index. Range 0-100.")
        self.lbl_stoch = self.add_row(self.grid_frame, "Stoch RSI", 1, "Stochastic RSI.\n\nMore sensitive than standard RSI.\nUse this to time specific entries/exits within a trend.\n0.0 = Max Oversold, 1.0 = Max Overbought.")
        self.lbl_macd = self.add_row(self.grid_frame, "MACD", 2, "Moving Average Convergence Divergence.")
        self.lbl_bb = self.add_row(self.grid_frame, "Bollinger Bands", 3, "20-day SMA +/- 2 STDs.")
        self.lbl_atr = self.add_row(self.grid_frame, "ATR (Volatility)", 4, "Average True Range (Daily Move in $).")
        self.lbl_vol = self.add_row(self.grid_frame, "Vol (HV vs EWMA)", 5, "HV: 30d Historical Volatility.\nEWMA: Exponentially Weighted Moving Average Vol Forecast (decay=0.94).")
        if self.use_sentiment:
            self.lbl_sent = self.add_row(self.grid_frame, "AI Sentiment", 6, "Headline sentiment scored 0-1.")
            self.lbl_return = self.add_row(self.grid_frame, "Return (Period)", 7, "Total return over selected period.")
            self.lbl_vwap = self.add_row(self.grid_frame, "VWAP Gap", 8, "Volume Weighted Average Price.\n\n'Who is winning today?'\nPrice > VWAP: Buyers are in control (Bullish).\nPrice < VWAP: Sellers are in control (Bearish).")
        else:
            self.lbl_return = self.add_row(self.grid_frame, "Return (Period)", 6, "Total return over selected period.")
            self.lbl_vwap = self.add_row(self.grid_frame, "VWAP Gap", 7, "Volume Weighted Average Price.\n\n'Who is winning today?'\nPrice > VWAP: Buyers are in control (Bullish).\nPrice < VWAP: Sellers are in control (Bearish).")


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
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

# --- SYSTEM LOG & CONTROLS ---
        if self.use_sentiment:
            log_frame = ttk.LabelFrame(root, text="System Log & AI Controls", padding=5)

        else:
            log_frame = ttk.LabelFrame(root, text="System Log", padding=5)
            
        log_frame.pack(fill="x", padx=10, pady=5)

        ctrl_panel = ttk.Frame(log_frame)
        ctrl_panel.pack(fill="x", pady=2)
        
        self.btn_log = ttk.Button(ctrl_panel, text="Show Log", command=self.toggle_log, width=10)
        self.btn_log.pack(side="right", padx=5)
        if self.use_sentiment:
            ttk.Label(ctrl_panel, text="Active Model:").pack(side="left")
        
            self.lbl_model_status = ttk.Label(ctrl_panel, text="Status: Init...", foreground="orange")
            self.lbl_model_status.pack(side="left", padx=10)

        self.log_box = tk.Text(log_frame, height=6, font=("Consolas", 9), 
                               bg="#1e1e1e", fg="#00ff00", # Matrix Green Text
                               insertbackground="white") # Cursor color
        self.log_scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box['yscrollcommand'] = self.log_scroll.set
        
        # Track visibility state
        self.log_visible = False
        
        self.current_ticker = None
        self.stock = None
        self.current_price = 0
        self.hv_30 = 0
        self.projected_earnings = []
        self.lbl_adx = self.add_row(self.grid_frame, "ADX (Trend)", 9, "Trend Strength (0-100).\n\n< 20: Weak/Choppy market. (DANGER: Do not buy options here, theta will kill you).\n> 25: Trending market. (SAFE: Good for directional trades).")
        self.lbl_obv = self.add_row(self.grid_frame, "OBV Trend", 10, "On-Balance Volume.\n\nTracks 'Smart Money' flow.\nBullish: OBV rising with Price.\nDivergence: If Price rises but OBV falls, the rally is a trap.")
        self.lbl_pe = self.add_row(self.grid_frame, "P/E Ratio", 11, "Price-to-Earnings Ratio (TTM vs Forward).")
        self.lbl_pe_percentile = self.add_row(self.grid_frame, "P/E Percentile", 12, 
                                     "How expensive the current P/E is vs the last 5 years (0-100%).")
        self.lbl_peg = self.add_row(self.grid_frame, "PEG Ratio", 13, "Price/Earnings-to-Growth Ratio. < 1.0 generally implies undervaluation.")
        self.lbl_williams = self.add_row(self.grid_frame, "Williams %R", 14, "Momentum oscillator (-100 to 0).\n\nOversold < -80 (Buy signal)\nOverbought > -20 (Sell signal)")
        self.lbl_cci = self.add_row(self.grid_frame, "CCI (20)", 15, "Commodity Channel Index.\n\nOversold < -100\nOverbought > +100\nMeasures deviation from mean price.")
        # Only initialize the transformer if the toggle is True
        if self.use_sentiment:
            self.log("App Started. Defaulting to FinBERT.")
            threading.Thread(target=self.init_model_bg, args=("FinBERT",), daemon=True).start()
        else:
            self.log("AI Sentiment is currently disabled.")
        self.scan_data = []
        self._scan_lock = threading.Lock()
        self._data_cache_lock = threading.Lock()
        self._sent_cache_lock = threading.Lock()
        self._valuation_cache_lock = threading.Lock()
        self._earnings_lock = threading.Lock()
        self._chart_request_id = 0
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

    def load_data(self, period="5d", interval="5m"):
        """Refreshes chart data and updates fundamentals only on ticker change."""
        new_ticker = self.entry_ticker.get().upper().strip()
        if not new_ticker: return

        # Only reload the Heavy Ticker Object if the symbol changed
        if new_ticker != self.current_ticker:
            self.current_ticker = new_ticker
            
            # Use the shared session for the ticker
            self.stock = self.data_provider.create_ticker(new_ticker)
            
            # Clear caches and reset drift for the new stock
            self.data_provider.clear_cache()
            with self._data_cache_lock:
                self.data_cache.clear()
            with self._sent_cache_lock:
                self.sent_cache.clear()
            self.projected_earnings = []
            
            self.lbl_pe.config(text="Updating...", foreground="orange")
            self.lbl_pe_percentile.config(text="Updating...", foreground="orange")
            self.lbl_peg.config(text="Updating...", foreground="orange")
            self.pe_fwd = None
            self.pe_ttm = None
            self.peg_ratio = None
            self.pe_percentile = None
            self.earnings_growth = None
            self.valuation_status = {}
            self.current_price = 0.0
            self.hv_30 = 0.0
            self.btn_opt.config(state="disabled", text="🔎 Options Explorer")
            self.btn_news.config(state="disabled")
            
            # Start background fundamental fetch
            threading.Thread(target=self.get_info, daemon=True).start()
            self.log(f"Ticker changed: {new_ticker}. Session reused.")

        # Always refresh the chart (light logic)
        self.load_chart(period, interval)

    def load_chart(self, period, interval):
        ticker = self.entry_ticker.get().upper().strip()
        if not ticker: return

        # A timeframe button can be clicked after editing the entry without
        # pressing Enter. Initialize the matching Ticker object before fetching.
        if ticker != self.current_ticker or self.stock is None:
            self.load_data(period, interval)
            return

        self.last_interval = interval
        self._chart_request_id += 1
        request_id = self._chart_request_id
        stock = self.stock
        self.lbl_status.config(text=f"Loading {ticker} ({period})...")
        self.log(f"Requesting Chart: {ticker} ({period})")
        threading.Thread(
            target=self.fetch_and_plot,
            args=(ticker, period, interval, stock, request_id),
            daemon=True
        ).start()

    def _is_current_chart_request(self, ticker, request_id):
        return ticker == self.current_ticker and request_id == self._chart_request_id

    def fetch_history_with_retry(self, stock, period, interval, retries=2, delay=1.0):
        return self.data_provider.fetch_history(
            stock, period, interval, retries=retries, delay=delay, log=self.log
        )

    def get_cached_df(self, ticker, period, interval):
        return self.data_provider.get_cached_df(ticker, period, interval)

    def save_df_cache(self, ticker, period, interval, df):
        self.data_provider.save_df_cache(ticker, period, interval, df)

    def _to_finite_float(self, value):
        """Best-effort numeric normalization for provider fields."""
        try:
            f_val = float(value)
            if not math.isfinite(f_val):
                return None
            return f_val
        except (TypeError, ValueError):
            return None

    def _is_number(self, value):
        return isinstance(value, (int, float, np.floating)) and not math.isnan(float(value))

    def _is_finite_number(self, value):
        return self._is_number(value) and math.isfinite(float(value))

    def _format_metric(self, value, decimals=2, fallback="N/A"):
        if self._is_number(value):
            if math.isinf(float(value)):
                return "Inf" if float(value) > 0 else "-Inf"
            return f"{float(value):.{decimals}f}"
        return fallback

    def _valuation_status_text(self, key, default="N/A"):
        reason = self.valuation_status.get(key)
        if not reason:
            return default
        reason_map = {
            "NO_CURRENT_PE_TTM": "N/A (No TTM P/E)",
            "NO_PRICE_HISTORY": "N/A (No 5Y Price)",
            "NO_EARNINGS_HISTORY": "N/A (No EPS History)",
            "INSUFFICIENT_EARNINGS_HISTORY": "N/A (EPS < 4 Qtrs)",
            "NO_VALID_PE_HISTORY": "N/A (No Valid P/E)",
            "MISSING_PEG_INPUTS": "Not Calculable",
            "ZERO_GROWTH": "Inf (Zero Growth)",
        }
        return reason_map.get(reason, default)

    def _get_historical_ttm_eps(self, ticker_obj):
        """Builds historical TTM EPS timeline from reported quarterly EPS."""
        cache_key = (self.current_ticker, "hist_ttm_eps")
        with self._valuation_cache_lock:
            cached = self.valuation_cache.get(cache_key)
        if cached and (time.time() - cached[1] < self.VALUATION_CACHE_DURATION):
            return cached[0], None

        try:
            earnings_df = ticker_obj.get_earnings_dates(limit=80)
            if earnings_df is None or earnings_df.empty:
                return None, "NO_EARNINGS_HISTORY"

            reported_col = None
            for col in earnings_df.columns:
                col_name = str(col).lower()
                if "reported" in col_name and "eps" in col_name:
                    reported_col = col
                    break

            if reported_col is None:
                return None, "NO_EARNINGS_HISTORY"

            eps_series = pd.to_numeric(earnings_df[reported_col], errors='coerce').dropna()
            if eps_series.empty:
                return None, "NO_EARNINGS_HISTORY"

            eps_df = pd.DataFrame({"reported_eps": eps_series})
            eps_df["report_date"] = pd.to_datetime(eps_df.index, errors='coerce', utc=True)
            eps_df = eps_df.dropna(subset=["report_date"]).sort_values("report_date")
            eps_df["report_date"] = eps_df["report_date"].dt.tz_convert(None)
            eps_df = eps_df.drop_duplicates(subset=["report_date"], keep="last")
            eps_df["ttm_eps"] = eps_df["reported_eps"].rolling(4).sum()
            eps_df = eps_df.dropna(subset=["ttm_eps"])
            if eps_df.empty:
                return None, "INSUFFICIENT_EARNINGS_HISTORY"

            eps_timeline = eps_df[["report_date", "ttm_eps"]].copy()
            with self._valuation_cache_lock:
                self.valuation_cache[cache_key] = (eps_timeline, time.time())
            return eps_timeline, None
        except Exception as e:
            self.log(f"Historical EPS fetch error: {e}")
            return None, "NO_EARNINGS_HISTORY"

    def calculate_pe_percentile(self, ticker_obj):
        """Computes a strict TTM-based P/E percentile using historical reported EPS."""
        self.pe_percentile = None
        self.valuation_status["pe_percentile_reason"] = None

        current_pe_ttm = self._to_finite_float(self.pe_ttm)
        if current_pe_ttm is None:
            self.valuation_status["pe_percentile_reason"] = "NO_CURRENT_PE_TTM"
            return

        try:
            hist = self.data_provider.fetch_history(ticker_obj, "5y", "1d", log=self.log)
            if hist is None or hist.empty or "Close" not in hist.columns:
                self.valuation_status["pe_percentile_reason"] = "NO_PRICE_HISTORY"
                return

            eps_timeline, eps_reason = self._get_historical_ttm_eps(ticker_obj)
            if eps_timeline is None or eps_timeline.empty:
                self.valuation_status["pe_percentile_reason"] = eps_reason or "NO_EARNINGS_HISTORY"
                return

            hist_df = hist[["Close"]].copy().dropna()
            hist_df["date"] = pd.to_datetime(hist_df.index, errors='coerce', utc=True).tz_convert(None)
            hist_df = hist_df.dropna(subset=["date"]).sort_values("date")

            merged = pd.merge_asof(
                hist_df[["date", "Close"]],
                eps_timeline.sort_values("report_date"),
                left_on="date",
                right_on="report_date",
                direction="backward"
            )

            pe_series = pd.to_numeric(merged["Close"], errors='coerce') / pd.to_numeric(merged["ttm_eps"], errors='coerce')
            pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()

            # For percentile comparability, use positive P/E history only.
            pe_series = pe_series[pe_series > 0]
            if pe_series.empty:
                self.valuation_status["pe_percentile_reason"] = "NO_VALID_PE_HISTORY"
                return

            self.pe_percentile = float((pe_series < current_pe_ttm).mean() * 100.0)
            self.valuation_status["pe_percentile_reason"] = None
        except Exception as e:
            self.log(f"P/E Percentile error: {e}")
            self.valuation_status["pe_percentile_reason"] = "NO_VALID_PE_HISTORY"

    def compute_peg_ratio(self):
        """Computes PEG with provider-first fallback to derived forward PEG."""
        self.valuation_status["peg_reason"] = None

        provider_peg = self._to_finite_float(self.peg_ratio)
        if provider_peg is not None:
            self.peg_ratio = provider_peg
            return

        pe_for_peg = self._to_finite_float(self.pe_fwd)
        if pe_for_peg is None:
            pe_for_peg = self._to_finite_float(self.pe_ttm)

        growth_dec = self._to_finite_float(self.earnings_growth)
        if pe_for_peg is None or growth_dec is None:
            self.peg_ratio = None
            self.valuation_status["peg_reason"] = "MISSING_PEG_INPUTS"
            return

        growth_pct = growth_dec * 100.0
        if abs(growth_pct) < 1e-9:
            self.peg_ratio = math.inf if pe_for_peg >= 0 else -math.inf
            self.valuation_status["peg_reason"] = "ZERO_GROWTH"
            return

        self.peg_ratio = pe_for_peg / growth_pct

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

                    def _elem_text(elem):
                        return elem.text if (elem is not None and elem.text) else ""

                    for item in items:
                        title = _elem_text(item.find('title'))
                        link = _elem_text(item.find('link'))
                        pub_date_str = _elem_text(item.find('pubDate'))

                        raw_desc = _elem_text(item.find('description'))
                        
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
        with self._sent_cache_lock:
            if ticker in self.sent_cache:
                val, news_items, ts = self.sent_cache[ticker]
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

        with self._sent_cache_lock:
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
        for index, (_, k) in enumerate(l):
            tv.move(k, '', index)

        # Update the heading command to toggle the sort direction next time
        tv.heading(col, command=lambda _col=col: self.treeview_sort_column(tv, _col, not reverse))
    
    def open_news_window(self):
        if not self.current_ticker: return
        
        # Retrieve rich data from cache
        with self._sent_cache_lock:
            if self.current_ticker not in self.sent_cache:
                messagebox.showinfo("News", "No news loaded yet for this ticker.")
                return

            _, news_items, _ = self.sent_cache[self.current_ticker]
        
        if not news_items:
            messagebox.showinfo("News", "No headlines found.")
            return

        win = Toplevel(self.root)
        win.title(f"News Feed: {self.current_ticker}")
        win.geometry("900x600")
        win.configure(bg=DARK_BG)

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
    def fetch_and_plot(self, ticker, period, interval, stock=None, request_id=None):
        try:
            # Keep a request-local stock snapshot so a ticker change cannot make
            # this worker fetch another symbol under the old cache key.
            stock = stock or self.stock
            if request_id is not None and not self._is_current_chart_request(ticker, request_id):
                return
            
            # Chart Data - check cache first
            df, _ = self.get_cached_df(ticker, period, interval)
            
            # --- Earnings Cycle Detection (Optimized) ---
            # We only need to fetch the calendar once per ticker change.
            # If self.projected_earnings is already populated, we skip this.
            # Lock so concurrent chart-period clicks don't double-fetch.
            with self._earnings_lock:
                if not self.projected_earnings:
                    try:
                        cal = self.data_provider.get_calendar(stock)
                        anchor_date = None
                        if isinstance(cal, dict) and 'Earnings Date' in cal:
                            dates = cal['Earnings Date']
                            if dates: anchor_date = pd.to_datetime(dates[0]).date()
                        elif cal is not None and not cal.empty:
                            anchor_date = pd.to_datetime(cal.iloc[0].values[0]).date()

                        if (anchor_date and
                                (request_id is None or self._is_current_chart_request(ticker, request_id))):
                            projected = [anchor_date]
                            for i in range(1, 4):
                                projected.append(anchor_date + timedelta(days=91*i))
                            self.projected_earnings = projected
                            self.log(f"Earnings Cycle Detected: {self.projected_earnings}")
                    except Exception as e:
                        self.log(f"Earnings fetch skipped: {e}")

            # --- Main Price Data Fetching ---
            if df is None:
                df = self.fetch_history_with_retry(stock, period, interval)
                if df.empty: 
                    self.log("No price data found.")
                    if request_id is None or self._is_current_chart_request(ticker, request_id):
                        self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                    return
                
                # Vectorized EMA calculations
                for span in [5, 21, 63, 200]:
                    df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
                
                # --- CHART VWAP with Daily Reset (matches calculate_technicals) ---
                try:
                    tp = (df['High'] + df['Low'] + df['Close']) / 3
                    trade_date = df.index.normalize()
                    tp_vol = tp * df['Volume']
                    df['VWAP'] = (tp_vol.groupby(trade_date).cumsum() /
                                  df['Volume'].groupby(trade_date).cumsum())
                except Exception as e:
                    self.log(f"VWAP Calc Error: {e}")
                # -----------------------------------------------------------------
                self.save_df_cache(ticker, period, interval, df)
                status_msg = f"Live Data ({interval})"
            else:
                status_msg = f"Cached Data ({interval})"
            
            # Calculate Period Return
            period_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] if not df.empty else 0.0
            # --- 2. Technical Data & EWMA Vol (1y Daily) ---
            df_tech, _ = self.get_cached_df(ticker, "1y", "1d")
            if df_tech is None:
                df_tech = self.fetch_history_with_retry(stock, "1y", "1d")
                df_tech = calculate_technicals(df_tech)
                df_tech['log_ret'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1))
                self.save_df_cache(ticker, "1y", "1d", df_tech)
            
            last = df_tech.iloc[-1]
            
            # 1. Try to get the absolute latest tick from fast_info
            real_time_price = None
            try:
                # fast_info is lighter/faster than .info and usually has the latest metadata
                real_time_price = self.data_provider.get_fast_last_price(stock)
            except Exception:
                pass

            # 2. Assign Current Price (Priority: Fast Info -> Intraday Chart -> Daily Cache)
            real_time_price = self._to_finite_float(real_time_price)
            if real_time_price is not None and real_time_price > 0:
                current_price = real_time_price
            elif not df.empty:
                current_price = df['Close'].iloc[-1]
            else:
                current_price = last['Close']

            if request_id is not None and not self._is_current_chart_request(ticker, request_id):
                return

            # Volatility Logic
            recent_returns = df_tech['log_ret'].dropna().tail(30)
            hv_30 = (float(recent_returns.std()) * np.sqrt(252)
                     if len(recent_returns) >= 2 else 0.0)
            log_rets = df_tech['log_ret'].dropna().values
            ewma_vol = VegaChimpCore.ewma_vol_forecast(log_rets)
            garch_vol, garch_info = garch11_vol_forecast(log_rets)
            
            # --- 4. Sentiment Analysis ---
            sentiment_score, headlines = self.calculate_sentiment(ticker, stock)

            # --- 5. UI Updates ---
            last_copy = last.copy()
            last_copy['Close'] = current_price

            # A daily-reset VWAP on one-row-per-day data is just that candle's
            # typical price, not a useful session VWAP. Use the active intraday
            # chart's latest VWAP and suppress the metric on non-intraday views.
            intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
            chart_vwap = (df['VWAP'].iloc[-1]
                          if interval in intraday_intervals and 'VWAP' in df
                          else np.nan)
            last_copy['VWAP'] = chart_vwap if self._is_finite_number(chart_vwap) else np.nan
            
            def publish_result():
                if request_id is not None and not self._is_current_chart_request(ticker, request_id):
                    return
                self.current_price = current_price
                self.hv_30 = hv_30
                self.garch_vol = garch_vol
                self.garch_info = garch_info
                self.lbl_status.config(text=status_msg)
                self.btn_news.config(state="normal" if headlines else "disabled")
                self.update_technicals(
                    last_copy, hv_30, ewma_vol, sentiment_score, period_return,
                    garch_vol=garch_vol)
                self.update_chart(df, ticker, period)

            self.root.after(0, publish_result)

        except Exception as e:
            self.log(f"CRITICAL ERROR in fetch_and_plot: {e}")
            if request_id is None or self._is_current_chart_request(ticker, request_id):
                self.root.after(0, lambda: self.lbl_status.config(text="Error"))
            
    def setup_dark_theme(self):
        """Compat wrapper — theme lives in ui.theme."""
        self.style = apply_dark_theme(self.root)
            
    def update_chart(self, df, ticker, period):
        if not hasattr(self, 'ax') or self.ax is None: return

        try:
            if df is None or df.empty:
                self.log("Chart skipped: no data")
                self.root.after(0, lambda: self.lbl_status.config(text="No data"))
                return
            
            plot_df = df.copy()
            interval = getattr(self, "last_interval", None)
            
            # --- 1. Intraday Filtering (Keep existing logic) ---
            # We still filter the DATAFRAME for market hours, even if plotting logic changes
            intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
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

            # --- 2. X-Axis Preparation ---
            # We now use index-based plotting for ALL charts to control tick counts precisely
            times_for_labels = plot_df.index
            if getattr(times_for_labels, "tz", None):
                times_for_labels = times_for_labels.tz_convert("America/New_York")
            else:
                times_for_labels = times_for_labels.tz_localize("UTC").tz_convert("America/New_York")
            
            x_vals = np.arange(len(plot_df))

            # --- 3. PLOTTING ---
            self.ax.clear()
            self.hover_annot = None 

            # Price
            self.ax.plot(x_vals, plot_df['Close'], label='Price', color='#00e6ff', linewidth=1.5)
            
            # EMAs
            if 'EMA_5' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_5'], label='EMA 5', color='#ff00ff', linewidth=1, alpha=0.8)
            if 'EMA_21' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_21'], label='EMA 21', color='#ffe100', linewidth=1, alpha=0.8)
            if 'EMA_63' in plot_df.columns: 
                self.ax.plot(x_vals, plot_df['EMA_63'], label='EMA 63', color='#9900ff', linewidth=1, alpha=0.8)
            if 'EMA_200' in plot_df.columns and plot_df['EMA_200'].notna().sum() > 0:
                self.ax.plot(x_vals, plot_df['EMA_200'], label='EMA 200', color='#ff3333', linewidth=1.5)
            
            # VWAP
            if 'VWAP' in plot_df.columns and plot_df['VWAP'].notna().sum() > 0:
                 self.ax.plot(x_vals, plot_df['VWAP'], label='VWAP', color='#ffd700', linewidth=1.5, linestyle='--')

            # --- Fibonacci Retracement Levels (D4) ---
            fib_high = plot_df['High'].max()
            fib_low = plot_df['Low'].min()
            fib_range = fib_high - fib_low
            if fib_range > 0:
                fib_levels = {
                    '23.6%': fib_high - 0.236 * fib_range,
                    '38.2%': fib_high - 0.382 * fib_range,
                    '50.0%': fib_high - 0.500 * fib_range,
                    '61.8%': fib_high - 0.618 * fib_range,
                }
                fib_colors = {'23.6%': '#ff9800', '38.2%': '#e91e63', '50.0%': '#9c27b0', '61.8%': '#2196f3'}
                for level_name, level_val in fib_levels.items():
                    self.ax.axhline(y=level_val, color=fib_colors[level_name], linestyle=':', linewidth=0.7, alpha=0.6)
                    self.ax.text(x_vals[-1], level_val, f" {level_name}", color=fib_colors[level_name],
                                fontsize=6, va='center', alpha=0.8)

            # Styling
            self.ax.set_xlim(left=x_vals[0], right=x_vals[-1])
            self.ax.set_title(f"{ticker} Price Action ({period})", color="white", fontweight="bold")
            self.ax.legend(loc='upper right', fontsize='small', frameon=False, labelcolor='white')
            self.ax.grid(True, color='#2a2a2a', linestyle='-', linewidth=0.5)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['bottom'].set_color('#444444')
            self.ax.spines['left'].set_color('#444444')
            self.ax.tick_params(axis='x', colors='gray')
            self.ax.tick_params(axis='y', colors='gray')

            # --- 4. CUSTOM TICK LOGIC ---
            if len(times_for_labels) > 0:
                # A. Determine Target Tick Count based on Period
                if period == "1mo":
                    target_count = 21
                elif period == "3mo":
                    target_count = 63
                elif period == "1y":
                    target_count = 52
                elif period == "5y" or period == "10y":
                    target_count = 60
                elif period == "25y":
                    target_count = 25
                elif period == "1d":
                    target_count = 7
                else:
                    # Fallback for 5d, 3m, etc.
                    target_count = 6

                # Ensure we don't ask for more ticks than data points
                tick_count = min(target_count, len(times_for_labels))
                
                # B. Generate Indices
                tick_idx = np.linspace(0, len(times_for_labels) - 1, tick_count, dtype=int)
                self.ax.set_xticks(tick_idx)
                
                # C. Generate Labels with Special Logic
                final_labels = []
                for i in tick_idx:
                    ts = times_for_labels[i]
                    
                    if period == "1d":
                        # 1D Chart: Show Time
                        label = ts.strftime("%H:%M")
                    else:
                        # All other charts: Show Date Only
                        
                        # logic: if 15:55 -> show next day
                        if ts.hour == 15 and ts.minute == 55:
                            ts = ts + timedelta(days=1)
                        
                        label = ts.strftime("%Y-%m-%d")
                        
                    final_labels.append(label)

                self.ax.set_xticklabels(final_labels, rotation=45, ha='right', fontsize=8)

            # --- Finalize ---
            self.ax.set_facecolor('#121212')
            self.figure.patch.set_facecolor('#121212')

            self.canvas.draw()
            
            # Update state for hover
            self.last_plot_df = plot_df

        except Exception as e:
            self.log(f"Chart Render Error: {e}")

    def update_technicals(self, data, hv, ewma, sentiment, period_return, garch_vol=None):
        self.lbl_price.config(text=f"${data['Close']:.2f}")
        
        rsi_val = data['RSI']
        rsi_c = "green" if rsi_val < 30 else "red" if rsi_val > 70 else "white"
        self.lbl_rsi.config(text=f"{rsi_val:.2f}", foreground=rsi_c)
        
        stoch_val = data['StochRSI']
        stoch_c = "green" if stoch_val < 0.2 else "red" if stoch_val > 0.8 else "white"
        if pd.notna(data.get('StochRSI_D')):
            self.lbl_stoch.config(text=f"K:{stoch_val:.2f} D:{data['StochRSI_D']:.2f}", foreground=stoch_c)
        else:
            self.lbl_stoch.config(text=f"{stoch_val:.2f}", foreground=stoch_c)

        macd_val = data['MACD']
        macd_hist = data.get('MACD_Hist', 0)
        macd_c = "green" if macd_val > 0 else "red"
        hist_arrow = "+" if pd.notna(macd_hist) and macd_hist > 0 else "-"
        self.lbl_macd.config(text=f"{macd_val:.2f} (H:{hist_arrow}{abs(macd_hist):.2f})" if pd.notna(macd_hist) else f"{macd_val:.2f}", foreground=macd_c)
        
        # 1. VWAP Display
        # Show percentage distance from VWAP
        if 'VWAP' in data and not pd.isna(data['VWAP']):
            vwap = data['VWAP']
            price = data['Close']
            diff = ((price - vwap) / vwap) * 100
            
            # Green if price is ABOVE VWAP (Bullish), Red if BELOW (Bearish)
            vwap_c = "green" if diff > 0 else "red"
            self.lbl_vwap.config(text=f"{diff:+.2f}%", foreground=vwap_c)
        else:
            self.lbl_vwap.config(text="N/A", foreground="gray")

        # 2. ADX Display with Direction (B5)
        if 'ADX' in data and not pd.isna(data['ADX']):
            adx = data['ADX']
            adx_c = "#ffd700" if adx > 25 else "gray" if adx < 20 else "white"
            strength = "Strong" if adx > 25 else "Weak" if adx < 20 else "Neutral"
            # Show direction from +DI/-DI
            plus_di = data.get('+DI', 0)
            minus_di = data.get('-DI', 0)
            if not pd.isna(plus_di) and not pd.isna(minus_di):
                direction = "Bull" if plus_di > minus_di else "Bear"
            else:
                direction = ""
            self.lbl_adx.config(text=f"{adx:.1f} {strength} ({direction})", foreground=adx_c)
        else:
            self.lbl_adx.config(text="N/A", foreground="gray")

        # 3. OBV Display
        # Compare current OBV to its moving average to determine trend
        if 'OBV' in data and 'OBV_SMA' in data:
            obv_val = data['OBV']
            obv_avg = data['OBV_SMA']
            
            if pd.isna(obv_val) or pd.isna(obv_avg):
                self.lbl_obv.config(text="Wait...", foreground="gray")
            else:
                # Green if Volume is supporting price upward
                obv_c = "green" if obv_val > obv_avg else "red"
                obv_txt = "Bullish" if obv_val > obv_avg else "Bearish"
                self.lbl_obv.config(text=obv_txt, foreground=obv_c)
        
        self.update_pe_display()
        
        bb_pos = "Inside"
        bb_c = "white"
        if data['Close'] > data['BB_Upper']:
            bb_pos = "Overbought"; bb_c = "red"
        elif data['Close'] < data['BB_Lower']:
            bb_pos = "Oversold"; bb_c = "green"
        pctb = data.get('BB_PctB')
        bw = data.get('BB_Width')
        bb_extra = ""
        if pd.notna(pctb) and pd.notna(bw):
            bb_extra = f" | %B:{pctb:.2f} BW:{bw:.2f}"
        self.lbl_bb.config(text=f"{bb_pos}{bb_extra}\n[{data['BB_Lower']:.2f}-{data['BB_Upper']:.2f}]", foreground=bb_c)
        
        self.lbl_atr.config(text=f"${data['ATR']:.2f}")
        
        # --- EWMA & HV DISPLAY ---
        garch_txt = ""
        if garch_vol is not None and garch_vol > 0:
            garch_txt = f" | GARCH: {garch_vol:.1%}"
        self.lbl_vol.config(text=f"HV: {hv:.1%} | EWMA: {ewma:.1%}{garch_txt}")
        
        if self.use_sentiment:
            if sentiment is not None:
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

        # --- Williams %R Display ---
        williams = data.get('Williams_R')
        if williams is not None and pd.notna(williams):
            wr_c = "green" if williams < -80 else "red" if williams > -20 else "white"
            self.lbl_williams.config(text=f"{williams:.1f}", foreground=wr_c)
        else:
            self.lbl_williams.config(text="N/A", foreground="gray")

        # --- CCI Display ---
        cci = data.get('CCI')
        if cci is not None and pd.notna(cci):
            cci_c = "green" if cci < -100 else "red" if cci > 100 else "white"
            self.lbl_cci.config(text=f"{cci:.1f}", foreground=cci_c)
        else:
            self.lbl_cci.config(text="N/A", foreground="gray")

        self.btn_opt.config(state="normal", text=f"🔎 Open {self.current_ticker} Option Scanner")
    
    def visualize_3d(self, option_type):
        """Interactive 3D Plot with Uniform Color Types (Fixes Ragged Array Error)."""
        # Snapshot scan_data under the lock since a scan thread may be appending.
        with self._scan_lock:
            if not getattr(self, 'scan_data', None):
                messagebox.showinfo("3D Plot", "No data to plot. Please run a Scan first.")
                return
            base_data = [dict(row) for row in self.scan_data if row['type'] == option_type]
        if not base_data:
            messagebox.showinfo("3D Plot", f"No {option_type} data found.")
            return

        vis_win = Toplevel(self.root)
        vis_win.title(f"3D Analysis: {self.current_ticker} {option_type}s")
        vis_win.geometry("1000x850")
        vis_win.configure(bg="#1e1e1e")

        # --- CONTROLS ---
        ctrl_frame = ttk.LabelFrame(vis_win, text="Filter Conditions", padding=10)
        ctrl_frame.pack(side="top", fill="x", padx=10, pady=5)

        var_earn_under = tk.BooleanVar(value=True)
        var_earn_over  = tk.BooleanVar(value=True)
        var_reg_under  = tk.BooleanVar(value=True)
        var_reg_over   = tk.BooleanVar(value=True)

        # --- FIGURE ---
        fig = Figure(figsize=(8, 6), dpi=100, facecolor="#1e1e1e")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#1e1e1e")
        
        canvas = FigureCanvasTkAgg(fig, master=vis_win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- BOTTOM BUTTON ---
        btn_frame = ttk.Frame(vis_win)
        btn_frame.pack(fill="x", pady=10)
        btn_export = ttk.Button(btn_frame, text="Save HTML (Current View)", state="normal")
        btn_export.pack(fill="x", padx=50)

        # --- REFRESH FUNCTION ---
        def refresh_plot():
            ax.clear()
            
            dates_x, strikes, evs, colors, sizes = [], [], [], [], []
            date_labels, vols = [], []
            
            today = datetime.now()
            all_evs = [r['ev'] for r in base_data]
            cmap = plt.get_cmap('RdYlGn')
            norm = plt.Normalize(vmin=min(all_evs), vmax=max(all_evs))

            for row in base_data:
                is_earn = row['is_earnings']
                is_good = row['is_good']
                
                visible = False
                c = (0.5, 0.5, 0.5, 1.0) # Default gray tuple
                s = 20

                # Filter Logic
                if is_earn:
                    if is_good:
                        if var_earn_under.get():
                            visible = True
                            c = mcolors.to_rgba('#00ffff') 
                            s = 50
                    else:
                        if var_earn_over.get():
                            visible = True
                            c = mcolors.to_rgba('#af00ff') 
                            s = 50
                else:
                    if is_good:
                        if var_reg_under.get():
                            visible = True
                            c = cmap(norm(row['ev']))
                    else:
                        if var_reg_over.get():
                            visible = True
                            c = cmap(norm(row['ev']))

                if visible:
                    try:
                        dt = datetime.strptime(row['date'], "%Y-%m-%d")
                        days = (dt - today).days
                        
                        dates_x.append(days)
                        strikes.append(float(row['strike']))
                        evs.append(float(row['ev']))
                        colors.append(c) # Now this list only contains Tuples!
                        sizes.append(s)
                        date_labels.append(row['date'])
                        vols.append(row['vol'])
                    except:
                        continue

            # Plotting
            if dates_x:
                ax.scatter(dates_x, strikes, evs, c=colors, s=sizes, 
                           edgecolors='black', linewidth=0.5, alpha=0.9)

            ax.set_xlabel('Days', color='white'); ax.set_ylabel('Strike', color='white'); ax.set_zlabel('EV', color='white')
            ax.set_title(f"{self.current_ticker} {option_type} Landscape", color='white')
            ax.tick_params(colors='white'); ax.grid(color='gray', linestyle='--', linewidth=0.5)
            
            # --- SNAPSHOT DATA FOR EXPORT ---
            btn_export.config(command=lambda 
                d=dates_x, s=strikes, e=evs, dl=date_labels, v=vols, c=colors: 
                self.save_3d_html(option_type, d, s, e, dl, v, c)
            )

            canvas.draw()

        # --- CHECKBOXES ---
        cb_eu = ttk.Checkbutton(ctrl_frame, text="Earnings (Good)", variable=var_earn_under, command=refresh_plot)
        cb_eo = ttk.Checkbutton(ctrl_frame, text="Earnings (Bad)", variable=var_earn_over, command=refresh_plot)
        cb_ru = ttk.Checkbutton(ctrl_frame, text="Undervalued (Regular)", variable=var_reg_under, command=refresh_plot)
        cb_ro = ttk.Checkbutton(ctrl_frame, text="Overvalued (Regular)", variable=var_reg_over, command=refresh_plot)

        ttk.Label(ctrl_frame, text="[Cyan]", foreground="#00ffff").pack(side="left")
        cb_eu.pack(side="left", padx=10)
        ttk.Label(ctrl_frame, text="[Purple]", foreground="#af00ff").pack(side="left")
        cb_eo.pack(side="left", padx=10)
        ttk.Label(ctrl_frame, text="[Greenish]", foreground="#90ee90").pack(side="left")
        cb_ru.pack(side="left", padx=10)
        ttk.Label(ctrl_frame, text="[Reddish]", foreground="#ffcccb").pack(side="left")
        cb_ro.pack(side="left", padx=10)

        # Initial Render
        refresh_plot()
    
    def save_3d_html(self, option_type, dates, strikes, evs, date_labels, vol, colors_list):
        if not PLOTLY_AVAILABLE:
            messagebox.showerror("Error", "Plotly not installed.")
            return

        filename = filedialog.asksaveasfilename(
            initialfile=f"{self.current_ticker}_{option_type}_3D_Analysis.html",
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html")]
        )
        if not filename: return

        try:
            # Convert Matplotlib Tuples (0.0-1.0) to Plotly CSS Strings (rgb(0-255))
            plotly_colors = []
            for c in colors_list:
                r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
                plotly_colors.append(f"rgb({r}, {g}, {b})")

            hover_texts = []
            for d_str, stk, val, vols, c_code in zip(date_labels, strikes, evs, vol, plotly_colors):
                # Check for Cyan (0, 255, 255)
                if "0, 255, 255" in str(c_code):
                    type_str = "<b style='color:cyan'>EARNINGS (GOOD)</b>"
                # Check for Purple (175, 0, 255) -> derived from #af00ff
                elif "175, 0, 255" in str(c_code):
                    type_str = "<b style='color:magenta'>EARNINGS (BAD)</b>"
                else:
                    type_str = "<b>REGULAR</b>"

                txt = (f"{type_str}<br><b>Date:</b> {d_str}<br>"
                       f"<b>Strike:</b> ${stk}<br><b>Vol:</b> {int(vols)}<br><b>EV:</b> {val:+.2f}")
                hover_texts.append(txt)

            fig = go.Figure(data=[go.Scatter3d(
                x=dates, y=strikes, z=evs,
                mode='markers',
                marker=dict(
                    size=5,
                    color=plotly_colors, 
                    opacity=0.9
                ),
                text=hover_texts, 
                hoverinfo="text"
            )])

            fig.update_layout(
                title=f"{self.current_ticker} {option_type} Landscape (Filtered)",
                scene=dict(
                    xaxis_title='Days to Expiry', yaxis_title='Strike', zaxis_title='EV',
                    bgcolor='#1e1e1e',
                    xaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                    yaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                    zaxis=dict(backgroundcolor="#1e1e1e", color="white"),
                ),
                paper_bgcolor='#1e1e1e', font=dict(color="white")
            )

            plot(fig, filename=filename, auto_open=True)

        except Exception as e:
            self.log(f"HTML Export Error: {e}")
    def on_hover(self, event):
        if event.inaxes != self.ax or self.last_plot_df is None or self.last_plot_df.empty:
            if self.hover_annot:
                self.hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return

        try:
            idx = int(round(event.xdata))
            idx = int(np.clip(idx, 0, len(self.last_plot_df) - 1))
            xval = idx

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
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
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
    
    
    def _normalize_div_yield(self, div):
        """Normalize a dividend yield to decimal form (e.g. 0.0294 for 2.94%).

        yfinance versions vary: some return decimal (0.0294), others return
        percent (2.94). Treat any value > 1 as percent and divide by 100.
        """
        if div is None:
            return None
        try:
            d = float(div)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(d) or d < 0:
            return None
        if d > 1:
            d = d / 100.0
        return d

    def get_info(self):
        """Consolidated fundamental fetch called when ticker changes."""
        try:
            stock = self.stock
            info = self.data_provider.get_info(stock)
            
            # 1. Basic Fundamental Extraction
            self.pe_fwd = self._to_finite_float(info.get('forwardPE'))
            self.pe_ttm = self._to_finite_float(info.get('trailingPE'))
            self.peg_ratio = self._to_finite_float(info.get('trailingPegRatio'))
            self.earnings_growth = self._to_finite_float(info.get('earningsGrowth'))
            
            # 2. Compute derived valuation metrics
            self.calculate_pe_percentile(stock)
            self.compute_peg_ratio()
            
            # 3. Force UI update now that data is ready
            self.root.after(0, self.update_pe_display)
            
        except Exception as e:
            self.log(f"Fundamental fetch error: {e}")
            self.root.after(0, self.update_pe_display)

    def get_smart_dividend(self, stock_obj):
        """
        Retrieves the dividend yield (as a decimal) using a priority queue:
        1. fast_info (Fastest, lightest)
        2. info.dividendYield (Standard)
        3. info.trailingAnnualDividendYield (Fallback)

        All sources are normalized via _normalize_div_yield so percent/decimal
        ambiguity across yfinance versions is handled uniformly. Always returns
        a finite float (0.0 when nothing is available).
        """
        try:
            # 1. fast_info (milliseconds)
            fast_div = self._normalize_div_yield(
                self.data_provider.get_fast_info(stock_obj).get('dividend_yield')
            )
            if fast_div is not None:
                print(f"[DEBUG] Found Dividend (fast_info): {fast_div:.4%}")
                return fast_div

            # 2. Full info blob
            info = self.data_provider.get_info(stock_obj)
            div = self._normalize_div_yield(info.get('dividendYield'))
            if div is not None:
                print(f"[DEBUG] Found Dividend (info/dividendYield): {div:.4%}")
                return div
            div = self._normalize_div_yield(info.get('trailingAnnualDividendYield'))
            if div is not None:
                print(f"[DEBUG] Found Dividend (info/trailing): {div:.4%}")
                return div

        except Exception as e:
            print(f"[DEBUG] Div fetch error: {e}")

        # Fallthrough: no dividend data available (or error). Return 0.0 so
        # callers that do math.exp(-q*T) don't blow up on None.
        return 0.0

    def open_options_window(self):
        if not self.current_ticker: return
        win = Toplevel(self.root)
        win.title(f"Options Explorer: {self.current_ticker}")
        win.geometry("1200x800")
        win.configure(bg=DARK_BG)

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
        
        # --- EXPANDED COLUMNS: Greeks, POP, OI, Spread ---
        cols = ("Date", "Type", "Strike", "Vol", "OI", "Price", "Spread%",
                "Breakeven", "Imp Vol", "Fair", "EV", "Delta", "Gamma", "Theta", "Vega", "POP", "Verdict")
        self.tree = ttk.Treeview(right_panel, columns=cols, show="headings")

        col_widths = {"Date": 90, "Breakeven": 75, "Verdict": 65,
                      "Delta": 55, "Gamma": 55, "Theta": 55, "Vega": 55, "POP": 50,
                      "OI": 55, "Spread%": 55}
        for c in cols:
            self.tree.heading(c, text=c, command=lambda _c=c: self.treeview_sort_column(self.tree, _c, False))
            w = col_widths.get(c, 60)
            self.tree.column(c, width=w, anchor="center")
        
        scr = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scr.set); self.tree.pack(side="left", fill="both", expand=True); scr.pack(side="right", fill="y")
        
        # Green (Undervalued): "Sage Green" 
        # Old: #d4f8d4 (Too bright)
        self.tree.tag_configure("green", background="#8fbc8f", foreground="black")
        
        # Red (Overvalued): "Muted Salmon"
        # Old: #f8d4d4 (Too bright)
        self.tree.tag_configure("red", background="#e57373", foreground="black")   
    
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
                cols = ("Date", "Type", "Strike", "Vol", "OI", "Price", "Spread%",
                        "Breakeven", "Imp Vol", "Fair", "EV", "Delta", "Gamma", "Theta", "Vega", "POP", "Verdict")
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
        self.all_exps = self.data_provider.get_option_expirations(stock)
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

    def _fetch_rate_curve(self):
        """Fetches ^IRX (short) and ^TNX (long) rates once. Returns (short_rate, long_rate)."""
        return self.data_provider.fetch_rate_curve()

    def update_pe_display(self):
        """Updates only the P/E labels. Called when get_info finishes."""
        try:
            # Update TTM/Fwd Label
            pe_ttm_str = self._format_metric(self.pe_ttm, decimals=2)
            pe_fwd_str = self._format_metric(self.pe_fwd, decimals=2)
            self.lbl_pe.config(text=f"TTM: {pe_ttm_str} | Fwd: {pe_fwd_str}", foreground="white")

            # Update Percentile Label
            if self._is_finite_number(self.pe_percentile):
                p_val = float(self.pe_percentile)
                p_color = "red" if p_val > 80 else "green" if p_val < 20 else "white"
                self.lbl_pe_percentile.config(text=f"{p_val:.1f}% (TTM)", foreground=p_color)
            else:
                self.lbl_pe_percentile.config(
                    text=self._valuation_status_text("pe_percentile_reason", default="Loading..."),
                    foreground="gray"
                )

            # Update PEG Ratio Label
            if self._is_number(self.peg_ratio):
                peg_val = float(self.peg_ratio)
                if math.isinf(peg_val):
                    peg_color = "orange"
                elif peg_val < 0:
                    peg_color = "orange"
                else:
                    peg_color = "green" if peg_val < 1.0 else "red" if peg_val > 2.0 else "white"
                self.lbl_peg.config(text=self._format_metric(peg_val, decimals=2), foreground=peg_color)
            else:
                self.lbl_peg.config(
                    text=self._valuation_status_text("peg_reason", default="Not Calculable"),
                    foreground="gray"
                )
                
        except Exception as e:
            self.log(f"UI Update Error (Display): {e}")


    def _flush_option_rows(self, rows):
        """Insert a batch of option-scan rows on the UI thread."""
        tree = self.tree
        for vals, tag in rows:
            tree.insert("", "end", values=vals, tags=(tag,))

    def fetch_options_batch(self, dates, filter_under_only=False):
        with self._scan_lock:
            self.scan_data = []

        stock = self.stock
        spot = float(self.current_price)
        DIV_YIELD = self.get_smart_dividend(stock)

        # Fetch rate curve once for the entire batch (provider TTL-caches ^IRX/^TNX)
        _short_rate, _long_rate = self._fetch_rate_curve()

        earnings_contracts = set()
        if self.projected_earnings and hasattr(self, 'all_exps') and self.all_exps:
            for p_date in self.projected_earnings:
                p_str = p_date.strftime("%Y-%m-%d")
                valid_exps = [e for e in self.all_exps if e >= p_str]
                if valid_exps:
                    earnings_contracts.add(min(valid_exps))

        historical_vol_base = self._to_finite_float(self.hv_30)
        today = datetime.now().date()
        ui_batch = []
        UI_BATCH_SIZE = 40
        scan_buf = []

        def flush_ui():
            nonlocal ui_batch
            if not ui_batch:
                return
            batch = ui_batch
            ui_batch = []
            self.root.after(0, lambda b=batch: self._flush_option_rows(b))

        for date in dates:
            try:
                exp_date = datetime.strptime(date, "%Y-%m-%d").date()
                trading_days = int(np.busday_count(today, exp_date))
                T = max(trading_days / 252.0, 1 / 252)

                if T <= 0.25:
                    RFR = _short_rate
                else:
                    t_clamped = min(max(T, 0.25), 10.0)
                    weight = (t_clamped - 0.25) / (10.0 - 0.25)
                    RFR = _short_rate + weight * (_long_rate - _short_rate)

                chain = self.data_provider.get_option_chain(stock, date)
                calls = chain.calls.assign(Type="CALL")
                puts = chain.puts.assign(Type="PUT")
                all_options = pd.concat([calls, puts], ignore_index=True)
                if all_options.empty:
                    continue
                if 'volume' in all_options.columns:
                    all_options = all_options.sort_values('volume', ascending=False, kind='mergesort')

                # Vectorized mid prices for parity map + scan filters
                bid = pd.to_numeric(all_options.get('bid', 0), errors='coerce').fillna(0.0).to_numpy(dtype=float)
                ask = pd.to_numeric(all_options.get('ask', 0), errors='coerce').fillna(0.0).to_numpy(dtype=float)
                last = pd.to_numeric(all_options.get('lastPrice', 0), errors='coerce').fillna(0.0).to_numpy(dtype=float)
                vol = pd.to_numeric(all_options.get('volume', 0), errors='coerce').fillna(0.0).to_numpy(dtype=float)
                oi_arr = pd.to_numeric(all_options.get('openInterest', 0), errors='coerce').fillna(0.0).to_numpy(dtype=float)
                iv_arr = pd.to_numeric(all_options.get('impliedVolatility', 0), errors='coerce').to_numpy(dtype=float)
                strikes = pd.to_numeric(all_options['strike'], errors='coerce').to_numpy(dtype=float)
                types = all_options['Type'].to_numpy()

                has_ba = (bid > 0) & (ask > 0)
                mid = np.where(has_ba, 0.5 * (bid + ask), last)
                spread_pct = np.where(
                    has_ba & (mid > 0),
                    ((ask - bid) / mid) * 100.0,
                    999.0,
                )

                # Parity map from vectorized mids
                parity_map = {}
                for i in range(len(strikes)):
                    if mid[i] <= 0 or not np.isfinite(mid[i]):
                        continue
                    s = float(strikes[i])
                    bucket = parity_map.setdefault(s, {})
                    bucket[str(types[i]).lower()] = float(mid[i])

                iv_weight = min(T * 4.0, 0.8)
                hv_weight = 1.0 - iv_weight
                sqrtT = math.sqrt(T)
                is_earnings = date in earnings_contracts
                threshold = 0.25 if is_earnings else 0.15
                parity_bounds = VegaChimpCore.american_put_call_parity_bounds
                Ncdf = VegaChimpCore.N

                # ---- Filter eligible contracts (vector masks) ----
                kind_is_call = (types == "CALL")
                valid = (
                    (vol > 0) & np.isfinite(vol)
                    & (mid > 0) & np.isfinite(mid)
                    & np.isfinite(iv_arr) & (iv_arr >= 0.01)
                    & np.isfinite(strikes) & (strikes > 0)
                )
                # Wide spread + zero OI filter (match prior scalar rule)
                oi_int = np.where(np.isfinite(oi_arr), oi_arr, 0.0)
                valid &= ~((spread_pct > 50.0) & (oi_int == 0))

                idx = np.flatnonzero(valid)
                if idx.size == 0:
                    continue

                strikes_v = strikes[idx]
                mid_v = mid[idx]
                iv_v = iv_arr[idx]
                vol_v = vol[idx]
                oi_v = oi_int[idx]
                sp_v = spread_pct[idx]
                is_call_v = kind_is_call[idx]
                types_v = types[idx]

                historical_vol = historical_vol_base
                if historical_vol is None or historical_vol <= 0:
                    historical_vol = float(np.nanmedian(iv_v)) if iv_v.size else 0.25
                # Optional GARCH blend into the historical leg (default off)
                if self.use_garch_blend and getattr(self, 'garch_vol', 0) > 0:
                    historical_vol = 0.5 * historical_vol + 0.5 * float(self.garch_vol)

                # Optional quadratic smile smoother for the IV leg
                iv_for_blend = iv_v.copy()
                forward = spot * math.exp((RFR - DIV_YIELD) * T)
                smile_coef = None
                if self.use_smile_vol:
                    smile_coef = fit_quadratic_smile(strikes_v, iv_v, forward)
                    if smile_coef is not None:
                        iv_for_blend = smile_vol_arr(strikes_v, forward, smile_coef)

                vol_input = np.where(
                    iv_for_blend < 0.05,
                    historical_vol,
                    iv_weight * iv_for_blend + hv_weight * historical_vol,
                )

                kinds = np.where(is_call_v, 'call', 'put')
                # Batch numpy path wins above ~64 contracts; below that scalar
                # BS2002 is faster (numpy overhead dominates small n).
                _BATCH_N = 64
                if idx.size >= _BATCH_N:
                    fair_v = VegaChimpCore.bjerksund_stensland_batch(
                        spot, strikes_v, T, RFR, DIV_YIELD, vol_input, kinds,
                    )
                else:
                    fair_v = np.array([
                        VegaChimpCore.bjerksund_stensland(
                            spot, float(strikes_v[j]), T, RFR, DIV_YIELD,
                            float(vol_input[j]), kinds[j],
                        )
                        for j in range(idx.size)
                    ], dtype=float)

                # Greeks: American FD (batch when large) or European BS
                greeks_map = None
                if self.use_american_greeks and idx.size >= _BATCH_N:
                    greeks_map = VegaChimpCore.american_greeks_batch(
                        spot, strikes_v, RFR, DIV_YIELD, iv_v, T, kinds,
                    )

                for j, i in enumerate(idx):
                    fair = float(fair_v[j])
                    if fair <= 0 or not math.isfinite(fair):
                        continue
                    market_price = float(mid_v[j])
                    strike = float(strikes_v[j])
                    iv = float(iv_v[j])
                    kind_str = kinds[j]
                    ev = fair - market_price
                    oi = int(oi_v[j])
                    sp = float(sp_v[j])

                    if greeks_map is not None:
                        greeks = {
                            'delta': float(greeks_map['delta'][j]),
                            'gamma': float(greeks_map['gamma'][j]),
                            'theta': float(greeks_map['theta'][j]),
                            'vega': float(greeks_map['vega'][j]),
                        }
                    elif self.use_american_greeks:
                        greeks = VegaChimpCore.american_greeks(
                            spot, strike, RFR, DIV_YIELD, iv, T, kind_str,
                        )
                    else:
                        greeks = VegaChimpCore.bs_greeks(
                            spot, strike, RFR, DIV_YIELD, iv, T, kind_str,
                        )

                    try:
                        if kind_str == "call":
                            breakeven_price = strike + market_price
                        else:
                            breakeven_price = strike - market_price
                        if breakeven_price <= 0:
                            pop = 0.0
                        else:
                            d2_pop = (
                                math.log(spot / breakeven_price)
                                + (RFR - DIV_YIELD - 0.5 * iv * iv) * T
                            ) / (iv * sqrtT)
                            pop = Ncdf(d2_pop) * 100 if kind_str == "call" else Ncdf(-d2_pop) * 100
                            pop = max(0.0, min(100.0, pop))
                    except Exception:
                        pop = 0.0

                    parity_warn = ""
                    parity_data = parity_map.get(strike, {})
                    if 'call' in parity_data and 'put' in parity_data:
                        lower, upper = parity_bounds(spot, strike, RFR, DIV_YIELD, T)
                        observed = parity_data['call'] - parity_data['put']
                        if observed < lower - 0.10 or observed > upper + 0.10:
                            parity_warn = "!"

                    is_undervalued = ev > threshold
                    scan_buf.append({
                        'date': date,
                        'type': types_v[j],
                        'strike': strike,
                        'ev': ev,
                        'vol': float(vol_v[j]),
                        'is_earnings': is_earnings,
                        'is_good': is_undervalued,
                    })

                    breakeven = strike + market_price if kind_str == "call" else strike - market_price
                    verdict = "Fair"
                    tag = ""
                    if is_undervalued:
                        verdict = "Earnings Under" if is_earnings else "Under"
                        tag = "green"
                    elif ev < -0.05:
                        verdict = "Earnings Over" if is_earnings else "Over"
                        tag = "red"
                    if parity_warn:
                        verdict += " " + parity_warn
                    if filter_under_only and "Under" not in verdict:
                        continue

                    vals = (
                        date, types_v[j], f"{strike:.2f}", int(vol_v[j]),
                        oi, f"{market_price:.2f}", f"{sp:.0f}%",
                        f"{breakeven:.2f}", f"{iv:.1%}",
                        f"{fair:.2f}", f"{ev:+.2f}",
                        f"{greeks['delta']:.3f}", f"{greeks['gamma']:.4f}",
                        f"{greeks['theta']:.3f}", f"{greeks['vega']:.3f}",
                        f"{pop:.0f}%", verdict,
                    )
                    ui_batch.append((vals, tag))
                    if len(ui_batch) >= UI_BATCH_SIZE:
                        flush_ui()

            except Exception as e:
                self.log(f"Options fetch error for {date}: {e}")

        flush_ui()
        if scan_buf:
            with self._scan_lock:
                self.scan_data.extend(scan_buf)

