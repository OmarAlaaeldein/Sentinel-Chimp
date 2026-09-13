# September reliability fixes

Options scans now capture GUI inputs before launching workers. Only callbacks for
the current ticker, options window and scan request can publish rows; opening a
new options window replaces the old one. Fundamentals compute on isolated state
and publish atomically on the Tk thread. Existing 3D windows keep their ticker
snapshot for subsequent redraws and exports.

Dividend resolution records its source and individual source failures in CLI JSON
(`dividend_status`); failed fast-info requests still try ordinary/trailing yield.
Missing yield is distinguishable from a confirmed zero. Option rows expose bid,
ask and a canonical category separately from the display verdict. Long breakeven
and risk-neutral PoP use ask as entry price. Fair contracts remain Fair in 3D views
and have their own filter.

Remaining expiry time uses the NYSE session calendar, including holidays and early
closes. `scan_option_chains(valuation_time=...)` accepts an aware valuation time for
repeatable calculations. All rate and volatility terms share the existing trading
year convention: remaining session seconds divided by 252 × 6.5 hours. This is an
explicit model convention, not an ACT/365 conversion or an independently calibrated
pricing model. After expiry close a chain is skipped. This calendar applies to US
equity options; AM-settled/index contracts require a product-specific adapter.

GUI settings use `SENTINEL_CONFIG_DIR`, or platform user storage (Application Support,
APPDATA or XDG_CONFIG_HOME). JSON replacement is atomic; failed saves are observable,
and an intentionally empty watchlist stays empty. Legacy settings in the source
folder can be copied to the new directory once to preserve prior customizations.

News requests retain HTTPS certificate checks and normalize timestamps to UTC.
Chart date labels retain actual session dates and earnings markers stay inside the
visible date range. Peer payloads retain all direct relationship edges and expose
source, target and direction; CLI/GUI displays show endpoints. Default graph summary
JSON exposes Sectivia cache merge status and attribution. Native build scripts include
the cached graph data and the new exchange calendar resources.

Offline regressions cover stale/closed GUI requests, dividend fallbacks, ask-based
metrics, calendar boundaries, chart labels, storage failure, relationship direction,
Fair plot categories, TLS failure and shell input validation. Native builds and
live provider behavior still require platform/live verification.
