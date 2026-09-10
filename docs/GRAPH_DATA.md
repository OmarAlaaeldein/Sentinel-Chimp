# Stock Relationship Graph — data sources

Sentinel’s relationship network is a **curated static graph** plus an optional
**Sectivia** supply-chain overlay. It is **not** a live scraped web graph.

## Attribution

```
Supply-chain data: Sectivia (https://sectivia.com), CC BY 4.0
```

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  
Dataset page: https://sectivia.com/dataset/

## What ships in the app

| Layer | Location | Contents |
| --- | --- | --- |
| Curated base | `core/graph_data.py` | NASDAQ-100 + S&P titans nodes; competitors, suppliers, infrastructure/power partners, ETF correlated peers |
| Sectivia overlay | `data/sectivia/` (cached CSV/JSON) | Hand-mapped **supplier → customer** edges (~350) across AI, energy, defense, medicine, quantum, finance |
| Merge | `core/sectivia_import.py` + `build_default_graph()` | Union merge; duplicates keep curated |

### Conflict policy

- Edges are keyed by `(source, target, relation)`.
- If curated already has that key, **curated wins** (Sectivia duplicate skipped).
- Sectivia-only edges are added as:
  - `SUPPLIER_TO` — supplier sells to customer
  - `CUSTOMER_OF` — reverse perspective (same link, complementary label)
- Sectivia weight defaults to `0.85` (`SECTIVIA_EDGE_WEIGHT`).

## Sectivia cache (offline-friendly)

Vendored/cached under `data/sectivia/`:

- `sectivia-relations.csv`
- `sectivia-companies.csv`
- `sectivia-supply-chain.json` (optional convenience)
- `ATTRIBUTION.txt`, `fetched_at.txt`

Refresh (network):

```python
from core.sectivia_import import refresh_sectivia_cache
refresh_sectivia_cache()
```

`build_default_graph()` loads the cache when present; missing cache is a soft
no-op so Lite Mode / offline installs still start.

## How curated edges were chosen

Prefer accuracy over spam. Typical provenance for hand edits in `graph_data.py`:

- Company **10-K / 10-Q** customer concentration and supplier risk disclosures
- Investor presentations and product documentation
- Widely reported hyperscaler / foundry / auto Tier-1 relationships
- ETF holdings as `CORRELATED_PEER` (e.g. SMH, XLE, XLY) — correlation, not contracts

## Other public / future sources (not wired as live scrapers)

| Source | Notes |
| --- | --- |
| **Sectivia** (primary) | Free CC BY 4.0 CSV/JSON; commercial OK with attribution — **ingested** |
| [MonarchCastleTech/supplychain](https://github.com/MonarchCastleTech/supplychain) | Top-100 map with per-edge provenance/confidence; MIT code, check third-party notices before bulk reuse |
| **Wikidata** | Ownership / industry / subsidiary structure — weak for supplier lists |
| **OpenCorporates** | Legal entities, not product supply chains |
| Vendor “customer lists” / 10-K mentions | Good for curated one-offs |
| **FactSet Revere** / similar | Paid institutional supply-chain graphs — future commercial option |
| [lang2org/stock-supply-chain](https://github.com/lang2org/stock-supply-chain) | Per-company JSON with supplies / competes_with — future ingest candidate |

Future work: optional CLI `graph refresh-sectivia`, selective merge flags, and
importers for Monarch / Wikidata ownership edges — **not** a full web scraper.

## GUI

In the main window, click **Graph** (next to News) after loading a ticker to open
peers, a short 1‑month divergence summary, and **Open 2D/3D HTML** (temp Plotly
file + system browser). Lite Mode already includes Plotly for 3D options.
