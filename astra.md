**Sentinel Chimp: practical updates for local Ollama and a CLI for humans and agents**

Prepared 2026-09-07 against checkout `3a8ac1b`. This is a proposed implementation plan, based on reading the application source, tests, documentation, launchers, dependencies, and CI/release workflows. Commands described as proposed do not exist yet. Existing functionality and newly observed issues are distinguished below.

**Implementation update, 2026-09-07:** The audit below describes the starting checkout.
The first implementation now provides local `models list/show/select/current`,
`profiles list/use`, `config`, and bounded `ask` commands with JSON output.
See the current [CLI usage in README.md](README.md#local-ollama-cli) and the experiment
record at the end of this file. Remaining roadmap commands below are proposals;
`explain`, GUI selection, loaded-model status and full config management are not implemented.

**The best first release is model discovery, named configurations, dependable JSON, and one useful AI command.** Keep the existing argparse CLI and shared numerical services. Let users see the exact local model names, inspect their settings, save a configuration, and select it either interactively or with explicit arguments. Use that model to explain an existing analysis result. Extend news sentiment after extracting its orchestration from the GUI.

An external agent should be able to discover commands and consume numerical results without using a language model inside Sentinel. Ollama is an optional application capability; the CLI contract should work equally well when AI is disabled.

**What the app already provides gives us most of the foundation.**

| Area reviewed | Current behavior | Integration opportunity |
| --- | --- | --- |
| [sentinel.py](sentinel.py), [sentinel_cli.py](sentinel_cli.py), [main/__init__.py](main/__init__.py) | No arguments to `sentinel.py` launch Tk; arguments dispatch to CLI; module aliases exist. | Keep these entry points; add an installed `sentinel` command and explicit `gui` command. |
| [main/cli.py](main/cli.py) | `analyze`, `scan`, `verify`; text/JSON analysis and scan; CSV scan export. | Extend argparse; centralize formatting, validation, errors, and command discovery. |
| [core/scan_service.py](core/scan_service.py) | Shared scanner for GUI/CLI, dataclass results, ticker analysis, dividend/rate/earnings helpers. | Reuse these services for commands, saved results, and AI context. |
| [core/options_scan.py](core/options_scan.py) | Shared liquidity, moneyness, and tradeable-edge rules. | Make rules inspectable and configurable through a validated rules object. |
| [core/pricing.py](core/pricing.py) | European pricing/IV, American BS2002, scalar/batch Greeks, lattice cross-check, EWMA. | Expose offline calculators and diagnostics without rewriting pricing. |
| [core/technicals.py](core/technicals.py), [core/vol_models.py](core/vol_models.py) | Technicals, daily EMA mapping, Ichimoku, swing Fib, GARCH, smile, cones. | Expose already computed fields and cone data through the CLI. |
| [core/data.py](core/data.py) | `DataProvider` and Yahoo implementation; in-memory bounded caches. | Add data provenance and later a snapshot provider. CLI processes currently lose these caches on exit. |
| [core/sentiment.py](core/sentiment.py) | Only `FinBERT` / `ProsusAI/finbert`; 0–1 sentiment; model weights loaded on demand. | Introduce an optional Ollama provider and a common typed sentiment result. |
| [main/app.py](main/app.py) | Controller still owns news retrieval, sentiment orchestration, valuation, exports, and worker coordination. `use_sentiment = False` is hardcoded. | Extract reusable services; add real settings and protect results against ticker/model changes. |
| [ui/prefs.py](ui/prefs.py), [ui/watchlist.py](ui/watchlist.py) | JSON persistence near application source; preferences accept only predefined boolean keys. | Move persistence to a headless module with shared config paths and atomic writes. |
| [ui/chart.py](ui/chart.py), [ui/options_3d.py](ui/options_3d.py) | Chart rendering and reusable Plotly figure construction already exist. | Add file exports once package imports are decoupled from Tk. |
| [ui/news.py](ui/news.py), [ui/options_explorer.py](ui/options_explorer.py), [ui/theme.py](ui/theme.py), [ui/tooltip.py](ui/tooltip.py) | Display helpers, readable theme, scan legend, and news reader. | Reuse their terminology; add a compact model/profile selector and status. |
| [requirements.txt](requirements.txt), [requirements-ci.txt](requirements-ci.txt), [requirements-release.txt](requirements-release.txt) | Source requirements include torch/transformers; CI/release sets exclude them. | Add an explicitly supported lean CLI install and optional dependency groups. |
| [scripts](scripts), build scripts, [.github/workflows](.github/workflows) | Native Lite GUI releases; CI tests on Python 3.12. | Ship a console entry point and test packaged commands through captured stdout/stderr. |
| [tests](tests), [README.md](README.md), [CLAUDE.md](CLAUDE.md), [docs/LOGIC_REVIEW.md](docs/LOGIC_REVIEW.md), [plan.md](plan.md), [to_do.md](to_do.md) | Broad numerical tests and an existing roadmap; older audit text contains superseded recommendations. | Add CLI/provider contract tests and keep new work separate from already shipped features. |

Watchlists, Ichimoku, earnings markers, daily EMA mapping, probability cones, GARCH, the quadratic smile, American Greeks, and the shared scanner already exist. They should not be presented as missing features. This plan focuses on exposing and strengthening them.

**Your local Ollama files provide concrete model names for the selector.**

The following inventory was read from `~/.ollama/models/manifests` and their small configuration/parameter blobs. All referenced layer files were present. This establishes local file availability, not a successful inference test or proof that the running server uses this model directory.

| Exact name reconstructed from the manifest path | Stored family / parameter label | Stored quantization information | Approximate declared layer size |
| --- | --- | --- | --- |
| `qwen3.5:4b` | `qwen35` / `4.7B` | Config explicitly reports `Q4_K_M`. | 3.39 GB |
| `spark2.5:latest` | `spark2_5` / `4.11B` | Config reports `unknown`; model layer matches the Spark entry below. | 2.60 GB |
| `hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M` | `spark2_5` / `4.11B` | Tag says `Q4_K_M`; config reports `unknown`. | 2.60 GB |
| `hf.co/IFM/K2-Horizon-0.9B-GGUF:BF16` | `k2-horizon` / `1.08B` | Tag says `BF16`; config reports `unknown`. | 2.16 GB |

Sizes use decimal GB and sum declared layer sizes; they are not RAM requirements or unique disk usage. Both Spark names reference the same layer hashes, so preserve both aliases without double-counting their storage. The Qwen config contains `requires: 0.17.1`; report compatibility against the actual server version rather than assuming installation means usability.

None of these four manifests includes an explicit parameter layer. Do not invent their temperature, context size, or generation defaults. Inspect the server when connected and label omitted settings as inherited. Sentinel currently has no saved Ollama profiles and does not use any of these models. Their appearance here is an inventory, not a model-quality ranking.

Production discovery should use the configured Ollama API, not depend on these private filesystem layouts. This also accommodates another model directory or server setup.

**Make “models” and “profiles” two distinct, inspectable concepts.**

A model is an exact Ollama name/tag and its installed metadata. A profile is a Sentinel configuration that selects a model plus inference settings. The same model may have several profiles, such as `research` and `headlines`, without duplicating its weights.

Proposed commands:

| Command | Meaning | Writes state? |
| --- | --- | --- |
| `sentinel models list` | List models, local availability, active-profile marker, size, family, and quantization. | No |
| `sentinel models list --json` | Return exact names and metadata for agents. | No |
| `sentinel models show MODEL` | Inspect one model's capabilities, declared context limit, template/parameters, and version requirement when available. | No |
| `sentinel models current` | Show the effective profile/model and where each setting came from. | No |
| `sentinel models select` | TTY-only picker; choose a model and a profile name, then save/select it. | Yes |
| `sentinel models select MODEL --save-as NAME` | Create and activate a named profile from explicit arguments. Existing names require `--replace`. | Yes |
| `sentinel profiles list` / `profiles show NAME` | List saved configurations or inspect one. Works while Ollama is stopped. | No |
| `sentinel profiles use NAME` | Persist the default profile after validating its configuration; report server availability separately. | Yes |
| `sentinel profiles set NAME --temperature 0 --num-ctx 4096` | Update only the supplied, validated settings. | Yes |
| `sentinel config show` / `config path` / `config validate` | Inspect effective configuration, its storage location, or validation errors. | No |
| `sentinel doctor --json` | Check runtime, dependencies, config validity, and local Ollama metadata endpoints. | No |

Every command should support `--json`; interactive selection must fail with a structured missing-argument error in JSON/noninteractive mode. `--save-as` is required when selection runs without a terminal. Entering `qwen3.5:4b` must preserve that exact tag; displaying a short friendly label must not change its identity.

Example workflow after implementation:

```bash
sentinel models list
sentinel models show qwen3.5:4b
sentinel models select qwen3.5:4b --save-as research
sentinel profiles set research --temperature 0 --num-ctx 4096
sentinel models select spark2.5:latest --save-as headlines
sentinel profiles use research
sentinel models current --json
```

These examples use your observed model names but do not assert that either model is best for the named task. For the first implementation, a plain numbered terminal picker is sufficient. Searchable selection and shell completion can follow.

**Use a small Ollama HTTP adapter with explicit behavior.**

Add `core/ollama.py` with `list_models()`, `show_model(name)`, `running_models()`, `server_version()`, and `chat(...)`. Reuse `requests`, which is already a dependency. Keep HTTP transport injectable so tests can supply responses. An additional agent framework, vector database, or Ollama Python SDK is unnecessary for this initial scope.

| API | Proposed use | Primary reference |
| --- | --- | --- |
| `GET /api/tags` | Discover available names, digests, sizes, and basic details. | [Ollama model listing](https://docs.ollama.com/api/tags) |
| `POST /api/show` with a model name | Inspect capabilities, template, parameter text, and model metadata. Preserve raw parameter text alongside selected parsed values. | [Ollama model details](https://docs.ollama.com/api-reference/show-model-details) |
| `GET /api/ps` | Distinguish installed models from models currently loaded; show runtime context and memory information when reported. | [Ollama running models](https://docs.ollama.com/api/ps) |
| `GET /api/version` | Diagnose server compatibility. | [Ollama version](https://docs.ollama.com/api-reference/get-version) |
| `POST /api/chat` | Generate explanations and structured sentiment. Start with `stream: false`; supply a JSON schema through `format` for structured tasks. | [Ollama chat API](https://docs.ollama.com/api/chat) |

Model detail calls should be fetched on demand or cached by model digest; listing should not load every model into memory. Keep “installed,” “loaded,” “compatible,” and “selected” as separate states. Metadata inspection must not trigger inference or model downloads.

Use separate connection and inference timeouts: a suggested starting point is 2 seconds to connect, 5 seconds for metadata, and a configurable 120-second inference deadline. These are product defaults to measure, not performance guarantees. Bound any queue and retry only appropriate transient transport failures. Do not retry malformed output indefinitely. Check response status, response size, expected fields, and final completion state.

`/api/show` may expose capabilities without proving that a model reliably follows a particular schema. Provide an explicit `models test MODEL --task sentiment` diagnostic later, using a small built-in fixture and reporting latency, schema validity, and completion status. Keep this inference test out of normal discovery.

**Define offline behavior precisely.**

The initial feature should promise local AI inference while normal quote/news commands continue using their current online providers. A later `--offline --input FILE` workflow should prohibit live market/news requests and operate only on supplied snapshots. It may still contact the loopback Ollama service.

Default `ollama.host` to `http://127.0.0.1:11434` and `ollama.local_only` to `true`. Validate the endpoint, reject non-loopback destinations in this mode, disable proxy inheritance for loopback requests, and reject redirects that could change the destination. An explicit host override must not silently disable the local-only policy. A server bind address such as `0.0.0.0` needs an actionable message asking for a client address.

A local server can represent cloud-backed models, so localhost alone is insufficient. Ollama's API types include `remote_model` and `remote_host`; reject entries with remote backing before submitting prompts, and use version-aware checks for metadata omissions. Unknown compatibility/locality should be visible rather than labelled verified. [Ollama API type definitions](https://github.com/ollama/ollama/blob/main/api/types.go)

Ollama also documents a server-wide cloud disable setting, `disable_ollama_cloud`, or `OLLAMA_NO_CLOUD=1`, followed by a restart. Document that operator-controlled setting for a strict local setup. Setting an environment variable in Sentinel does not reconfigure a server that is already running. [Ollama local-only configuration](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features)

When Ollama is stopped, return `OLLAMA_UNAVAILABLE` with the endpoint and a practical next step. When a model is missing, return `MODEL_NOT_INSTALLED`. Ordinary analysis should continue working with AI disabled; an explicitly requested AI operation should report its failure. Never silently switch to another model, invoke a cloud service, pull weights, or start a background daemon during selection.

**Store configuration outside the application bundle and make resolution predictable.**

The existing preference file only handles booleans and silently ignores unsupported keys. Introduce `core/config.py`; do not put model strings into that boolean-only loader. Use an OS-appropriate per-user configuration directory, with `--config PATH` and `SENTINEL_CONFIG` overrides. Expose the resolved path through `config path` so users and agents need not guess.

Prefer JSON for the first version because the app already uses it. Example proposed file, with suggested settings rather than detected current model defaults:

```json
{
  "schema_version": 1,
  "default_profile": "research",
  "ollama": {
    "host": "http://127.0.0.1:11434",
    "local_only": true,
    "connect_timeout_seconds": 2,
    "metadata_timeout_seconds": 5,
    "inference_timeout_seconds": 120
  },
  "ai": {
    "enabled_by_default": false,
    "max_headlines": 20,
    "max_concurrency": 1
  },
  "profiles": {
    "research": {
      "provider": "ollama",
      "model": "qwen3.5:4b",
      "options": {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 512
      },
      "keep_alive": "5m"
    },
    "headlines": {
      "provider": "ollama",
      "model": "spark2.5:latest",
      "options": {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 512
      },
      "keep_alive": "5m"
    }
  }
}
```

Map only supported generation settings into Ollama request options; `keep_alive` is a separate chat-request field. Keep task prompts under application control with a recorded prompt version. Add model-specific options, including thinking controls, only after capability checks. [Ollama chat request fields](https://docs.ollama.com/api/chat)

Use these resolution rules:

1. Resolve the config file from `--config`, then `SENTINEL_CONFIG`, then the standard user path. Do not automatically load arbitrary configuration from the caller's current directory.
2. Select the profile from `--profile`, then `SENTINEL_PROFILE`, then `default_profile`.
3. Apply only explicitly supplied command flags over profile settings, then built-in defaults. An omitted flag must not overwrite a saved value; allow explicit false values for booleans.
4. Resolve the host from `--ollama-host`, then `SENTINEL_OLLAMA_HOST`, then the config, then normalized `OLLAMA_HOST`, then the loopback default. Always enforce the local-only policy after resolution.
5. `--model MODEL` overrides only that run's model. It never saves a preference. Selection/profile mutation commands are the explicit persistence operations.
6. Treat `--profile` on an AI command as profile selection, not as permission to enable AI on every subsequent numerical command.

`config show --json` should include value origins, configured model parameters, and runtime-discovered metadata in distinct fields. Expose unknown inherited server values honestly; do not fabricate a completely resolved server configuration.

Validate field types, bounded numeric settings, profile references, supported provider names, and unknown configuration keys. Reject malformed configuration with a precise field path. Write a temporary sibling file and atomically replace the destination. Add a short lock around read/modify/write to prevent concurrent CLI and GUI updates from losing changes; atomic replacement alone does not solve that race.

Move watchlist/preferences helpers into a module that does not import `ui`. Preserve their existing import locations through compatibility wrappers. Importing `ui.watchlist` currently executes `ui/__init__.py`, which imports Tk helpers. Migrate old `user_prefs.json` and `watchlist.json` only from known legacy locations, once, when the new files do not exist. Preserve the original files and report migration/persistence errors.

**Start AI integration with an explanation of already computed results.**

Proposed initial AI command:

```bash
sentinel analyze AMD --json > amd-analysis.json
sentinel explain --input amd-analysis.json --profile research
sentinel explain --input amd-analysis.json --profile research --json
```

`explain` should validate the input document, construct a bounded context from its numeric fields and recorded warnings, and return a structured explanation with references to input field paths. Keep numeric results unchanged and associate the explanation with the input's hash, ticker, timestamps, model name/digest, profile settings, and prompt version. It must say when required data is missing. A low temperature reduces variation but is not a guarantee of identical responses.

An initial explanation schema can contain `summary`, `observations`, `limitations`, and `source_fields`. Validate referenced fields against the supplied document. The model should explain existing calculations rather than invent prices, replace scan verdicts, or manufacture news sources. Treat article text and imported result strings as untrusted content, with no shell execution or application-setting changes available to the model.

Ollama supports schema-constrained responses, but Sentinel should still parse and validate the returned content, including required fields and types. Reject incomplete/truncated output and report an explicit error. [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

Next, extract `get_google_news_rss()` and the retrieval portion of `calculate_sentiment()` into `core/news_service.py`, and add:

```bash
sentinel news AMD --limit 20 --json
sentinel sentiment AMD --profile headlines --limit 20 --json
sentinel sentiment --input headlines.json --profile headlines --offline --json
```

Provide a small sentiment-provider interface with a FinBERT adapter and an Ollama adapter. Do not require Ollama to emulate tokenizer/model objects or the mutable `SentimentEngine.models` dictionary. Replace mixed numeric/`"Pending"` results at the new service boundary with objects containing `status`, `score`, `label`, and `reason`; adapt those back to the current GUI as needed.

If FinBERT is selected in offline mode, require existing local weights and prevent its current first-run download path. Missing weights should produce an explicit unavailable-model result.

Preserve headline IDs and input ordering. Require one result per supplied headline, validate uniqueness and score range, and report processed/failed counts. Start with small batches and a configurable headline cap; generative model cost differs from FinBERT's existing batches of 32. Use `null` with an unavailable status for failed classification, not a fabricated neutral score.

Retain the existing 0–1 presentation where 0.5 is neutral, but mark scores with their provider and method: an Ollama-generated score is not automatically comparable to FinBERT's probability-derived score. Evaluate a small reviewed headline fixture before choosing task defaults. Keep news retrieval usable when sentiment is disabled.

Cache AI results by model digest, effective settings, task/prompt version, and input hash. The GUI currently keys its sentiment cache by ticker, which is insufficient once configurations can change. Bound cache size and record when cached results were generated. Show model-load/generation time and token counts when returned, without putting diagnostics into machine-readable stdout.

**Give all CLI commands one dependable output and error contract.**

The existing `--json` flags are a useful start. Centralize rendering in `main/cli_output.py`, keep text output readable, and define the machine interface separately from widget labels.

| Contract | Proposed behavior |
| --- | --- |
| Output selection | `--format` accepts `text`, `json`, `jsonl`, or `csv`; preserve `--json` as an alias. Initially implement text/JSON everywhere and add JSONL/CSV only on applicable commands. Reject conflicting formats. |
| stdout | Exactly the requested data format. No progress, debug prints, spinners, prompts, or export confirmations mixed into it. |
| stderr | Human diagnostics/progress, controlled by `--quiet`/`--verbose`; stack traces only with `--debug`. |
| JSON validity | Explicitly serialize timestamps and supported numeric types. Convert unavailable nonfinite numbers to `null` with a reason and enforce `allow_nan=False`. Avoid blanket `default=str`. |
| Schema | Publish versioned input/output schemas with field types, units, required fields, enums, and null semantics. |
| Noninteractive use | `--non-interactive`, also implied by JSON output or non-TTY input. Missing values produce errors; commands never wait for stdin unless `--input -` was supplied. |
| Ordering | Deterministic sorting with a documented tie-breaker. Report requested filters and effective scan rules. |
| Row limits | `--limit` controls emitted rows after filtering/sorting; report both total matches and emitted count. A row limit does not imply less scan work. |
| CSV | Define whether output is limited. Recommended: `--csv PATH` exports all matched rows, as its current help promises; add `--export-limit N` for an explicit subset. Write headers for empty results. |
| Errors | Typed error codes with concise message, retryability, and relevant details; never make agents parse prose. |
| Partial results | Preserve successful rows, report each failed expiry/ticker, and return a documented nonzero partial-result status. |
| Cancellation | Handle Ctrl-C, close client resources, stop queued work, return 130. JSONL may include completed records plus a terminal cancellation event. |
| Bounded execution | Separate connection/read timeouts and an overall deadline; pass cancellation/deadline checks into shared services. Do not imply killing a waiting thread cancels a network call. |

Document the current units explicitly before renaming fields in a new schema:

| Current field | Current interpretation |
| --- | --- |
| `hv_30`, `ewma_vol`, `garch_vol`, `forecast_vol`, `iv` | Annualized volatility as a decimal fraction; `0.25` means 25%. |
| `dividend_yield`, rate inputs | Decimal annual rates. |
| `edge_pct` | Decimal fraction, using midpoint as denominator; its side depends on the verdict. |
| `spread_pct`, `pop` | Percentage points on a 0–100 scale. POP uses the code's risk-neutral calculation at midpoint-based breakeven. |
| `fair`, `mid`, `ev_at_ask` | Option premium/edge per underlying unit as priced by the core; no contract multiplier is applied. |
| `theta`, `vega` | Price change per calendar day and per one percentage-point volatility change, respectively. |

Add currency, contract identity/multiplier where the provider supplies them, pricing-model name, forecast-vol method, and Greek method to the new output. Distinguish `ev_at_ask` from sell-side edge; an Over verdict can have positive sell-side edge while `ev_at_ask` is negative. Explain these existing meanings in `schema scan` so agents do not infer units from names.

Proposed new JSON envelope, illustrated with synthetic values:

```json
{
  "schema_version": "1.0",
  "command": "scan",
  "status": "partial",
  "data": {
    "ticker": "AMD",
    "rows": [],
    "total_matches": 0,
    "returned_count": 0
  },
  "meta": {
    "data_provider": "yfinance",
    "fetched_at": "2026-09-07T12:00:00Z",
    "source_as_of": null,
    "requested_expiries": ["2026-09-11", "2026-09-18"],
    "completed_expiries": ["2026-09-11"],
    "failed_expiries": ["2026-09-18"],
    "profile": null,
    "ai": null
  },
  "warnings": [],
  "errors": [
    {
      "code": "OPTION_CHAIN_FETCH_FAILED",
      "message": "Could not retrieve one requested expiration.",
      "retryable": true,
      "details": {"expiry": "2026-09-18"}
    }
  ]
}
```

Distinguish local fetch time from source quote/bar time; never manufacture a provider timestamp. A valid empty scan with all requested chains fetched is `ok`. A scan in which every chain request failed is `error`. The example is partial because one chain completed and one failed.

Recommended exit codes: `0` complete success, including valid empty results; `1` unexpected internal failure; `2` invalid arguments/config; `3` unavailable dependency/service/model or incompatible capability; `4` failed data retrieval with no usable result; `5` partial result; `6` invalid model response; `130` cancellation. Make the exact mapping part of the public contract. In JSON mode, emit the error envelope on stdout even on failure; emit supplemental diagnostics on stderr. Catch parser errors as well as handler errors without converting `--help` into a failure.

Changing existing JSON shape can break scripts. Ship stdout contamination and error fixes immediately, and introduce the new envelope through an explicit schema option such as `--schema-version 1` during a documented transition. Mark existing output as legacy and keep it stable until a deliberate major CLI change. Examples for new agent integrations should pin schema version 1.

For discovery, add `sentinel commands --json`, `sentinel schema scan`, `sentinel schema models.list`, and `sentinel --version`. Command descriptions should include required inputs, supported formats, data/network requirements, state-writing behavior, and examples. Keep the manifest and argparse definitions aligned through shared command metadata. Add `docs/cli-for-agents.md` with short working examples; an MCP server can wait until a concrete client needs it.

**Several small fixes should land before agents depend on this interface.**

| Priority | Observed issue and evidence | Concrete update | Scope |
| --- | --- | --- | --- |
| P0 | `main/cli.py::_write_csv()` prints after JSON. A mocked `scan TEST --json --csv PATH` returned exit 0 but `json.loads(stdout)` failed. | Move export status to stderr or envelope metadata; make one renderer own stdout. Also replace the pricing fallback `print()` in `core/pricing.py` with a diagnostic callback/log. | Small |
| P0 | `scan_option_chains()` catches per-expiry exceptions and only logs. JSON mode suppresses that log; a mocked all-chain failure returned count 0 and exit 0 with no error field. | Add structured failures and completed/failed expiry lists to `ScanResult`; map empty/partial/error outcomes at the CLI boundary. Preserve GUI diagnostics. | Small–medium |
| P0 | `--max-expiries -1` is accepted and effectively leaves scanning unlimited; `--limit -1` is also accepted. Invalid `--div` escapes as `ArgumentTypeError` from the handler. | Validate at parsing/service boundaries before constructing providers or doing I/O. Require positive expiry caps, nonnegative row limits, finite yields, and valid expiry filters. | Small |
| P0 | `core/__init__.py` and `sentinel.py` eagerly import sentiment; `core/sentiment.py` attempts torch/transformers imports at module scope. A Python 3.11 CLI import actually loaded torch. | Make package re-exports lazy and import numerical/provider libraries inside their command paths. `--help`, schemas, and config commands should not load AI/numerical libraries. Preserve compatibility imports. | Small–medium |
| P0 | Five batch-pricing tests fail on the system Python 3.9.6 with `'staticmethod' object is not callable`. | Declare the supported Python version and fail clearly before use. If retaining 3.9 support, remove module-level staticmethod wrappers from helper functions and attach them only on the class. Expand `verify` to exercise batch pricing. | Small |
| P1 | CSV help promises full results, but `cmd_scan()` applies `--limit` before exporting; empty results produce no CSV file. | Apply the documented export contract, explicit UTF-8, stable field ordering, and header-only empty exports. | Small |
| P1 | `OptionScanRow` omits original bid/ask, market IV versus smoothed display IV, contract symbol, and structured parity status; percentage units differ by field. | Add raw values and documented units. Preserve current fields during schema migration; represent verdict/side/earnings/parity separately from display text. | Small–medium |
| P1 | `all_options.get(column, 0)` can return a scalar, followed by `.fillna()`/`.to_numpy()`, causing an expiry to be swallowed when columns are missing. | Validate required quote columns; use index-aligned defaults for optional columns and explicit data-quality warnings. | Small |
| P1 | Rate fetches silently fall back to 0.045/short rate; dividend/volatility helpers also use fallbacks without provenance in results. | Return source/fallback reasons and timestamps with resolved inputs. Let agents distinguish observed inputs from defaults. | Small–medium |
| P1 | Prefs/watchlists write directly, suppress I/O errors, and use source-relative locations. | Shared validated config/persistence, atomic writes, clear failures, and user-directory migration. | Medium |
| P1 | Saving `[]` to the watchlist reloads the default five tickers; this was reproduced. | Distinguish an intentionally empty list from a missing or corrupt file. | Small |
| P1 | News timestamps mix timezone-aware RSS parsing and naive Yahoo dates; sorting can fail for mixed inputs. RSS explicitly uses `verify=False` and global warning suppression. | Normalize news times to UTC, handle unknown dates explicitly, restore TLS verification, remove global warning suppression, and surface fetch failures. Validate current Yahoo payload shapes through fixtures. | Small–medium |
| P1 | AI enablement is hardcoded, and the existing sentiment cache is keyed only by ticker. | Real `off`/FinBERT/Ollama settings, shared profile resolution, and model-aware result caching. | Medium |
| P1 | Chart workers have request IDs, but fundamentals and option workers still publish through mutable controller state. | Snapshot ticker/profile/options at job start; reject stale completions and bind callbacks to the originating window. Use bounded workers and cancellation for model switches. | Medium |
| P1 | Windows release scripts build `--noconsole` GUI executables; no separate console artifact exists. | Build/test a CLI executable from `sentinel_cli.py`; use a package console entry point for source installs. | Medium |
| P2 | 3D category helpers have only Under/Over and classify every non-Under row as Over, including Fair rows passed in `scan_buf`. | Carry the actual verdict into plot data and introduce an explicit Fair category/filter. | Small |
| P2 | `ui/options_explorer.py` says smile-smoothed IV is used for Greeks, while the scanner correctly passes original `iv_mkt` to Greeks. | Correct the tooltip and expose `market_iv`/`display_iv` distinctly. | Small |
| P2 | `_font()` advertises a font stack but always chooses its first entry; header help mapping is not wired to actual column hover. | Select an installed font and implement column-specific help. Keep this separate from CLI work. | Small |
| P2 | `build_macos.sh --full` still requires `accelerate`, which is absent from source requirements; `Stocks.cmd` hardcodes Conda, uses `Sentinel.py`, and drops arguments. | Align dependency checks; prefer the selected Python environment, correct filename casing, forward arguments, and preserve exit status. | Small |
| P2 | `watchlist.json` is not ignored; documentation claims torch imports are lazy although package initialization currently defeats that. | Ignore user state, ship separate example files, and update startup/install documentation after fixing imports. | Small |

“Small” means a localized change with focused validation. “Medium” means a service/config boundary or more than one UI/CLI surface. These labels describe relative scope, not guaranteed delivery dates. Findings that were read from source but not reproduced are not presented as runtime test results.

**Expose existing functionality through additional commands in a sensible order.**

| Proposed addition | Existing code to reuse | Value and effort |
| --- | --- | --- |
| `expiries TICKER --json` | `DataProvider.get_option_expirations()` | Discover valid expiry values before scanning; small. |
| `rules show --json` | `core/options_scan.py` | Explain all active thresholds with names/units; small. |
| Configurable scan presets | Shared scan service and rules helpers | Save common scan flags and thresholds; medium because scalar/vector filters and the rules log must use the same rules object. |
| `analyze --fields ...` / richer analysis JSON | `calculate_technicals()`, daily EMAs, existing dataclass | Expose Stoch %D, Williams %R, CCI, OBV, BB width, EMA values, and missing-data reasons; small–medium. |
| `cone TICKER --days 30 --json` | Cone/forecast functions | Machine-readable cone bands; small. Add explicit spot/vol inputs for a fully offline calculator. |
| `price` / `iv` with explicit numeric inputs | Existing numerical functions | Offline calculation with model and unit labels; small. Validate inputs consistently before invoking the core. |
| `watchlist list`, `watchlist add`, `watchlist remove` with `--json` | Watchlist helpers after extraction | Shared human/agent watchlists; small after persistence work. |
| `analyze --watchlist` / `scan --tickers AMD,SPY` | Shared analysis/scan services | Multi-ticker runs with per-ticker outcomes; medium. Begin sequentially and add bounded concurrency only if measured useful. |
| `fundamentals TICKER --json` | `MarketApp.get_info()`, P/E percentile and PEG logic | Access metrics currently confined to the GUI; medium extraction with structured status reasons. |
| `news` / `sentiment` | Extracted news service and sentiment adapters | Reusable news without a window and selectable local classification; medium. |
| `explain --input FILE` | Existing result dataclasses plus Ollama adapter | First useful generative-AI workflow; medium. |
| `export --input FILE --html PATH` | Existing Plotly builder | Share a saved scan without GUI/browser dialogs; small–medium after import/serialization cleanup. CLI exports should not auto-open a browser. |
| `snapshot save` and `--offline --input FILE` | `DataProvider` abstraction | Repeatable analysis without quote/news access; medium. Raw recomputation snapshots need history, chains, rates, dividends, calendar, provenance, and a frozen valuation time. |
| `--sort`, `--fields`, `--limit`, `--format jsonl` | Result dataclasses | Compact, composable output; small–medium. Keep deterministic ordering and distinguish row streaming from token streaming. |
| `--deadline` and cancellation | Provider/service loops | Prevent hanging automation and stop obsolete GUI work; medium. |
| `completion bash`, `completion zsh`, `completion fish` | Shared command metadata | Human discoverability; small–medium, useful after command names stabilize. |
| `scan --watch --interval SECONDS` | Shared service loop | Foreground repeated scans suitable for a scheduler; medium, after deadlines and per-run outcomes exist. Default one-shot behavior stays explicit. |

For offline snapshots, distinguish saving a final analysis for explanation from saving enough raw inputs to recompute it. Current scan calculations use `datetime.now()` internally, so reproducible replay also requires an injected valuation date/time. A disk cache by itself does not provide reproducible snapshots or permission to present stale quotes as current.

Expose configurable scan thresholds through a dataclass such as `ScanRules`. Pass it through scalar helpers, vector masks, CLI, GUI, and metadata. Keep the existing defaults initially; do not change pricing economics as a side effect of adding profile support. Check lower/upper bounds and generate the displayed rules from the same object.

**The GUI should use the same settings and services.**

Add a small AI settings panel with provider `Off / Ollama / FinBERT`, a saved-profile selector, exact model name, Refresh Models, and a status message. Persist explicit selection and offer “Explain analysis” as a requested action. Model selection alone need not load weights. Make settings available even when sentiment is off; current widget construction depends on the hardcoded flag.

Use the same `core/config.py` and `core/ollama.py` as the CLI. Perform discovery/inference in bounded background jobs; publish only if ticker, profile, and job ID still match. Keep a frozen job context so changing the global model halfway through work cannot mix results. Reuse the existing `root.after(...)` publication pattern.

Ollama client support can be included in Lite builds because inference runs in the separately installed Ollama service. Keep weights, torch, and transformers outside that feature's dependency path. Update release wording to distinguish bundled Ollama client support from optional FinBERT rather than saying all AI requires the source build.

Ship a real console executable for automation, especially on Windows: PyInstaller's windowed mode leaves standard streams unavailable there. Do not assume the current GUI executable can become a reliable CLI merely by accepting arguments. [PyInstaller standard-stream behavior](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#sys-stdin-sys-stdout-and-sys-stderr-in-noconsole-windowed-applications-windows-only)

Add packaging metadata with a `sentinel = main.cli:main` console entry point and documented GUI/FinBERT extras. Make the no-argument `sentinel` console command print help; retain `python sentinel.py` for the established GUI behavior. Smoke-test installation, `--help`, `verify`, JSON parseability, exit codes, and execution from another working directory. Resource paths and user-state paths must remain distinct.

**Implement this as a sequence of reviewable changes.**

| Order | Deliverable | Main files | Completion evidence |
| --- | --- | --- | --- |
| 1 | Reliable existing CLI | `main/cli.py`, proposed `main/cli_output.py`, `core/scan_service.py`, package initializers | JSON+CSV parses; expected failures produce structured errors; invalid arguments do no network work; metadata/help paths avoid heavy imports. |
| 2 | Config and model discovery | Proposed `core/config.py`, `core/ollama.py`, CLI commands | Models list/show/current; profiles can be saved/selected; config works from any cwd; daemon absence and missing models are explicit outcomes. |
| 3 | First useful local AI feature | Proposed `core/ai_service.py`, `explain` command | A saved analysis gets a validated local explanation with model/profile/input provenance; numerical output is preserved. |
| 4 | GUI settings and reusable news | `main/app.py`, `core/sentiment.py`, proposed news/sentiment adapters and `ui/ai_settings.py` | CLI/GUI select the same profile; stale jobs cannot overwrite current results; sentiment handles partial failures. |
| 5 | Broader CLI access | Existing numerical services; proposed fundamentals/watchlist modules | Expiries, rules, richer technicals, calculators, watchlist commands, then batch input and exports. |
| 6 | Distribution and replay | Packaging metadata, build scripts, workflows, snapshot provider | Console builds work on supported platforms; lean install does not require FinBERT; offline replay makes no market/news calls. |

Implement only modules needed by each change; a full rewrite of the 1,858-line GUI controller is unnecessary. Before phase 4, the first three deliver a useful CLI independently. Console packaging can be pulled forward if prebuilt distribution is required for that first release.

**Validate behavior at the integration boundaries.**

The current test suite has strong numerical coverage, while CLI tests mostly cover argument parsing and the existing math self-test. Add focused regression tests with injected providers/HTTP transport rather than requiring internet access or downloading models in CI.

- Model discovery: no models, aliases, Unicode/long names, names containing `/` and `:`, loaded versus installed state, stopped daemon, malformed metadata, remote-backed entries, unsupported model version/capability, and missing optional metadata.
- Configuration: no file, corrupt file, invalid types, precedence including explicit `false`, per-run override without persistence, missing profile, write failure, concurrent edits, legacy migration, and running from a different cwd.
- CLI output: whole stdout parses as JSON; JSON plus CSV export remains valid; empty exports have headers; invalid flags/config produce the documented envelope and status; strict JSON contains no NaN/Infinity; help succeeds without optional dependencies.
- Scan failures: one failed expiry, every expiry failed, valid empty scan, unmatched explicit expiry, missing chain columns, fallback input provenance, and total-match versus returned-row counts.
- AI: bounded input, schema/type/range checks, truncated output, missing/duplicate headline IDs, timeout, model disappearance, model/profile cache separation, and article text attempting to supply instructions.
- Local execution: record outbound requests through the injected transport; local-only rejects non-loopback hosts/redirects/proxies/remote models before a prompt is sent; offline snapshots never instantiate live market/news fetches.
- GUI: ticker/model/window changes during background work cannot publish stale results; model discovery does not block Tk or trigger inference.
- Packaging: console streams/exit codes on each supported OS, clean lean installation, lazy optional dependencies, and the full numerical suite after service changes.

Keep a separate opt-in live Ollama smoke test: list models, select a known installed model for that test run, generate one tiny response, validate it, and record runtime/version/digest. Do not use this as a universal model-quality benchmark or make the default test suite depend on it.

**The audit established this baseline.**

| Check executed | Result |
| --- | --- |
| `python3.11 -m pytest tests/ -q --tb=short` | **224 passed, 1 skipped** in 43.39 seconds. Plotly is absent in that interpreter, so its figure-construction test was skipped. |
| `python3 -m pytest tests/ -q --tb=short` using system Python 3.9.6 | **219 passed, 5 failed, 1 skipped**. All five failures were batch-pricing/Greeks calls hitting a module-level `staticmethod` helper. |
| `python3 sentinel.py verify` | All 21 current checks passed, even on Python 3.9; it does not cover the failing batch path. |
| `python3 -m sentinel_cli --help` | Succeeded; advertised `analyze`, `scan`, `verify`. |
| Mocked CLI/provider checks | Reproduced JSON contamination from CSV status, success on total chain failure, accepted negative limits, and uncaught dividend parsing error. |
| Temporary watchlist check | Reproduced an intentionally empty watchlist reloading the default list. |
| CLI import inspection under Python 3.11 | Tk/matplotlib stayed unloaded, but torch and `core.sentiment` loaded through package initialization. |
| Local Ollama manifest inspection | Four names, all referenced layers present; no explicit parameter layers. No server query, inference benchmark, or model download was performed. |

Python documents the ability to call `staticmethod` objects directly as a change in 3.10, which explains the observed interpreter difference. This is a runtime-support issue rather than evidence that batch pricing is broken on the CI interpreter. [Python staticmethod documentation](https://docs.python.org/3/library/functions.html#staticmethod)

The review used existing local dependencies and did not install the currently declared requirement sets or build release binaries. GUI behavior and external Yahoo payloads were inspected in code rather than exercised against a live market feed. These results support the integration plan; they do not replace future packaged/live smoke tests.

**Keep larger research projects behind this work.**

Fundamental bias scoring, semantic peer arbitrage, FOMC/CPI feeds, alternate market-data providers, tray/webhook scheduling, SVI, and analytic BS2002 Greeks remain the existing broader backlog. They need additional data, modeling, or operational work and are not prerequisites for local model selection or an agent-friendly CLI. A chat-driven command executor, MCP server, vector search, or automatic model download manager would also expand scope substantially.

Trading-calendar/0DTE precision, consistent day-count conventions, historical-data warm-up, and extended Ichimoku plotting deserve separate numerical/display reviews. Record current assumptions in output now; avoid changing them casually while implementing the CLI. The immediate target is a tool whose available models, active configuration, inputs, outputs, and failures are easy to inspect and reproduce.


**First implementation and live experiments (2026-09-07).**

- Installed Plotly **7.0.0** into the Python **3.11** user environment. All five
  `tests/test_options_3d.py` tests pass, including the formerly skipped Plotly test.
- Added standard-library-only `core/ollama.py`, `core/model_config.py` and
  `main/model_cli.py`. Config writes validate a versioned schema, use process locks,
  and atomically replace the file. Existing profiles require explicit `--replace`.
- Added exact model-name discovery, inspection, saved profile selection and local
  generation using saved temperature, context size and token limits. Config and
  profile inspection do not contact the daemon. HTTP requests disable proxies and
  redirects; local-only checks reject remote model metadata before inference.
- Made `core` and the launcher exports lazy. Importing the CLI no longer imports
  NumPy, pandas, Torch or Tk. Analyze/scan/verify import their dependencies on use.
- Fixed JSON/CSV stdout contamination, full-row CSV export despite `--limit`, empty
  CSV headers, JSON NaN conversion, negative scan limits, dividend input errors,
  and structured expiry failures with nonzero exit status. Added total row counts.
- Removed module-level `staticmethod` wrappers that broke batch pricing on Python
  3.9; class static methods and formulas are retained. Pricing fallback notices now
  go to stderr.

Live discovery confirmed four installed names and the two Spark aliases sharing
the same digest. Experiments used `/private/tmp/sentinel-astra-experiment.json`,
not the user's default configuration. No model was downloaded.

| Experiment | Observed result |
| --- | --- |
| Select `spark2.5:latest`, context 4096, temperature 0, max 64 tokens | Metadata validation and temporary profile persistence succeeded. |
| Generate through Spark | Ollama HTTP 500: `unknown model architecture: 'spark2_5'`. Installed metadata does not guarantee runtime compatibility. CLI returned a structured error and exit 4. |
| Select `qwen3.5:4b` with the same settings and ask for a fixed short reply | Returned exactly `Local model selection works.`, stop reason `stop`, 6 generated tokens, reported evaluation duration 353,022,000 ns (~0.353 s). This timing excludes model loading and is not a benchmark. |

The initial implementation deliberately keeps host configuration in CLI flags and
environment variables. It does not yet implement the larger proposed host/timeout
config schema, loaded-model reporting, model filesystem inventory, doctor, profile
deletion, interactive menus, structured market explanations or GUI integration.
The new `ask` output is model-generated text, not a numerical engine input.
Existing FinBERT sentiment behavior remains separate. Scan hardening still needs
explicit unmatched-expiry validation, richer data provenance and missing-column
handling as listed in the roadmap.

Validation after implementation: **248 tests passed** under Python 3.11 in
12.03 seconds, with no skipped tests. The Python 3.9 compatibility run before the
last four error-path tests passed **243 tests with one Plotly skip**; all five
previous batch failures were resolved. Python compilation, CLI help and
`git diff --check` passed. Live GUI behavior, release binaries and Yahoo market
requests were not exercised in this implementation slice.
