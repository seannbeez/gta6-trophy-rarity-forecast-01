# GTA VI Trophy Rarity Forecast — v1

Entry for Make.fun competition `gta6-trophy-rarity-forecast-01`: forecast,
for each of the 55 PS5 trophies, the 5/25/50/75/95% quantiles of its
Exophase earn rate at the target time, from the history of Exophase page
reads in the case file.

## Model

Earners / players decomposition, fitted from the case history at runtime —
nothing about any trophy is hardcoded.

- Each read gives players tracked `P` and each trophy's earn rate `r`.
  Earners are `E = r/100 × P`.
- **Players** `P(T)`: `ln P` against `ln(1 + s/c)` (hours since the first
  read), with the clock `c` (1h or 24h) chosen by the history itself,
  clamped to `[P_last, 50 × P_last]`.
- **Per trophy, two structural forms are fitted** (saturating adoption
  curves, τ grid 12h–1536h, shapes k = 1, 2; τ shrunk toward the global
  median τ), and the history picks between them by in-sample squared
  error in logit space (a near-tie is averaged in logit space):
  - **Form 1 — earners track players**: the rate follows the adoption
    curve, `R(s) = C·(1 − exp(−(s/τ)^k))`.
  - **Form 2 — earners saturate absolutely**: `E(s) = E0 + D·(1 −
    exp(−(s/τ)^k))`, rate `E(T)/P(T)` — the rate peaks, then declines
    from dilution as players flood in.
- **Guards**: if neither form beats persistence on the history by ≥10%
  of squared logit error, the median is persistence; the fitted move is
  damped by history size and by `span/(span + horizon)`; the logit move
  is capped at ±3.5.
- **Quantile spread** (logit scale, asymmetric): random walk
  `0.5·√days` + fit residual + refresh-step term (half the typical
  per-refresh jump in `ln P`) + model-disagreement term (half the
  distance between the two forms' medians) + 0.10 floor. Asymmetry from
  players-vs-earners log growth (dilution widens the downside), capped
  at ±0.35, zero when the guard falls back to persistence.
- **Flat history** (all reads identical, as in the practice cases):
  persistence median with a wide 1.2-logit ignorance spread.

## Validation (local, no submits)

- Practice cases p00/p01 (flat launch-night histories, answers
  reconstructed by dilution only): **v1 loss 0.9869, typical miss
  0.73pp, 90% coverage 0.9818** vs house persistence starter 1.6877 /
  0.73pp / 0.0000 and the house curve baseline 1.6877 / 0.73pp / 0.0000.
  The practice numbers are a proxy: the reconstructed answers are
  approximate and slightly pessimistic for persistence models.
- Synthetic long-horizon stress tests (2-day history, 5-day target,
  5 scenarios: player-flood dilution, heavy dilution, slow growth,
  earners saturate + rate declines, power-law earners with no
  saturation): **mean loss v1 0.8917** vs starter 1.0337 vs curve
  baseline 1.1051; 90% coverage 0.93–1.00 in every scenario.
- Determinism: two runs on the same case produce byte-identical
  outputs. Runtime: 0.34s for both practice cases; 0.90s for a dense
  291-read case (limit 8s per case).
- Edge cases verified: empty history, single read, null rates, 0% /
  100% rates, zero horizon, target before the first read — all produce
  valid output, never crash.

## Files

- `main.py` — the entry (Python 3.13, standard library only).
- `LICENSE` — MIT, copied from the competition kit starter.

## Running

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

Reads every `*.in` in the input dir, writes `<id>.out` per case:
`{"trophies": [{"id": N, "q": [q05, q25, q50, q75, q95]}]}`.
