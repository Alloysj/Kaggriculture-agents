# kaggriculture-agents

Rule-based agents for Kaggle's [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) competition — a farming-simulation environment where an agent manages crops, labor, land, and a live market to maximize its final bank over a 30-day (720-turn) season.

## Repo structure

```
agents/
  v1_scaffold_agent.py   # first pass — schema guessed, not yet run against the real env
  v2_agent.py             # grounded in the real observation/action schema; single-crop heuristic
  v3_agent.py             # diversified portfolio, trickle-selling, aggressive land buying
notebooks/
  kaggriculture_rule_agent_v3.ipynb   # v3 walked through + benchmarked, exports agents/main.py for submission
scripts/
  diagnose_kaggriculture.py   # turn-by-turn money/crew/seed/shed trace for debugging local result mismatches
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Agent history and benchmarks

Each version fixed something concrete found by actually running the environment locally, not by guessing:

| Version | Change | Avg final bank vs `starter` (8 seeds) |
|---|---|---|
| v1 | Scaffold only — observation/action keys guessed from the competition description, never run against the real env | — |
| v2 | Rebuilt against the real schema (`kaggle_environments.envs.kaggriculture.kaggriculture`). Fixed a bug where non-ongoing crops (wheat/carrot/melon) start with `yield_units=1` at planting but aren't legally harvestable until `first_yield_day` — checking `yield_units > 0` alone got the agent stuck repeatedly attempting an invalid harvest instead of watering. Single "best crop" heuristic. | ~$4,600 |
| v3 | Added three findings from the community notebook *[Kaggriculture, Visualized: What Every Crop Pays](https://www.kaggle.com/code/georgymamarin/kaggriculture-visualized-what-every-crop-pays)*: (1) diversify planting across the top 3 crops by live profit/tile-day instead of one crop, since a single crop crashes its own market as you sell into it; (2) trickle-sell (capped per-turn quantity) instead of dumping the whole shed and walking the price down; (3) buy land against the real `LAND_PRICES` tiers as soon as affordable, instead of a flat guessed threshold. | ~$17,800 |

Benchmarks were run with `kaggle-environments==1.32.6/1.32.7` across 8 fixed seeds against the built-in `random`, `starter`, and `pass` agents — see the notebook for the exact benchmark cell.

## Running / benchmarking an agent

```bash
python agents/v3_agent.py
```

Runs a smoke test and prints final bank vs. `random`, `starter`, and `pass`.

## Known open issue

Local runs of v3 have shown results ranging from ~$1,300 to ~$21,000 average across different machines/environment setups for reasons not yet fully diagnosed — `kaggle-environments` version was tested and ruled out as the sole cause (an older version scored *higher*, not lower). `scripts/diagnose_kaggriculture.py` traces money/crew/seeds/shed turn-by-turn to help pin down the divergence; if you hit this, run it and compare against the notebook's benchmark cell output.

## Not yet implemented

- **Livestock** (`BUILD_COOP`/`BUILD_PASTURE`, `BUY_ANIMAL`, `FEED`, `CARE`, `COLLECT_FERTILIZER`) — a stateful pickup-wheat → feed → care → harvest → drop routine per unit. Animals are compounding assets and the only source of fertilizer, which is needed for wheat to reach its yield cap.
- **Fertilizer** on crops directly.
- **Dynamic sell-batch sizing** - currently a flat cap; could instead invert the market's price-impact curve per product.
- **Proper task allocation** - units are currently assigned crops by round-robin index rather than by nearest-unit-to-nearest-task.

## Submitting

The notebook's last two cells `%%writefile` the agent to `agents/main.py` (gitignored - it's a build artifact) and replay-test it before you upload it to the competition.
