"""
Kaggriculture — Rule-Based Starter Agent (v2)
===============================================

Grounded in the ACTUAL kaggle_environments source (kaggriculture.py), not
guesses. Key mechanics this agent leans on:

- CROPS / ANIMALS / market pricing all come from the real module constants.
- Watering is near-mandatory maintenance: 2 consecutive unwatered days turns
  a tile into a WEED, independent of whether watering boosts yield_units
  that particular day.
- HARVEST is legal any time yield_units > 0 (harvest first, always).
- Hire cost resets every in-game day and follows the Fibonacci sequence
  (1, 1, 2, 3, 5, 8, ...), so the first 1-2 hands each day are essentially
  free. This agent re-hires up to a target crew size every day.
- Land unlock order is fixed: NE -> SW -> SE at 1000 -> 2000 -> 4000. The
  agent speculatively issues BUY_LAND once cash clears a reserve; the
  engine silently no-ops if you can't afford it, so this is safe to try.
- Harvested goods sit in the acting unit's personal inventory until they're
  auto-dropped to the shed at end of day (_drop_inventories_to_shed), so we
  don't need explicit DROP/PICKUP moves for a first pass.

Not covered yet (left as clearly marked TODOs — extend once you want the
extra complexity): livestock (BUILD_COOP/PASTURE, BUY_ANIMAL, FEED, CARE,
COLLECT_FERTILIZER, FERTILIZE) and manual DROP/PICKUP for intraday selling.

Run this file directly for a local smoke test against the built-in
"random" agent.
"""

import statistics

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------
CASH_RESERVE = 100          # never let planned spend eat into this buffer
TARGET_CREW_SIZE = 4        # hands to maintain; Fib hire cost makes this cheap
SEED_BUY_BATCH = 3          # max seeds to queue per turn for the chosen crop
LAND_BUY_ATTEMPT_CASH = 1200  # speculative BUY_LAND trigger (cheapest tier is 1000)
PRICE_HISTORY_WINDOW = 15   # turns of price memory per product for the sell rule
MIN_HISTORY_FOR_THRESHOLD = 4  # sell opportunistically until we have this many samples

# Rolling price history persists across turns via module-level state (the
# same Python process handles the whole episode).
_price_history = {}


def _update_price_history(product, price):
    hist = _price_history.setdefault(product, [])
    hist.append(price)
    if len(hist) > PRICE_HISTORY_WINDOW:
        hist.pop(0)


def _should_sell(product, price):
    hist = _price_history.get(product, [])
    if len(hist) < MIN_HISTORY_FOR_THRESHOLD:
        return True  # not enough data yet — take the cash rather than hoard
    return price >= statistics.mean(hist)


def _best_crop(money, prices):
    """
    Rank plantable crops by rough profit-per-day-in-ground:
        (max_yield * current_price - seed_cost) / max_yield_day
    This ignores the fact that selling many units will push price down and
    that non-ongoing crops need active watering to hit max_yield — it's a
    starting heuristic, not a simulator. Only considers crops we can afford
    to seed right now.
    """
    scored = []
    for crop, info in CROPS.items():
        if info["seed"] > money - CASH_RESERVE:
            continue
        price = prices.get(crop, 0)
        est_profit = info["max_yield"] * price - info["seed"]
        rate = est_profit / max(1, info["max_yield_day"])
        scored.append((rate, crop))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _is_mature(tile, day):
    """
    Non-ongoing crops start with yield_units=1 the instant they're planted,
    but the engine still rejects HARVEST until `first_yield_day` has passed
    (day - planted_day >= first_yield_day). Checking yield_units alone is
    not enough — a bot that ignores this gets stuck repeatedly attempting
    an invalid HARVEST on a still-growing plant instead of watering it.
    """
    crop_info = CROPS.get(tile.get("crop"))
    if crop_info is None:
        return False
    return day - tile.get("planted_day", day) >= crop_info["first_yield_day"]


def _scan_targets(farm, board_size, best_crop, have_seed, day):
    """
    Every actionable tile on the board, tagged by purpose. Shared across all
    units (farmer + hands) — each unit independently picks its own nearest
    target, so multiple units may converge on the same tile. That's fine:
    HARVEST/WATER/PLANT are idempotent no-ops once done, and duplicate PLANT
    requests beyond available seeds are dropped atomically by the engine.
    """
    harvestable, needs_water, plantable = [], [], []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                if have_seed:
                    plantable.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue  # "LOCKED"
            if tile.get("kind") != "PLANT":
                continue  # ignore animal structures for this crop-only pass
            if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
                harvestable.append((x, y))
            elif not tile.get("watered_today", False):
                needs_water.append((x, y))
    return harvestable, needs_water, plantable if best_crop else []


def _unit_action(pos, farm, board_size, day, best_crop, have_seed):
    """Decide the single next action for one unit (farmer or a hand)."""
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    # Standing on a mature, harvestable tile -> take it.
    if (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("yield_units", 0) > 0
        and _is_mature(tile, day)
    ):
        return ["HARVEST"]

    # Standing on our own unwatered plant -> water it (prevents weeds, and
    # boosts yield for non-ongoing crops in their watering window).
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
        return ["WATER"]

    # Standing on empty, owned land with seed in hand -> plant.
    if tile is None and have_seed and best_crop:
        return ["PLANT", best_crop]

    # Otherwise, walk toward the nearest useful tile: harvest > water > plant.
    harvestable, needs_water, plantable = _scan_targets(farm, board_size, best_crop, have_seed, day)
    for group in (harvestable, needs_water, plantable):
        if not group:
            continue
        tx, ty = min(group, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]

    return ["PASS"]


def agent(obs, configuration=None):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    day = obs.get("day", 0)
    money = farm["money"]
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    market_prices = (obs.get("market", {}) or {}).get("prices", {})

    # --- keep price memory fresh ------------------------------------------------
    for product, price in market_prices.items():
        _update_price_history(product, price)

    # --- decide this turn's target crop -----------------------------------------
    best_crop = _best_crop(money, market_prices)

    # --- market orders -----------------------------------------------------------
    market_orders = []

    # Sell whatever's sitting in the shed and priced well.
    for product, qty in shed.items():
        if qty > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, qty])

    # Keep seed stock topped up for the crop we're currently favoring.
    if best_crop and seeds.get(best_crop, 0) < SEED_BUY_BATCH and money - CASH_RESERVE >= CROPS[best_crop]["seed"]:
        market_orders.append(["BUY_SEED", best_crop, 1])

    # Re-hire up to target crew size every day — cheap because hire cost
    # resets daily and follows Fibonacci (first couple hands cost ~1 coin).
    if len(farm["hands"]) < TARGET_CREW_SIZE and money - CASH_RESERVE >= 1:
        market_orders.append(["HIRE"])

    # Speculative land grab once cash is healthy; no-ops safely if we can't
    # afford the next tier or have already unlocked everything.
    if money >= LAND_BUY_ATTEMPT_CASH and len(farm["unlocked_quadrants"]) < 4:
        market_orders.append(["BUY_LAND"])

    # --- unit actions (farmer + every hired hand) --------------------------------
    have_seed = bool(best_crop) and seeds.get(best_crop, 0) > 0
    farmer_action = _unit_action(farm["farmer"], farm, board_size, day, best_crop, have_seed)
    hands_actions = [
        _unit_action(hand_pos, farm, board_size, day, best_crop, have_seed)
        for hand_pos in farm["hands"]
    ]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}


# --------------------------------------------------------------------------
# Local smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from kaggle_environments import make

    env = make("kaggriculture", debug=True)
    env.run([agent, "random"])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
