"""
Kaggriculture — Rule-Based Starter Agent (v3)
===============================================
"""

import statistics

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, LAND_ORDER, LAND_PRICES

# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------
CASH_RESERVE = 100            # never let planned spend eat into this buffer
TARGET_CREW_SIZE = 6          # hands to maintain daily; Fib hire cost keeps this cheap
SEED_BUY_BATCH = 3            # max seeds to queue per turn, per crop in the portfolio
PORTFOLIO_SIZE = 3            # spread planting across this many top crops
SELL_BATCH_CAP = 20           # max units of one product sold in a single turn (trickle, not dump)
PRICE_HISTORY_WINDOW = 15     # turns of price memory per product for the sell-timing rule
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


def _one_shot_units(crop):
    c = CROPS[crop]
    window = range((c["max_yield_day"] + 1) // 2, c["max_yield_day"] + 1)
    return min(c["max_yield"], 1 + len(list(window)))


def _one_shot_days(crop):
    c = CROPS[crop]
    cap_day = (c["max_yield_day"] + 1) // 2 + c["max_yield"] - 2
    return max(c["first_yield_day"], min(cap_day, c["max_yield_day"]))


def _ongoing_days(crop):
    c = CROPS[crop]
    return [c["first_yield_day"] + k * c["interval"] for k in range(c["max_yield"])]


def _crop_profit_per_tile_day(crop, price):
    """
    Same occupied-days accounting as the source notebook's crop_econ(), but
    using the CURRENT market price instead of the base price — a live
    read rather than a static table, since prices move all game.
    """
    c = CROPS[crop]
    if c["ongoing"]:
        days = _ongoing_days(crop)
        units, occupied = c["max_yield"], days[-1]
    else:
        units, occupied = _one_shot_units(crop), _one_shot_days(crop)
    revenue = units * price
    profit = revenue - c["seed"]
    return profit / max(1, occupied)


def _crop_portfolio(money, prices):
    """
    Rank every crop we can currently afford to seed by live profit/tile-day
    and return the top PORTFOLIO_SIZE. Planting gets spread across this
    list (see agent()) instead of piling into a single "best" crop.
    """
    scored = []
    for crop, info in CROPS.items():
        if info["seed"] > money - CASH_RESERVE:
            continue
        price = prices.get(crop, 0)
        scored.append((_crop_profit_per_tile_day(crop, price), crop))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [crop for _, crop in scored[:PORTFOLIO_SIZE]]


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
    but the engine still rejects HARVEST until `first_yield_day` has
    passed. Checking yield_units alone gets a bot stuck repeatedly
    attempting an invalid HARVEST on a still-growing plant instead of
    watering it.
    """
    crop_info = CROPS.get(tile.get("crop"))
    if crop_info is None:
        return False
    return day - tile.get("planted_day", day) >= crop_info["first_yield_day"]


def _scan_targets(farm, board_size, plantable_crops, day):
    """
    Every actionable tile on the board, tagged by purpose. Shared across
    all units (farmer + hands) — each independently picks its own nearest
    target from the same snapshot, so multiple units may converge on the
    same tile in one turn. That's a mild inefficiency (a second HARVEST or
    PLANT on an already-handled tile is a silent no-op), not a crash.
    """
    harvestable, needs_water, plantable = [], [], []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                if plantable_crops:
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
    return harvestable, needs_water, plantable


def _unit_action(pos, farm, board_size, day, assigned_crop, have_seed, plantable_crops):
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

    # Standing on empty, owned land with our assigned crop's seed in hand.
    if tile is None and have_seed and assigned_crop:
        return ["PLANT", assigned_crop]

    # Otherwise, walk toward the nearest useful tile: harvest > water > plant.
    harvestable, needs_water, plantable = _scan_targets(farm, board_size, plantable_crops, day)
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

    # --- this turn's crop portfolio (top few by live profit/tile-day) -----------
    portfolio = _crop_portfolio(money, market_prices)

    # --- market orders -----------------------------------------------------------
    market_orders = []

    # Trickle-sell: cap how much of one product goes out in a single turn
    # instead of dumping the whole shed and walking the price down ourselves.
    for product, qty in shed.items():
        if qty > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, min(qty, SELL_BATCH_CAP)])

    # Keep seed stock topped up across the whole portfolio, not just one crop.
    for crop in portfolio:
        if seeds.get(crop, 0) < SEED_BUY_BATCH and money - CASH_RESERVE >= CROPS[crop]["seed"]:
            market_orders.append(["BUY_SEED", crop, 1])

    # Re-hire up to target crew size every day — cheap because hire cost
    # resets daily and follows Fibonacci (first couple hands cost ~1 coin).
    if len(farm["hands"]) < TARGET_CREW_SIZE and money - CASH_RESERVE >= 1:
        market_orders.append(["HIRE"])

    # Buy the NEXT real land tier as soon as we can afford it with reserve
    # to spare — checked against the actual price, not a flat guess.
    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER):
        next_land_price = LAND_PRICES[n_extra_unlocked]
        if money - CASH_RESERVE >= next_land_price:
            market_orders.append(["BUY_LAND"])

    # --- unit actions (farmer + every hired hand) --------------------------------
    # Round-robin the portfolio across units so different units work
    # different crops — this is the actual diversification, not just having
    # a ranked list.
    def crop_for_unit(idx):
        if not portfolio:
            return None
        return portfolio[idx % len(portfolio)]

    plantable_crops = bool(portfolio)

    farmer_crop = crop_for_unit(0)
    farmer_have_seed = bool(farmer_crop) and seeds.get(farmer_crop, 0) > 0
    farmer_action = _unit_action(farm["farmer"], farm, board_size, day, farmer_crop, farmer_have_seed, plantable_crops)

    hands_actions = []
    for i, hand_pos in enumerate(farm["hands"], start=1):
        hand_crop = crop_for_unit(i)
        hand_have_seed = bool(hand_crop) and seeds.get(hand_crop, 0) > 0
        hands_actions.append(
            _unit_action(hand_pos, farm, board_size, day, hand_crop, hand_have_seed, plantable_crops)
        )

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}


# --------------------------------------------------------------------------
# Local smoke test / benchmark
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from kaggle_environments import make

    for opponent in ["random", "starter", "pass"]:
        env = make("kaggriculture", debug=False)
        env.run([agent, opponent])
        r0 = env.steps[-1][0]["reward"]
        r1 = env.steps[-1][1]["reward"]
        print(f"vs {opponent:8s}  agent=${r0:,.0f}   opponent=${r1:,.0f}")
