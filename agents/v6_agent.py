import statistics
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, ANIMALS, LAND_ORDER, LAND_PRICES, _is_shed_adjacent, _shed_access_tiles,
)

OPERATING_RESERVE = 50
CASH_RESERVE = 200
TILES_PER_HAND = 5
LAND_UTILIZATION_THRESHOLD = 0.75
SEED_BUY_BATCH = 3
PORTFOLIO_SIZE = 3
SELL_BATCH_CAP = 20
PRICE_HISTORY_WINDOW = 15
MIN_HISTORY_FOR_THRESHOLD = 4

ANIMAL_TARGET = "GOOSE"
STRUCTURE_KIND = ANIMALS[ANIMAL_TARGET]["structure"]
RANCHER_COUNT = 1
ANIMAL_READY_CASH = ANIMALS[ANIMAL_TARGET]["cost"] + CASH_RESERVE + 200
WHEAT_FEED_BUFFER = 3
WHEAT_SELL_RESERVE = WHEAT_FEED_BUFFER * 2

_price_history = {}

def _update_price_history(product, price):
    hist = _price_history.setdefault(product, [])
    hist.append(price)
    if len(hist) > PRICE_HISTORY_WINDOW:
        hist.pop(0)

def _should_sell(product, price):
    hist = _price_history.get(product, [])
    if len(hist) < MIN_HISTORY_FOR_THRESHOLD:
        return True
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
    scored = []
    for crop, info in CROPS.items():
        if info["seed"] > money - OPERATING_RESERVE:
            continue
        price = prices.get(crop, 0)
        scored.append((_crop_profit_per_tile_day(crop, price), crop))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [crop for _, crop in scored[:PORTFOLIO_SIZE]]

def _step_toward(fx, fy, tx, ty):
    if fx > tx: return "WEST"
    if fx < tx: return "EAST"
    if fy > ty: return "NORTH"
    if fy < ty: return "SOUTH"
    return None

def _is_mature(tile, day):
    crop_info = CROPS.get(tile.get("crop"))
    if crop_info is None:
        return False
    return day - tile.get("planted_day", day) >= crop_info["first_yield_day"]

def _scan_targets(farm, board_size, plantable_crops, day):
    harvestable, needs_water, weeded, plantable = [], [], [], []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                if plantable_crops:
                    plantable.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                weeded.append((x, y))
                continue
            if tile.get("kind") != "PLANT":
                continue
            if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
                harvestable.append((x, y))
            elif not tile.get("watered_today", False):
                needs_water.append((x, y))
    return harvestable, needs_water, weeded, plantable

def _unit_action(pos, farm, board_size, day, assigned_crop, have_seed, plantable_crops):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    if (isinstance(tile, dict) and tile.get("kind") == "PLANT"
            and tile.get("yield_units", 0) > 0 and _is_mature(tile, day)):
        return ["HARVEST"]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
        return ["WATER"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None and have_seed and assigned_crop:
        return ["PLANT", assigned_crop]
    harvestable, needs_water, weeded, plantable = _scan_targets(farm, board_size, plantable_crops, day)
    for group in (harvestable, needs_water, weeded, plantable):
        if not group:
            continue
        tx, ty = min(group, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]
    return ["PASS"]

def _wheat_fallback_action(pos, farm, board_size, day, have_seed):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
        if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None and have_seed:
        return ["PLANT", "WHEAT"]
    candidates = []
    for y in range(board_size):
        for x in range(board_size):
            t = farm["tiles"][y][x]
            if t is None:
                if have_seed:
                    candidates.append((x, y, 3))
            elif isinstance(t, dict) and t.get("kind") == "WEED":
                candidates.append((x, y, 2))
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == "WHEAT":
                if t.get("yield_units", 0) > 0 and _is_mature(t, day):
                    candidates.append((x, y, 0))
                elif not t.get("watered_today", False):
                    candidates.append((x, y, 1))
    if not candidates:
        return ["PASS"]
    candidates.sort(key=lambda c: (c[2], abs(c[0] - fx) + abs(c[1] - fy)))
    tx, ty, _ = candidates[0]
    step = _step_toward(fx, fy, tx, ty)
    return [step] if step else ["PASS"]

def _find_structures(farm, board_size, structure_kind):
    found = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == structure_kind:
                found.append((x, y, tile))
    return found

def _find_empty_tile(farm, board_size, fx, fy):
    candidates = [
        (x, y) for y in range(board_size) for x in range(board_size)
        if farm["tiles"][y][x] is None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))

def _nearest_shed_tile(fx, fy, board_size):
    tiles = _shed_access_tiles(board_size)
    return min(tiles, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))

def _rancher_action(pos, unit_inv, shed, farm, board_size):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    if isinstance(tile, dict) and "animal" in tile:
        if tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
        if not tile.get("fed_today") and unit_inv.get("WHEAT", 0) > 0:
            return ["FEED"]
        if not tile.get("cared_today"):
            return ["CARE"]
    structures = _find_structures(farm, board_size, STRUCTURE_KIND)
    if (isinstance(tile, dict) and tile.get("kind") == STRUCTURE_KIND
            and "animal" not in tile and unit_inv.get(ANIMAL_TARGET, 0) > 0):
        return ["PLACE", ANIMAL_TARGET]
    if not structures:
        if tile is None:
            return ["BUILD_COOP"] if STRUCTURE_KIND == "COOP" else ["BUILD_PASTURE"]
        target = _find_empty_tile(farm, board_size, fx, fy)
        if target:
            step = _step_toward(fx, fy, target[0], target[1])
            if step:
                return [step]
        return ["PASS"]
    empty_structures = [s for s in structures if "animal" not in s[2]]
    living_structures = [s for s in structures if "animal" in s[2]]
    need_animal_pickup = bool(empty_structures) and unit_inv.get(ANIMAL_TARGET, 0) <= 0 and shed.get(ANIMAL_TARGET, 0) > 0
    need_wheat = unit_inv.get("WHEAT", 0) < WHEAT_FEED_BUFFER and shed.get("WHEAT", 0) > 0
    if need_animal_pickup or need_wheat:
        if _is_shed_adjacent((fx, fy), board_size):
            if need_animal_pickup:
                return ["PICKUP", ANIMAL_TARGET, 1]
            return ["PICKUP", "WHEAT", WHEAT_FEED_BUFFER]
        tx, ty = _nearest_shed_tile(fx, fy, board_size)
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]
    if empty_structures:
        tx, ty, _ = min(empty_structures, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]
    for sx, sy, s in living_structures:
        pending = (
            (s.get("yield_units", 0) > 0)
            or s.get("fertilizer_available")
            or (unit_inv.get("WHEAT", 0) > 0 and not s.get("fed_today"))
            or not s.get("cared_today")
        )
        if pending:
            step = _step_toward(fx, fy, sx, sy)
            if step:
                return [step]
    return ["PASS"]

def _count_owned_and_planted(farm, board_size):
    owned, planted = 0, 0
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                owned += 1
            elif isinstance(tile, dict):
                owned += 1
                if tile.get("kind") == "PLANT":
                    planted += 1
    return owned, planted

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
    inventories = private.get("inventories", [{}])
    market_prices = (obs.get("market", {}) or {}).get("prices", {})

    for product, price in market_prices.items():
        _update_price_history(product, price)

    structures = _find_structures(farm, board_size, STRUCTURE_KIND)
    already_expanded = len(farm["unlocked_quadrants"]) >= 2
    ranch_active = already_expanded and (bool(structures) or money >= ANIMAL_READY_CASH)

    portfolio = _crop_portfolio(money, market_prices)
    market_orders = []

    wheat_reserve = WHEAT_SELL_RESERVE if ranch_active else 0
    for product, qty in shed.items():
        sellable = qty - wheat_reserve if product == "WHEAT" else qty
        if sellable > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, min(sellable, SELL_BATCH_CAP)])

    money_left = money

    for crop in portfolio:
        cost = CROPS[crop]["seed"]
        if seeds.get(crop, 0) < SEED_BUY_BATCH and money_left - OPERATING_RESERVE >= cost:
            market_orders.append(["BUY_SEED", crop, 1])
            money_left -= cost

    if ranch_active and seeds.get("WHEAT", 0) < SEED_BUY_BATCH and money_left - OPERATING_RESERVE >= CROPS["WHEAT"]["seed"]:
        market_orders.append(["BUY_SEED", "WHEAT", 1])
        money_left -= CROPS["WHEAT"]["seed"]

    total_owned_tiles, planted_tiles = _count_owned_and_planted(farm, board_size)
    target_crew_size = max(1, total_owned_tiles // TILES_PER_HAND)
    if len(farm["hands"]) < target_crew_size and money_left - OPERATING_RESERVE >= 1:
        market_orders.append(["HIRE"])
        money_left -= 1

    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER):
        next_land_price = LAND_PRICES[n_extra_unlocked]
        utilization = planted_tiles / total_owned_tiles if total_owned_tiles else 1.0
        if money_left - CASH_RESERVE >= next_land_price and utilization >= LAND_UTILIZATION_THRESHOLD:
            market_orders.append(["BUY_LAND"])
            money_left -= next_land_price

    coop_empty = any("animal" not in s[2] for s in structures)
    animal_pending = shed.get(ANIMAL_TARGET, 0) > 0 or any(inv.get(ANIMAL_TARGET, 0) > 0 for inv in inventories)
    animal_cost = ANIMALS[ANIMAL_TARGET]["cost"]
    if money >= ANIMAL_READY_CASH and coop_empty and not animal_pending and money_left - CASH_RESERVE >= animal_cost:
        market_orders.append(["BUY_ANIMAL", ANIMAL_TARGET, 1])
        money_left -= animal_cost

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
        is_rancher = ranch_active and i <= RANCHER_COUNT
        if is_rancher:
            unit_inv = inventories[i] if i < len(inventories) else {}
            action = _rancher_action(hand_pos, unit_inv, shed, farm, board_size)
            if action != ["PASS"]:
                hands_actions.append(action)
                continue
            wheat_have_seed = seeds.get("WHEAT", 0) > 0
            hands_actions.append(_wheat_fallback_action(hand_pos, farm, board_size, day, wheat_have_seed))
            continue
        hand_crop = crop_for_unit(i)
        hand_have_seed = bool(hand_crop) and seeds.get(hand_crop, 0) > 0
        hands_actions.append(
            _unit_action(hand_pos, farm, board_size, day, hand_crop, hand_have_seed, plantable_crops)
        )

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}
