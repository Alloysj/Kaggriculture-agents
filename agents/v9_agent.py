"""
Kaggriculture — Rule-Based Starter Agent (v10)
================================================

v10 targets the four biggest gaps vs. top-ladder replays (episode
107847522 was the reference):

  1. AGGRESSIVE EARLY HIRING. The reference replay spent nearly all
     starting cash on labor by Day 1 -- 5 hands on Day 1, 4 on Day 2,
     5 on Day 3, 4 on Day 4 -- because the Fibonacci hire cost resets
     daily and the first two hands cost ~1 coin each. v9 hired one
     hand per turn, gated crew size on total_owned_tiles // 5 (only 5
     hands for a 25-tile quadrant), and used OPERATING_RESERVE=50 as
     the cash floor. v10 hires in bursts (up to HIRES_PER_TURN), floors
     crew at MIN_CREW_SIZE, and scales crew faster with land.

  2. MELON-WEIGHTED PLANTING. The reference replay planted ~60-70% of
     its tiles in MELON on Day 0 (highest profit/tile-day, longest
     cycle, needs the earliest start). v9 round-robined across the
     portfolio (1/3 melon, 1/3 wheat, 1/3 whatever ranked 3rd), which
     diluted the best crop. v10 assigns crops by weight: top crop gets
     ~60% of units, second ~25%, third ~15%.

  3. EARLY MULTI-ANIMAL LIVESTOCK (COW + SHEEP). The reference replay
     bought 2 cows + 2 sheep on Day 2 and had them placed by Day 3.
     v9 had ANIMAL_COUNT_CAP=1, gated ranching on already_expanded
     (>=2 quadrants unlocked), and only knew about COW. v10 removes
     the land gate for the first animal, raises the cap to
     ANIMAL_COUNT_CAP per animal type, and runs TWO ranch tracks
     (COW and SHEEP) in parallel with dedicated rancher slots.

  4. BULK SEED BUYING. The reference replay bought seeds in batches
     (BUY_SEED MELON 2, then 3, etc.). v9 bought 1 seed/crop/turn,
     so with 5+ hands planting, seeds constantly ran dry. v10 buys up
     to SEED_BUY_BATCH in a single order per crop per turn.

Everything from v9 that was working (endgame cutoffs, reserve floors,
weed DIG, price-history trickle-sell, rancher identity stability,
rancher's narrow wheat fallback, WHEAT_SELL_RESERVE) is preserved.

--- v9 docstring, kept for reference ---

v9 targeted the livestock + weed-clearing gaps described in its own
docstring below. The v9 docstring in turn preserved the v5 docstring
about the reserve-floor freeze and land-bought-ahead-of-labor bugs.
Those are still the right lessons; they're just not what v10 focuses
on. The v5/v9 fixes are all still in the code.
"""

import statistics

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS,
    ANIMALS,
    LAND_ORDER,
    LAND_PRICES,
    _is_shed_adjacent,
    _shed_access_tiles,
)

# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------
OPERATING_RESERVE = 50        # floor for cheap recurring costs: seeds, hire
CASH_RESERVE = 200            # floor for big one-off purchases: land, animals
TILES_PER_HAND = 4            # v10: was 5; crew now scales faster with land
MIN_CREW_SIZE = 5             # v10: floor so early quadrant is fully staffed
HIRES_PER_TURN = 5            # v10: burst-hire up to this many per turn
LAND_UTILIZATION_THRESHOLD = 0.75
SEED_BUY_BATCH = 3            # v10: now bought in bulk, not 1-at-a-time
PORTFOLIO_SIZE = 3
SELL_BATCH_CAP = 20
PRICE_HISTORY_WINDOW = 15
MIN_HISTORY_FOR_THRESHOLD = 4

# --- endgame cutoffs ---
DEFAULT_TOTAL_DAYS = 30
LAST_PLANT_DAY_MARGIN = 1
MIN_DAYS_FOR_LAND_PURCHASE = 3
MIN_DAYS_FOR_NEW_ANIMAL = 2

# --- livestock (v10: two parallel tracks) ---
# v10 runs COW and SHEEP in parallel. COW was chosen in v9 for its
# $80/tile-day base return; SHEEP adds wool (~$200 base) and further
# fertilizer, and was observed in the reference replay alongside cows.
ANIMAL_TARGETS = ("COW", "SHEEP")   # v10: was ("COW",)
ANIMAL_STRUCTURE = {a: ANIMALS[a]["structure"] for a in ANIMAL_TARGETS}

RANCHER_COUNT_PER_ANIMAL = 2   # ~1 rancher slot per this many animals
RANCHER_COUNT_MAX = 6          # v10: was 3; two tracks need more slots
ANIMAL_COUNT_CAP_PER_TYPE = 2  # v10: was a single global cap of 1
ANIMAL_READY_CASH = 250        # enough for one cow + reserve
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


def _crop_portfolio(money, prices, days_left):
    scored = []
    for crop, info in CROPS.items():
        if info["seed"] > money - OPERATING_RESERVE:
            continue
        if info["first_yield_day"] + LAST_PLANT_DAY_MARGIN > days_left:
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


def _unit_immediate_action(pos, farm, day, assigned_crop, have_seed):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("yield_units", 0) > 0
        and _is_mature(tile, day)
    ):
        return ["HARVEST"]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
        return ["WATER"]

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    if tile is None and have_seed and assigned_crop:
        return ["PLANT", assigned_crop]

    return None


def _unit_action(pos, farm, board_size, day, assigned_crop, have_seed, plantable_crops):
    """
    v10: watering urgency added. v9's scan order (harvest > water > weed >
    plant) meant that as soon as any tile was harvestable, every idle
    unit would converge on harvest and ignore a plant that had gone
    unwatered for 2+ days -- and an unwatered plant that hits its
    consecutive_unwatered cap is dead. v10 checks for URGENT water
    (consecutive_unwatered >= 1) before harvest.
    """
    immediate = _unit_immediate_action(pos, farm, day, assigned_crop, have_seed)
    if immediate:
        return immediate

    fx, fy = pos
    harvestable, needs_water, weeded, plantable = _scan_targets(farm, board_size, plantable_crops, day)

    urgent_water = [
        t for t in needs_water
        if farm["tiles"][t[1]][t[0]].get("consecutive_unwatered", 0) >= 1
    ]

    for group in (urgent_water, harvestable, needs_water, weeded, plantable):
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
        (x, y)
        for y in range(board_size)
        for x in range(board_size)
        if farm["tiles"][y][x] is None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))


def _nearest_shed_tile(fx, fy, board_size):
    tiles = _shed_access_tiles(board_size)
    return min(tiles, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))


def _rancher_action(pos, unit_inv, shed, farm, board_size, animal_target, animal_target_count):
    """
    v10: parameterized over animal_target (COW or SHEEP) so we can run
    two ranch tracks in parallel. Structure kind derived from the target.
    Logic otherwise identical to v9's single-track version -- the
    rancher for COW only ever looks at PASTUREs, only ever picks up
    COW, and only ever feeds with WHEAT (same feed for both species).
    """
    structure_kind = ANIMAL_STRUCTURE[animal_target]
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    # 1. Standing on our own living animal (this track's structure kind).
    if isinstance(tile, dict) and tile.get("kind") == structure_kind and "animal" in tile:
        if tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
        if not tile.get("fed_today") and unit_inv.get("WHEAT", 0) > 0:
            return ["FEED"]
        if not tile.get("cared_today"):
            return ["CARE"]

    structures = _find_structures(farm, board_size, structure_kind)
    empty_structures = [s for s in structures if "animal" not in s[2]]

    # 2. Holding a bought animal of THIS track, standing on matching empty structure.
    if (
        isinstance(tile, dict)
        and tile.get("kind") == structure_kind
        and "animal" not in tile
        and unit_inv.get(animal_target, 0) > 0
    ):
        return ["PLACE", animal_target]

    # 3. Build another structure for THIS track if under target.
    if len(structures) < animal_target_count and not empty_structures:
        if tile is None:
            return ["BUILD_COOP"] if structure_kind == "COOP" else ["BUILD_PASTURE"]
        target = _find_empty_tile(farm, board_size, fx, fy)
        if target:
            step = _step_toward(fx, fy, target[0], target[1])
            if step:
                return [step]
        return ["PASS"]

    # 4. Shed errands.
    need_animal_pickup = (
        bool(empty_structures)
        and unit_inv.get(animal_target, 0) <= 0
        and shed.get(animal_target, 0) > 0
    )
    need_wheat = unit_inv.get("WHEAT", 0) < WHEAT_FEED_BUFFER and shed.get("WHEAT", 0) > 0
    if need_animal_pickup or need_wheat:
        if _is_shed_adjacent((fx, fy), board_size):
            if need_animal_pickup:
                return ["PICKUP", animal_target, 1]
            return ["PICKUP", "WHEAT", WHEAT_FEED_BUFFER]
        tx, ty = _nearest_shed_tile(fx, fy, board_size)
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]

    # 5. Walk toward an empty structure waiting to be filled.
    if empty_structures:
        tx, ty, _ = min(empty_structures, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]

    # 6. Walk toward a living structure with pending work.
    for sx, sy, s in structures:
        if "animal" not in s:
            continue
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


def _crop_for_unit(idx, portfolio, total_units):
    """
    v10: weighted crop assignment. v9 round-robined (idx % len), which
    gave the #1 crop only 1/N of units -- so with the reference replay
    planting ~60-70% melon, v9 planted ~33% melon on the same tile pool
    and lost a big chunk of revenue. v10 weights: top crop gets ~60% of
    units, second ~25%, third ~15%. Falls back gracefully for a
    portfolio of size 1 or 2.
    """
    if not portfolio:
        return None
    if len(portfolio) == 1:
        return portfolio[0]
    weights = [0.60, 0.25, 0.15][:len(portfolio)]
    s = sum(weights)
    weights = [w / s for w in weights]
    frac = idx / max(1, total_units)
    cum = 0.0
    for crop, w in zip(portfolio, weights):
        cum += w
        if frac < cum:
            return crop
    return portfolio[-1]


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

    cfg = configuration or {}
    total_days = cfg.get("episodeSteps", 720) // cfg.get("turnsPerDay", 24)
    days_left = (total_days - 1) - day

    for product, price in market_prices.items():
        _update_price_history(product, price)

    # --- livestock presence check (v10: both tracks) --------------------------
    ranch_structures = {
        a: _find_structures(farm, board_size, ANIMAL_STRUCTURE[a])
        for a in ANIMAL_TARGETS
    }
    enough_season_left_to_start = (
        days_left >= max(ANIMALS[a]["first_yield_day"] for a in ANIMAL_TARGETS) + LAST_PLANT_DAY_MARGIN
    )

    # v10: removed the "already_expanded" gate for the FIRST animal. A
    # single pasture + one cow is ~$200 total and pays back in ~3 days;
    # gating it behind a land purchase (which v9 did) delayed livestock
    # to Day 5-7 at the earliest. First animal can go down on Day 2 if
    # affordable.
    any_structures = any(bool(ranch_structures[a]) for a in ANIMAL_TARGETS)
    ranch_active = enough_season_left_to_start and (
        any_structures or money >= ANIMAL_READY_CASH
    )

    # v10: per-type target. Scales with quadrant count as before, but
    # now applies to EACH type independently and the cap is higher.
    animal_target_count_per_type = 0
    if ranch_active:
        animal_target_count_per_type = min(
            ANIMAL_COUNT_CAP_PER_TYPE,
            max(1, len(farm["unlocked_quadrants"]) - 1),
        )

    total_animal_target = animal_target_count_per_type * len(ANIMAL_TARGETS)
    rancher_count = 0
    if total_animal_target > 0:
        rancher_count = min(
            RANCHER_COUNT_MAX,
            -(-total_animal_target // RANCHER_COUNT_PER_ANIMAL),  # ceil div
        )

    portfolio = _crop_portfolio(money, market_prices, days_left)

    market_orders = []

    # --- trickle sell (preserve v9 WHEAT reserve) -----------------------------
    wheat_reserve = WHEAT_SELL_RESERVE if ranch_active else 0
    for product, qty in shed.items():
        sellable = qty - wheat_reserve if product == "WHEAT" else qty
        if sellable > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, min(sellable, SELL_BATCH_CAP)])

    money_left = money

    # --- v10: BULK seed buying -------------------------------------------------
    # v9 bought 1 seed per crop per turn (BUY_SEED x 1). With 5+ hands
    # planting, the 3-seed stock was consumed in a single turn. v10 buys
    # the full SEED_BUY_BATCH shortfall in one order.
    for crop in portfolio:
        cost = CROPS[crop]["seed"]
        need = SEED_BUY_BATCH - seeds.get(crop, 0)
        if need <= 0:
            continue
        affordable = min(need, int((money_left - OPERATING_RESERVE) // cost))
        if affordable > 0:
            market_orders.append(["BUY_SEED", crop, affordable])
            money_left -= cost * affordable

    # Wheat seed specifically for the rancher's fallback patch.
    if ranch_active:
        need_w = SEED_BUY_BATCH - seeds.get("WHEAT", 0)
        if need_w > 0:
            cost_w = CROPS["WHEAT"]["seed"]
            affordable_w = min(need_w, int((money_left - OPERATING_RESERVE) // cost_w))
            if affordable_w > 0:
                market_orders.append(["BUY_SEED", "WHEAT", affordable_w])
                money_left -= cost_w * affordable_w

    # --- v10: BURST HIRING -----------------------------------------------------
    # v9: crew_target = total_owned // 5, one HIRE per turn, and 5 hands
    # total on a 25-tile quadrant. v10: floor of MIN_CREW_SIZE, scales
    # with TILES_PER_HAND=4, hires up to HIRES_PER_TURN per turn. This
    # matches the reference replay's Day-1 hiring bursts.
    total_owned_tiles, planted_tiles = _count_owned_and_planted(farm, board_size)
    target_crew_size = max(MIN_CREW_SIZE, total_owned_tiles // TILES_PER_HAND)
    hires_needed = target_crew_size - len(farm["hands"])
    for _ in range(min(hires_needed, HIRES_PER_TURN)):
        if money_left - OPERATING_RESERVE < 1:
            break
        market_orders.append(["HIRE"])
        money_left -= 1  # conservative lower bound; real Fib cost is 1 for first hires

    # --- land buying (unchanged from v9) --------------------------------------
    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER) and days_left >= MIN_DAYS_FOR_LAND_PURCHASE:
        next_land_price = LAND_PRICES[n_extra_unlocked]
        utilization = planted_tiles / total_owned_tiles if total_owned_tiles else 1.0
        if money_left - CASH_RESERVE >= next_land_price and utilization >= LAND_UTILIZATION_THRESHOLD:
            market_orders.append(["BUY_LAND"])
            money_left -= next_land_price

    # --- v10: BUY ANIMALS FOR BOTH TRACKS -------------------------------------
    # One buy per type per turn; skip a type if it already has an animal
    # bought-but-unplaced (in shed or in anyone's hands) so we don't
    # double-buy before the rancher can fetch the first one.
    animals_owned = {
        a: sum(1 for s in ranch_structures[a] if "animal" in s[2])
        for a in ANIMAL_TARGETS
    }
    for a in ANIMAL_TARGETS:
        animal_cost = ANIMALS[a]["cost"]
        animal_pending = shed.get(a, 0) > 0 or any(inv.get(a, 0) > 0 for inv in inventories)
        if (
            animal_target_count_per_type > animals_owned[a]
            and not animal_pending
            and enough_season_left_to_start
            and money_left - CASH_RESERVE >= animal_cost
        ):
            market_orders.append(["BUY_ANIMAL", a, 1])
            money_left -= animal_cost

    # --- unit actions ---------------------------------------------------------
    n_units = 1 + len(farm["hands"])

    def crop_for_unit(idx):
        return _crop_for_unit(idx, portfolio, n_units)

    plantable_crops = bool(portfolio)

    # Rancher slots: first `rancher_count` hands (indices 1..rancher_count).
    # Split them between COW and SHEEP tracks by alternating.
    rancher_indices = set(range(1, rancher_count + 1))

    crop_units = []
    hands_actions = [None] * len(farm["hands"])

    farmer_crop = crop_for_unit(0)
    farmer_have_seed = bool(farmer_crop) and seeds.get(farmer_crop, 0) > 0
    farmer_immediate = _unit_immediate_action(farm["farmer"], farm, day, farmer_crop, farmer_have_seed)
    farmer_action = farmer_immediate
    if farmer_immediate is None:
        crop_units.append({
            "idx": "farmer",
            "pos": tuple(farm["farmer"]),
            "have_seed": farmer_have_seed,
            "crop": farmer_crop,
        })

    # v10: assign each rancher hand to a specific animal track.
    # Alternating assignment: rancher 1 -> COW, rancher 2 -> SHEEP,
    # rancher 3 -> COW, rancher 4 -> SHEEP, etc. Track identity is
    # stable for the day because hands only get appended, never
    # reordered mid-day (same guarantee v9 relied on).
    rancher_track = {}
    if rancher_count > 0 and animal_target_count_per_type > 0:
        for i in range(1, rancher_count + 1):
            rancher_track[i] = ANIMAL_TARGETS[(i - 1) % len(ANIMAL_TARGETS)]

    for i, hand_pos in enumerate(farm["hands"], start=1):
        if i in rancher_indices and i in rancher_track:
            track = rancher_track[i]
            unit_inv = inventories[i] if i < len(inventories) else {}
            action = _rancher_action(
                hand_pos, unit_inv, shed, farm, board_size,
                track, animal_target_count_per_type,
            )
            if action == ["PASS"]:
                wheat_have_seed = seeds.get("WHEAT", 0) > 0
                action = _wheat_fallback_action(hand_pos, farm, board_size, day, wheat_have_seed)
            hands_actions[i - 1] = action
            continue
        hand_crop = crop_for_unit(i)
        hand_have_seed = bool(hand_crop) and seeds.get(hand_crop, 0) > 0
        immediate = _unit_immediate_action(hand_pos, farm, day, hand_crop, hand_have_seed)
        if immediate is not None:
            hands_actions[i - 1] = immediate
        else:
            crop_units.append({
                "idx": i - 1,
                "pos": tuple(hand_pos),
                "have_seed": hand_have_seed,
                "crop": hand_crop,
            })

    if crop_units:
        for u in crop_units:
            action = _unit_action(u["pos"], farm, board_size, day, u["crop"], u["have_seed"], plantable_crops)
            if u["idx"] == "farmer":
                farmer_action = action
            else:
                hands_actions[u["idx"]] = action

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}


# --------------------------------------------------------------------------
# Local smoke test / benchmark
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from kaggle_environments import make

    seeds = [0, 1, 2, 853407103, 89716423]
    print("=== v10 multi-seed benchmark vs starter ===")
    wins = 0
    total_reward = 0
    for seed in seeds:
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run([agent, "starter"])
        r0 = env.steps[-1][0]["reward"] or 0
        r1 = env.steps[-1][1]["reward"] or 0
        total_reward += r0
        if r0 > r1:
            wins += 1
        print(f"seed {seed:>10}: agent=${r0:>10,.0f}  starter=${r1:>10,.0f}  "
              f"{'WIN' if r0 > r1 else 'loss'}")
    print(f"\nwins: {wins}/{len(seeds)}   avg agent reward: ${total_reward/len(seeds):,.0f}")

    print()
    print("=== vs random / pass ===")
    for opponent in ["random", "pass"]:
        env = make("kaggriculture", debug=False)
        env.run([agent, opponent])
        r0 = env.steps[-1][0]["reward"] or 0
        r1 = env.steps[-1][1]["reward"] or 0
        print(f"vs {opponent:8s}  agent=${r0:>10,.0f}  opponent=${r1:>10,.0f}")

    print()
    print("=== livestock lifecycle check (seed 0) ===")
    env = make("kaggriculture", configuration={"seed": 0}, debug=True)
    env.run([agent, "starter"])
    for animal in ANIMAL_TARGETS:
        structure = ANIMAL_STRUCTURE[animal]
        placed_turn = first_yield_turn = None
        escapes = []
        prev_had = False
        for i, step in enumerate(env.steps):
            f = step[0]["observation"]["farms"][0]
            for y in range(len(f["tiles"])):
                for x in range(len(f["tiles"][y])):
                    t = f["tiles"][y][x]
                    if isinstance(t, dict) and t.get("kind") == structure:
                        has = "animal" in t and t.get("animal") == animal
                        if has and placed_turn is None:
                            placed_turn = i
                        if has and t.get("yield_units", 0) > 0 and first_yield_turn is None:
                            first_yield_turn = i
                        if prev_had and not has:
                            escapes.append(i)
                        prev_had = has
        product = ANIMALS[animal]["product"]
        print(f"{animal:>6}: placed turn={placed_turn}  first {product} turn={first_yield_turn}  "
              f"escapes={len(escapes)}")
    priv = env.steps[-1][0]["observation"]["private"]
    for animal in ANIMAL_TARGETS:
        product = ANIMALS[animal]["product"]
        print(f"final shed {product}: {priv['shed'].get(product)}")
    print(f"final reward: agent=${env.steps[-1][0]['reward']:,.0f}  "
          f"starter=${env.steps[-1][1]['reward']:,.0f}")