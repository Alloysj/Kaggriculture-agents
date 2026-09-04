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
CASH_RESERVE = 100            # protects big one-off spends (BUY_LAND, BUY_ANIMAL)
OPERATING_RESERVE = 20        # lower floor for cheap recurring costs (seeds, hire)
TARGET_CREW_SIZE = 6          # hands to maintain daily; Fib hire cost keeps this cheap
SEED_BUY_BATCH = 3            # max seeds to queue per turn, per crop in the portfolio
PORTFOLIO_SIZE = 3            # spread planting across this many top crops
SELL_BATCH_CAP = 20           # max units of one product sold in a single turn (trickle, not dump)
PRICE_HISTORY_WINDOW = 15     # turns of price memory per product for the sell-timing rule
MIN_HISTORY_FOR_THRESHOLD = 4  # sell opportunistically until we have this many samples

# --- livestock ---
ANIMAL_TARGET = "GOOSE"
STRUCTURE_KIND = ANIMALS[ANIMAL_TARGET]["structure"]  # "COOP"
RANCHER_COUNT = 1              # hands carved out for ranching, once crew allows
ANIMAL_READY_CASH = ANIMALS[ANIMAL_TARGET]["cost"] + CASH_RESERVE + 200  # don't start until comfortably affordable
WHEAT_FEED_BUFFER = 3          # wheat a rancher tries to carry so it isn't fetching every single day

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
    crop_info = CROPS.get(tile.get("crop"))
    if crop_info is None:
        return False
    return day - tile.get("planted_day", day) >= crop_info["first_yield_day"]


def _scan_targets(farm, board_size, plantable_crops, day):
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
                continue  # crop tiles only; animal structures handled separately
            if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
                harvestable.append((x, y))
            elif not tile.get("watered_today", False):
                needs_water.append((x, y))
    return harvestable, needs_water, plantable


def _unit_action(pos, farm, board_size, day, assigned_crop, have_seed, plantable_crops):
    """Crop-worker decision (farmer, or a hand not currently ranching)."""
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

    if tile is None and have_seed and assigned_crop:
        return ["PLANT", assigned_crop]

    harvestable, needs_water, plantable = _scan_targets(farm, board_size, plantable_crops, day)
    for group in (harvestable, needs_water, plantable):
        if not group:
            continue
        tx, ty = min(group, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]

    return ["PASS"]


# --------------------------------------------------------------------------
# Livestock
# --------------------------------------------------------------------------
def _find_structures(farm, board_size, structure_kind):
    """All tiles matching the given structure kind (COOP/PASTURE), with or
    without an animal placed. Returns list of (x, y, tile_dict)."""
    found = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == structure_kind:
                found.append((x, y, tile))
    return found


def _find_empty_tile(farm, board_size, fx, fy):
    """Nearest empty (buildable) owned tile, for siting a new structure."""
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


def _wheat_fallback_action(pos, farm, board_size, day, have_seed):
    """
    A rancher's idle-time wheat routine -- deliberately NARROWER than
    _unit_action's board-wide scan. Only considers the rancher's own wheat
    tiles and empty tiles, never other units' crops. Sharing the general
    scan (harvestable/needs_water/plantable across ALL crops) caused a
    genuine bug in testing: two tiles at equal distance can alternate
    being "nearest" as the unit moves between them, and since the scan is
    recomputed fresh every turn with no memory of prior intent, the unit
    got stuck in an infinite NORTH/SOUTH oscillation between them instead
    of ever reaching a wheat/empty tile. Restricting scope removes the
    competing, constantly-changing targets that triggered it.
    """
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
        if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]

    if tile is None and have_seed:
        return ["PLANT", "WHEAT"]

    candidates = []  # (x, y, priority) -- lower priority value wins
    for y in range(board_size):
        for x in range(board_size):
            t = farm["tiles"][y][x]
            if t is None:
                if have_seed:
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


def _rancher_action(pos, unit_inv, shed, farm, board_size):
    """
    Dedicated livestock routine for one unit. Priority, every turn,
    inferred purely from the current observation:

      1. Standing on our own living animal: harvest > collect fertilizer >
         feed > care, in that order (harvest/fertilizer never expire and
         cost nothing to grab; feed is the one with a real 2-day penalty
         for skipping, so it outranks the free CARE action).
      2. Holding a bought animal, standing on the matching empty structure
         -> place it.
      3. No structure exists yet -> build one (walking to an empty tile
         first if not already standing on one).
      4. Need a shed errand (fetch the bought animal, or top up wheat for
         feeding) -> do it if adjacent, otherwise walk to the shed.
      5. Structure exists with a living animal -> walk toward it to resume
         the daily loop.
      6. Nothing to do -> PASS (agent() falls back to crop work for this
         unit when this happens).
    """
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    # 1. Standing on our own living animal.
    if isinstance(tile, dict) and "animal" in tile:
        if tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
        if not tile.get("fed_today") and unit_inv.get("WHEAT", 0) > 0:
            return ["FEED"]
        if not tile.get("cared_today"):
            return ["CARE"]
        # Nothing left to do here today; fall through (may still need to
        # go restock wheat for tomorrow, handled below).

    structures = _find_structures(farm, board_size, STRUCTURE_KIND)

    # 2. Holding a bought animal, standing on the matching empty structure.
    if (
        isinstance(tile, dict)
        and tile.get("kind") == STRUCTURE_KIND
        and "animal" not in tile
        and unit_inv.get(ANIMAL_TARGET, 0) > 0
    ):
        return ["PLACE", ANIMAL_TARGET]

    # 3. No structure yet -> build one.
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

    # 4. Shed errands: fetch a bought-but-not-placed animal, or top up wheat.
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

    # 5. Walk toward an empty structure (waiting to place) or a living one
    # (to resume the daily loop).
    for group in (empty_structures, living_structures):
        if not group:
            continue
        tx, ty, _ = min(group, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
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
    inventories = private.get("inventories", [{}])
    market_prices = (obs.get("market", {}) or {}).get("prices", {})

    for product, price in market_prices.items():
        _update_price_history(product, price)

    structures = _find_structures(farm, board_size, STRUCTURE_KIND)

    # Once ranching is active (a coop exists, or we're about to build one),
    # wheat MUST be in the portfolio regardless of its profit/tile-day
    # ranking -- it's the only feed source, and ranking alone rarely puts
    # it in the top PORTFOLIO_SIZE. Missing this meant the goose had
    # nothing to eat in testing, no matter how well-positioned the rancher
    # was.
    ranch_active = bool(structures) or money >= ANIMAL_READY_CASH
    portfolio = _crop_portfolio(money, market_prices)
    if ranch_active and "WHEAT" not in portfolio:
        portfolio = portfolio + ["WHEAT"]

    market_orders = []

    # Trickle-sell. WHEAT is exempt from its own sell-all-surplus rule once
    # ranching is active: it's the feed supply, and testing showed the
    # generic sell loop was grabbing wheat out of the shed the instant it
    # landed there -- selling our own feed out from under the rancher
    # before it ever got a PICKUP chance. That was the actual cause behind
    # every escape, not a positioning or timing issue.
    wheat_reserve = WHEAT_FEED_BUFFER * 2 if ranch_active else 0
    for product, qty in shed.items():
        sellable = qty - wheat_reserve if product == "WHEAT" else qty
        if sellable > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, min(sellable, SELL_BATCH_CAP)])

    # Every spend below is checked against a RUNNING remaining-money
    # figure, decremented as we queue each one -- not all against the same
    # starting `money`. The engine resolves our order list sequentially and
    # rejects anything it can't afford at that point, so checking them
    # independently against one static snapshot let several spends land in
    # the same turn and jointly overshoot a single shared floor.
    #
    # Two different floors, not one: seeds and hires are cheap and
    # recurring -- gating them behind the same CASH_RESERVE used for
    # BUY_LAND/BUY_ANIMAL meant that once money drifted down to exactly
    # CASH_RESERVE (easy to do with a $300+ animal purchase in the mix),
    # even a $10 wheat seed failed the check, permanently zeroing the crop
    # portfolio with no way to earn back above reserve -- caught in
    # testing below. OPERATING_RESERVE is a much smaller floor that keeps
    # cheap, essential spending (especially wheat -- the feed supply)
    # alive even when money is tight.
    money_left = money

    # Seeds across the portfolio.
    for crop in portfolio:
        cost = CROPS[crop]["seed"]
        if seeds.get(crop, 0) < SEED_BUY_BATCH and money_left - OPERATING_RESERVE >= cost:
            market_orders.append(["BUY_SEED", crop, 1])
            money_left -= cost

    # Daily re-hire (Fibonacci cost resets every day).
    if len(farm["hands"]) < TARGET_CREW_SIZE and money_left - OPERATING_RESERVE >= 1:
        market_orders.append(["HIRE"])
        money_left -= 1  # conservative estimate; real Fib cost is looked up engine-side

    # Land, against real tier prices.
    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER):
        next_land_price = LAND_PRICES[n_extra_unlocked]
        if money_left - CASH_RESERVE >= next_land_price:
            market_orders.append(["BUY_LAND"])
            money_left -= next_land_price

    # --- livestock: buy the animal once a coop exists, is empty, and one
    # isn't already purchased-and-pending (in the shed or in someone's hands) ---
    coop_empty = any("animal" not in s[2] for s in structures)
    animal_pending = shed.get(ANIMAL_TARGET, 0) > 0 or any(inv.get(ANIMAL_TARGET, 0) > 0 for inv in inventories)
    animal_cost = ANIMALS[ANIMAL_TARGET]["cost"]
    if money >= ANIMAL_READY_CASH and coop_empty and not animal_pending and money_left - CASH_RESERVE >= animal_cost:
        market_orders.append(["BUY_ANIMAL", ANIMAL_TARGET, 1])
        money_left -= animal_cost

    # --- unit assignment ---
    def crop_for_unit(idx):
        if not portfolio:
            return None
        return portfolio[idx % len(portfolio)]

    plantable_crops = bool(portfolio)
    ranch_ready = ranch_active  # same condition; kept as a separate name for readability below

    # NOTE: an earlier version of this pinned the farmer permanently to
    # WHEAT once ranching started, reasoning that the farmer (unlike hands)
    # always exists so it's the reliable feed source. Testing showed that
    # was a bad trade: it sacrificed the one guaranteed unit's full-game
    # productivity for the sake of one $50-base-price goose, and reward
    # cratered (~$17K -> ~$800). The ACTUAL bug was that our own trickle-
    # sell logic was selling the wheat out of the shed before the rancher
    # could fetch it (fixed above, in the sell loop). With that fixed, the
    # shared profit-ranked rotation supplies plenty of wheat for one
    # goose's 1-unit/day appetite -- no dedicated grower needed.
    farmer_crop = crop_for_unit(0)
    farmer_have_seed = bool(farmer_crop) and seeds.get(farmer_crop, 0) > 0
    farmer_action = _unit_action(farm["farmer"], farm, board_size, day, farmer_crop, farmer_have_seed, plantable_crops)

    # Hands are only ever *appended* within a day (never reordered or
    # removed until the next day's reset -- see _do_hire), so the first
    # RANCHER_COUNT hire-order slots are a STABLE identity for the whole
    # day. Picking "the last N hands" instead would reassign which
    # physical hand is "the rancher" every time a new hire lands later
    # the same day, orphaning mid-errand state (a goose or wheat it was
    # carrying) -- that was the actual v4 bug caught in testing below.
    hands_actions = []
    for i, hand_pos in enumerate(farm["hands"], start=1):
        is_rancher = ranch_ready and i <= RANCHER_COUNT
        if is_rancher:
            unit_inv = inventories[i] if i < len(inventories) else {}
            action = _rancher_action(hand_pos, unit_inv, shed, farm, board_size)
            if action != ["PASS"]:
                hands_actions.append(action)
                continue
            # Nothing ranch-related to do this turn. Falling back to the
            # shared crop rotation here was the actual cause of the
            # repeated escapes in testing: harvest/water priority is
            # crop-agnostic and shared across ALL units, so under board-
            # wide maintenance load (weeds, watering backlog) the one
            # rotation slot notionally assigned to wheat could get pulled
            # into tending other units' crops indefinitely and never
            # actually reach an empty tile to plant its own -- seed stock
            # just sat there unused while the animal starved. Instead, the
            # rancher defaults to WHEAT specifically when idle, coupling
            # supply directly to the unit that needs it rather than
            # depending on an unrelated rotation slot's luck.
            wheat_have_seed = seeds.get("WHEAT", 0) > 0
            hands_actions.append(
                _unit_action(hand_pos, farm, board_size, day, "WHEAT", wheat_have_seed, True)
            )
            continue
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
