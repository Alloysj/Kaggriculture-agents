"""
Kaggriculture — Rule-Based Starter Agent (v7)
===============================================

Builds on v5 (reserve-floor and land-utilization fixes, validated against
a real Kaggle replay -- see that docstring below) by adding:

1. LIVESTOCK, targeting COW (swapped from GOOSE in v7). The mechanics
   below were originally built and debugged against GOOSE, and that
   debugging narrative is kept in the comments below since the bugs and
   fixes are animal-agnostic (rancher stability, wheat self-sabotage, the
   tug-of-war oscillation, etc. -- all apply the same way regardless of
   which animal is placed). The swap itself is a one-line constant change
   (STRUCTURE_KIND and BUILD_COOP/BUILD_PASTURE already derive from
   ANIMAL_TARGET), motivated by base-price economics: at base price,
   GOOSE returns $50/tile-day vs COW's $80/tile-day (melon, for
   comparison, is $137.50 -- livestock was never going to beat melon
   outright, but COW is a strictly better livestock choice than GOOSE by
   this metric, and $100 cheaper to start than SHEEP with a shorter
   production interval). Grounded in mechanics confirmed
   directly by the user against the live rules (PLACE/FEED/HARVEST/
   COLLECT_FERTILIZER/CARE semantics, including the care-bonus banking:
   FEED+CARE the same day banks +1 to pending_care_bonus; a missed feed
   loses that day's banking; on a production day the WHOLE bank pays out
   if fed, or is wiped if unfed). One hired hand is carved out as a
   dedicated "rancher" -- but this repeats a feature v4 got wrong twice
   before it worked, so the fixes found there are applied from the start
   here, not rediscovered:
     - Rancher identity must be STABLE across a day. Hands are only ever
       *appended* within a day (never reordered -- see _do_hire), so
       "the first RANCHER_COUNT hires" is a stable slot; "the last N
       hands" is NOT, since that cutoff shifts every time a new hire
       lands later the same day, orphaning whatever the previous rancher
       was carrying (a goose or wheat) mid-errand.
     - The rancher's own trickle-sell must NOT sell the wheat it needs
       for feed. v4's generic "sell any shed surplus" logic sold wheat
       out of the shed the instant it landed there, before the rancher
       ever got a PICKUP chance -- that was the actual cause of every
       goose escaping, not a positioning bug.
     - Wheat supply must be COUPLED to the rancher's own idle time, not
       diluted across the shared profit-ranked crop rotation. Sharing the
       rotation meant only ~1-in-6 unit-turns ever touched wheat, and
       under board-wide maintenance load (weeds, watering backlog) that
       slot could get pulled into tending OTHER crops indefinitely and
       never reach an empty tile to plant its own.
     - The rancher's idle-time wheat routine must have a NARROW scan
       scope (its own wheat tiles + empty tiles only), not the shared
       board-wide harvest/water scan used by crop workers -- sharing that
       scan caused a real infinite NORTH/SOUTH oscillation bug (two
       tiles alternating "nearest" as the unit moved between them, with
       no memory of prior intent to break the tie).
     - Pinning the FARMER permanently to wheat (an earlier attempt at
       "guaranteed supply") is NOT done -- it sacrificed the one
       always-present unit's full-game crop productivity for one $50
       animal and cratered reward (~$17K -> ~$800 in testing). The
       rancher grows its own small patch instead.
   The rancher is one of the slots within v5's land-scaled crew target,
   not an addition on top of it -- reserving ranching capacity shouldn't
   reopen the "hired beyond what the land can support" problem v5 fixed.

2. DIG for weed clearing. Confirmed by the user: DIG removes a weed (or a
   plant, to free the tile, or an empty coop/pasture) with no yield. Since
   v5's land-buying is gated on a utilization metric (planted / owned
   tiles), actively clearing weeds back to empty, replantable tiles
   directly helps clear that bar instead of leaving dead tiles sitting
   there dragging utilization down.

--- v5 docstring, unchanged below ---

Fixes two bugs found by analyzing the JSON replay of an ACTUAL Kaggle
submission (episode 105633438, module_version 1.32.7) -- not sandbox
guesses. That replay's leaderboard-visible score (300, rank 6000/7000)
was far below its own in-episode final money (12,721, actually a win
over the opponent's 11,921), and tracing the replay turn-by-turn found
why the STRATEGY itself was unreliable across games, independent of
whatever the leaderboard's aggregation does with the number:

1. THE RESERVE-FLOOR FREEZE. The replay showed money sitting at EXACTLY
   $100 (the old flat CASH_RESERVE) for a full week of the 30-day season
   (days 2, 5-10) -- once money hit that floor, even a $10 wheat seed
   failed the `money - CASH_RESERVE >= cost` check, so NO further seeds
   or hires could happen, permanently zeroing the crop portfolio with no
   path to earn back above reserve. Fix: two separate floors --
   OPERATING_RESERVE ($50) gates cheap recurring costs (seeds, hire),
   CASH_RESERVE ($200) gates big one-off purchases (land, and now the
   animal purchase).

2. LAND BOUGHT FAR AHEAD OF LABOR. The same replay had all 4 quadrants
   unlocked by day 12 while 80 of 100 tiles sat completely empty. Fix:
   BUY_LAND now also requires LAND_UTILIZATION_THRESHOLD (75%) of owned
   land to be actively planted first, and crew size scales with owned
   land (1 hand per TILES_PER_HAND=5 tiles) instead of a flat target.

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
CASH_RESERVE = 200            # floor for big one-off purchases: land, animal
TILES_PER_HAND = 5            # crew target scales with owned land, not a flat number
LAND_UTILIZATION_THRESHOLD = 0.75  # don't unlock next quadrant until current land is this full
SEED_BUY_BATCH = 3            # max seeds to queue per turn, per crop in the portfolio
PORTFOLIO_SIZE = 3            # spread planting across this many top crops
SELL_BATCH_CAP = 20           # max units of one product sold in a single turn (trickle, not dump)
PRICE_HISTORY_WINDOW = 15     # turns of price memory per product for the sell-timing rule
MIN_HISTORY_FOR_THRESHOLD = 4  # sell opportunistically until we have this many samples

# --- livestock ---
ANIMAL_TARGET = "COW"
STRUCTURE_KIND = ANIMALS[ANIMAL_TARGET]["structure"]  # "PASTURE"
RANCHER_COUNT = 1              # crew slots (within the land-scaled target) reserved for ranching
ANIMAL_READY_CASH = ANIMALS[ANIMAL_TARGET]["cost"] + CASH_RESERVE + 200  # don't start until comfortably affordable
WHEAT_FEED_BUFFER = 3          # wheat a rancher tries to carry so it isn't fetching every single day
WHEAT_SELL_RESERVE = WHEAT_FEED_BUFFER * 2  # wheat exempt from trickle-sell once ranching is active

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
        if info["seed"] > money - OPERATING_RESERVE:
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

    Includes WEED tiles as DIG targets: v5's land-buying is gated on a
    utilization metric (planted / owned tiles), so actively clearing
    weeds back to empty, replantable tiles helps clear that bar instead
    of leaving dead tiles sitting there.
    """
    harvestable, needs_water, weeded, plantable = [], [], [], []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                if plantable_crops:
                    plantable.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue  # "LOCKED"
            if tile.get("kind") == "WEED":
                weeded.append((x, y))
                continue
            if tile.get("kind") != "PLANT":
                continue  # ignore animal structures for this crop-only pass
            if tile.get("yield_units", 0) > 0 and _is_mature(tile, day):
                harvestable.append((x, y))
            elif not tile.get("watered_today", False):
                needs_water.append((x, y))
    return harvestable, needs_water, weeded, plantable


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

    # Standing on a weed -> clear it, reclaiming the tile for replanting.
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    # Standing on empty, owned land with our assigned crop's seed in hand.
    if tile is None and have_seed and assigned_crop:
        return ["PLANT", assigned_crop]

    # Otherwise, walk toward the nearest useful tile: harvest > water > weed > plant.
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
    """
    A rancher's idle-time wheat routine -- deliberately NARROWER than
    _unit_action's board-wide scan. Only considers the rancher's own wheat
    tiles and empty tiles, never other units' crops. Sharing the general
    scan caused a genuine bug in v4 testing: two tiles at equal distance
    can alternate being "nearest" as the unit moves between them, and
    since the scan is recomputed fresh every turn with no memory of prior
    intent, the unit got stuck in an infinite NORTH/SOUTH oscillation
    between them instead of ever reaching a wheat/empty tile. Restricting
    scope removes the competing, constantly-changing targets that
    triggered it.
    """
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

    candidates = []  # (x, y, priority) -- lower priority value wins
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


def _rancher_action(pos, unit_inv, shed, farm, board_size):
    """
    Dedicated livestock routine for one unit. Priority, every turn,
    inferred purely from the current observation:

      1. Standing on our own living animal: harvest > collect fertilizer >
         feed > care. Feed outranks the free CARE action because a missed
         feed both risks the 2-day escape AND wipes that day's care-bonus
         banking (confirmed rule: FEED+CARE the same day banks +1 to
         pending_care_bonus; an unfed day banks nothing regardless of
         care -- "basic needs first").
      2. Holding a bought animal, standing on the matching empty structure
         -> place it.
      3. No structure exists yet -> build one (walking to an empty tile
         first if not already standing on one).
      4. Need a shed errand (fetch the bought animal, or top up wheat for
         feeding) -> do it if adjacent, otherwise walk to the shed.
      5. Structure exists with a living animal -> walk toward it to resume
         the daily loop.
      6. Nothing to do -> PASS (agent() gives this unit its own small
         wheat patch to tend when this happens -- see
         _wheat_fallback_action -- rather than joining the shared crop
         rotation, which starved wheat supply in earlier testing).
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
        # Nothing left to do here today; fall through.

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

    # 5. Walk toward an empty structure (waiting to place). Only walk
    # toward a LIVING structure if there's an actual pending task there
    # today -- otherwise this unconditionally pulled the rancher back to
    # the coop the instant it stepped off-tile (e.g., to go plant wheat),
    # directly fighting _wheat_fallback_action's pull toward an empty
    # tile and producing a genuine infinite WEST/EAST oscillation in
    # testing: nothing-to-do-here -> PASS -> wheat fallback walks away ->
    # next turn this step unconditionally walks back -> repeat forever.
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
    """Single board scan shared by land-utilization and crew-sizing."""
    owned, planted = 0, 0
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                owned += 1  # owned, empty
            elif isinstance(tile, dict):
                owned += 1  # owned (LOCKED is a plain string, not a dict)
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

    # --- keep price memory fresh ------------------------------------------------
    for product, price in market_prices.items():
        _update_price_history(product, price)

    structures = _find_structures(farm, board_size, STRUCTURE_KIND)
    # Ranching only starts once land has already expanded past the first
    # quadrant -- same "develop before expanding" philosophy applied to
    # livestock. Testing showed starting ranching on a single 25-tile
    # quadrant (crew of 5, 1 carved out for ranching) left only 4 units to
    # fight both routine maintenance AND a weed backlog on a small board,
    # and the crop economy cratered (~$26K -> ~$9K) even though the
    # livestock loop itself worked correctly. A larger, already-expanded
    # crew can afford to spare one unit; a small one-quadrant crew can't.
    already_expanded = len(farm["unlocked_quadrants"]) >= 2
    ranch_active = already_expanded and (bool(structures) or money >= ANIMAL_READY_CASH)

    # --- this turn's crop portfolio (top few by live profit/tile-day) -----------
    portfolio = _crop_portfolio(money, market_prices)

    # --- market orders -----------------------------------------------------------
    market_orders = []

    # Trickle-sell: cap how much of one product goes out in a single turn
    # instead of dumping the whole shed and walking the price down
    # ourselves. WHEAT is exempt from selling its own surplus once ranching
    # is active -- it's the feed supply, and v4 testing found the generic
    # sell-all-surplus rule was grabbing wheat out of the shed the instant
    # it landed there, before the rancher ever got a PICKUP chance. That
    # was the actual cause of every goose escaping, not a positioning bug.
    wheat_reserve = WHEAT_SELL_RESERVE if ranch_active else 0
    for product, qty in shed.items():
        sellable = qty - wheat_reserve if product == "WHEAT" else qty
        if sellable > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            market_orders.append(["SELL", product, min(sellable, SELL_BATCH_CAP)])

    # Every spend below is checked against a RUNNING remaining-money figure,
    # decremented as we queue each one -- not all against the same starting
    # `money`. Two different floors, not one: seeds and hires are cheap and
    # recurring, so gating them behind the same reserve used for big
    # purchases was the actual cause of the v5-motivating replay's
    # reserve-floor freeze. OPERATING_RESERVE is a much lower floor that
    # keeps cheap, essential spending alive even when cash is tight;
    # CASH_RESERVE only guards big one-off purchases (land, animal).
    money_left = money

    # Seed stock across the whole portfolio.
    for crop in portfolio:
        cost = CROPS[crop]["seed"]
        if seeds.get(crop, 0) < SEED_BUY_BATCH and money_left - OPERATING_RESERVE >= cost:
            market_orders.append(["BUY_SEED", crop, 1])
            money_left -= cost

    # WHEAT seed, specifically for the rancher's own dedicated patch
    # (_wheat_fallback_action), independent of whether WHEAT ranks in the
    # profit-based portfolio above. Missing this was a real bug caught in
    # testing: the rancher's wheat routine only plants if it's holding
    # seed, and if WHEAT never happens to be profitable enough to make the
    # top PORTFOLIO_SIZE, nothing ever buys it -- the rancher's patch sits
    # permanently unseeded, shed WHEAT never rises above 0, and the goose
    # starves no matter how well-positioned the rancher is.
    if ranch_active and seeds.get("WHEAT", 0) < SEED_BUY_BATCH and money_left - OPERATING_RESERVE >= CROPS["WHEAT"]["seed"]:
        market_orders.append(["BUY_SEED", "WHEAT", 1])
        money_left -= CROPS["WHEAT"]["seed"]

    # Re-hire up to a crew size that scales with owned land (1 hand per
    # TILES_PER_HAND tiles), not a flat number that can outpace whatever
    # land is actually unlocked -- cheap because hire cost resets daily
    # and follows Fibonacci (first couple hands cost ~1 coin). The rancher
    # (once ranching is active) is one of these slots, not an addition on
    # top of them -- reserving ranching capacity shouldn't reopen the
    # "hired beyond what the land can support" problem v5 fixed.
    total_owned_tiles, planted_tiles = _count_owned_and_planted(farm, board_size)
    target_crew_size = max(1, total_owned_tiles // TILES_PER_HAND)
    if len(farm["hands"]) < target_crew_size and money_left - OPERATING_RESERVE >= 1:
        market_orders.append(["HIRE"])
        money_left -= 1  # conservative estimate; real Fib cost is looked up engine-side

    # Buy the NEXT real land tier only once (a) we can afford it with the
    # bigger reserve to spare, AND (b) the land we already own is at least
    # LAND_UTILIZATION_THRESHOLD actively planted.
    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER):
        next_land_price = LAND_PRICES[n_extra_unlocked]
        utilization = planted_tiles / total_owned_tiles if total_owned_tiles else 1.0
        if money_left - CASH_RESERVE >= next_land_price and utilization >= LAND_UTILIZATION_THRESHOLD:
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

    # Hands are only ever *appended* within a day (never reordered or
    # removed until the next day's reset -- see _do_hire), so the first
    # RANCHER_COUNT hire-order slots are a STABLE identity for the whole
    # day. Picking "the last N hands" instead would reassign which
    # physical hand is "the rancher" every time a new hire lands later
    # the same day, orphaning mid-errand state (a goose or wheat it was
    # carrying) -- that was a real v4 bug caught in testing.
    hands_actions = []
    for i, hand_pos in enumerate(farm["hands"], start=1):
        is_rancher = ranch_active and i <= RANCHER_COUNT
        if is_rancher:
            unit_inv = inventories[i] if i < len(inventories) else {}
            action = _rancher_action(hand_pos, unit_inv, shed, farm, board_size)
            if action != ["PASS"]:
                hands_actions.append(action)
                continue
            # Nothing ranch-related to do this turn. The rancher tends its
            # OWN small wheat patch when idle, rather than joining the
            # shared crop rotation -- v4 testing found the shared rotation
            # only gave wheat ~1-in-6 unit-turns, and under board-wide
            # maintenance load that slot could get pulled into tending
            # OTHER crops indefinitely and never reach an empty tile to
            # plant its own, starving the animal.
            wheat_have_seed = seeds.get("WHEAT", 0) > 0
            hands_actions.append(_wheat_fallback_action(hand_pos, farm, board_size, day, wheat_have_seed))
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

    # Replay-seed check (episode 105633438, the one that motivated v5).
    print()
    print("--- replay-seed check (89716423) ---")
    env = make("kaggriculture", configuration={"seed": 89716423}, debug=True)
    env.run([agent, "starter"])
    money_at_100_or_below = sum(
        1 for step in env.steps if step[0]["observation"]["farms"][0]["money"] <= 105
    )
    print(f"turns with money <= 105 (was 155+/720 stuck at exactly 100 before v5): {money_at_100_or_below}/720")
    final_obs = env.steps[-1][0]["observation"]
    owned, planted = _count_owned_and_planted(final_obs["farms"][0], len(final_obs["farms"][0]["tiles"]))
    print(f"final land utilization: {planted}/{owned} = {planted/owned:.0%}")
    print(f"final unlocked quadrants: {final_obs['farms'][0]['unlocked_quadrants']}")
    print(f"final reward: agent=${env.steps[-1][0]['reward']:,.0f}  opponent=${env.steps[-1][1]['reward']:,.0f}")

    # Livestock lifecycle check: did the animal actually get placed, fed,
    # and produce output -- not just "did money not crash."
    print()
    print(f"--- livestock lifecycle check ({ANIMAL_TARGET} vs starter, seed 0) ---")
    env = make("kaggriculture", configuration={"seed": 0}, debug=True)
    env.run([agent, "starter"])
    animal_placed_turn = first_yield_turn = None
    escape_turns = []
    prev_had_animal = False
    for i, step in enumerate(env.steps):
        f = step[0]["observation"]["farms"][0]
        for y in range(len(f["tiles"])):
            for x in range(len(f["tiles"][y])):
                t = f["tiles"][y][x]
                if isinstance(t, dict) and t.get("kind") == STRUCTURE_KIND:
                    has_animal = "animal" in t
                    if has_animal and animal_placed_turn is None:
                        animal_placed_turn = i
                    if has_animal and t.get("yield_units", 0) > 0 and first_yield_turn is None:
                        first_yield_turn = i
                    if prev_had_animal and not has_animal:
                        escape_turns.append(i)
                    prev_had_animal = has_animal
    print(f"{ANIMAL_TARGET} placed at turn: {animal_placed_turn}")
    print(f"first {ANIMALS[ANIMAL_TARGET]['product']} at turn: {first_yield_turn}")
    print(f"escape turns: {escape_turns}")
    priv = env.steps[-1][0]["observation"]["private"]
    print(f"final shed {ANIMALS[ANIMAL_TARGET]['product']}: {priv['shed'].get(ANIMALS[ANIMAL_TARGET]['product'])}")
    print(f"final reward: agent=${env.steps[-1][0]['reward']:,.0f}  opponent=${env.steps[-1][1]['reward']:,.0f}")
