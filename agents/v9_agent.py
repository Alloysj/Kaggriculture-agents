"""
Kaggriculture — Rule-Based Starter Agent (v9)
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

Run this file directly for a local smoke test / benchmark against the
built-in "random", "starter", and "pass" agents, including a run on the
exact seed (89716423) from the v5-motivating replay.
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

# --- endgame cutoffs ---
# Diagnosed from a real lost ladder game: 27 tiles planted in the final
# days (day 20-29) never reached maturity before the season ended at day
# 29, wasting $1,890 in seed cost alone -- plus land and crew time spent
# on all of it, for zero return. The loss margin was $671. Crops and land
# bought too close to the end can never earn their cost back.
DEFAULT_TOTAL_DAYS = 30        # episodeSteps(720) / turnsPerDay(24); read from configuration when available
LAST_PLANT_DAY_MARGIN = 1      # extra buffer day so a harvest also has time to reach the shed and sell
MIN_DAYS_FOR_LAND_PURCHASE = 3  # don't buy a new quadrant if there's not even time for one fast crop cycle
MIN_DAYS_FOR_NEW_ANIMAL = 2    # don't start a new animal (build+buy) too close to the end either

# --- livestock ---
ANIMAL_TARGET = "COW"
STRUCTURE_KIND = ANIMALS[ANIMAL_TARGET]["structure"]  # "PASTURE"
RANCHER_COUNT_PER_ANIMAL = 1   # 1 dedicated rancher per animal -- see rank-based design note below
RANCHER_COUNT_MAX = 4          # hard cap on crew slots reserved for ranching, regardless of animal count
ANIMAL_COUNT_CAP = 4           # hard cap on animals
# Multiple animals, safely this time: each rancher is given a RANK (0, 1, 2...)
# and deterministically claims the structure at that rank once all structures
# are sorted by position -- e.g. rancher 0 always tends the first (by
# position) pasture, rancher 1 the second, etc. Since every rancher computes
# the same sort independently, they agree on ownership without needing to
# communicate, and a structure is only ever built by the rancher whose rank
# exactly equals the current structure count (so only one rancher builds at
# a time). This replaces an earlier "any rancher tends any pending animal"
# design that caused a real bug: multiple ranchers converging on the same
# animal while others starved, over 100 consecutive escape/replace cycles,
# crashing the economy from ~$25-29K to ~$200-1,500 in testing. Grounded in
# a real top-player replay showing successful agents managing many animals
# at once (one had 8 cows, 6 sheep, 3 geese by game end) -- this version
# scopes to one species (COW, still) at a higher count, rather than adding
# cross-species coordination in the same pass as a second compounding risk.
ANIMAL_READY_CASH = ANIMALS[ANIMAL_TARGET]["cost"] + OPERATING_RESERVE + 50  # affordable-now bar, not a big-buffer bar --
                                # a real top-player replay bought 2 cows + 2 sheep at turn 2, well
                                # before any land purchase, committing over half of starting cash
                                # immediately. The old CASH_RESERVE+200 threshold ($800) made that
                                # impossible this early; this is deliberately closer to "just
                                # affordable" than "comfortably affordable."
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


def _crop_portfolio(money, prices, days_left):
    """
    Rank every crop we can currently afford to seed by live profit/tile-day
    and return the top PORTFOLIO_SIZE. Planting gets spread across this
    list (see agent()) instead of piling into a single "best" crop.

    Excludes any crop that couldn't reach first_yield_day (plus a safety
    margin) before the season ends -- see the endgame-cutoff note above.
    `days_left` is the number of days remaining AFTER today.
    """
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
    """
    Kept for standalone testing/back-compat: decide one unit's action in
    isolation (no cross-unit coordination). agent() itself uses
    _unit_immediate_action + _assign_movement_targets instead, so several
    units don't independently converge on the same tile -- see that
    function's docstring for why.
    """
    immediate = _unit_immediate_action(pos, farm, day, assigned_crop, have_seed)
    if immediate:
        return immediate

    fx, fy = pos
    harvestable, needs_water, weeded, plantable = _scan_targets(farm, board_size, plantable_crops, day)
    for group in (harvestable, needs_water, weeded, plantable):
        if not group:
            continue
        tx, ty = min(group, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        step = _step_toward(fx, fy, tx, ty)
        if step:
            return [step]

    return ["PASS"]


def _unit_immediate_action(pos, farm, day, assigned_crop, have_seed):
    """
    What a unit should do if it's ALREADY standing on an actionable tile:
    harvest, water, dig a weed, or plant. Returns None if there's nothing
    to do right here -- meaning the unit needs to move, which is handled
    separately by _assign_movement_targets so multiple units don't
    independently converge on the same tile.
    """
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


def _assign_movement_targets(units, harvestable, needs_water, weeded, plantable):
    """
    Greedy nearest-unit-to-nearest-target assignment, run ONCE per turn
    across every unit that needs to move (has no immediate on-tile
    action) -- not independently per unit. Diagnosed from a real lost
    ladder game: roughly half of all unit-actions were pure movement, and
    with each unit picking its own nearest target from the same snapshot,
    multiple units could converge on the same tile (only one benefits;
    the rest walk there for nothing).

    `units` is a list of dicts: {"idx": key, "pos": (x, y), "have_seed": bool}.
    Priority mirrors _unit_action's old per-unit order: harvest > water >
    weed > plant. Harvest/water/weed are crop-agnostic -- any unit can do
    them. Plant targets only consider units currently holding seed (a
    unit with none can't use a plant target this turn).

    Returns a dict: idx -> (tx, ty), or idx -> None if nothing was
    available for that unit this turn.
    """
    assignments = {}
    remaining = list(units)

    def greedy_assign(pool, targets):
        available = list(targets)
        while pool and available:
            best_pair, best_dist = None, None
            for pi, u in enumerate(pool):
                for ti, t in enumerate(available):
                    d = abs(u["pos"][0] - t[0]) + abs(u["pos"][1] - t[1])
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_pair = (pi, ti)
            pi, ti = best_pair
            u = pool.pop(pi)
            t = available.pop(ti)
            assignments[u["idx"]] = t

    for group in (harvestable, needs_water, weeded):
        if group and remaining:
            greedy_assign(remaining, group)

    if plantable and remaining:
        seed_eligible = [u for u in remaining if u["have_seed"]]
        greedy_assign(seed_eligible, plantable)
        remaining = [u for u in remaining if u["idx"] not in assignments]

    for u in remaining:
        assignments[u["idx"]] = None

    return assignments


def _wheat_fallback_action(pos, farm, board_size, day, have_seed):
    """
    A rancher's idle-time wheat routine -- deliberately NARROWER than
    the crop-worker board-wide scan. Only considers the rancher's own wheat
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


def _rancher_action(pos, unit_inv, shed, farm, board_size, my_rank):
    """
    Dedicated livestock routine for one unit, responsible for exactly ONE
    structure -- the one at position `my_rank` once every structure of
    STRUCTURE_KIND is sorted by (y, x). Every rancher computes this same
    sort independently, so all ranchers agree on who owns what without
    needing to coordinate directly -- rancher 0 always tends the first
    (by position) structure, rancher 1 the second, and so on. A structure
    is only ever built by the rancher whose rank exactly equals the
    current structure count, so only one rancher builds at a time.

    This replaces an earlier "any rancher tends any pending animal"
    design that caused a real, serious bug in testing: with multiple
    ranchers and no ownership concept, they could all converge on the
    same animal while others starved, over 100 consecutive escape/replace
    cycles, crashing the economy from ~$25-29K to ~$200-1,500 on the same
    seeds. Grounded in a real top-player replay showing successful agents
    managing many animals at once (one had 17 total -- 8 cows, 6 sheep, 3
    geese -- by game end).

    Priority, every turn, inferred purely from the current observation:
      1. Standing on OUR structure's living animal: harvest > collect
         fertilizer > feed > care. Feed outranks the free CARE action
         because a missed feed both risks the 2-day escape AND wipes that
         day's care-bonus banking (confirmed rule: FEED+CARE the same day
         banks +1 to pending_care_bonus; an unfed day banks nothing
         regardless of care -- "basic needs first").
      2. Holding a bought animal, standing on our own empty structure ->
         place it. (Placement is opportunistic on ANY empty structure the
         unit happens to be standing on, not gated to "mine" -- once
         already there, there's no coordination risk in just using it.)
      3. Our own structure doesn't exist yet, and it's our turn to build
         it (exactly `my_rank` structures currently exist) -> build one.
      4. Need a shed errand (fetch the bought animal, or top up wheat for
         feeding) -> do it if adjacent, otherwise walk to the shed.
      5. Walk toward OUR OWN structure specifically -- not "the nearest
         one" -- if there's an actual pending task there today.
      6. Nothing to do -> PASS (agent() gives this unit its own small
         wheat patch to tend when this happens -- see
         _wheat_fallback_action -- rather than joining the shared crop
         rotation, which starved wheat supply in earlier testing).
    """
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    # 1. Standing on a living animal (ours or not -- opportunistic; normal
    # movement in step 5 only ever sends this unit toward its OWN
    # structure, so it shouldn't often end up elsewhere, but if it does,
    # there's no harm in tending whatever's here).
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

    all_structures = sorted(_find_structures(farm, board_size, STRUCTURE_KIND), key=lambda s: (s[1], s[0]))
    my_structure = all_structures[my_rank] if my_rank < len(all_structures) else None

    # 2. Holding a bought animal, standing on an empty structure -> place it.
    if (
        isinstance(tile, dict)
        and tile.get("kind") == STRUCTURE_KIND
        and "animal" not in tile
        and unit_inv.get(ANIMAL_TARGET, 0) > 0
    ):
        return ["PLACE", ANIMAL_TARGET]

    # 3. Build our own structure, if it doesn't exist yet and it's our turn.
    if my_structure is None and len(all_structures) == my_rank:
        if tile is None:
            return ["BUILD_COOP"] if STRUCTURE_KIND == "COOP" else ["BUILD_PASTURE"]
        target = _find_empty_tile(farm, board_size, fx, fy)
        if target:
            step = _step_toward(fx, fy, target[0], target[1])
            if step:
                return [step]
        return ["PASS"]

    # 4. Shed errands: fetch a bought-but-not-placed animal for OUR
    # structure specifically, or top up wheat.
    need_animal_pickup = (
        my_structure is not None
        and "animal" not in my_structure[2]
        and unit_inv.get(ANIMAL_TARGET, 0) <= 0
        and shed.get(ANIMAL_TARGET, 0) > 0
    )
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

    # 5. Walk toward OUR OWN structure -- not the nearest one -- if it
    # exists and has a pending task (empty, or fed/cared/harvest/
    # fertilizer pending). Only walking when there's an actual reason to
    # go is what fixed a real infinite oscillation bug in testing:
    # unconditionally pulling back the instant the unit stepped off-tile
    # directly fought the wheat-patch fallback's pull toward an empty
    # tile, producing a genuine infinite back-and-forth.
    if my_structure is not None:
        sx, sy, s = my_structure
        pending = (
            "animal" not in s
            or (s.get("yield_units", 0) > 0)
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

    # Days remaining AFTER today -- drives the endgame cutoffs below.
    # Diagnosed from a real lost ladder game: 27 tiles planted in the
    # final days never reached maturity before the season ended, wasting
    # $1,890 in seed cost alone (plus land and crew time), in a game lost
    # by only $671.
    cfg = configuration or {}
    total_days = cfg.get("episodeSteps", 720) // cfg.get("turnsPerDay", 24)
    days_left = (total_days - 1) - day

    # --- keep price memory fresh ------------------------------------------------
    for product, price in market_prices.items():
        _update_price_history(product, price)

    structures = _find_structures(farm, board_size, STRUCTURE_KIND)
    total_owned_tiles, planted_tiles = _count_owned_and_planted(farm, board_size)
    target_crew_size = max(1, total_owned_tiles // TILES_PER_HAND)

    # Ranching starts as soon as it's affordable, not gated behind land
    # expansion. An earlier version required 2+ quadrants first, reasoning
    # that a single 25-tile quadrant couldn't spare a crop worker -- but a
    # real top-player replay (final rewards ~$117K and ~$119K, vs our own
    # ~$27K average) showed the opening move buying 2 cows AND 2 sheep at
    # turn 2, literally the start of the game, well before any land
    # purchase. The actual lesson from the earlier crash wasn't "ranching
    # is premature on one quadrant" -- it was "don't reserve MORE ranching
    # capacity than the crew can spare." That's handled below by scaling
    # the animal target with crew size, not land tier.
    enough_season_left_to_start = days_left >= ANIMALS[ANIMAL_TARGET]["first_yield_day"] + LAST_PLANT_DAY_MARGIN
    ranch_active = bool(structures) or (money >= ANIMAL_READY_CASH and enough_season_left_to_start)

    # How many animals we're aiming for right now, scaled with CREW size
    # (roughly 1 rancher per 3 crop workers) rather than land tier, and
    # capped. Grounded in the same replay: the winning player had 17
    # animals (8 cows, 6 sheep, 3 geese) by game end -- v7 could only ever
    # manage one, since it only built a new structure when literally zero
    # existed.
    animal_target_count = 0
    if ranch_active:
        animal_target_count = min(ANIMAL_COUNT_CAP, max(1, target_crew_size // 3))
    rancher_count = min(RANCHER_COUNT_MAX, animal_target_count)

    # --- this turn's crop portfolio (top few by live profit/tile-day) -----------
    portfolio = _crop_portfolio(money, market_prices, days_left)

    # --- market orders -----------------------------------------------------------
    market_orders = []

    # Trickle-sell: cap how much of one product goes out in a single turn
    # instead of dumping the whole shed and walking the price down
    # ourselves. WHEAT is exempt from selling its own surplus once ranching
    # is active -- it's the feed supply, and v4 testing found the generic
    # sell-all-surplus rule was grabbing wheat out of the shed the instant
    # it landed there, before the rancher ever got a PICKUP chance. That
    # was the actual cause of every goose escaping, not a positioning bug.
    # FERTILIZER is NOT exempted -- selling it directly turned out to be a
    # major revenue stream in the same top-player replay (10,357 units
    # sold across the game), not just a minor byproduct to route back onto
    # crops via FERTILIZE.
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
    # and follows Fibonacci (first couple hands cost ~1 coin). The
    # rancher slots (once ranching is active) are within this target, not
    # an addition on top of it -- reserving ranching capacity shouldn't
    # reopen the "hired beyond what the land can support" problem v5 fixed.
    if len(farm["hands"]) < target_crew_size and money_left - OPERATING_RESERVE >= 1:
        market_orders.append(["HIRE"])
        money_left -= 1  # conservative estimate; real Fib cost is looked up engine-side

    # Buy the NEXT real land tier only once (a) we can afford it with the
    # bigger reserve to spare, (b) the land we already own is at least
    # LAND_UTILIZATION_THRESHOLD actively planted, AND (c) there's enough
    # season left for even a fast crop cycle on the new land to pay off --
    # a real lost ladder game bought a third quadrant with only 6 days
    # left and immediately planted melon (10-12 day cycle) on it, none of
    # which ever matured.
    n_extra_unlocked = len(farm["unlocked_quadrants"]) - 1
    if n_extra_unlocked < len(LAND_ORDER) and days_left >= MIN_DAYS_FOR_LAND_PURCHASE:
        next_land_price = LAND_PRICES[n_extra_unlocked]
        utilization = planted_tiles / total_owned_tiles if total_owned_tiles else 1.0
        if money_left - CASH_RESERVE >= next_land_price and utilization >= LAND_UTILIZATION_THRESHOLD:
            market_orders.append(["BUY_LAND"])
            money_left -= next_land_price

    # --- livestock: buy animals up to animal_target_count, one at a time,
    # as long as none is currently sitting bought-but-unplaced (in the
    # shed or someone's hands) ---
    animal_cost = ANIMALS[ANIMAL_TARGET]["cost"]
    animal_pending = shed.get(ANIMAL_TARGET, 0) > 0 or any(inv.get(ANIMAL_TARGET, 0) > 0 for inv in inventories)
    animals_owned = sum(1 for s in structures if "animal" in s[2])
    if (
        animal_target_count > animals_owned
        and not animal_pending
        and enough_season_left_to_start
        and money_left - CASH_RESERVE >= animal_cost
    ):
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

    # Hands are only ever *appended* within a day (never reordered or
    # removed until the next day's reset -- see _do_hire), so the first
    # `rancher_count` hire-order slots are a STABLE identity for the whole
    # day. Picking "the last N hands" instead would reassign which
    # physical hand is "the rancher" every time a new hire lands later
    # the same day, orphaning mid-errand state (an animal or wheat it was
    # carrying) -- that was a real v4 bug caught in testing. Each rancher
    # slot i (1-indexed here) maps to rank i-1, giving it sole ownership
    # of the structure at that rank -- see _rancher_action's docstring.
    rancher_indices = set(range(1, rancher_count + 1))

    # Phase 1: immediate on-tile actions (harvest/water/dig/plant-in-place)
    # for the farmer and every non-rancher hand, plus rancher/wheat-patch
    # actions for ranchers (unchanged from v7 -- ranchers have few enough
    # units and different-enough target types that the crop-side logic
    # below doesn't apply to them).
    #
    # NOTE ON MOVEMENT COORDINATION: a real lost ladder game showed
    # roughly half of all unit-actions were pure movement, motivating a
    # global nearest-unit-to-nearest-target assignment (_assign_movement_
    # targets, still defined above) instead of each unit independently
    # picking its own nearest target. That version was BUILT and
    # BENCHMARKED here, and caused a severe regression -- final money
    # crashed to ~$200-1,500 (from v7's ~$25-29K) on the same seeds. The
    # cause wasn't isolated before time ran out on this pass, so rather
    # than ship a fix with a known-serious-but-not-understood bug, this
    # reverts to each unit independently calling _unit_action (the
    # pre-v8 approach) -- correct and tested, just not yet addressing the
    # movement-efficiency finding. _assign_movement_targets is left in
    # the file, unused, as a documented starting point for whoever
    # revisits this rather than deleting the (very real) diagnostic work
    # that motivated it.
    crop_units = []
    hands_actions = [None] * len(farm["hands"])

    farmer_crop = crop_for_unit(0)
    farmer_have_seed = bool(farmer_crop) and seeds.get(farmer_crop, 0) > 0
    farmer_immediate = _unit_immediate_action(farm["farmer"], farm, day, farmer_crop, farmer_have_seed)
    farmer_action = farmer_immediate
    if farmer_immediate is None:
        crop_units.append({"idx": "farmer", "pos": tuple(farm["farmer"]), "have_seed": farmer_have_seed, "crop": farmer_crop})

    for i, hand_pos in enumerate(farm["hands"], start=1):
        if i in rancher_indices:
            unit_inv = inventories[i] if i < len(inventories) else {}
            action = _rancher_action(hand_pos, unit_inv, shed, farm, board_size, i - 1)
            if action == ["PASS"]:
                # Nothing ranch-related to do this turn. The rancher tends
                # its OWN small wheat patch when idle, rather than joining
                # the shared crop rotation -- v4 testing found the shared
                # rotation only gave wheat ~1-in-6 unit-turns, and under
                # board-wide maintenance load that slot could get pulled
                # into tending OTHER crops indefinitely and never reach an
                # empty tile to plant its own, starving the animal.
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
            crop_units.append({"idx": i - 1, "pos": tuple(hand_pos), "have_seed": hand_have_seed, "crop": hand_crop})

    # Phase 2 (reverted to independent per-unit decisions -- see note above).
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