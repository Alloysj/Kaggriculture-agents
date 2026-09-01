
import statistics

# ---------------------------------------------------------------------------
# PLACEHOLDER economics — replace with real figures from the crop-payout
# notebook / episode_features.csv before relying on this for ranking crops.
# profit_per_tile is a rough (revenue - cost) per tile per cycle; higher is better.
# ---------------------------------------------------------------------------
CROP_ECONOMICS = {
    # "crop_name": {"cost": ..., "revenue": ..., "grow_turns": ..., "profit_per_tile": ...},
    "wheat": {"cost": 10, "revenue": 25, "grow_turns": 72, "profit_per_tile": 15},
    "corn":  {"cost": 15, "revenue": 40, "grow_turns": 96, "profit_per_tile": 25},
    "soy":   {"cost": 12, "revenue": 30, "grow_turns": 84, "profit_per_tile": 18},
}

CASH_RESERVE = 50          # never spend below this buffer
LAND_EXPANSION_THRESHOLD = 300   # buy land once cash comfortably clears this
HIRE_COST_ASSUMED = 8             # placeholder cost per hire action
PRICE_HISTORY_WINDOW = 20         # turns to average for the sell rule

# rolling price history per product, kept across calls via function attribute
_price_history = {}


def _best_affordable_crop(cash, land_available):
    """Pick the highest profit-per-tile crop we can currently afford."""
    affordable = [
        (name, info) for name, info in CROP_ECONOMICS.items()
        if info["cost"] <= (cash - CASH_RESERVE) and land_available > 0
    ]
    if not affordable:
        return None
    affordable.sort(key=lambda kv: kv[1]["profit_per_tile"], reverse=True)
    return affordable[0][0]


def _update_price_history(product, price):
    hist = _price_history.setdefault(product, [])
    hist.append(price)
    if len(hist) > PRICE_HISTORY_WINDOW:
        hist.pop(0)


def _should_sell(product, price):
    hist = _price_history.get(product, [])
    if len(hist) < 3:
        return False  # not enough data yet, hold
    avg = statistics.mean(hist)
    return price >= avg


def agent(observation, configuration):
    """
    Main entrypoint. Signature follows the standard kaggle_environments
    pattern: agent(observation, configuration) -> action.

    Replace every obs.get(...) key below with the real field name once you've
    inspected a live observation object.
    """
    obs = observation
    cfg = configuration

    cash = obs.get("money", obs.get("bank", 0))
    land_available = obs.get("free_land", obs.get("empty_tiles", 0))
    crew = obs.get("crew", obs.get("hired_workers", 0))
    planted_tiles = obs.get("planted_tiles", 0)
    market_prices = obs.get("market_prices", {})  # e.g. {"wheat": 12.3, "corn": 18.1}
    inventory = obs.get("inventory", {})           # harvested goods ready to sell

    # 1. Update our price memory for every product we can see.
    for product, price in market_prices.items():
        _update_price_history(product, price)

    # 2. Sell anything in inventory that's priced at/above its recent average.
    sell_orders = {}
    for product, qty in inventory.items():
        if qty > 0 and product in market_prices and _should_sell(product, market_prices[product]):
            sell_orders[product] = qty

    if sell_orders:
        # Replace with the real SELL action shape once known.
        return {"action": "SELL", "orders": sell_orders}

    # 3. Hire more labor if we can afford it and have unplanted land to work.
    if land_available > planted_tiles and cash - CASH_RESERVE >= HIRE_COST_ASSUMED:
        return {"action": "HIRE", "count": 1}

    # 4. Plant the best available crop if we have workable land and cash.
    crop_choice = _best_affordable_crop(cash, land_available)
    if crop_choice:
        return {"action": "PLANT", "crop": crop_choice, "tiles": 1}

    # 5. Expand land once cash is comfortably above the reserve + threshold.
    if cash >= LAND_EXPANSION_THRESHOLD:
        return {"action": "BUY_LAND", "amount": 1}

    # 6. Nothing productive to do this turn — pass / no-op.
    #    Replace with whatever the real "do nothing" action is (often "WAIT"
    #    or an empty dict, depending on the env spec).
    return {"action": "WAIT"}


# ---------------------------------------------------------------------------
# Local smoke test (won't run inside the actual competition harness, but
# lets you sanity check the logic before wiring up kaggle_environments).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fake_obs = {
        "money": 200,
        "free_land": 5,
        "crew": 1,
        "planted_tiles": 2,
        "market_prices": {"wheat": 24, "corn": 39},
        "inventory": {"wheat": 3, "corn": 0},
    }
    fake_cfg = {}
    print(agent(fake_obs, fake_cfg))
