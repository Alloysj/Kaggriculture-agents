"""
Run this locally, exactly where you got the low results, and paste the
output back. It replays one game turn-by-turn for the first two in-game
days and prints money, crew size, seed stock, and shed contents so we can
see exactly where things diverge from what's expected.

Before running, first confirm your kaggle-environments version — paste
this number back too:
"""
from importlib import metadata
print("kaggle-environments version:", metadata.version("kaggle-environments"))

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

from kaggle_environments import make
from v3_agent import agent

env = make("kaggriculture", configuration={"seed": 0}, debug=True)
env.run([agent, "starter"])

print(f"\nFinal: Player 0={env.steps[-1][0]['reward']}, Player 1={env.steps[-1][1]['reward']}")

print("\n--- turn-by-turn trace, first 48 turns (2 days) ---")
for i, step in enumerate(env.steps[:48]):
    obs = step[0]["observation"]
    farm = obs["farms"][0]
    priv = obs.get("private", {})
    if i % 4 == 0:  # every 4th turn to keep it readable
        print(
            f"turn={i:3d} day={obs['day']} money={farm['money']:8.1f} "
            f"hands={len(farm['hands'])} unlocked={farm['unlocked_quadrants']} "
            f"seeds={dict((k, v) for k, v in priv.get('seeds', {}).items() if v)} "
            f"shed={dict((k, v) for k, v in priv.get('shed', {}).items() if v)}"
        )

# Also check what action our agent actually returns given the very first observation.
first_obs = env.steps[0][0]["observation"]
print("\n--- action on turn 0 ---")
print(agent(first_obs))
