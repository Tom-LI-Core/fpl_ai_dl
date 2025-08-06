"""
Simple integer-programming squad selector.
"""
import pulp, pandas as pd
from fpl_ai.config import BUDGET, POSITIONS, SQUAD_LIMIT, TEAM_LIMIT

def build_team(xp: pd.Series, meta: pd.DataFrame):
    players = xp.index
    prob = pulp.LpProblem("FPL", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("pick", players, 0, 1, cat="Binary")
    prob += pulp.lpSum(xp[p]*pick[p] for p in players)  # objective

    # constraints
    prob += pulp.lpSum(meta.loc[p,"now_cost"]*pick[p] for p in players) <= BUDGET*10
    prob += pulp.lpSum(pick[p] for p in players) == 15
    for pos,name in POSITIONS.items():
        prob += pulp.lpSum(pick[p] for p in players if meta.loc[p,"element_type"]==pos) == SQUAD_LIMIT[name]
    for t in meta.team.unique():
        prob += pulp.lpSum(pick[p] for p in players if meta.loc[p,"team"]==t) <= TEAM_LIMIT
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return [p for p in players if pick[p].value()==1]
