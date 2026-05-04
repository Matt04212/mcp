import pulp
from hypergraph import Hypergraph

def optimal_solu(hg, budget):
    prob = pulp.LpProblem('MCP', pulp.LpMaximize)

    # variables
    x = {h: pulp.LpVariable(f"x_{h}", cat='Binary') for h in hg.hedges}
    y = {v: pulp.LpVariable(f"y_{v}", cat='Binary') for v in hg.vtxs}

    # objective -- maximize covered vertices
    prob += pulp.lpSum(hg.vtx_weights[v] * y[v] for v in hg.vtxs)

    # budget constraint
    prob += pulp.lpSum(x[h] for h in hg.hedges) <= budget

    # coverage constraint -- vertex covered only if at least one edge covers it
    for v in hg.vtxs:
        prob += y[v] <= pulp.lpSum(x[h] for h in hg.vtxs_dict[v])

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    optimal_coverage = round(pulp.value(prob.objective), 4)
    return optimal_coverage