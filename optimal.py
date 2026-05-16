import pulp
from hypergraph import Hypergraph

def optimal_solu(hg, budget, time_limit=None, msg=False):
    result = optimal_solu_detail(hg, budget, time_limit=time_limit, msg=msg)
    return result["objective"]


def optimal_solu_detail(hg, budget, time_limit=None, msg=False):
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

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    status_code = prob.solve(solver)

    objective = pulp.value(prob.objective)
    objective = round(objective, 6) if objective is not None else None

    return {
        "objective": objective,
        "status": pulp.LpStatus.get(status_code, str(status_code)),
    }
