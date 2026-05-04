def greedy(hg, covered_vertices, removed_edges):

    while len(covered_vertices) < hg.nvtxs:
        best_score = 0
        best_edge = None

        for hedge in hg.hedges:
            if hedge in removed_edges:
                continue
            score = len(hg.hedges_dict[hedge] - covered_vertices)
            if score > best_score:
                best_score = score
                best_edge = hedge

        if best_edge is None or best_score == 0:
            break

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)


    return len(removed_edges), len(covered_vertices)

def pure_greedy_mcp(hg, budget, filename = None, **kwargs):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    while len(solu) < budget:
        best_score = 0
        best_edge = None

        for hedge in hg.hedges:
            if hedge in removed_edges:
                continue
            score = sum(hg.vtx_weights[v] for v in hg.hedges_dict[hedge] - covered_vertices)
            if score > best_score:
                best_score = score
                best_edge = hedge

        if best_edge is None or best_score == 0:
            break

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        solu.append(best_edge)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges

def pure_greedy_set_cover(hg, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    while len(covered_vertices) < hg.nvtxs:
        best_score = 0
        best_edge = None

        for hedge in hg.hedges:
            if hedge in removed_edges:
                continue
            score = sum(hg.vtx_weights[v] for v in hg.hedges_dict[hedge] - covered_vertices)
            if score > best_score:
                best_score = score
                best_edge = hedge

        if best_edge is None or best_score == 0:
            break  # remaining vertices uncoverable

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        solu.append(best_edge)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges