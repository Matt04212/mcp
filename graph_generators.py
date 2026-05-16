import random

import numpy as np

from hypergraph import Hypergraph


def edge_size_meta(hg):
    sizes = [len(vertices) for vertices in hg.hedges_dict.values()]
    weights = list(hg.vtx_weights.values())
    return {
        "min_edge_size": min(sizes),
        "mean_edge_size": round(sum(sizes) / len(sizes), 2),
        "max_edge_size": max(sizes),
        "min_vertex_weight": min(weights),
        "mean_vertex_weight": round(sum(weights) / len(weights), 2),
        "max_vertex_weight": max(weights),
    }


def _set_vertex_weights(hg, seed, weight_mode):
    rng = np.random.default_rng(seed)
    if weight_mode in ("uniform", "unweighted", None):
        hg.vtx_weights = {v: 1.0 for v in hg.vtxs}
    elif weight_mode in ("paper_1_10", "weighted_1_10", "weighted"):
        hg.vtx_weights = {v: float(rng.uniform(1, 10)) for v in hg.vtxs}
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")


def _auto_dis2_rmax(nvtxs):
    # Target edge size grows sublinearly with |U| so larger spatial instances
    # are not just dense versions of smaller ones.
    target_edge_size = min(14, max(7, int(0.12 * np.sqrt(nvtxs))))
    return float(np.sqrt(target_edge_size / (np.pi * nvtxs)))


def build_dis2_graph(nhedges, nvtxs, seed, rmax=0.035, weight_mode="uniform"):
    """
    Geometric random graph matching Hypergraph.generate(distribution='dis2').

    Vertices are random points in the unit square. Each hedge is centered at a
    random vertex and covers all vertices within radius rmax. This is the
    paper-like spatial/facility graph family.
    """
    random.seed(seed)
    np.random.seed(seed)
    if rmax == "auto":
        rmax = _auto_dis2_rmax(nvtxs)
    hg = Hypergraph(nhedges, nvtxs)
    hg.generate(distribution="dis2", rmax=rmax)
    _set_vertex_weights(hg, seed + 17, weight_mode)
    return hg, {
        "graph_family": "dis2",
        "weight_mode": weight_mode,
        "rmax": rmax,
        **edge_size_meta(hg),
    }


def _partition_vertices(nvtxs, n_communities, subcommunities_per_community):
    vertices = list(range(1, nvtxs + 1))
    communities = []
    subcommunities = []
    cursor = 0

    for community_idx in range(n_communities):
        remaining_communities = n_communities - community_idx
        community_size = (nvtxs - cursor) // remaining_communities
        community = vertices[cursor:cursor + community_size]
        cursor += community_size
        communities.append(community)

        subs = []
        sub_cursor = 0
        for sub_idx in range(subcommunities_per_community):
            remaining_subs = subcommunities_per_community - sub_idx
            sub_size = (len(community) - sub_cursor) // remaining_subs
            subs.append(community[sub_cursor:sub_cursor + sub_size])
            sub_cursor += sub_size
        subcommunities.append(subs)

    return communities, subcommunities


def _sample(rng, pool, count):
    if count <= 0 or not pool:
        return set()
    return set(rng.choice(pool, size=min(count, len(pool)), replace=False))


def _cap_edge(rng, edge, max_edge_size):
    if len(edge) <= max_edge_size:
        return edge
    return _sample(rng, list(edge), max_edge_size)


def _default_hierarchy_shape(nvtxs):
    if nvtxs <= 1200:
        return 6, 4
    if nvtxs <= 2400:
        return 8, 5
    if nvtxs <= 4800:
        return 12, 5
    return max(12, int(round(nvtxs / 500))), 6


def build_natural_hierarchy_graph(
    nhedges,
    nvtxs,
    seed,
    n_communities=None,
    subcommunities_per_community=None,
    max_edge_ratio=0.0,
    max_edge_cap=14,
):
    """
    Stochastic hierarchical community graph.

    Hidden communities/subcommunities are only sampling pools. The generated
    hedges are actual selectable coverage sets.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    if n_communities is None or subcommunities_per_community is None:
        default_communities, default_subs = _default_hierarchy_shape(nvtxs)
        n_communities = n_communities or default_communities
        subcommunities_per_community = subcommunities_per_community or default_subs

    communities, subcommunities = _partition_vertices(
        nvtxs, n_communities, subcommunities_per_community
    )

    community_popular = []
    community_tail = []
    for subs in subcommunities:
        popular = []
        tail = []
        for sub in subs:
            split = max(1, int(0.35 * len(sub)))
            popular.extend(sub[:split])
            tail.extend(sub[split:])
        community_popular.append(popular)
        community_tail.append(tail)

    # Keep hierarchy edges useful but avoid saturating the whole universe when
    # F=0.8|U|. The cap scales with |U|, but the lower bound stays modest so
    # small/medium instances do not become automatically full-covered.
    scaled_edge_cap = max(7, int(0.12 * np.sqrt(nvtxs)))
    if max_edge_ratio > 0:
        scaled_edge_cap = min(scaled_edge_cap, int(max_edge_ratio * nvtxs))
    max_edge_size = min(max_edge_cap, scaled_edge_cap)
    n_global = int(0.14 * nhedges)
    n_community = int(0.28 * nhedges)
    n_local = int(0.44 * nhedges)
    n_bridge = nhedges - n_global - n_community - n_local

    hedges = []

    # Global generalists: wide, shallow, biased toward popular vertices.
    for _ in range(n_global):
        edge = set()
        active_count = rng.integers(
            max(2, int(0.65 * n_communities)),
            n_communities + 1,
        )
        active = rng.choice(n_communities, size=active_count, replace=False)
        for community_idx in active:
            pop_pool = community_popular[community_idx]
            tail_pool = community_tail[community_idx]
            edge.update(_sample(rng, pop_pool, max(1, int(0.045 * len(pop_pool)))))
            edge.update(_sample(rng, tail_pool, max(1, int(0.006 * len(tail_pool)))))
        hedges.append(_cap_edge(rng, edge, max_edge_size))

    # Community edges: medium coverage within one community.
    for _ in range(n_community):
        community_idx = int(rng.integers(0, n_communities))
        subs = subcommunities[community_idx]
        edge = set()
        active_subs = rng.choice(
            len(subs),
            size=int(rng.integers(2, len(subs) + 1)),
            replace=False,
        )
        for sub_idx in active_subs:
            sub = subs[sub_idx]
            low = max(2, int(0.10 * len(sub)))
            high = max(3, int(0.22 * len(sub)))
            edge.update(_sample(rng, sub, int(rng.integers(low, high))))
        edge.update(
            _sample(
                rng,
                community_popular[community_idx],
                max(1, int(0.018 * len(community_popular[community_idx]))),
            )
        )
        hedges.append(_cap_edge(rng, edge, max_edge_size))

    # Local specialists: deep coverage within one subcommunity.
    for _ in range(n_local):
        community_idx = int(rng.integers(0, n_communities))
        sub_idx = int(rng.integers(0, subcommunities_per_community))
        sub = subcommunities[community_idx][sub_idx]
        local_ratio = rng.uniform(0.25, 0.48)
        edge = _sample(rng, sub, max(2, int(local_ratio * len(sub))))

        if rng.random() < 0.25:
            neighbor_idx = min(
                subcommunities_per_community - 1,
                max(0, sub_idx + rng.choice([-1, 1])),
            )
            neighbor = subcommunities[community_idx][neighbor_idx]
            edge.update(_sample(rng, neighbor, max(1, int(0.03 * len(neighbor)))))

        hedges.append(_cap_edge(rng, edge, max_edge_size))

    # Bridge/noise edges: small cross-community coverage.
    for _ in range(n_bridge):
        edge = set()
        first = int(rng.integers(0, n_communities))
        second = int(rng.integers(0, n_communities))
        while second == first:
            second = int(rng.integers(0, n_communities))

        for community_idx in (first, second):
            sub = subcommunities[community_idx][
                int(rng.integers(0, subcommunities_per_community))
            ]
            high = max(3, int(0.07 * len(sub)))
            edge.update(_sample(rng, sub, max(2, int(rng.integers(2, high)))))
        hedges.append(_cap_edge(rng, edge, max_edge_size))

    hedges = [set(edge) for edge in hedges[:nhedges] if edge]
    while len(hedges) < nhedges:
        hedges.append({int(rng.integers(1, nvtxs + 1))})

    hg = Hypergraph(nhedges, nvtxs)
    hg.hedges = list(range(1, nhedges + 1))
    hg.vtxs = list(range(1, nvtxs + 1))
    hg.nhedges = nhedges
    hg.nvtxs = nvtxs
    hg.vtx_weights = {v: 1.0 for v in hg.vtxs}
    hg.hedges_dict = {edge: hedges[edge - 1] for edge in hg.hedges}
    hg.vtxs_dict = {v: set() for v in hg.vtxs}
    for edge, vertices in hg.hedges_dict.items():
        for vertex in vertices:
            hg.vtxs_dict[vertex].add(edge)

    for vertex, incident in hg.vtxs_dict.items():
        if not incident:
            edge = int(rng.integers(1, nhedges + 1))
            hg.hedges_dict[edge].add(vertex)
            incident.add(edge)

    return hg, {
        "graph_family": "natural_hierarchy",
        "weight_mode": "uniform",
        "n_communities": n_communities,
        "subcommunities_per_community": subcommunities_per_community,
        "n_global_edges": n_global,
        "n_community_edges": n_community,
        "n_local_edges": n_local,
        "n_bridge_edges": n_bridge,
        **edge_size_meta(hg),
    }
