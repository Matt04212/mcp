import numpy as np
import random


def _beta_rvs(a, b, size):
    return np.random.beta(a, b, size=size)


def _scale_radius_for_density(radius, nvtxs, reference_nvtxs):
    return radius * np.sqrt(reference_nvtxs / nvtxs)


def _spatial_distances(elem_x, elem_y, cx, cy, toroidal=False):
    dx = np.abs(elem_x - cx)
    dy = np.abs(elem_y - cy)
    if toroidal:
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.minimum(dy, 1.0 - dy)
    return np.sqrt(dx * dx + dy * dy)


def _assign_isolated_vertices(vtxs, vtxs_dict, hedge_centers_x, hedge_centers_y,
                              hedge_ids, hedge_dict, elem_x, elem_y, toroidal=False):
    for vtx in vtxs_dict:
        if not vtxs_dict[vtx]:
            vx, vy = elem_x[vtx - 1], elem_y[vtx - 1]
            dists = _spatial_distances(
                hedge_centers_x,
                hedge_centers_y,
                vx,
                vy,
                toroidal=toroidal,
            )
            nearest_hedge = hedge_ids[int(np.argmin(dists))]
            hedge_dict[nearest_hedge].add(vtx)
            vtxs_dict[vtx].add(nearest_hedge)


def _assign_empty_hedges(hedges, hedge_dict, vtxs_dict, hedge_centers_x, hedge_centers_y,
                         elem_x, elem_y, toroidal=False):
    for hedge, fx, fy in zip(hedges, hedge_centers_x, hedge_centers_y):
        if hedge_dict[hedge]:
            continue
        dists = _spatial_distances(elem_x, elem_y, fx, fy, toroidal=toroidal)
        nearest_vtx = int(np.argmin(dists)) + 1
        hedge_dict[hedge].add(nearest_vtx)
        vtxs_dict[nearest_vtx].add(hedge)


def _build_spatial_radius_hypergraph(hg, radii, toroidal=False):
    elem_x = np.random.uniform(0, 1, hg.nvtxs)
    elem_y = np.random.uniform(0, 1, hg.nvtxs)
    center_x = np.random.uniform(0, 1, hg.nhedges)
    center_y = np.random.uniform(0, 1, hg.nhedges)

    hg.hedges_dict = {}
    hg.vtxs_dict = {vtx: set() for vtx in hg.vtxs}

    for hedge, fx, fy, radius in zip(hg.hedges, center_x, center_y, radii):
        dists = _spatial_distances(elem_x, elem_y, fx, fy, toroidal=toroidal)
        covered = set(np.where(dists <= radius)[0] + 1)
        hg.hedges_dict[hedge] = covered
        for vtx in covered:
            hg.vtxs_dict[vtx].add(hedge)

    _assign_isolated_vertices(
        hg.vtxs,
        hg.vtxs_dict,
        center_x,
        center_y,
        hg.hedges,
        hg.hedges_dict,
        elem_x,
        elem_y,
        toroidal=toroidal,
    )
    _assign_empty_hedges(
        hg.hedges,
        hg.hedges_dict,
        hg.vtxs_dict,
        center_x,
        center_y,
        elem_x,
        elem_y,
        toroidal=toroidal,
    )


def _build_spatial_knn_hypergraph(hg, hedge_sizes, toroidal=False):
    elem_x = np.random.uniform(0, 1, hg.nvtxs)
    elem_y = np.random.uniform(0, 1, hg.nvtxs)
    center_x = np.random.uniform(0, 1, hg.nhedges)
    center_y = np.random.uniform(0, 1, hg.nhedges)

    hg.hedges_dict = {}
    hg.vtxs_dict = {vtx: set() for vtx in hg.vtxs}

    for hedge, cx, cy, hedge_size in zip(hg.hedges, center_x, center_y, hedge_sizes):
        hedge_size = max(1, min(int(hedge_size), hg.nvtxs))
        dists = _spatial_distances(elem_x, elem_y, cx, cy, toroidal=toroidal)
        nearest_idx = np.argpartition(dists, hedge_size - 1)[:hedge_size]
        covered = set((nearest_idx + 1).tolist())
        hg.hedges_dict[hedge] = covered
        for vtx in covered:
            hg.vtxs_dict[vtx].add(hedge)

    _assign_isolated_vertices(
        hg.vtxs,
        hg.vtxs_dict,
        center_x,
        center_y,
        hg.hedges,
        hg.hedges_dict,
        elem_x,
        elem_y,
        toroidal=toroidal,
    )

class Hypergraph:
    def __init__(self, nhedges, nvtxs):
        self.nhedges = nhedges
        self.nvtxs = nvtxs

        self.hedges = list(range(1,nhedges+1))
        self.vtxs = list(range(1,nvtxs+1))

        self.vtx_weights = {v: random.uniform(1, 1) for v in self.vtxs} # adjust later

    def generate(self, distribution='gamma', **kwargs):
        # left-skewed (FrontLoad) — most edges small
        if distribution == 'beta_right':
            max_size = kwargs.get('max_size', 25)
            raw = _beta_rvs(2, 5, size=self.nhedges)
            hedge_size = np.clip((raw * max_size).astype(int), 1, self.nvtxs)
            # mean = 2/10 * 500 = 100

        # bell-shaped (symmetric) — replaces uniform
        elif distribution == 'beta_bell':
            max_size = kwargs.get('max_size', 25)
            raw = _beta_rvs(5, 5, size=self.nhedges)
            hedge_size = np.clip((raw * max_size).astype(int), 1, self.nvtxs)
            # mean = 5/10 * 500 = 250

        # right-skewed (BackLoad) — most edges large
        elif distribution == 'beta_left':
            max_size = kwargs.get('max_size', 25)
            raw = _beta_rvs(5, 2, size=self.nhedges)
            hedge_size = np.clip((raw * max_size).astype(int), 1, self.nvtxs)
            # mean = 8/10 * 500 = 400

        elif distribution == 'uniform':
            low = kwargs.get('low', 1)
            high = kwargs.get('high', 23)
            hedge_size = np.random.randint(low, high+1, size=self.nhedges)

        elif distribution == 'dis':
            rmax_small = kwargs.get('rmax_small', 0.03)
            rmax_medium = kwargs.get('rmax_medium', 0.05)
            rmax_large = kwargs.get('rmax_large', 0.08)

            # proportion of each size - many small, some medium, few large
            prop_small = kwargs.get('prop_small', 0.60)
            prop_medium = kwargs.get('prop_medium', 0.30)
            prop_large = kwargs.get('prop_large', 0.10)

            # generate coordinates for all vertices
            elem_x = np.random.uniform(0, 1, self.nvtxs)
            elem_y = np.random.uniform(0, 1, self.nvtxs)

            # place each facility at a random vertex location
            facility_idx = np.random.choice(self.nvtxs, self.nhedges, replace=False)

            # assign rmax to each facility based on proportions
            n_small = int(self.nhedges * prop_small)
            n_medium = int(self.nhedges * prop_medium)
            n_large = self.nhedges - n_small - n_medium

            rmax_list = (
                    [rmax_small] * n_small +
                    [rmax_medium] * n_medium +
                    [rmax_large] * n_large
            )

            random.shuffle(rmax_list)  # shuffle so large ones aren't all at end

            # reset dicts
            self.hedges_dict = {}
            self.vtxs_dict = {vtx: set() for vtx in self.vtxs}

            for hedge, fidx, rmax in zip(self.hedges, facility_idx, rmax_list):
                fx, fy = elem_x[fidx], elem_y[fidx]
                dists = np.sqrt((elem_x - fx) ** 2 + (elem_y - fy) ** 2)
                covered = set(np.where(dists <= rmax)[0] + 1)
                self.hedges_dict[hedge] = covered
                for vtx in covered:
                    self.vtxs_dict[vtx].add(hedge)

            # handle isolated vertices - assign to nearest facility
            for vtx in self.vtxs_dict:
                if not self.vtxs_dict[vtx]:
                    vx, vy = elem_x[vtx - 1], elem_y[vtx - 1]
                    dists = np.sqrt((elem_x[facility_idx] - vx) ** 2 +
                                    (elem_y[facility_idx] - vy) ** 2)
                    nearest_hedge = self.hedges[np.argmin(dists)]
                    self.hedges_dict[nearest_hedge].add(vtx)
                    self.vtxs_dict[vtx].add(nearest_hedge)

            return

        elif distribution == 'spatial_bell':
            rmax = kwargs.get('rmax', 0.019)
            auto_scale = kwargs.get('auto_scale', True)
            reference_nvtxs = kwargs.get('reference_nvtxs', 10000)
            if auto_scale:
                rmax = _scale_radius_for_density(rmax, self.nvtxs, reference_nvtxs)

            # generate coordinates for all vertices
            elem_x = np.random.uniform(0, 1, self.nvtxs)
            elem_y = np.random.uniform(0, 1, self.nvtxs)

            # place each facility at a random vertex location
            facility_idx = np.random.choice(self.nvtxs, self.nhedges, replace=False)

            # reset dicts
            self.hedges_dict = {}
            self.vtxs_dict = {vtx: set() for vtx in self.vtxs}
            for hedge, fidx in zip(self.hedges, facility_idx):
                fx, fy = elem_x[fidx], elem_y[fidx]

                # vectorized distance computation
                dists = np.sqrt((elem_x - fx) ** 2 + (elem_y - fy) ** 2)
                covered = set(np.where(dists <= rmax)[0] + 1)  # +1 for 1-indexed
                self.hedges_dict[hedge] = covered
                for vtx in covered:
                    self.vtxs_dict[vtx].add(hedge)

            _assign_isolated_vertices(
                self.vtxs,
                self.vtxs_dict,
                elem_x[facility_idx],
                elem_y[facility_idx],
                self.hedges,
                self.hedges_dict,
                elem_x,
                elem_y,
            )

            return

        elif distribution == 'spatial_right_skewed':
            max_size = kwargs.get('max_size', 25)
            beta_a = kwargs.get('beta_a', 2)
            beta_b = kwargs.get('beta_b', 5)
            toroidal = kwargs.get('toroidal', True)
            raw = _beta_rvs(beta_a, beta_b, size=self.nhedges)
            hedge_sizes = np.clip((raw * max_size).astype(int), 1, self.nvtxs)
            _build_spatial_knn_hypergraph(self, hedge_sizes, toroidal=toroidal)
            return

        elif distribution == 'spatial_left_skewed':
            max_size = kwargs.get('max_size', 25)
            beta_a = kwargs.get('beta_a', 5)
            beta_b = kwargs.get('beta_b', 2)
            toroidal = kwargs.get('toroidal', True)
            raw = _beta_rvs(beta_a, beta_b, size=self.nhedges)
            hedge_sizes = np.clip((raw * max_size).astype(int), 1, self.nvtxs)
            _build_spatial_knn_hypergraph(self, hedge_sizes, toroidal=toroidal)
            return

        elif distribution == 'spatial_uniform':
            low = kwargs.get('low', 1)
            high = kwargs.get('high', 23)
            toroidal = kwargs.get('toroidal', True)
            hedge_sizes = np.random.randint(low, high + 1, size=self.nhedges)
            _build_spatial_knn_hypergraph(self, hedge_sizes, toroidal=toroidal)
            return

        # clip to valid range
        hedge_size = np.clip(hedge_size, 1, self.nvtxs)

        # reset dicts
        self.hedges_dict = {}
        self.vtxs_dict = {vtx: set() for vtx in self.vtxs}

        # assign vertices to edges
        for hedge, size in zip(self.hedges, hedge_size):
            c = set(random.sample(self.vtxs, size))
            self.hedges_dict[hedge] = c
            for vtx in c:
                self.vtxs_dict[vtx].add(hedge)

        for vtx in self.vtxs_dict:
            if not self.vtxs_dict[vtx]:
                r = random.choice(self.hedges)
                self.hedges_dict[r].add(vtx)
                self.vtxs_dict[vtx].add(r)


    def output(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{self.nvtxs} {self.nhedges}\n")

            for vtxs in self.vtxs_dict:
                hedges = self.vtxs_dict[vtxs]
                line = " ".join(str(n) for n in hedges)
                f.write(line + "\n")
