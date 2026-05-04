import numpy as np
import random
from scipy.stats import powerlaw

class Hypergraph:
    def __init__(self, nhedges, nvtxs):
        self.nhedges = nhedges
        self.nvtxs = nvtxs

        self.hedges = list(range(1,nhedges+1))
        self.vtxs = list(range(1,nvtxs+1))

        self.vtx_weights = {v: random.uniform(1, 1) for v in self.vtxs} # adjust later

    def generate(self, distribution='exponential', **kwargs):
        if distribution == 'exponential':
            scale = kwargs.get('scale', 20)
            hedge_size = np.ceil(np.random.exponential(scale=scale, size=self.nhedges)).astype(int)

        elif distribution == 'uniform':
            low = kwargs.get('low', 10)
            high = kwargs.get('high', 500)
            hedge_size = np.random.randint(low, high+1, size=self.nhedges)

        elif distribution == 'gamma':
            hedge_size = np.random.gamma(4, 40, size=self.nhedges).astype(int)



        elif distribution == 'distance':
            rmax_small = kwargs.get('rmax_small', 0.03)
            rmax_medium = kwargs.get('rmax_medium', 0.07)
            rmax_large = kwargs.get('rmax_large', 0.10)

            # proportion of each size - many small, some medium, few large
            prop_small = kwargs.get('prop_small', 0.68)
            prop_medium = kwargs.get('prop_medium', 0.22)
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

        elif distribution == 'distance2':
            rmax = kwargs.get('rmax', 0.1)

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
            f.write(f"{self.nhedges} {self.nvtxs}\n")

            for hedge in self.hedges_dict:
                vtxs = self.hedges_dict[hedge]
                line = " ".join(str(n) for n in vtxs)
                f.write(line + "\n")







