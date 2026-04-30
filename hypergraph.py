import numpy as np
import random
from scipy.stats import powerlaw

class Hypergraph:
    def __init__(self, nhedges, nvtxs):
        self.nhedges = nhedges
        self.nvtxs = nvtxs

        self.hedges = list(range(1,nhedges+1))
        self.vtxs = list(range(1,nvtxs+1))

    def generate(self, distribution='exponential', **kwargs):
        if distribution == 'exponential':
            scale = kwargs.get('scale', 40)
            hedge_size = np.ceil(np.random.exponential(scale=scale, size=self.nhedges)).astype(int)

        elif distribution == 'uniform':
            low = kwargs.get('low', 10)
            high = kwargs.get('high', 500)
            hedge_size = np.random.randint(low, high+1, size=self.nhedges)

        elif distribution == 'gamma':
            hedge_size = np.random.gamma(5, 70, size=self.nhedges).astype(int)


        elif distribution == 'spatial':
            rmax = kwargs.get('rmax', 0.02)

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

            return  # skip the rest of generate()

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







