import numpy as np
import random

class hypergraph:
    def __init__(self, nhedges, nvtxs):
        self.nhedges = nhedges
        self.nvtxs = nvtxs

        self.hedges = list(range(1,nhedges+1))
        self.vtxs = list(range(1, nvtxs+1))

    def distribution(self):
        hedge_size = np.ceil(np.random.exponential(scale=20.0, size=self.nhedges)).astype(int)
        self.hedge_size = np.clip(hedge_size, 1, self.nvtxs)
        self.hedges_dict = {}
        self.vtxs_dict = {vtx: set() for vtx in self.vtxs}
        for hedge, size in zip(self.hedges, self.hedge_size):
            c = set(random.sample(self.vtxs, size))
            self.hedges_dict[hedge] = c
            for vtx in c:
                self.vtxs_dict[vtx].add(hedge)

        #insure not isolated nodes
        for vtx in self.vtxs_dict:
            if not self.vtxs_dict[vtx]:
                r = random.choice(self.hedges)
                self.hedges_dict[r].add(vtx)
                self.vtxs_dict[vtx].add(r)






