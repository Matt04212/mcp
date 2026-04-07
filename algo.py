"""import matplotlib.pyplot as plt

filename = "data1.hgr"
hedge_size = []

with open("data1.hgr", "r") as f:
    lines = f.readlines()

for line in lines[1:]:
    vtxs = line.strip().split()
    hedge_size.append(len(vtxs))

plt.hist(hedge_size, bins=100)
plt.show()"""
import subprocess

def algo(hg):
    subprocess.run("./hmetis data1.hgr 2 5 10 2 1 0 0 0", shell=True)

    with open("data1.hgr.part.2", "r") as f:
        line = f.read().splitlines()

    s0 = set()
    s1 = set()
    s2 = set()
    s3 = set()
    for vtx, part in enumerate(line, start=1):
        if part == "0":
            hedge = [h for h, v in hg.hedges_dict.items() if vtx in v]
            s0.update(hedge)
        elif part == "1":
            hedge = [h for h, v in hg.hedges_dict.items() if vtx in v]
            s1.update(hedge)
        elif part == "2":
            hedge = [h for h, v in hg.hedges_dict.items() if vtx in v]
            s2.update(hedge)
        else:
            hedge = [h for h, v in hg.hedges_dict.items() if vtx in v]
            s3.update(hedge)

    print(len(s0))
    print(len(s1))
    print(len(s2))
    print(len(s3))






