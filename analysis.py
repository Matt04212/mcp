import numpy as np
import matplotlib.pyplot as plt

def analyze_graph(hg, dist):
    sizes = [len(hg.hedges_dict[h]) for h in hg.hedges_dict]
    degrees = [len(hg.vtxs_dict[v]) for v in hg.vtxs_dict]

    stats = {
        'distribution': dist,

        #edge stat
        'mean_edge_size': float('%.3f'% np.mean(sizes)),
        'std_edge_size': float('%.3f'% np.std(sizes)),
        'max_edge_size': int(np.max(sizes)),
        'min_edge_size': int(np.min(sizes)),
        'median_edge_size': float('%.3f'% np.median(sizes)),

        #coverage potential
        'total_pins': int(sum(sizes)),
        'avg_coverage_per_edge': float('%.3f'% (sum(sizes) / hg.nhedges)),
        'overlap_ratio': float((sum(sizes) - hg.nvtxs) / sum(sizes))
    }

    return stats, sizes, degrees


def plot_graph(hg, dist):
    stats, sizes, degrees = analyze_graph(hg, dist)

    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f'Distribution - {dist}')

    # 1. edge size distribution
    plt.hist(sizes, bins="auto")
    plt.title('Edge Size Distribution')
    plt.xlabel('Edge Size')
    plt.ylabel('Count')

    plt.savefig(f'graph_analysis_{dist}.png')
    plt.show()

    return stats

