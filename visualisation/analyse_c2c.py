import copy

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cdist
from classes import Run, DSU
from utils.general_utils import get_dsus
from utils.loader_utils import load_topology_runs, load_events
from utils.plt_utils import add_colorbar

np.set_printoptions(edgeitems=30, linewidth=100000)

matplotlib.rc('font', **{
    'family': 'sans',
    'size': 24})

basepath = Path(__file__).parent


def estimate_c2c(cores: [DSU], num_cores: int) -> np.array:
    core_map = np.zeros((num_cores,2))

    for core in cores:
        core_map[core.coreids[0]] = (core.x, core.y)
        core_map[core.coreids[1]] = (core.x, core.y)

    c2c = np.zeros((num_cores,num_cores))
    for i in range(num_cores):
        for j in range(num_cores):
            if i == j: continue
            pos_i = core_map[i]
            pos_j = core_map[j]
            distance = cityblock(pos_i, pos_j)
            c2c[i][j] = distance

    for i in range(num_cores):
        c2c[i][i] = np.nan
    return c2c

def plot_c2c(c2c: np.array, save: bool = False,
             norm=None, cmap="viridis",
             ticker_locator=matplotlib.ticker.MultipleLocator(4), zoom: (int, int) = None):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams['axes.titley'] = 1.1
    im = ax.imshow(c2c, norm=norm, cmap=cmap)
    cbar = add_colorbar(im)
    cbar.set_label("Core-to-Core Latency [ns]")
    ax.tick_params(axis='x', labelrotation=90, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    if ticker_locator:
        ax.xaxis.set_major_locator(ticker_locator)
        ax.yaxis.set_major_locator(ticker_locator)

    ax.set_xlabel("Core ID")
    ax.set_ylabel("Core ID")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if zoom:
        ax.set_xlim(zoom[0] - .5, zoom[1] + .5)
        ax.set_ylim(zoom[1] + .5, zoom[0] - .5)
    fig.tight_layout()
    if save:
        ...
        #plt.savefig(basepath / "figures" / f"c2c.pdf")
    else:
        plt.show()

def main():
    runs = {}

    for f in list(filter(lambda x: x.suffix == ".csv", (basepath / "data" / "c2c").iterdir())):
        df = pd.read_csv(f, header=None)
        df = df.combine_first(df.T)
        name = f.stem.replace(".csv", "")

        r = Run(name=name, df=df)
        runs[r.name] = r

    save = False

    # basepath_measurements = Path("/mnt/opencube/code/cmn_topology_tool/data/")
    # run_name = "determine_topology/cn02_2024-01-02T1707"
    # mesh_size = (8, 6)
    # num_cores = 80
    # c2c_df = runs["cn03c1_monolithic"]

    basepath_measurements = Path("/mnt/aam/code/cmn_topology_tool/data/")
    run_name = "determine_topology/i10se12_2024-01-02T1418"
    mesh_size = (8, 8)
    num_cores = 128
    c2c_df = runs["aam1c1_monolithic"]


    plot_c2c(c2c_df.df.values,
             save=save)
    return

    events = load_events(basepath_measurements / run_name / "events.csv")
    #layout_df = load_static_layout(basepath_measurements / run_name, events=events)
    runs = load_topology_runs(run_name, events, basepath_measurements)
    cores = get_dsus(runs=runs, mesh_size=mesh_size)
    estimated_c2c = estimate_c2c(cores, num_cores)

    cores_array = np.array([[c.x, c.y] for c in cores])
    core_dist_mat = cdist(cores_array, cores_array, "cityblock")
    plot_c2c(c2c_df.df)

    max_hops = np.max(core_dist_mat)
    min_hops = np.min(core_dist_mat)

    core_pairs = {i:[] for i in range(int(min_hops), int(max_hops)+1)}

    for i in range(core_dist_mat.shape[0]):
        for j in range(core_dist_mat.shape[1]):
            if i == j: continue
            dist = core_dist_mat[i,j]
            core_i = cores[i]
            core_j = cores[j]
            core_pairs[dist].append(c2c_df.df.loc[core_i.coreids[0], core_j.coreids[0]])

    dist_per_n_hops = { hops: np.mean(vals) for hops, vals in core_pairs.items() }

    dist_per_n_hops_normed = [(hop, dist-dist_per_n_hops[0]) for hop,dist in dist_per_n_hops.items() if hop > 0]

    dist_steps = [(j_dist-i_dist)/(j_hop-i_hop) for ((i_hop, i_dist),(j_hop, j_dist)) in zip(dist_per_n_hops_normed[:-1], dist_per_n_hops_normed[1:]) ]
    print(f"Step median: {np.median(dist_steps)}")
    print(f"Step mean  : {np.mean(dist_steps)}")

    norm = matplotlib.colors.Normalize(vmin=c2c_df.df.min().min(), vmax=c2c_df.df.max().max())
    normed_c2c = norm(c2c_df.df.values)
    normed_c2c *= np.nanmax(estimated_c2c)
#    plot_c2c(normed_c2c - estimated_c2c)

    plot_c2c(estimated_c2c)
    estimated_c2c_dist = np.median(dist_steps)*estimated_c2c + dist_per_n_hops[0]
    for i in range((num_cores//2)):
        j = i*2
        estimated_c2c_dist[j+1][j  ] = 27
        estimated_c2c_dist[j  ][j+1] = 27
    plot_c2c(estimated_c2c_dist)
    # plot_c2c(c2c_df.df.values)


if __name__ == "__main__": main()