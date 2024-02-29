import functools
import types
import typing
from typing import Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

from benchmarks.benchmark_loaders import RunType, load
from classes import Run

matplotlib.rc('font', **{
    'family' : 'sans',
    'size'   : 24})


def plot(runs: [[Run]], basepath: Path,
         metric: Union[Union[str, typing.Callable], list[Union[str, typing.Callable]]], metric_name: str,
         x_axis_groups: Optional[list[tuple[float,str]]] = None,
         xlabels: Optional[list[str]] = None,
         normalize: Optional[str] = None,
         save: bool = False,
         title: bool = True,
         figsize = (12,8),
         fname: Optional[str] = None):

    if isinstance(runs[0], Run):
        runs = [runs]

    if not isinstance(metric, list):
        metric = [metric]

    N = len(runs)
    fig, axes = plt.subplots(ncols=N, figsize=figsize)
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])

    fig: plt.Figure
    _metric_name = metric_name
    for i, run_set in enumerate(runs):
        ax: plt.Axes = axes.flat[i]
        for m_i, _metric in enumerate(metric):
            if isinstance(_metric, str):
                data = np.array([r.df[_metric] for r in run_set])
            else:
                data = np.array([_metric(r.df) for r in run_set])
            if isinstance(normalize, str):
                match normalize:
                    case "max":
                        vmax = np.max(list(map(np.median, data)))
                        data /= vmax

                    case "min":
                        vmin = np.min(list(map(np.median, data)))
                        data /= vmin

                data -= 1
                data *= 100
                _metric_name = "Slowdown (%)"

            boxplot_kwargs = {}
            if m_i != 0:
                boxplot_kwargs["showfliers"] = False
                boxplot_kwargs["whis"] = False

            if isinstance(xlabels, types.NoneType):
                labels = [r.name.split(" ")[-1] for r in run_set]
                for j, label in enumerate(labels):
                    if len(str(label)) > 20:
                        labels[j] = f"#{len(label.split(','))}"
            else:
                labels = xlabels

            ax.boxplot(data.T, labels=labels, **boxplot_kwargs)
            if len(labels[0]) > 8:
                ticks = ax.get_xticks()
                ax.set_xticks(ticks, labels, rotation=45, ha="right")
        if not isinstance(x_axis_groups, types.NoneType):
            sec = ax.secondary_xaxis(-0.14)
            sec.spines["bottom"].set_visible(False)
            sec.set_xticks([tup[0] for tup in x_axis_groups], labels=[f"{tup[1]}" for tup in x_axis_groups])
            sec.tick_params("x", length=0)

            locs = [tup[0] for tup in x_axis_groups]
            if len(locs) > 1:
                for loc_right, loc_left in zip(locs[:-1], locs[1:]):
                    loc = np.mean([loc_right, loc_left])
                    ax.axvline(x=loc, color="black", alpha=.5, linestyle=":")
        if title:
            ax.set_title(run_set[0].name.split(" ")[0])

        if i == 0:
            ax.set_ylabel(_metric_name)
        ax.yaxis.grid(True)
        ax.yaxis.grid(True, which="minor", alpha=.2)

        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    fig.tight_layout(pad=0)


    if save:
        if not isinstance(fname, str):
            fname = ""
        else:
            fname = f"_{fname}"
        bdir = basepath / "figures" / "benchmark"
        if not bdir.is_dir():
            bdir.mkdir(parents=True)
        plt.savefig(bdir / f"{runs[0][0].name.split(' ')[0]}{fname}.pdf")
    else:
        plt.show()


def main_paper():
    basepath = Path(__file__).parent.parent # double .parent to get out of benchmark/
    basepath_data = Path("/mnt/aam/scripts/topology/data/epcc")
    _load = functools.partial(load, basepath_data=basepath_data)
    save = True
    figsize = (12,4)

    # BARRIER 2 cores cleaned
    plot([
        [
            _load(folder="measurements_i10se19_syncbench_24-01-12T1931", run_type=RunType.EPCC), #60,61
            _load(folder="measurements_i10se19_syncbench_24-01-12T2018", run_type=RunType.EPCC), #2,3

            _load(folder="measurements_i10se19_syncbench_24-01-12T1942", run_type=RunType.EPCC), #60,124
            _load(folder="measurements_i10se19_syncbench_24-01-12T2030", run_type=RunType.EPCC), #2,66

            _load(folder="measurements_i10se19_syncbench_24-01-12T1948", run_type=RunType.EPCC), #60,52
            _load(folder="measurements_i10se19_syncbench_24-01-12T2035", run_type=RunType.EPCC), #2,18
        ]
    ], basepath, save=save, fname="2", title=False, figsize=figsize, metric="overhead", metric_name="Overhead (μs)",
    x_axis_groups=[(1.5, "1 DSU 1 MXP"), (3.5, "2 DSUs 1 MXP"), (5.5, "2 DSUs 2 MXPs")],
    xlabels=["top","center"]*3
    )

    # # BARRIER 4 cores
    plot([
        [
            _load(folder="measurements_i10se19_syncbench_24-01-12T2121", run_type=RunType.EPCC),  # 56 57 120 121
            _load(folder="measurements_i10se19_syncbench_24-01-12T2041", run_type=RunType.EPCC), # 2 3 66 67

            _load(folder="measurements_i10se19_syncbench_24-01-12T2126", run_type=RunType.EPCC), # 56 57 58 59
            _load(folder="measurements_i10se19_syncbench_24-01-12T2047", run_type=RunType.EPCC), # 2 3 6 7

            _load(folder="measurements_i10se19_syncbench_24-01-12T2138", run_type=RunType.EPCC),  # 56 120 48 112
            _load(folder="measurements_i10se19_syncbench_24-01-12T2058", run_type=RunType.EPCC),  # 2 66 7 70

            _load(folder="measurements_i10se19_syncbench_24-01-12T2132", run_type=RunType.EPCC), # 56 48 50 58
            _load(folder="measurements_i10se19_syncbench_24-01-12T2053", run_type=RunType.EPCC), # 2 64 4 6

        ]
    ], basepath, save=save, fname="4", figsize=figsize, title=False, metric="overhead", metric_name="Overhead (μs)",
        x_axis_groups=[(1.5, "2 DSUs 1 MXP"), (3.5, "2 DSUs/MXPs"), (5.5, "4 DSUs 2 MXPs"), (7.5, "4 DSUs/MXPs")],
       xlabels=["top", "center"] * 4
    )
    #
    # # # LULESH 2 cores
    plot([
        [
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2143", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2211", run_type=RunType.LULESH),

            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2150", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2217", run_type=RunType.LULESH),

            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2153", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2221", run_type=RunType.LULESH),
        ],
    ], basepath, save=save, fname="2", title=False, figsize=figsize, metric="fom", metric_name="FOM (z/s)",
    x_axis_groups=[(1.5, "1 DSU 1 MXP"), (3.5, "2 DSUs 1 MXP"), (5.5, "2 DSUs 2 MXPs")],
    xlabels=["top","center"]*3)

    # # LULESH 32,48,64 cores
    plot([
        [
            ## center of chip
            # 12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1104", run_type=RunType.LULESH), #32
            # 20,21,22,23,84,85,86,87,12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75,16,17,18,19,80,81,82,83
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1140", run_type=RunType.LULESH), #48
            # 28,29,20,21,22,23,30,31,92,93,84,85,86,87,94,95,12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75,24,25,16,17,18,19,26,27,88,89,80,81,82,83,90,91
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1205", run_type=RunType.LULESH), #64

            ## top pf chip
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1115", run_type=RunType.LULESH), #32
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119,44,45,36,37,38,39,46,47,108,109,100,101,102,103,110,111
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1152", run_type=RunType.LULESH), #48
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119,44,45,36,37,38,39,46,47,108,109,100,101,102,103,110,111,28,29,20,21,22,23,30,31,92,93,84,85,86,87,94,95
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1221", run_type=RunType.LULESH), #64
        ]
    ], basepath, save=save, fname="32_48_64", title=False, figsize=figsize, metric="fom", metric_name="FOM (z/s)",
    x_axis_groups=[(2, "Center of Chip"), (5, "Top of Chip")])

    # STREAM 2 cores
    plot([
        [
            _load(folder="measurements_i10se19_stream.150M_24-02-02T0955", run_type=RunType.STREAM), #60,61
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1004", run_type=RunType.STREAM), #2,3

            _load(folder="measurements_i10se19_stream.150M_24-02-02T0957", run_type=RunType.STREAM), #60,124
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1006", run_type=RunType.STREAM), #2,66

            _load(folder="measurements_i10se19_stream.150M_24-02-02T0958", run_type=RunType.STREAM), #60,52
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1007", run_type=RunType.STREAM), #2,18
        ],
    ], basepath, save=save, fname="2", title=False, figsize=figsize, metric="copy", metric_name="Copy Rate (MB/s)",
    x_axis_groups=[(1.5, "1 DSU 1 MXP"), (3.5, "2 DSUs 1 MXP"), (5.5, "2 DSUs 2 MXPs")],
    xlabels=["top","center"]*3
    )
    # #
    ## STREAM 4 cores
    plot([
        [
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1015", run_type=RunType.STREAM), # 56 57 120 121
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1008", run_type=RunType.STREAM), # 2 3 66 67

            _load(folder="measurements_i10se19_stream.150M_24-02-02T1016", run_type=RunType.STREAM), # 56 57 58 59
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1009", run_type=RunType.STREAM), # 2 3 6 7

            _load(folder="measurements_i10se19_stream.150M_24-02-02T1018", run_type=RunType.STREAM),  # 56 120 48 112
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1011", run_type=RunType.STREAM),  # 2 66 7 70

            _load(folder="measurements_i10se19_stream.150M_24-02-02T1017", run_type=RunType.STREAM), # 56 48 50 58
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1010", run_type=RunType.STREAM), # 2 64 4 6
        ]
    ], basepath, figsize=figsize, title=False, save=save, fname="4", metric="copy", metric_name="Copy Rate (MB/s)",
        x_axis_groups=[(1.5, "2 DSUs 1 MXP"), (3.5, "2 DSUs/MXPs"),  (5.5, "4 DSUs 2 MXPs"), (7.5, "4 DSUs/MXPs")],
        xlabels=["top","center"]*4
    )
    #
    #
    # STREAM 32, 48, 64 cores
    plot([
        [

            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1602", run_type=RunType.STREAM), # 32
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1603", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1605", run_type=RunType.STREAM), # 48
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1606", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1607", run_type=RunType.STREAM), # 64
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1608", run_type=RunType.STREAM),
        ],
    ], basepath, save=save, fname="32_48_64", title=False, figsize=figsize, metric="copy", metric_name="Copy Rate (MB/s)")

    # # Netperf
    # plot([
    #     [
    #         _load(folder="measurements_i10se19_taskset_24-02-05T1404", run_type=RunType.NETPERF, run_id="62-62"),
    #         _load(folder="measurements_i10se19_taskset_24-02-05T1428", run_type=RunType.NETPERF, run_id="40-62"),
    #         _load(folder="measurements_i10se19_taskset_24-02-05T1415", run_type=RunType.NETPERF, run_id="40-40"),
    #     ],
    # ], basepath, save=save, fname="", title=False, figsize=figsize, metric="mean_latency_us", metric_name="Mean Latency (μs)")
    #
    # # NPB FT B 2 cores
    # plot([
    #         _load(folder="measurements_i10se19_ft.B.x_24-02-09T1750", run_type=RunType.NPB_FT),
    #         _load(folder="measurements_i10se19_ft.B.x_24-02-09T1755", run_type=RunType.NPB_FT),
    #         _load(folder="measurements_i10se19_ft.B.x_24-02-09T1801", run_type=RunType.NPB_FT),
    #         _load(folder="measurements_i10se19_ft.B.x_24-02-09T1806", run_type=RunType.NPB_FT),
    #
    # ], basepath, save=save, fname="2", title=False, figsize=figsize, metric="mop/s/thread", metric_name="mop/s/thread")
    #
    # # NPB FT C 32,48,64 cores
    # plot([
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1619", run_type=RunType.NPB_FT),
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1620", run_type=RunType.NPB_FT),
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1621", run_type=RunType.NPB_FT),
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1622", run_type=RunType.NPB_FT),
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1623", run_type=RunType.NPB_FT),
    #     _load(folder="measurements_i10se19_ft.C.x_24-02-16T1624", run_type=RunType.NPB_FT),
    #
    # ], basepath, save=save, fname="32_48_64", title=False, figsize=figsize, metric="mop/s/thread", metric_name="mop/s/thread")


def main_aam():
    basepath = Path(__file__).parent.parent # double .parent to get out of benchmark/
    basepath_data = Path("/mnt/aam/scripts/topology/data/epcc")
    _load = functools.partial(load, basepath_data=basepath_data)
    save = None

    # # BARRIER 2 cores cleaned
    plot([
        [
            _load(folder="measurements_i10se19_syncbench_24-01-12T1931", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T1937", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T1942", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T1948", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T1954", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2000", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2006", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2012", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2018", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2024", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2030", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2035", run_type=RunType.EPCC),
        ]
    ], basepath, save="barrier_2", metric="overhead", metric_name="Overhead (μs)")
    #
    # # BARRIER 4 cores
    plot([
        [
            _load(folder="measurements_i10se19_syncbench_24-01-12T2041", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2047", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2053", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2058", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2104", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2109", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2115", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2121", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2126", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2132", run_type=RunType.EPCC),
            _load(folder="measurements_i10se19_syncbench_24-01-12T2138", run_type=RunType.EPCC),
        ]
    ], basepath, save="barrier_4", metric="overhead", metric_name="Overhead (μs)")


    # # BARRIER 2 cores walk top->bottom left
    # plot([
    #     [
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1015", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1016", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1018", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1019", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1021", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1022", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1023", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1025", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1026", run_type=RunType.EPCC),
    #         _load(folder="measurements_i10se19_syncbench_24-02-05T1028", run_type=RunType.EPCC),
    #     ],
    #
    # ], basepath, save=save, metric="overhead", metric_name="Overhead (μs)")



    # # LULESH 2 cores
    plot([
        [
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2143", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2146", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2150", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2153", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2157", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2200", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2204", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2207", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2211", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2214", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2217", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2221", run_type=RunType.LULESH),

            # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1133", run_type=RunType.LULESH),
            # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1137", run_type=RunType.LULESH),
            # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1140", run_type=RunType.LULESH),
            # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1144", run_type=RunType.LULESH),
        ],
    ], basepath, save="lulesh_2", metric="fom", metric_name="FOM (z/s)", normalize="max")

    ## LULESH 4 cores
    # plot([
    #     [
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2224", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2228", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2231", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2235", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2238", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2242", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2246", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2250", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2253", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2257", run_type=RunType.LULESH),
    #         _load(folder="measurements_i10se19_lulesh2.0_24-01-12T2301", run_type=RunType.LULESH),
    #
    #         # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1147", run_type=RunType.LULESH),
    #         # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1151", run_type=RunType.LULESH),
    #         # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1154", run_type=RunType.LULESH),
    #         # _load(folder="measurements_i10se19_lulesh2.0_24-02-02T1158", run_type=RunType.LULESH),
    #     ]
    # ], basepath, save=save, metric="fom", metric_name="FOM (z/s)", normalize="max")


    # # LULESH 32,48,64 cores
    plot([
        [
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1104", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1115", run_type=RunType.LULESH),
        ],[    _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1140", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1152", run_type=RunType.LULESH),
       ], [    _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1205", run_type=RunType.LULESH),
            _load(folder="measurements_i10se19_lulesh2.0_24-02-12T1221", run_type=RunType.LULESH),
        ]
    ], basepath, save="lulesh_32_48_64", metric="fom", metric_name="FOM (z/s)", normalize="max")


    # STREAM 2 cores
    plot([
        [
            _load(folder="measurements_i10se19_stream.150M_24-02-02T0955", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T0957", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T0958", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1000", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1001", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1003", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1004", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1006", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M_24-02-02T1007", run_type=RunType.STREAM),
        ],
    ], basepath, save="stream_2", metric="copy", metric_name="Copy Rate (MB/s)", normalize="max")

    ## STREAM 4 cores
    # plot([
    #     [
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1008", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1009", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1010", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1011", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1012", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1013", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1014", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1015", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1016", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1017", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1018", run_type=RunType.STREAM),
    #
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1259", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1300", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1301", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1302", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1303", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1304", run_type=RunType.STREAM),
    #         _load(folder="measurements_i10se19_stream.150M_24-02-02T1305", run_type=RunType.STREAM),
    #     ]
    # ], basepath, save=save, metric="copy", metric_name="Copy Rate (MB/s)", normalize="max")

    # STREAM 32, 48, 64 cores
    plot([
        [
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1602", run_type=RunType.STREAM), # 32
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1603", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1605", run_type=RunType.STREAM), # 48
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1606", run_type=RunType.STREAM),
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1607", run_type=RunType.STREAM), # 64
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1608", run_type=RunType.STREAM),
        ],
    ], basepath, save="stream_32_48_64", metric="copy", metric_name="Copy Rate (MB/s)")

    # Netperf
    plot([
        [
            _load(folder="measurements_i10se19_taskset_24-02-05T1404", run_type=RunType.NETPERF, run_id="62-62"),
            _load(folder="measurements_i10se19_taskset_24-02-05T1428", run_type=RunType.NETPERF, run_id="40-62"),
            _load(folder="measurements_i10se19_taskset_24-02-05T1415", run_type=RunType.NETPERF, run_id="40-40"),
        ],
    ], basepath, save="netperf", metric="mean_latency_us", metric_name="Mean Latency (μs)", normalize="max")

    # NPB FT B 2 cores
    plot([
            _load(folder="measurements_i10se19_ft.B.x_24-02-09T1750", run_type=RunType.NPB_FT),
            _load(folder="measurements_i10se19_ft.B.x_24-02-09T1755", run_type=RunType.NPB_FT),
            _load(folder="measurements_i10se19_ft.B.x_24-02-09T1801", run_type=RunType.NPB_FT),
            _load(folder="measurements_i10se19_ft.B.x_24-02-09T1806", run_type=RunType.NPB_FT),

    ], basepath, save="npb_ft_b_2", metric="mop/s/thread", metric_name="mop/s/thread", normalize="max")

    # NPB FT C 32,48,64 cores
    plot([
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1619", run_type=RunType.NPB_FT),
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1620", run_type=RunType.NPB_FT),
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1621", run_type=RunType.NPB_FT),
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1622", run_type=RunType.NPB_FT),
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1623", run_type=RunType.NPB_FT),
        _load(folder="measurements_i10se19_ft.C.x_24-02-16T1624", run_type=RunType.NPB_FT),

    ], basepath, save="npb_ft_c_32_48_64", metric="mop/s/thread", metric_name="mop/s/thread")


def main_opencube():
    basepath = Path(__file__).parent
    basepath_data = Path("/mnt/opencube/scripts/cxi/data")
    _load = functools.partial(load, basepath_data=basepath_data)
    save = None

    ## CXI Read LAT
    # plot([
    #     [
    #         _load(folder="measurements_cn04_cxi_read_lat_24-02-13T1109", search_field="RDMA", run_type=RunType.CXI, run_id="36-36"),
    #         _load(folder="measurements_cn04_cxi_read_lat_24-02-13T1110-1", search_field="RDMA", run_type=RunType.CXI, run_id="30-30"),
    #               ],
    # ], basepath, save=save, metric=lambda df: df.loc[(slice(None), 16), "Mean[us]"], metric_name="Mean Latency (μs)", normalize="max")

    ## CXI Send LAT
    # plot([
    #     [
    #         _load(folder="measurements_cn03_cxi_send_lat_24-02-13T1156", search_field="Send", run_type=RunType.CXI),
    #         _load(folder="measurements_cn03_cxi_send_lat_24-02-13T1155", search_field="Send", run_type=RunType.CXI),
    #               ],
    # ], basepath, save=save, metric=lambda df: df.loc[(slice(None), 32), "Mean[us]"], metric_name="Mean Latency (μs)")

    # ## CXI send LAT pressure
    # plot([
    #     [
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1444-1", search_field="Send", run_type=RunType.CXI),
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1445", search_field="Send", run_type=RunType.CXI),
    #
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1452", search_field="Send", run_type=RunType.CXI), # pressured not same MXP
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1453", search_field="Send", run_type=RunType.CXI), # pressured not same MXP
    #
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1457", search_field="Send", run_type=RunType.CXI), # pressured same DSU
    #         _load(folder="measurements_cn01_cxi_send_lat_24-02-14T1457-1", search_field="Send", run_type=RunType.CXI), # pressured same DSU
    #
    #     ],
    # ], basepath, save=save, metric=[lambda df: df.loc[(slice(None), 32), "Mean[us]"]], metric_name="Mean Latency (μs)")


    ## CXI Read BW
    # plot([
    #     [
    #         _load(folder="measurements_cn01_cxi_read_bw_24-02-13T1525", search_field="RDMA Size[B]",
    #               run_type=RunType.CXI),
    #         _load(folder="measurements_cn01_cxi_read_bw_24-02-13T1523-1", search_field="RDMA Size[B]", run_type=RunType.CXI), # pressure not on cores on same MXP
    #         _load(folder="measurements_cn01_cxi_read_bw_24-02-13T1535", search_field="RDMA Size[B]", run_type=RunType.CXI), # pressure on cores on same MXP, but not DSU
    #         _load(folder="measurements_cn01_cxi_read_bw_24-02-13T1537", search_field="RDMA Size[B]",
    #               run_type=RunType.CXI),  # pressure on cores on same MXP and on same DSU (but not same core)
    #     ],
    # ], basepath, save=save, metric=lambda df: df.loc[(slice(None), 65536), "BW[MB/s]"], metric_name="Avg Bandwidth (MB/s)")


    basepath_data = Path("/mnt/opencube/scripts/osu/data")
    _load = functools.partial(load, basepath_data=basepath_data)
    # ## OSU alltoall
    # plot([
    #     [
    #         _load(folder="measurements_cn04_osu_alltoall_24-02-13T1419-1", run_type=RunType.OSU),
    #         _load(folder="measurements_cn04_osu_alltoall_24-02-13T1423", run_type=RunType.OSU),
    #               ],
    # ], basepath, save=save, metric=lambda df: df.loc[(slice(None), 16 ), "Avg Latency(us)"], metric_name="Mean Latency (μs)")

    # ## OSU lateny
    # plot([
    #     [
    #         _load(folder="measurements_cn04_osu_latency_24-02-13T1431", run_type=RunType.OSU),
    #         _load(folder="measurements_cn04_osu_latency_24-02-13T1433", run_type=RunType.OSU),
    #               ],
    # ], basepath, save=save, metric=lambda df: df.loc[(slice(None), 16 ), "Latency (us)"], metric_name="Mean Latency (μs)")

def main():
    #main_opencube()
    # main_aam()
    main_paper()

if __name__ == "__main__": main()