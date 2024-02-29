import functools
import types
import typing
from typing import Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

from visualisation.benchmarks.benchmark_loaders import RunType, load
from visualisation.classes import Run

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
         figsize: tuple[int,int] = (12,8),
         fname: Optional[str] = None):
    """
    Visualise benchmark Runs as boxplots

    :param runs: list of lists of runs (for each list of runs, add new subplot)
    :param basepath: basepath to store figure output to
    :param metric: which metric to plot (str for simple key, depends on importer, or callable on a DataFrame for more complex lookups), can also be list of metrics
    :param metric_name: metric name to plot on yaxis
    :param x_axis_groups: optional second x-labels per group, format is (tick_location, name)
    :param xlabels: optional list of x labels
    :param normalize: whether to normalise, supported are "max" and "min":
    :param save: whether to save plot to file (will plt.show() on false)
    :param title: whether to add title
    :param figsize: figsize
    :param fname: optional figure name suffix
    """

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
            # get data, either via string metric or callable
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
                # generate x labels (core IDs) from run, truuncate if too long
                labels = [r.name.split(" ")[-1] for r in run_set]
                for j, label in enumerate(labels):
                    if len(str(label)) > 20:
                        labels[j] = f"#{len(label.split(','))}"
            else:
                labels = xlabels

            ax.boxplot(data.T, labels=labels, **boxplot_kwargs)
            if len(labels[0]) > 8: # rotate ticks if longer than 8 characters
                ticks = ax.get_xticks()
                ax.set_xticks(ticks, labels, rotation=45, ha="right")

        # add secondary x axis for groups
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
        suffix = f"_{fname}" if isinstance(fname, str) else ""
        output_dir = basepath / "figures" / "benchmark"
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        plt.savefig(output_dir / f"{runs[0][0].name.split(' ')[0]}{suffix}.pdf")
    else:
        plt.show()


def main():
    basepath = Path(__file__).parent.parent # double .parent to get out of benchmark/
    basepath_data = basepath / "data" / "benchmarks"
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

    # BARRIER 4 cores
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

    # LULESH 2 cores
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

    # LULESH 32,48,64 cores
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

    # STREAM 32, 48, 64 cores
    plot([
        [
            ## center of chip
            # 12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1602", run_type=RunType.STREAM),  # 32
            # 20,21,22,23,84,85,86,87,12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75,16,17,18,19,80,81,82,83
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1605", run_type=RunType.STREAM),  # 48
            # 28,29,20,21,22,23,30,31,92,93,84,85,86,87,94,95,12,13,4,5,6,7,14,15,76,77,68,69,70,71,78,79,8,9,0,1,2,3,10,11,72,73,64,65,66,67,74,75,24,25,16,17,18,19,26,27,88,89,80,81,82,83,90,91
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1607", run_type=RunType.STREAM),  # 64

            ## top pf chip
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1603", run_type=RunType.STREAM), # 32
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119,44,45,36,37,38,39,46,47,108,109,100,101,102,103,110,111
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1606", run_type=RunType.STREAM), # 48
            # 60,61,56,57,58,59,62,63,124,125,120,121,122,123,126,127,52,53,48,49,50,51,54,55,116,117,112,113,114,115,118,119,44,45,36,37,38,39,46,47,108,109,100,101,102,103,110,111,28,29,20,21,22,23,30,31,92,93,84,85,86,87,94,95
            _load(folder="measurements_i10se19_stream.150M.100_24-02-16T1608", run_type=RunType.STREAM), # 64

        ],
    ], basepath, save=save, fname="32_48_64", title=False, figsize=figsize, metric="copy", metric_name="Copy Rate (MB/s)",
        x_axis_groups=[(2, "Center of Chip"), (5, "Top of Chip")]
    )

if __name__ == "__main__": main()