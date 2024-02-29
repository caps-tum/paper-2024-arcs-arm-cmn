import functools
import types

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.colors
import matplotlib.ticker
import matplotlib.patheffects
import matplotlib.pyplot as plt
import re
import json
from typing import Union, Optional

from mpl_toolkits.axes_grid1 import ImageGrid

from utils import loader_utils, plt_utils
from utils.loader_utils import load_run
from classes import Run


pd.options.display.width=1920
pd.options.display.max_columns=99

matplotlib.rc('font', **{
    'family' : 'sans',
    'size'   : 24})

matplotlib.rc("axes",
              titley=1.05)
basepath = Path(__file__).parent


def _show_vals(_vals, _title, _ax, _mesh_size,
               _cbar_title,
               _norm: Optional[matplotlib.colors.Normalize] = None,
               add_cb: bool = False):
    _vals = _vals.reshape(_mesh_size).T

    if isinstance(_norm, types.NoneType):
        if not np.any(_vals):  # all-zero
            _norm = matplotlib.colors.CenteredNorm()
        else:
            _norm = matplotlib.colors.LogNorm()
    im = _ax.imshow(_vals, norm=_norm)

    _title = _title.replace("_dat_txflit_valid", "").replace("mxp_", "")
    _ax.set_title(_title)
    _ax.set_ylim(*_ax.get_ylim()[::-1])
    if add_cb:
        plt_utils.add_colorbar(ax=_ax, im=im, title="# events" if _cbar_title else None)
    return im

def plot_run(run: Run,
             eventids: list[Union[str,re.Pattern]] = None,
             show_remaining: bool = False,
             mesh_size: (int, int) = (8, 6),
             save: bool = False,
             cbar_title: Optional[bool] = None,
             suptitle: bool = True,
             display_meta: bool = True,
             nrows: Optional[int] = None,
             ncols: Optional[int] = None,
             base_size_x: float = 7,
             base_size_y: float = 6,
             )             :
    """
    Plot CMN performance counter measurement run as coloured matrix

    Note: If eventids contains a regex pattern, then all measurements matching this pattern will be summed and
           then plotted as one measurement. If no eventids are given, display all events separately.

    :param run: Run instance to plot
    :param eventids: list of eventids as string or regex (default: None -> all events in run)
    :param show_remaining: whether to include event IDs in `run` but not in `eventids` - useful for specifying patterns in `eventids` and including the rest like a wildcard (default: False)
    :param mesh_size: mesh size (default: 8x8)
    :param save: whether to save to disk (default: False)
    :param cbar_title: whether to add colourbar title (default: True if len(eventids)==1)
    """

    if not isinstance(eventids, list):
        eventids = run.df.event_name.unique()

    if not isinstance(cbar_title, bool):
        cbar_title = len(eventids) == 1

    df = run.df.copy()
    df["counts"] = pd.to_numeric(df["counts"]
                                 .replace("<not counted>", 0)
                                 .replace("<not supported>", -1))

    included_eventids = set()
    vals_array = []
    for i, eventid in enumerate(eventids):
        if isinstance(eventid, re.Pattern):
            _events = [e for e in df.event_name.unique() if eventid.match(e)]
            included_eventids |= set(_events)
            vals = np.zeros(shape=(len(df[df.event_name == _events[0]]),))
            for event in _events:
                for _, group in df[df.event_name == event].groupby("node_port"):
                    vals += group["counts"].values
            title = eventid.pattern
        else:
            included_eventids |= {eventid}
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))

            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            title = eventid
        vals_array.append((title, vals))

    if show_remaining:
        for i, eventid in enumerate(list(set(run.df.event_name.unique()) - included_eventids)):
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))
            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            vals_array.append((eventid, vals))

    N = len(vals_array)
    if display_meta:
        N += 1

    if not isinstance(nrows, int):
        nrows = int(np.ceil(np.sqrt(N)))
    if not isinstance(ncols, int):
        ncols = int(np.ceil(N / nrows))

    _data = np.array([tup[1] for tup in vals_array]).flat
    norm = matplotlib.colors.LogNorm(vmin=np.min(list(filter(lambda x: x>0, _data))), vmax=np.max(_data))

    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size_x*ncols, base_size_y*nrows))
    if nrows == ncols == 1:
        axes = np.array([axes])

    for i, (title, vals) in enumerate(vals_array):
        _show_vals(vals, title, axes.flat[i], _mesh_size=mesh_size, _cbar_title=cbar_title,
                   _norm=norm, add_cb=i % ncols == ncols-1)

        axes.flat[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axes.flat[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))


    for ax in axes.flat[N:]:
        fig.delaxes(ax)

    for vax in axes:
        if not isinstance(vax, plt.Axes):
            for ax in vax[1:]:
                ax.yaxis.set_visible(False)

    if display_meta:
        ax: plt.Axes = axes.flat[-1]
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.text(x=0.1,y=0,s=run.meta, fontdict={"fontsize": 11}, wrap=True)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # if suptitle:
    #     plt.suptitle(f"{run.name}")
    fig.tight_layout()
    if save: plt.savefig(basepath / "figures" / "raw_runs" / f"{run.name.replace('/','-')}.pdf")
    else:    plt.show()


def plot_run_one_row(run: Run,
             eventids: list[Union[str,re.Pattern]] = None,
             show_remaining: bool = False,
             mesh_size: (int, int) = (8, 6),
             save: bool = False,
             suptitle: bool = True,
             ncols: Optional[int] = None,
             base_size_x: int = 6,
             base_size_y: int = 6,
             )             :



    if not isinstance(eventids, list):
        eventids = run.df.event_name.unique()

    df = run.df.copy()
    df["counts"] = pd.to_numeric(df["counts"]
                                 .replace("<not counted>", 0)
                                 .replace("<not supported>", -1))

    included_eventids = set()
    vals_array = []
    for i, eventid in enumerate(eventids):
        if isinstance(eventid, re.Pattern):
            _events = [e for e in df.event_name.unique() if eventid.match(e)]
            included_eventids |= set(_events)
            vals = np.zeros(shape=(len(df[df.event_name == _events[0]]),))
            for event in _events:
                for _, group in df[df.event_name == event].groupby("node_port"):
                    vals += group["counts"].values
            title = eventid.pattern
        else:
            included_eventids |= {eventid}
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))

            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            title = eventid
        vals_array.append((title, vals))

    if show_remaining:
        for i, eventid in enumerate(list(set(run.df.event_name.unique()) - included_eventids)):
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))
            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            vals_array.append((eventid, vals))

    _data = np.array([tup[1] for tup in vals_array]).flat
    norm = matplotlib.colors.LogNorm(vmin=np.min(list(filter(lambda x: x>0, _data))), vmax=np.max(_data))


    fig = plt.figure(figsize=(base_size_x*ncols, base_size_y*1))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, ncols),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15,
                     )

    for i, (title, vals) in enumerate(vals_array):
        ax = grid[i]
        im = _show_vals(vals, title, ax, _mesh_size=mesh_size, _norm=norm, add_cb=False, _cbar_title=False)

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    cb = grid[-1].cax.colorbar(im)
    cb.set_label("# events")
    grid[0].set_ylabel("Mesh Y")
    grid[len(grid)//2].set_xlabel("Mesh X")


    if suptitle:
        plt.suptitle(f"{run.name}")
    plt.tight_layout(pad=0)
    if save: plt.savefig(basepath / "figures" / "raw_runs" / f"{run.name.replace('/','-')}.pdf")
    else:    plt.show()


def main_opencube():
    basepath_measurements = Path("/mnt/opencube/code/cmn_topology_tool/data/")

    ## network
    run_name = "launch/cn02_2024-01-02T1744"
    run = load_run(run_name, basepath=basepath_measurements,
                           events=loader_utils.load_events(basepath_measurements / run_name / "events.csv"))
    plot_run(run, eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8,6))

    ## memory
    # run_name = "launch/cn02_2024-01-02T1751"
    # run = load_run(run_name, basepath=basepath_measurements,
    #                        events=loader_utils.load_events(basepath_measurements / run_name / "events.csv"))
    # plot_run(run, eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8,6))

    ## storage
    # run_name = "launch/cn02_2024-01-02T1755"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8, 6))

    # ## hsn
    # run_name = "launch/cn03_2024-02-13T1004"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[
    #              re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          ],
    #          show_remaining=True, mesh_size=(8, 6),
    #          save=False)

    ## testing ground
    # run_name = "launch/cn03_2024-02-05T1339"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[
    #              re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          ],
    #          show_remaining=True, mesh_size=(8, 6),
    #          save=False)

def main_aam():
    basepath_measurements = Path("/mnt/aam/code/cmn_topology_tool/data/")


#     basepath_measurements = Path("/mnt/aam/code/cmn_topology_tool/data/")
#     run_name = "measurements_i10se12_2023-12-15T1623"
#     mesh_size = (8, 8)
#     events = utils.load_events(basepath_measurements / run_name / "events.csv")
#     run_key = "0-60"
# #    layout_df = utils.load_static_layout(basepath_measurements / run_name, events=events)
#     runs = utils.load_topology_runs_new(run_name, events, basepath_measurements, key="edges", run_key=run_key)
#     print(runs.keys())
#     plot_run(runs[run_key],              eventids=[
#                  re.compile("mxp_[nesw]_dat_txflit_valid"),
#              ],
#              show_remaining=True,mesh_size=mesh_size)
#     return

    #run_name = "measurements-23-12-13T1713-i10se19" # dd core 44 quad
    #run_name = "measurements-23-12-13T1711-i10se19" # dd core 85 quad
    #run_name = "measurements-23-12-18T1513-i10se12-dd" # dd core 12 mono

    #run_name = "measurements-23-12-13T1557-i10se19" # iperf3 24 quad
    #run_name = "measurements-23-12-13T1557-i10se19" # iperf3 24 quad
    #run_name = "measurements-23-12-18T1522-i10se12-iperf3" # iperf3 12 mono

    #run_name = "measurements-23-12-13T1734-i10se19" # fio 2 quad
    #run_name = "measurements-23-12-13T1737-i10se19" # fio 2 quad with lulesh disruption
    #run_name = "measurements-23-12-13T1833-i10se12" # fio 42 mono

    #run_name = "measurements-23-12-13T1920-i10se19" # sysbench memory 85 quad
    #run_name = "measurements-23-12-18T1606-i10se12-sysbench" # sysbench memory 42 mono
    #run_name = "measurements-23-12-18T1635-i10se19-sysbench" # sysbench memory 4,108 quad
    #run_name = "measurements-23-12-18T1712-i10se19-sysbench" # sysbench memory 1T 108 quad

    #run_name = "measurements-23-12-18T1727-i10se19-mbw" # mbw 10G 12 quad

    #run_name = "measurements-23-12-14T1650-i10se12-ep.C.x"
    #run_name = "measurements-23-12-14T1716-i10se12-ep.C.x"

    #run_name = "measurements-23-12-18T1636-i10se19-sleep" # sleep 4 quad

    # run_name = "determine_launch_multi/i10se12_2024-01-02T1344"
    #
    # events = loader_utils.load_events(basepath_measurements / run_name / "events.csv")
    #
    # _load_run = functools.partial(loader_utils.load_run, events=events, basepath=basepath_measurements)
    # run =       _load_run(run_name)
    #
    # plot_run(run,
    #          eventids=[
    #              re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          ],
    #          show_remaining=True,
    #          mesh_size=(8,8))




    # ## iperf3
    # run_name = "launch/i10se19_2024-02-05T1403"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
    #          show_remaining=True, mesh_size=(8, 8), save=False)
    #
    # ## stream
    # run_name = "launch/i10se19_2024-01-10T1530"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
    #          show_remaining=True, mesh_size=(8, 8), save=False)

    # # c2c 60,42
    # run_name = "launch/i10se19_2024-02-09T1643"
    # plot_run_one_row(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
    #          show_remaining=True, mesh_size=(8, 8), save=True, suptitle=False,
    #          ncols=3, base_size_x=4.5, base_size_y=5.5)

    # plot_run_one(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventid=re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          mesh_size=(8, 8), save=True, display_meta=False, suptitle=False, xlabel=None, ylabel="Node ID")
    #
    # plot_run_one(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventid=re.compile("mxp_p0_dat_txflit_valid"),
    #          mesh_size=(8, 8), save=True, display_meta=False, suptitle=False, xlabel="Node ID", yticks=False)
    #
    # plot_run_one(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventid=re.compile("mxp_p1_dat_txflit_valid"),
    #          mesh_size=(8, 8), save=True, display_meta=False, suptitle=False, yticks=False)




    # testing ground
    run_name = "launch/i10se19_2024-02-28T0932"
    plot_run(load_run(run_name, basepath=basepath_measurements,
                      events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
             eventids=[
                 re.compile("mxp_[nesw]_dat_txflit_valid"),
             ],
             show_remaining=True, mesh_size=(8, 8),
             save=False)


def main_opencube():
    basepath_measurements = Path("/mnt/opencube/code/cmn_topology_tool/data/")

    ## network
    run_name = "launch/cn02_2024-01-02T1744"
    run = load_run(run_name, basepath=basepath_measurements,
                           events=loader_utils.load_events(basepath_measurements / run_name / "events.csv"))
    plot_run(run, eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8,6))

    ## memory
    # run_name = "launch/cn02_2024-01-02T1751"
    # run = load_run(run_name, basepath=basepath_measurements,
    #                        events=loader_utils.load_events(basepath_measurements / run_name / "events.csv"))
    # plot_run(run, eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8,6))

    ## storage
    # run_name = "launch/cn02_2024-01-02T1755"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[re.compile("mxp_[nesw]_dat_txflit_valid")], show_remaining=True, mesh_size=(8, 6))

    # ## hsn
    # run_name = "launch/cn03_2024-02-13T1004"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[
    #              re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          ],
    #          show_remaining=True, mesh_size=(8, 6),
    #          save=False)

    ## testing ground
    # run_name = "launch/cn03_2024-02-05T1339"
    # plot_run(load_run(run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[
    #              re.compile("mxp_[nesw]_dat_txflit_valid"),
    #          ],
    #          show_remaining=True, mesh_size=(8, 6),
    #          save=False)

def main_paper():
    basepath_measurements = Path("/mnt/aam/code/cmn_topology_tool/data/")

    # c2c 60,42
    run_name = "launch/i10se19_2024-02-09T1643"
    plot_run_one_row(load_run(run_name, basepath=basepath_measurements,
                      events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
             eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
             show_remaining=True, mesh_size=(8, 8), save=True, suptitle=False,
             ncols=3, base_size_x=4.5, base_size_y=5.5)



def main():
    # main_opencube()
    main_aam()

if __name__ == "__main__": main()