import colorsys
import copy
import dataclasses
import itertools
import logging
import types
import typing
from typing import Optional

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.colorbar
import matplotlib.patheffects
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.text
import matplotlib.patches
import matplotlib.font_manager
import scipy.spatial.distance
from mpl_toolkits import axes_grid1
import logging
log = logging.getLogger("visualise_topology")
logging.basicConfig(level=logging.WARNING)

from utils import plt_utils, general_utils
from classes import DSU, Run
from utils.loader_utils import load_topology_runs, load_events, load_static_layout
from utils.plt_utils import highlight_cell

from scipy.spatial.distance import cityblock

pd.options.display.width=1920
pd.options.display.max_columns=99


basepath = Path(__file__).parent

@dataclasses.dataclass
class Distance:
    distances: dict[tuple[int,int],float]
    name: str
    source_nodes: list[tuple[int,int]]


def _add_cell(ax: plt.Axes, core: DSU, fontsize: int,
              _numa_ranges: Optional[list[int]] = None,
              _cell_color: Optional[typing.Union[str, typing.Iterable]] = None,
              _font_color: typing.Union[str, typing.Iterable] = "black",
              _annotations: bool = True,
              _cell_size: float = .5):
    if not isinstance(_numa_ranges, list):
        _numa_ranges = [0]
    domain = 0
    for i, v in enumerate(_numa_ranges[1:]):
        if core.coreids[1] < v:
            domain = i + 1
            break
    if isinstance(_cell_color, types.NoneType):
        _cell_color = matplotlib.colors.hex2color(list(matplotlib.colors.TABLEAU_COLORS.values())[domain])
        _cell_color = plt_utils.brighten_color(*_cell_color[:3], by=0)
        label_color = _font_color

    else:
        if isinstance(_cell_color, typing.Iterable):
            l = (0.2126 * _cell_color[0] + 0.7152 * _cell_color[1] + 0.0722 * _cell_color[2])  # luminance based off of ITU BT.709
            label_color = "black" if l > .5 else "white"


    highlight_cell(core.x, core.y, ax=ax, size=_cell_size, color=_cell_color,
                   alpha=1,
                   zorder=12)
    if _annotations:
        ax.annotate(f"{core.coreids[0]}", xy=np.array((core.x, core.y)) + .5,
                    xytext=(core.x + .48, core.y + .5 + 0.1 * (-1 if core.p else 1)),
                    fontsize=fontsize, ha="right", va="center", color=label_color, zorder=15)
        ax.annotate(f"{core.coreids[1]}", xy=np.array((core.x, core.y)) + .5,
                    xytext=(core.x + .52, core.y + .5 + 0.1 * (-1 if core.p else 1)),
                    fontsize=fontsize, ha="left", va="center", color=label_color, zorder=15)



def visualise_layout(runs: dict[str, Run], layout_static: pd.DataFrame,
                     mesh_size: (int,int),
                     numa_ranges: Optional[list[int]] = None,
                     highlight: list[int] = None,
                     distance_to_target: Optional[Distance] = None,
                     legend: bool = True,
                     annotations: bool = True,
                     additional_annotations: Optional[dict] = None,
                     cell_size: float = .64,
                     font_size_cells: int = 18,
                     fig_size_base: int = 14,
                     axis_labels: bool = True,
                     save: Optional[str] = None):


    if isinstance(additional_annotations, types.NoneType):
        additional_annotations = {}

    # adjust fig size aspect depending on mesh size
    #  calculate ratio of mesh_x/mesh_y and scale fig_size_y accordingly
    #  add: 1 to leave room for one-row legend, 2 for two-row legend
    #       one row for <= 6 entries, two for >
    #       legend entries: 2 (MXP, cores) + #additional_annotations
    fig_size_x = fig_size_base
    fig_size_y = fig_size_base * (np.min(mesh_size) / np.max(mesh_size))
    if legend:
        fig_size_y += 1 if 2+len(additional_annotations.keys()) <= 6 else 2

    fig,ax = plt.subplots( figsize=(fig_size_x, fig_size_y))

    # plot MXP nodes
    mxp = layout_static[layout_static["event_type"] == "mxp"]
    xmax = mxp.node_x.max()+1
    ymax = mxp.node_y.max()+1

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # connecting lines
    for i in range(0,xmax): ax.plot((i+.5, i+.5   ), (0.5,  ymax-.5), color="black", zorder=3)
    for j in range(0,ymax): ax.plot((0.5,  xmax-.5), (j+.5, j+.5   ), color="black", zorder=3)

    # keep track of which elements to display in legend
    #  initially: boxes for MXP and Core/DSU nodes
    #  later    : highlighted nodes, such as an RNID for storage, SNF for memory, etc.
    legend_entries = {"mxp": [matplotlib.patches.Patch(facecolor="lightgrey", label="Crosspoint",
                                                       path_effects=[matplotlib.patheffects.Stroke(linewidth=1,
                                                                                                   foreground="black")])],
                      "dsu": [matplotlib.patches.Patch(facecolor="lightblue", label="Cores",
                                                       path_effects=[matplotlib.patheffects.Stroke(linewidth=1,
                                                                                                   foreground="black")])]}

    # flatten annotations
    #  from name: {content<core,color,filter> }
    #  to   core: { filter : { content<color,name> } }
    # for easier lookup
    tab_colors = list(matplotlib.colors.TABLEAU_COLORS.keys())[len(numa_ranges):] # remove first N, as they are reserved for NUMA
    flat_annotations = {}
    if isinstance(additional_annotations, dict):
        for i, (name, annot) in enumerate(additional_annotations.items()):
            if _color := annot.get("color"):
                font_color = _color
            else:
                font_color = tab_colors[i % len(tab_colors)]
            for dsu in annot["nodes"]:
                if dsu not in flat_annotations: flat_annotations[dsu] = {}
                flat_annotations[dsu][annot["name_filter"]] = {"name": name, "color": font_color}


    # add MXP labels & potentially highlight the node label
    for i,g in layout_static[layout_static["event_type"] != "mxp"].groupby(["node_x","node_y"]):
        pos = np.array((i[0], i[1]))

        for port, group in g.groupby("node_port"):
            g_type = group["event_type"].iloc[0]
            name = f'{g_type} {port}'
            color = "black"
            if (annotation := flat_annotations.get((i[0], i[1]))) and (annot_dict := annotation.get(g_type)):
                color = annot_dict["color"]
                legend_entries[annot_dict["name"]] = [matplotlib.patches.Patch(alpha=0, label=annot_dict["name"]), color]

            if annotations:
                ax.annotate(name, xy=pos+.5, xytext=[pos[0]+.5, pos[1]+.5 + 0.1 * (-1 if port else 1)],
                            fontsize=font_size_cells, color=color, ha="center", va="center",
                            zorder=20)

    # determine and add core MXP nodes
    # also keep track of which mesh node (x,y) maps to which core ID (i) in mesh[x,y]=i
    mesh = np.zeros((mesh_size[0], mesh_size[1],2))
    mesh -= 1

    dsus = general_utils.get_dsus(runs=runs, mesh_size=mesh_size)


    if isinstance(distance_to_target, Distance):
        colors = matplotlib.colormaps["inferno_r"]
        norm = matplotlib.colors.Normalize(vmin=np.min(list(distance_to_target.distances.values())),
                                         vmax=np.max(list(distance_to_target.distances.values())))
        for k, v in distance_to_target.distances.items():
            distance_to_target.distances[k] = colors(norm(v))
        divider = axes_grid1.make_axes_locatable(ax)
        width = axes_grid1.axes_size.AxesY(ax, aspect=1. / 40)
        pad = axes_grid1.axes_size.Fraction(.5, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=colors, norm=norm, orientation='vertical')
        cb1.locator = matplotlib.ticker.MaxNLocator(7)
        cb1.set_label(f"Avg. Distance to {distance_to_target.name}")
        plt.gcf().add_axes(cax)

    for dsu in dsus:
        font_color = "black"
        if isinstance(highlight, list) and (dsu.coreids[0] in highlight or dsu.coreids[1] in highlight):
            font_color = "red"

        cell_color = None
        if isinstance(distance_to_target, Distance):
            cell_color = distance_to_target.distances.get((dsu.x, dsu.y), None)

        _add_cell(ax, dsu, _numa_ranges=numa_ranges, _cell_size=cell_size, _cell_color=cell_color, fontsize=font_size_cells,
                  _font_color=font_color, _annotations=annotations)

        # TODO: this assumes monolithic mode
        if dsu.coreids[0] < len(dsus): # 2 DSUs per MXP; only select lower ones here
            mesh[dsu.x, dsu.y, 0] = dsu.coreids[0]
        else:
            mesh[dsu.x, dsu.y, 1] = dsu.coreids[0]


    # add non-core MXP nodes
    #  remove core-MXPs by comparing their mesh IDs
    for i, row in mxp[~mxp.apply(lambda x: (x["node_x"], x["node_y"]), axis=1).isin([(c.x, c.y) for c in dsus])].iterrows():
        if isinstance(distance_to_target, Distance) and (row.node_x, row.node_y) in distance_to_target.source_nodes:
            highlight_cell(row.node_x, row.node_y, ax=ax, color="darkgrey", size=cell_size, fill=True, zorder=10)

        else:
            highlight_cell(row.node_x, row.node_y, ax=ax, color="lightgrey", size=cell_size, fill=True, zorder=10)

    # annotate core-to-core latency, if given
    # if isinstance(c2c, np.ndarray):
    #     # [[xpos, ypos, rotation, latency]]
    #     c2c_pairs: list[list[tuple[float,float], tuple[float,float], int, float]] = []
    #     c2c_pairs_intra = []
    #     for x in range(mesh_size[0]):
    #         for y in range(mesh_size[1]):
    #             if int(mesh[x,y,0]) > 0 and int(mesh[x,y,1]) > 0:
    #                 offset_outer = .25
    #                 offset_inner = .12
    #                 y_offset = .37
    #                 if x > 3:
    #                     offset_outer = 1-offset_outer
    #                     offset_inner = 1-offset_inner
    #                     y_offset = 1-y_offset
    #                 c2c_pairs_intra.append(([
    #                     (x+offset_outer, x+offset_inner, x+offset_inner, x+offset_outer),
    #                     (y+1-y_offset, y+1-y_offset, y+y_offset, y+y_offset),
    #                     (x+(offset_inner+.01), y+.5),
    #                     90,
    #                     np.mean([c2c[int(mesh[x, y, 0])  , int(mesh[x, y, 1])  ],
    #                              c2c[int(mesh[x, y, 0])  , int(mesh[x, y, 1])+1],
    #                              c2c[int(mesh[x, y, 0])+1, int(mesh[x, y, 1])  ],
    #                              c2c[int(mesh[x, y, 0])+1, int(mesh[x, y, 1])+1],
    #                              ]) ])) # average over all four core latencies in the two DSUs for the current MXP
    #
    #             cell_from = int(mesh[x,y,0])
    #             if cell_from < 0: # some CMN cells are not cores, mesh[$that,$cell] == -1 in this case
    #                 continue
    #             for lane in [0,1]:
    #                 offset = 0.07 * (-1 if lane else 1)
    #                 func = np.min if lane else np.max
    #                 if x > 0 and mesh[x-1,y, 0] >= 0: # look left
    #                     c2c_pairs.append([(x-1+.75, x+.25), (y+.5+offset, y+.5+offset ), 0 ,
    #                                       func([c2c[int(mesh[x, y, 0]), int(mesh[x-1, y, 0])],
    #                                             c2c[int(mesh[x, y, 0]), int(mesh[x-1, y, 1])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x-1, y, 0])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x-1, y, 1])] ])
    #                                       ])
    #                 if x < 7 and mesh[x+1,y, 0] >= 0: # look right
    #                     c2c_pairs.append([(x+1+.25, x+.75), (y+.5+offset, y+.5+offset ), 0 ,
    #                                       func([c2c[int(mesh[x, y, 0]), int(mesh[x+1, y, 0])],
    #                                             c2c[int(mesh[x, y, 0]), int(mesh[x+1, y, 1])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x+1, y, 0])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x+1, y, 1])] ])
    #                                       ])
    #                 if y > 0 and mesh[x,y-1, 0] >= 0: # look down
    #                     c2c_pairs.append([(x+.5+offset, x+ .5+offset), (y-1+.75, y+.25), 90,
    #                                       func([c2c[int(mesh[x, y, 0]), int(mesh[x, y-1, 0])],
    #                                             c2c[int(mesh[x, y, 0]), int(mesh[x, y-1, 1])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x, y-1, 0])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x, y-1, 1])] ])
    #                                       ])
    #                 if y < 7 and mesh[x,y+1, 0] >= 0: # look up
    #                     c2c_pairs.append([(x+.5+offset, x+ .5+offset), (y+1+.25, y+.75), 90,
    #                                       func([c2c[int(mesh[x, y, 0]), int(mesh[x, y+1, 0])],
    #                                             c2c[int(mesh[x, y, 0]), int(mesh[x, y+1, 1])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x, y+1, 0])],
    #                                             c2c[int(mesh[x, y, 1]), int(mesh[x, y+1, 1])] ])
    #                                       ])
    #
    #     c2c_pairs_data = [c[-1] for c in c2c_pairs] + [c[-1] for c in c2c_pairs_intra]
    #     norm = matplotlib.colors.Normalize(vmin=np.nanmin(c2c_pairs_data), vmax=np.nanmax(c2c_pairs_data))  # noqa
    #     colors = matplotlib.colormaps["viridis"]
    #
    #     # inter-MXP c2c latency
    #     for xpos, ypos, rotation, latency in c2c_pairs:
    #         color = colors(norm(latency))
    #         _,l,_ = colorsys.rgb_to_hls(*color[:3])
    #
    #         ax.plot(xpos, ypos, color=color, linewidth=10, zorder=5)
    #         ax.annotate(f"{latency:.1f}",
    #                     xy=(np.mean(xpos)+0.01, np.mean(ypos)-0.01), # +-.001 to properly center label - unsure why ha/va="center" doesn't to that properly
    #                     fontsize=10, color="black" if l > .45 else "white",
    #                     horizontalalignment="center", verticalalignment="center",
    #                     zorder=20, rotation=rotation)
    #
    #     # intra-MXP c2c latency
    #     for xpos, ypos, textxy, rotation, latency in c2c_pairs_intra:
    #         color = colors(norm(latency))
    #         _,l,_ = colorsys.rgb_to_hls(*color[:3])
    #
    #         ax.plot(xpos, ypos, color=color, linewidth=9, zorder=5)
    #         ax.annotate(f"{latency:.1f}",
    #                     xy=textxy,
    #                     fontsize=9, color="black" if l > .45 else "white",
    #                     horizontalalignment="center", verticalalignment="center",
    #                     zorder=20, rotation=rotation)
    #
    #     divider = axes_grid1.make_axes_locatable(ax)
    #     width = axes_grid1.axes_size.AxesY(ax, aspect=1. / 40)
    #     pad = axes_grid1.axes_size.Fraction(.5, width)
    #     current_ax = plt.gca()
    #     cax = divider.append_axes("right", size=width, pad=pad)
    #     plt.sca(current_ax)
    #     cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=colors, norm=norm, orientation='vertical')
    #     cb1.locator = matplotlib.ticker.MaxNLocator(7)
    #     cb1.set_label(f"Core-to-Core Latency [ns]")
    #     plt.gcf().add_axes(cax)

    # axis formatting
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda x,pos: f"{int(x-.5)}"))
        axis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axis.set_minor_locator(matplotlib.ticker.FixedLocator([i+.5 for i in range(xmax)]))

    ax.tick_params(axis="y", which="major",left=False)
    ax.tick_params(axis="x", which="major",bottom=False)

    if axis_labels:
        ax.set_xlabel("Mesh X")
        ax.set_ylabel("Mesh Y")
    ax.set_aspect(1)
    fig.tight_layout(pad=.1)

    # handle custom legend entries

    if legend:
        legend_elements = [ t[0] for t in legend_entries.values() ]
        # switch to two rows of legend if there are more than 6 entries
        ncols = len(legend_elements) if len(legend_entries) <= 6 else np.ceil(len(legend_entries)/2)
        box = ax.get_position()
        ax.set_position([box.x0,    box.y0,
                         box.width, box.height])

        legend = ax.legend(handles=legend_elements, loc='center',
                           bbox_to_anchor=(.5, 1.03 if len(legend_entries) <= 6 else 1.05), # adjust bbox anchor for two-row mode
                           fontsize=20,
                           ncols=ncols,
                           handletextpad=0)
        for key, text in zip(legend_entries.keys(), legend.get_texts()):
            if len(legend_entries[key]) > 1:
                text.set_color(legend_entries[key][1])

        for i, patch in enumerate(legend.get_patches()):
            patch.set_width(25)

    if isinstance(save, str): plt.savefig(basepath / "figures" / f"{save}.pdf")
    else:    plt.show()




def determine_distances(mesh_size, target_locations, runs: dict[str,Run], name: str) -> Distance:
    dsus = general_utils.get_dsus(runs=runs, mesh_size=mesh_size)
    dsu_distances = {}
    for dsu in dsus:
        distances = []
        for location in np.array(target_locations).reshape((-1,2)):
            distances.append(cityblock((dsu.x, dsu.y), location))
        dsu_distances[(dsu.x,dsu.y)] = np.mean(distances)

    return Distance(distances=dsu_distances, name=name, source_nodes=target_locations)

def main_aam():
    basepath_measurements = Path("/mnt/aam/code/cmn_topology_tool/data/")
    numa_type = None
    highlight = None
    # highlight = [
    #
    #     60, 61, 56, 57, 58, 59, 62, 63, 124, 125, 120, 121, 122, 123, 126, 127, 52, 53, 48, 49, 50, 51, 54, 55, 116,
    #     117, 112, 113, 114, 115, 118, 119, 44, 45, 36, 37, 38, 39, 46, 47, 108, 109, 100, 101, 102, 103, 110, 111, 28,
    #     29, 20, 21, 22, 23, 30, 31, 92, 93, 84, 85, 86, 87, 94, 95
    #
    # ]

    run_name = "determine_topology/i10se12_2024-01-02T1418"
    numa_ranges = [0]
    # numa_type = "aam1_monolithic_noc2c"

    # run_name = "determine_topology/i10se12_2024-02-28T1003"
    # numa_ranges = [0,64]
    # numa_type = "aam1_hemisphere"
    #
    # run_name = "determine_topology/i10se12_2024-02-28T1021"
    # numa_ranges = [0,32,64,96]
    # numa_type = "aam1_quadrant"

    # memory / SNF are not measurable directly via perf_event, so hunt them down visually (see notes)
    #  and add them manually to the layout
    memory_locations = [ [(j,i) for i in range(1,5)] for j in [0,7] ]
    memory_df = pd.DataFrame([
        {"node_x": x, "node_y": y, "node_port": 0,
         "event_name": "0x0", "event_id": "0x0", "event_type": "snf", "cmn_idx": None,
         "counts": 0}
        for (x,y) in itertools.chain(*memory_locations)
    ])

    additional_annotations = {
        "Storage": {"name_filter": "rnid", "color": "tab:brown", "nodes": [(0, 7)]},
        "Network": {"name_filter": "rnid", "color": "tab:blue", "nodes": [(0, 0)]},
        "Memory":  {"name_filter": "snf",  "color": "tab:red", "nodes": itertools.chain(*memory_locations)},
        "Cache":   {"name_filter": "hnf",  "color": "tab:green", "nodes": itertools.chain(*[ [(j,i) for i in range(1,5)] for j in [0,2,5,7] ])}
    }

    mesh_size = (8,8)
    events = load_events(basepath_measurements / run_name / "events.csv")
    layout_df = load_static_layout(basepath_measurements / run_name, events=events)
    layout_df = pd.concat([layout_df, memory_df])
    runs = load_topology_runs(run_name, events, basepath_measurements)

    distance_cache = determine_distances(mesh_size=mesh_size,
                                             target_locations=list(copy.deepcopy(additional_annotations["Cache"]["nodes"])),
                                             runs=runs, name="HNF")
    distance_memory = determine_distances(mesh_size=mesh_size,
                                          target_locations=list(
                                              copy.deepcopy(additional_annotations["Memory"]["nodes"])),
                                          runs=runs, name="SNF")
    matplotlib.rc('font', **{
        'family': 'sans',
        #"sans-serif": ["Roboto"],
        # "family": "SF Pro Display",
        'size': 28})
    visualise_layout(runs, mesh_size=mesh_size, layout_static=layout_df, save="aam1_monolithic",
                     numa_ranges=numa_ranges, highlight=highlight,
                     additional_annotations=additional_annotations,
                     cell_size=.71,
                     font_size_cells=20,
                     )

    matplotlib.rc('font', **{
        'family': 'sans',
        'size': 42})
    visualise_layout(runs,
                     mesh_size=mesh_size,
                     layout_static=layout_df,
                     save="aam1_monolithic_cache",
                     numa_ranges=numa_ranges,
                     highlight=highlight,
                     additional_annotations=None,
                     distance_to_target=distance_cache,
                     legend=False,
                     annotations=False,
                     cell_size=.75,
                     axis_labels=False,
                     )


    visualise_layout(runs,
                     mesh_size=mesh_size,
                     layout_static=layout_df,
                     save="aam1_monolithic_memory",
                     numa_ranges=numa_ranges,
                     highlight=highlight,
                     additional_annotations=None,
                     distance_to_target=distance_memory,
                     legend=False,
                     annotations=False,
                     cell_size=.75,
                     axis_labels=False,
                     )


def main_opencube():
    basepath_measurements = Path("/mnt/opencube/code/cmn_topology_tool/data/")

    numa_type = None
    highlight = None

    run_name = "determine_topology/cn02_2024-01-02T1707"
    numa_ranges = [0]
    #numa_type = "cn_monolithic"

    # memory / SNF are not measurable directly via perf_event, so hunt them down visually (see notes)
    #  and add them manually to the layout
    memory_locations = [[(j, i) for i in range(1, 5)] for j in [0, 7]]
    memory_df = pd.DataFrame([
        {"node_x": x, "node_y": y, "node_port": 0,
         "event_name": "0x0", "event_id": "0x0", "event_type": "snf", "cmn_idx": None,
         "counts": 0}
        for (x, y) in itertools.chain(*memory_locations)
    ])

    additional_annotations = {
        "Storage": {"name_filter": "rnid", "nodes": [(2, 5)]},
        "Network": {"name_filter": "rnid", "nodes": [(7, 0)]},
        "Memory": {"name_filter": "snf", "nodes": itertools.chain(*memory_locations)},
        "Cache": {"name_filter": "hnf", "nodes": itertools.chain(*[[(j, i) for i in range(1, 5)] for j in [0, 2, 5, 7]])},
        "HSN": {"name_filter": "rnid", "nodes": [(7, 5)]},
    }

    mesh_size = (8, 6)
    events = load_events(basepath_measurements / run_name / "events.csv")
    layout_df = load_static_layout(basepath_measurements / run_name, events=events)
    layout_df = pd.concat([layout_df, memory_df])
    runs = load_topology_runs(run_name, events, basepath_measurements)

    distance_to_target = determine_distances(mesh_size=mesh_size, target_locations=list(copy.deepcopy(additional_annotations["Memory"]["nodes"])),
                        runs=runs)

    visualise_layout(runs, mesh_size=mesh_size, layout_static=layout_df,
                     save=numa_type,
                     numa_ranges=numa_ranges, highlight=highlight,
                     additional_annotations=additional_annotations,
                     )

if __name__ == "__main__":
    main_aam()
    #main_opencube()