import copy
import dataclasses
import itertools
import types
import typing
import typing_extensions
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
from mpl_toolkits import axes_grid1
import logging
log = logging.getLogger("visualise_topology")
logging.basicConfig(level=logging.WARNING)

from utils import plt_utils, general_utils
from classes import DSU
from utils.loader_utils import load_topology_runs, load_events, load_static_layout

from scipy.spatial.distance import cityblock

pd.options.display.width=1920
pd.options.display.max_columns=99

matplotlib.rc('font', **{
        'family': 'sans',
        'size': 28})

basepath = Path(__file__).parent

@dataclasses.dataclass
class Distance:
    distances: dict[tuple[int,int],float]
    name: str
    source_nodes: list[tuple[int,int]]

def add_rect(x: float, y: float, ax: plt.Axes, size: float = 1, **kwargs):
    """
    Add matplotlib rect to plot at (x,y)
    :param x: X position
    :param y: Y position
    :param ax: Ax to add rect to
    :param size: Size of rect
    :param kwargs: Any other args to plt.Rectangle()
    """
    rect = plt.Rectangle(xy=(x + (1-size)/2, y + (1-size)/2), width=size, height=size, **kwargs)
    rect.set_path_effects([matplotlib.patheffects.Stroke(linewidth=1, foreground="black")])
    ax.add_patch(rect)

    return rect


def add_cell(ax: plt.Axes, dsu: DSU, fontsize: int,
             numa_ranges: Optional[list[int]] = None,
             cell_color: Optional[typing.Union[str, typing.Iterable]] = None,
             font_color: typing.Union[str, typing.Iterable] = "black",
             annotations: bool = True,
             cell_size: float = .5,
             hatch: Optional[str] = None):
    """
    Add cell for DSU, which involves adding the rectangle (add_rect) and optionally adding annotations and handling several colouring options.
    :param ax: Main matplotlib axis
    :param dsu: DSU to add cell for
    :param fontsize: Fontsize of cell label
    :param numa_ranges: Optional NUMA ranges, changes cell background depending on NUMA domain
    :param cell_color: Basic cell colour (overwritten by numa_ranges)
    :param font_color: Basic font colour (overwritten if cell_color is set)
    :param annotations: Whether to add annotation label
    :param cell_size: Size of cell
    """
    if not isinstance(numa_ranges, list):
        numa_ranges = [0]
    domain = 0
    # find NUMA domain for DSU - Note: could be replaced by some smarter searching tool, but numa_ranges is most likely <10, so don't bother
    for i, v in enumerate(numa_ranges[1:]):
        if dsu.coreids[1] < v:
            domain = i + 1
            break

    label_color = font_color
    if isinstance(cell_color, types.NoneType):
        cell_color = matplotlib.colors.hex2color(list(matplotlib.colors.TABLEAU_COLORS.values())[domain])
        cell_color = plt_utils.adapt_hls_color(*cell_color[:3], s_by=-.25, l_by=.25)

    else:
        if isinstance(cell_color, typing_extensions.Sequence):
            l = (0.2126 * cell_color[0] + 0.7152 * cell_color[1] + 0.0722 * cell_color[2])  # luminance based off of ITU BT.709
            label_color = "black" if l > .5 else "white"
        elif isinstance(cell_color, np.ndarray):
            print("Warning! cell_color is an np.ndarray, which is not handled here! Image might contain unexpected results")

    add_rect(x=dsu.x, y=dsu.y, ax=ax, size=cell_size, color=cell_color, zorder=12)
    if hatch:
        add_rect(x=dsu.x, y=dsu.y, ax=ax, size=cell_size, color="black", zorder=13, alpha=.15, hatch=hatch, fill=False)

    if annotations:
        ax.annotate(text=f"{dsu.coreids[0]}", xy=(dsu.x + .5, dsu.y + .5),
                    xytext=(dsu.x + .48, dsu.y + .5 + 0.125 * (-1 if dsu.p else 1)),
                    fontsize=fontsize, ha="right", va="center", color=label_color, zorder=15,
                         path_effects=[matplotlib.patheffects.Stroke(linewidth=5, foreground=cell_color),
                                       matplotlib.patheffects.Normal()])
        ax.annotate(text=f"{dsu.coreids[1]}", xy=(dsu.x + .5, dsu.y + .5),
                    xytext=(dsu.x + .52, dsu.y + .5 + 0.125 * (-1 if dsu.p else 1)),
                    fontsize=fontsize, ha="left", va="center", color=label_color, zorder=15,
                         path_effects=[matplotlib.patheffects.Stroke(linewidth=5, foreground=cell_color),
                                       matplotlib.patheffects.Normal()])


def visualise_layout(dsus: list[DSU],
                     layout_static: pd.DataFrame,
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
                     out_fname: Optional[str] = None):
    """
    Visualise the topological layout of an ARM CMN

    :param dsus: List of DSUs on that mesh
    :param layout_static: DataFrame containing location of static components (MXP, HNF, etc.)
    :param mesh_size: Mesh size of underlying CMN mesh
    :param numa_ranges: Optional list of NUMA domains, given as ID of first core per domain; used for highlighting
    :param highlight: Optional list of core IDs to be highlighted on the visualisation
    :param distance_to_target: Optional Distance instance used to colour DSU boxes with mean distance to target nodes (see determine_distances)
    :param legend: Whether to plot legend
    :param annotations: Whether to include MXP, DSU, etc. annotations/labels
    :param additional_annotations: Whether to additionally highlight MXP labels
    :param cell_size: Size of each MXP cell
    :param font_size_cells: Font size of label inside MXP cell
    :param fig_size_base: Base size of Figure, final fig size is (fig_size_base, fig_size_base * <aspect_ratio>)
    :param axis_labels: Whether to plot X,Y axis labels
    :param out_fname: Optional string for output file name - will show instead of save if set to None
    """

    if isinstance(additional_annotations, types.NoneType):
        additional_annotations = {}

    enable_distance = isinstance(distance_to_target, Distance)

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

    xmax = mesh_size[0]
    ymax = mesh_size[1]

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # plt lines connecting each MXP, will draw boxes over it later
    for i in range(0,xmax): ax.plot((i+.5, i+.5   ), (0.5,  ymax-.5), color="black", zorder=3)
    for j in range(0,ymax): ax.plot((0.5,  xmax-.5), (j+.5, j+.5   ), color="black", zorder=3)



    # keep track of which elements to display in legend
    #  initially: boxes for MXP and Core/DSU nodes
    #  later    : highlighted nodes, such as an RNID for storage, SNF for memory, etc.
    legend_entries = {"mxp": [matplotlib.patches.Patch(facecolor="lightgrey", label="Crosspoint",
                                                       path_effects=[matplotlib.patheffects.Stroke(linewidth=1,
                                                                                                   foreground="black")])],
                      "dsu": [matplotlib.patches.Patch(facecolor="#82b1d1", label="Cores", hatch="\\\\",
                                                       path_effects=[matplotlib.patheffects.Stroke(linewidth=1,
                                                                                                   foreground="black")])]}

    # prepare additional_annotations by flattening it
    #  from name: {content<core,color,filter> }
    #  to   core: { filter : { content<color,name> } }
    # for easier lookup
    tab_colors = list(matplotlib.colors.TABLEAU_COLORS.keys())[len(numa_ranges):] # remove first N, as they are reserved for NUMA
    flat_annotations = {}
    if isinstance(additional_annotations, dict):
        for i, (name, annot) in enumerate(additional_annotations.items()):
            if color := annot.get("color"):
                font_color = color
            else:
                font_color = tab_colors[i % len(tab_colors)]
            for dsu in annot["nodes"]:
                if dsu not in flat_annotations: flat_annotations[dsu] = {}
                flat_annotations[dsu][annot["name_filter"]] = {"name": name, "color": font_color, "symbol": annot["symbol"]}


    # add non-core MXP cells
    #  remove core-MXPs by comparing their mesh IDs (they'll be added later with a different cell colour)
    mxp = layout_static[layout_static["event_type"] == "mxp"]
    for i, row in mxp[~mxp.apply(lambda x: (x["node_x"], x["node_y"]), axis=1).isin([(c.x, c.y) for c in dsus])].iterrows():
        if enable_distance:
            add_rect(row.node_x, row.node_y, ax=ax, color="white", size=cell_size, zorder=10)
            if (row.node_x, row.node_y) in distance_to_target.source_nodes:
                add_rect(row.node_x, row.node_y, ax=ax, color="black", size=cell_size,  zorder=11, alpha=0.75, hatch="\\", fill=False)
        else:
            add_rect(row.node_x, row.node_y, ax=ax, color="lightgrey", size=cell_size,  zorder=10)

    # add MXP labels & potentially highlight the node label
    for i,g in layout_static[layout_static["event_type"] != "mxp"].groupby(["node_x","node_y"]):
        for port, group in g.groupby("node_port"):
            g_type = group["event_type"].iloc[0]
            #name = f'{g_type} {port}'
            name = f'{g_type}'
            color = "black"
            if (annotation := flat_annotations.get((i[0], i[1]))) and (annot_dict := annotation.get(g_type)):
                color = annot_dict["color"]
                legend_entries[annot_dict["name"]] = [
                    matplotlib.patches.Patch(alpha=0, label=f'{annot_dict["symbol"]} {annot_dict["name"]}'), color
                ]
                name = f"{annot_dict['symbol']} {name}"
            if annotations:
                ax.annotate(name, xy=(i[0]+.5, i[1]+.5), xytext=[i[0]+.5, i[1]+.5 + 0.125 * (-1 if port else 1)],
                            fontsize=font_size_cells, color=color, ha="center", va="center",
                            zorder=20)


    # if given, prepare distance_to_target structure by normalising & color-mapping the distances for plotting
    if enable_distance:
        cmap = matplotlib.colormaps["inferno_r"]
        norm = matplotlib.colors.Normalize(vmin=np.min(list(distance_to_target.distances.values())),
                                           vmax=np.max(list(distance_to_target.distances.values())))
        for k, v in distance_to_target.distances.items():
            distance_to_target.distances[k] = cmap(norm(v))
        divider = axes_grid1.make_axes_locatable(ax)
        width = axes_grid1.axes_size.AxesY(ax, aspect=1. / 40)
        pad = axes_grid1.axes_size.Fraction(.5, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.locator = matplotlib.ticker.MaxNLocator(7)
        cb1.set_label(f"Avg. Distance to {distance_to_target.name}")
        plt.gcf().add_axes(cax)

    # add DSU cells
    for dsu in dsus:
        font_color = "black"
        if isinstance(highlight, list) and (dsu.coreids[0] in highlight or dsu.coreids[1] in highlight):
            font_color = "red"

        cell_color = None
        if enable_distance:
            cell_color = distance_to_target.distances.get((dsu.x, dsu.y), None)

        add_cell(ax=ax, dsu=dsu, numa_ranges=numa_ranges, cell_size=cell_size, cell_color=cell_color,
                 fontsize=font_size_cells,
                 font_color=font_color, annotations=annotations, hatch="\\" if not enable_distance else None)


    # format x,y axes
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
                           handletextpad=0,
                           columnspacing=.75)
        for key, text in zip(legend_entries.keys(), legend.get_texts()):
            if len(legend_entries[key]) > 1:
                text.set_color(legend_entries[key][1])

        for i, patch in enumerate(legend.get_patches()):
            patch.set_width(25)

    if isinstance(out_fname, str): plt.savefig(basepath / "figures" / "topology" / f"{out_fname}.pdf")
    else:                          plt.show()




def determine_distances(dsus: list[DSU], name: str, target_locations: list[tuple[int,int]]) -> Distance:
    """
    Given a list of DSUs a mesh, and a list of target locations (a,b) on that mesh:
     Determine the mean Manhattan distance from each DSU to all target locations
    """
    dsu_distances = {}
    for dsu in dsus:
        distances = []
        for location in np.array(target_locations).reshape((-1,2)):
            distances.append(cityblock((dsu.x, dsu.y), location))
        dsu_distances[(dsu.x,dsu.y)] = np.mean(distances)

    return Distance(distances=dsu_distances, name=name, source_nodes=target_locations)

def main():
    basepath_measurements = basepath / "data" / "topology"

    run_name = "i10se12_2024-01-02T1418"
    numa_ranges = [0]
    fname = "aam1_monolithic"

    # run_name = "i10se12_2024-02-28T1003"
    # numa_ranges = [0,64]
    # fname = "aam1_hemisphere"

    # run_name = "i10se12_2024-02-28T1021"
    # numa_ranges = [0,32,64,96]
    # fname = "aam1_quadrant"

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
        "Storage": {"symbol": "(S)", "name_filter": "rnid", "color": "tab:brown", "nodes": [(0, 7)]},
        "Network": {"symbol": "(N)", "name_filter": "rnid", "color": "tab:blue", "nodes": [(0, 0)]},
        "Memory":  {"symbol": "(M)", "name_filter": "snf",  "color": "tab:red", "nodes": itertools.chain(*memory_locations)},
        "Cache":   {"symbol": "(C)", "name_filter": "hnf",  "color": "tab:green", "nodes": itertools.chain(*[ [(j,i) for i in range(1,5)] for j in [0,2,5,7] ])}
    }

    mesh_size = (8,8)
    events = load_events(basepath_measurements / run_name / "events.csv")
    layout_df = load_static_layout(basepath_measurements / run_name, events=events)
    layout_df = pd.concat([layout_df, memory_df])
    runs = load_topology_runs(run_name, events, basepath_measurements)
    dsus = general_utils.get_dsus(runs=runs, mesh_size=mesh_size)

    # temp = pd.read_csv("/home/pfriese/Desktop/arm.csv", sep=",")
    # dsus = [
    #     DSU(x=row["x"], y=row["y"], coreids=(row["Cluster ID"],row["Core ID"]),
    #         p=0 if row["Core ID"] < 64 else 1)
    #     for _, row in temp.iterrows()
    #     if row["Core ID"] % 2 == 1
    # ]


    distance_cache = determine_distances(target_locations=list(copy.deepcopy(additional_annotations["Cache"]["nodes"])),
                                         dsus=dsus, name="HNF")
    distance_memory = determine_distances(target_locations=list( copy.deepcopy(additional_annotations["Memory"]["nodes"])),
                                          dsus=dsus, name="SNF")

    visualise_layout(dsus=dsus, mesh_size=mesh_size, layout_static=layout_df, out_fname=f"{fname}",
                     numa_ranges=numa_ranges,
                     additional_annotations=additional_annotations,
                     cell_size=.71,
                     font_size_cells=20)

    matplotlib.rc('font', size=42)
    visualise_layout(dsus=dsus,
                     mesh_size=mesh_size,
                     layout_static=layout_df,
                     out_fname=f"{fname}_cache",
                     numa_ranges=numa_ranges,
                     distance_to_target=distance_cache,
                     legend=False,
                     annotations=False,
                     cell_size=.75,
                     axis_labels=False)


    visualise_layout(dsus=dsus,
                     mesh_size=mesh_size,
                     layout_static=layout_df,
                     out_fname=f"{fname}_memory",
                     numa_ranges=numa_ranges,
                     distance_to_target=distance_memory,
                     legend=False,
                     annotations=False,
                     cell_size=.75,
                     axis_labels=False)

if __name__ == "__main__":
    main()
