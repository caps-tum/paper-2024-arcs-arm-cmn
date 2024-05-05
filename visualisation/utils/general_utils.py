import copy
from typing import Union, Optional


import numpy as np
import pandas as pd

from visualisation.classes import Run, DSU

from functionCacher.Cacher import Cacher
cacher = Cacher()
def from_nodeid(nodeid: Union[int,str], field_length=3):
    """
    Arm CMN nodes are coded into either 7 or 9-bit long bitstrings (cf. Arm CMN 600 TRM, Sec. 2.4 Node ID mapping)
    The format is:
    | 6/8:4/5 | 4/5:3 |  2   |  1:0  |
    |    x    |   y   | port | 0b00 |

    field_length determines whether ID is 7 or 9 bits long.
    7-bit IDs are used for mesh dimensions <=4x4, 9-bit for anything larger

    :param nodeid: hex string or int of nodeid
    :param field_length: length of x/y fields (3 for 9-bit nodeids, 2 for 7-bit nodeids)
    :returns: x, y, and port
    """

    nodeid_int = nodeid
    if isinstance(nodeid, str):
        nodeid_int = int(nodeid_int, 16)

    port = (nodeid_int & 0b000000100) >> 2
    mask=(~(~0b0 << field_length))
    y = (nodeid_int >> 3) & mask
    x = (nodeid_int >> (3+field_length)) & mask
    return x,y,port


def parse_eventid(df: pd.DataFrame, events: pd.DataFrame):
    df.insert(column="node_x", value=0, loc=len(df.columns)-1)
    df.insert(column="node_y", value=0, loc=len(df.columns)-1)
    df.insert(column="node_port", value=0, loc=len(df.columns)-1)

    # resolve node_id into parts
    df[["node_x","node_y","node_port"]] = np.array(df["node_id"].apply(from_nodeid).values.tolist())

    # add event_name from events df
    df = df.merge(events[["event_type","event_id","event_name"]], on=["event_type", "event_id"])
    return df


def get_dsus(runs: dict[str,Run], mesh_size: (int, int)) -> [DSU]:
    """
    Determine position of all DSUs in chip

    :param runs: dict[run_name: run] for all loaded runs
    :param mesh_size: size of CMN mesh
    :returns: list of DSU instances
    """

    # for two measurements both starting with core 0, that core (or DSU) must occur twice
    #  determine it.
    dsus = []
    r1 = runs["0-2"]
    dsus.append(get_dsu(r1, mesh_size=mesh_size))
    dsus.append(get_dsu(r1, mesh_size=mesh_size, dsu_0=dsus[0]))

    r2 = runs["0-4"]
    dsus.append(get_dsu(r2, mesh_size=mesh_size))
    dsus.append(get_dsu(r2, mesh_size=mesh_size, dsu_0=dsus[2]))

    counts = []
    for c in set(dsus):
        counts.append((c, len(list(filter(lambda x: x==c, dsus)))))
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    dsu_0 = counts[0][0]
    dsu_0.coreids = (0, 1)

    cores_out = [dsu_0]

    # find & annotate remaining cores
    for i,run in enumerate(runs.values()):
        core = get_dsu(run, mesh_size=mesh_size, dsu_0=dsu_0)
        parts = run.name.split("-")
        f,t = int(parts[0]), int(parts[1])
        core.coreids = (t, t + 1)
        cores_out.append(core)
    return cores_out

def get_dsu(run: Run, mesh_size: (int, int), dsu_0: Optional[DSU] = None) -> DSU:
    """
    Determine "brightest" core

    Each run measures the MXP perf event dat_txflit_valid for both ports p0 and p1. The measured code communicates between
     two cores, therefore two MXP nodes should light up, corresponding to these two cores.
    Given the two ports, this could mean any two cores on any of the two port-"planes" could light up.
    Go through both port planes and find the first and second "brightest" core. Out of all four bright-spots,
     pick the two globally brightest cores.
    If dsu_0 is set, exclude it from the four bright-spots - measurements measure between core 0 and X and we only
     want to get core X.

    :param run: Measurement run
    :param mesh_size: size of mesh
    :param dsu_0: Optional dsu_0
    :returns: Brightest core
    """
    hits = []

    for p in [0,1]:
        vals = run.df[(run.df.event_name == f"mxp_p{p}_dat_txflit_valid") & (run.df.counts >= 0)]["counts"]\
        .values.reshape(mesh_size).T
        ymax = np.argmax(vals, axis=1)
        y_idx = int(np.argmax([vals[i][m] for i,m in enumerate(ymax)]))
        x_idx = ymax[y_idx]
        hits.append((DSU(p=p, x=x_idx, y=y_idx), vals[y_idx, x_idx]))

        vals_second = copy.copy(vals)
        vals_second[y_idx,x_idx] = 0
        ymax = np.argmax(vals_second, axis=1)
        y_idx = int(np.argmax([vals_second[i][m] for i,m in enumerate(ymax)]))
        x_idx = ymax[y_idx]
        hits.append((DSU(p=p, x=x_idx, y=y_idx), vals_second[y_idx, x_idx]))

    if isinstance(dsu_0, DSU):
        hits = list(filter(lambda h: h[0] != dsu_0, hits))
    hits = sorted(hits, key=lambda x: x[1], reverse=True)
    return hits[0][0]
