from pathlib import Path

import pandas as pd

from classes import Run
from utils.general_utils import cacher, parse_eventid

@cacher.cache
def load_run(name: str, events: pd.DataFrame, basepath: Path) -> Run:
    df = pd.read_csv(basepath/name/"measurements.csv",sep=";", index_col=False)
    df = parse_eventid(df, events=events)
    run = Run(name=name, df=df)
    with open(basepath/name/"meta.md", "r") as infile:
        run.meta = infile.read()
    return run


@cacher.cache
def load_topology_runs(name: str, events: pd.DataFrame, basepath: Path, key: str = "cores") -> dict[str, Run]:
    runs = {}
    files = list(filter(lambda x: x.stem.startswith(f"{key}_"), (basepath/name/key).iterdir()))
    for file in files:
        df = pd.read_csv(file, sep=";", index_col=False)
        df = parse_eventid(df, events=events)
        run = Run(name=file.stem.replace(f"{key}_", "").replace("_", "-"), df=df)
        runs[run.name] = run
    return runs


@cacher.cache
def load_events(path: Path) -> pd.DataFrame:
    with open(path, "r") as infile:
        lines = infile.readlines()

    data = []
    for line in lines:
        datum = {}
        l,r = line.replace('"', "").split(";")

        datum["event_name"] = l
        r_parts = r.split(",")

        for r_p in r_parts:
            r_p_parts = r_p.split("=")
            match r_p_parts[0]:
                case "type":
                    datum["event_type"] = r_p_parts[1].strip()
                case "eventid":
                    datum["event_id"] = r_p_parts[1].strip()
        data.append(datum)
    return pd.DataFrame(data).drop_duplicates(keep="first",subset="event_name")


@cacher.cache
def load_static_layout(path: Path, events: pd.DataFrame):

    df = pd.concat([pd.read_csv(path / "nodes.csv", sep=";"),
                    pd.read_csv(path / "mxp.csv", sep=";")])

    df = df[df.counts >= 0]  # remove <not supported> / -1 entries
    df = parse_eventid(df, events=events)
    df["event_type"] = df["event_name"].apply(lambda x: x.split("_")[0]) # overwrite hex code with "name" (e.g. hnf, mxp, etc.)
    return df


