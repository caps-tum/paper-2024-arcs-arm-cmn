import enum
import re
import io
from pathlib import Path

import pandas as pd
from functionCacher.Cacher import Cacher

from visualisation.utils.classes import Run
cacher = Cacher()

class RunType(enum.Enum):
    EPCC=0
    LULESH=1
    STREAM=2
    NETPERF=3
    NPB_FT=4
    CXI=5
    OSU=6
    AMG=7

@cacher.cache
def load_epcc(folder: str, basepath_data: Path):
    pattern_overhead = re.compile(r"\w+ overhead \s+= ([\d.]+) microseconds \+\/- ([\d.]+)", flags=re.MULTILINE)
    pattern_median_ovrhd = re.compile(r"\w+ median_ovrhd\s+= ([\d.]+) microseconds", flags=re.MULTILINE)
    pattern_time = re.compile(r"\w+ time \s+= ([\d.]+) microseconds \+\/- ([\d.]+)", flags=re.MULTILINE)
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        if "Test: " in content:  # TODO remove on remeasure
            name = re.compile("Test: `(.*)`").search(content).group(1)
        else:
            name = re.compile("--measureonly (.*)").search(content).group(1)
        name += " " + re.compile("Cores: `(.*)`").search(content).group(1)
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1])}

            match = pattern_overhead.search(content)
            datum["overhead"] = float(match.group(1))
            datum["overhead_std"] = float(match.group(2))

            match = pattern_time.search(content)
            datum["time"] = float(match.group(1))
            datum["time_std"] = float(match.group(2))

            match = pattern_median_ovrhd.search(content)
            datum["median_ovrhd"] = float(match.group(1))
            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_lulesh(folder: str, basepath_data: Path) -> Run:
    pattern_fom = re.compile(r"FOM\s+=\s+?([\d\.]+) \(z\/s\)", flags=re.MULTILINE)
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()

        cores = re.compile("Cores: `(.*)`").search(content).group(1)
        # if len(cores) > 15:
        #     cores = len(cores.split(","))
        name = "LULESH " + f"{cores}"
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1])}

            match = pattern_fom.search(content)
            datum["fom"] = float(match.group(1))

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_stream(folder: str, basepath_data: Path) -> Run:
    pattern_copy = re.compile(r"Copy:\s+([\d\.]+).+", flags=re.MULTILINE)
    # pattern_copy = re.compile(r"Copy:\s+([\d\.]+).+", flags=re.MULTILINE)
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()

        name = "STREAM " + re.compile("Cores: `(.*)`").search(content).group(1)
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1])}

            match = pattern_copy.search(content)
            datum["copy"] = float(match.group(1))

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_netperf(folder: str, basepath_data: Path, **kwargs) -> Run:
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = "NETPERF "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Binary Arguments: `-c (\d+)").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            content = infile.readlines()[2].split(",")
            datum = {"i": int(file.stem.split("_")[1]),
                     "min_latency_us": float(content[0]),
                     "mean_latency_us": float(content[1]),
                     "max_latency_us": float(content[2]),
                     "std_latency_us": float(content[3])}

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_npb_ft(folder: str, basepath_data: Path, **kwargs) -> Run:
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        benchmark_class = re.compile(r"Binary:.*?\/bin\/(.*?)\.x",
                                     flags=re.MULTILINE).search(content).group(1)
        name = f"NPB {benchmark_class} "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            pattern_copy = re.compile(r"Mop/s/thread\s+=\s+([\d\.]+).+",
                                      flags=re.MULTILINE)
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1]),
                     "mop/s/thread": float(pattern_copy.search(content).group(1))}

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_amg(folder: str, basepath_data: Path, **kwargs) -> Run:
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = f"AMG2013"
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            pattern_fom = re.compile(r"System Size \* Iterations / Solve Phase Time:\s+(.*?)$",
                                      flags=re.MULTILINE)
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1]),
                     "FOM": float(pattern_fom.search(content).group(1))}

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_cxi(folder: str, basepath_data: Path, **kwargs) -> Run:
    dfs = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = re.compile("`(cxi_.*) ").search(content).group(1) + " "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    search_field = kwargs.get("search_field", "RDMA")
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            lines = infile.readlines()
            rdma_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith(search_field):
                    rdma_idx = idx
                    break
            if rdma_idx == -1:
                raise IndexError(f"Could not find {search_field} in {file}")
            measurement_lines = lines[rdma_idx:-1]
            sio = io.StringIO()
            pattern = re.compile("(\s{2,})")
            for line in measurement_lines:
                sio.write(re.sub(pattern, "\\t", line).lstrip())
            sio.seek(0)
            df = pd.read_csv(sio, sep="\t")
            df.insert(loc=len(df.columns)-1, column="i", value=int(file.stem.split("_")[1]))
            dfs.append(df)
    out_df = pd.concat(dfs).set_index(["i", df.columns[0]]).sort_index()

    return Run(df=out_df, name=name)

@cacher.cache
def load_osu(folder: str, basepath_data: Path, **kwargs) -> Run:
    dfs = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = re.compile("Binary: `.*\/(.*?)`").search(content).group(1) + " "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            lines = infile.readlines()
            rdma_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith("# Size"):
                    rdma_idx = idx
                    break
            if rdma_idx == -1:
                raise IndexError(f"Could not find '# Size' in {file}")
            measurement_lines = lines[rdma_idx:-1]
            if measurement_lines[1].startswith("#"):
                measurement_lines = [measurement_lines[0]] + measurement_lines[2:]
            sio = io.StringIO()
            pattern = re.compile("(\s{2,})")
            for line in measurement_lines:
                sio.write(re.sub(pattern, "\\t", line).lstrip())
            sio.seek(0)
            df = pd.read_csv(sio, sep="\t")
            df.insert(loc=len(df.columns)-1, column="i", value=int(file.stem.split("_")[1]))
            dfs.append(df)
    out_df = pd.concat(dfs).set_index(["i", df.columns[0]]).sort_index()

    return Run(df=out_df, name=name)


def load(folder: str, basepath_data: Path, run_type: RunType, **kwargs) -> Run:
    """
    Return Run instance as loaded depending on the RunType / custom loader
    """
    match run_type:
        case RunType.EPCC:
            fn = load_epcc
        case RunType.LULESH:
            fn = load_lulesh
        case RunType.STREAM:
            fn = load_stream
        case RunType.NETPERF:
            fn = load_netperf
        case RunType.NPB_FT:
            fn = load_npb_ft
        case RunType.CXI:
            fn = load_cxi
        case RunType.OSU:
            fn = load_osu
        case RunType.AMG:
            fn = load_amg
        case _:
            raise Exception()

    return fn(folder, basepath_data, **kwargs)