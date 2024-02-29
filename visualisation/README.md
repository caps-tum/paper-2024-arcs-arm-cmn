# Visualisation code for ARM CMN Topology paper

The code in this repository is used to reproduce the figures of the paper
> < paper title here >

The following sections illustrate how to reproduce all figures.

Note: All relative paths below are relative to `$root/visualisation`!

## Figure 1

Self-made, find the .pdf in `figures/MXP.pdf` and the corresponding Affinity Design 2 file in `figures/<file-here>`.

## Figure 2

To reproduce this figure, run:

```python3 visualise_raw_runs.py```


This should place (or update if already present) a file in `figures/raw_runs/i10se19_2024_02-09T1643.pdf` containing the visualisation.
Please consult `visualise_raw_runs.py` for more information. The `main()` file controls the visualisation.
Figure 2 requires the result of a core-to-core latency measurement between cores 60,42, monitored using the CMN Topology Tool.
You can measure this yourself on an ARM CMN system using the following command:
```bash
cd <path/to/cmn_topology_tool/path/to/binary>
./measurement \
        --events "mxp_n_dat_txflit_valid,mxp_e_dat_txflit_valid,mxp_s_dat_txflit_valid,mxp_w_dat_txflit_valid,mxp_p0_dat_txflit_valid,mxp_p1_dat_txflit_valid,hnf_txdat_stall,hnf_snp_sent" \
        launch \
        --binary <path/to/cmn_topology_tool/path/to/binary> \
    --args "-c 60,42 50000 5000"
```

Consult the README of the CMN Topology tool for more information.
This will place the corresponding measurement folder with format `$hostname_$date/` in `path/to/cmn_topology_tool/data/launch/`.
In `visualise_raw_runs.py:main`, point `name` to this folder and `basepath_measurements` to this folder and directory respectively.

## Figures 3,4

To reproduce Figures 3,4, run:

```python3 visualise_topology.py```

This should place (or update if already present) three file in `figures/topology/`:
- `aam1_monolithic.pdf` (Figure 3)
- `aam1_monolithic_cache.pdf` (Figure 4a)
- `aam1_monolithic_memory.pdf` (Figure 4b)

Please consult `visualise_topology.py` for more information. The `main()` file controls the visualisation.
The code requires the result the CMN topology tool.
You can measure this yourself on an ARM CMN system using the following command:

```bash
cd path/to/cmn_topology_tool/path/to/binary
./measurement determine-topology \
    --numa-config "monolithic" \
    --benchmark-binary-path ./benchmark \
    --benchmark-binary-args "--num-iterations 10000 --num-samples 2000" 
```

This will place the corresponding measurement folder with format `$hostname_$date/` in `path/to/cmn_topology_tool/data/determine_topology/`.
In `visualise_topology.py:main`, point `name` to this folder and `basepath_measurements` to this folder and directory respectively.

## Figures 5-10

To reproduce Figures 5-10, run:

```python3 benchmarks/analyse_benchmarks.py```

This should place (or update if already present) several files in `figures/benchmarks/`:
- `BARRIER_2.pdf` (Figure 5)
- `BARRIER_4.pdf` (Figure 6)
- `STREAM_2.pdf` (Figure 7)
- `STREAM_4.pdf` (Figure 8)
- `LULESH_2.pdf` (Figure 9)
- `LULESH_32_48_64.pdf` (Figure 10)

Please consult `benchmarks/analyse_benchmarks.py` for more information. The `main()` file controls the visualisation.

Data for these measurements are not gathered using the CMN topology tool, but instead using another simple launcher script.
This is because the CMN topology tool is responsible for gathering and processing perf-related information, which is not necessary nor desired here.

The launcher script is located at `$root/software/launcher/launcher.sh`. 
Please consult the [README](/software/launcher/README.md) of that script!

Find an example invocation for the barrier benchmark with two cores shown in Figure 5:

```bash
bash epcc-omp.sh --iterations 25 --cores '60,61'  --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000" && \
bash epcc-omp.sh --iterations 25 --cores '2,3'    --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000" && \
bash epcc-omp.sh --iterations 25 --cores '60,124' --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000" && \
bash epcc-omp.sh --iterations 25 --cores '2,66'   --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000" && \
bash epcc-omp.sh --iterations 25 --cores '60,52'  --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000" && \
bash epcc-omp.sh --iterations 25 --cores '2,18'   --binary path/to/syncbench --binary-args "--measureonly BARRIER --outer-repetitions 500 --test-time 5000"
```

Note: The `--cores` argument does not support abbreviations like `0-3`, you have to specify the whole list `0,1,2,3`
This script internally determines and sets the correct OpenMP-related variables `OMP_NUM_THREADS` and `GOMP_CPU_AFFINITY` and writes the 

## Notes

For all run folders both generated using the instructions outlined above and included in this repository, 
a `meta.md` file is included. This file contains information on the benchmark parameters, like the binary path, arguments,
added environment variables, etc. Example: `data/benchmarks/[measurements_i10se19_lulesh2.0_24-01-12T2143/meta.md`

## License

Code under MIT, images under CC-BY 4.0.