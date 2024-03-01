# Simple OpenMP Launcher

The file `laucher.sh` contains a simple wrapper script to launch OpenMP applications on a specific core subset. This script
was used to generate several datapoints used in this paper.

Use the following syntax:

```bash
bash launcher.sh \
  --iterations $iterations \
  --cores $list_of_cores \
  --binary $binary \
  --binary-args $binary_args
```

Note: The `--cores` argument does not support abbreviations like `0-3`, you have to specify the whole list `0,1,2,3`.
This script internally determines and sets the correct OpenMP-related variables `OMP_NUM_THREADS` and `GOMP_CPU_AFFINITY`.
You can however use the syntax `--cores #4` to only set `OMP_NUM_THREADS` and not pin to specific cores.

This script will populate a folder at `$pwd/data/$hostname_$timestamp` with stdout of each iteration.