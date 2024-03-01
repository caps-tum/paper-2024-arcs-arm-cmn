#!/usr/bin/bash
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo '`getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=binary:,binary-args:,cores:,iterations:,
OPTIONS=
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi

ITERATIONS=10
eval set -- "$PARSED"
while true; do
    case "$1" in
    --binary)
      BINARY="$2"
      shift 2 ;;
    --binary-args)
	    BINARY_ARGS="$2"
      shift 2 ;;
    --cores)
      CORES="$2"
      shift 2 ;;
    --iterations)
      ITERATIONS="$2"
      shift 2 ;;
    --)
      shift 
      break 
      ;;
    *)
      echo "Programming error, remaining args: $@"
      exit 3
      ;;
    esac
done

BASEDIR="measurements_$(hostname)_$(basename $BINARY)_$(date +%y-%m-%dT%H%M)"
OUTDIR="$(pwd)/data"

# if $OUTDIR/$BASEDIR exists, try $OUTDIR/$BASEDIR-$i until not exists
_basedir="$BASEDIR"
i=1
while true; do
  if [[ -d "$OUTDIR/$_basedir" ]]; then
        _basedir="$BASEDIR-$i"
    i=$(($i+1))
  else break
  fi
done
BASEDIR="$_basedir"
BASEPATH="$OUTDIR/$BASEDIR"


unset OMP_NUM_THREADS
unset __GOMP_CPU_AFFINITY
if [[ $CORES =~ \#.* ]]; then # test for ' --cores #X'
	export OMP_NUM_THREADS="${CORES#\#}"
else # split core-list-string into actual list
	IFS="," read -r -a CORES_ARRAY <<< "$CORES"
	export GOMP_CPU_AFFINITY="$CORES"	
	export OMP_NUM_THREADS="${#CORES_ARRAY[@]}"
fi

mkdir -p "$BASEPATH"

read -d '' meta_info << EOF
Measurement Start Timestamp: \`$(date +%c) ($(date +%s))\`
Host: $(hostname)
Binary: \`$BINARY\`
Binary Arguments: \`$BINARY_ARGS\`
Cores: \`$CORES\`
Iterations: \`$ITERATIONS\`
OMP_NUM_THREADS: \`$OMP_NUM_THREADS\`
GOMP_CPU_AFFINITY: \`$GOMP_CPU_AFFINITY\`
EOF
echo "$meta_info" > "$BASEPATH/meta.md"

for i in $(seq 1 "$ITERATIONS"); do
	printf "\r[%-2s/%s]" $i $ITERATIONS
	$BINARY $BINARY_ARGS > "$BASEPATH/measurement_$i.dat"
done

echo ""
echo "Written data to: $BASEDIR"
