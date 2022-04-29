#!/bin/bash

# Fit the upper envelope of synthetic stellar spectra and fit with RBF
# It needs to be run with a bunch of config files in a single config directory
# or with command line arguments passed in.

# Examples:
#   ./stellar/scripts/build_rbf.sh stellar-grid bosz
#   ./scripts/stellar/build_rbf.sh stellar-grid bosz --config ./configs/import/stellar/grid/bosz/rbf --in /.../grid/bosz_5000/ --out /.../rbf/bosz_5000

set -e

# Process command-line arguments

PYTHON_DEBUG=0

BINDIR="./bin"
CONFIGDIR="./configs/import/stellar/grid/bosz/rbf"
INDIR="${PFSSPEC_DATA}/models/stellar/grid/bosz/bosz_5000"
OUTDIR="${PFSSPEC_DATA}/models/stellar/grid/bosz/bosz_5000"
PARAMS=""

RUNMODE="run"
LASTJOBID=""
TYPE="$1"
SOURCE="$2"
shift 2

while (( "$#" )); do
    case "$1" in
        --debug)
            PYTHON_DEBUG=1
            PARAMS="$PARAMS --debug"
            shift
            ;;
        --sbatch)
            RUNMODE="sbatch"
            PARAMS="--sbatch $PARAMS"
            shift
            ;;
        --srun)
            RUNMODE="srun"
            PARAMS="--srun $PARAMS"
            shift
            ;;
        --config)
            CONFIGDIR="$2"
            shift 2
            ;;
        --in)
            INDIR="$2"
            shift 2
            ;;
        --out)
            OUTDIR="$2"
            shift 2
            ;;
        --) # end argument parsing
            shift
            break
            ;;
        #-*|--*=) # unsupported flags
            #  echo "Error: Unsupported flag $1" >&2
            #  exit 1
            #  ;;
        *) # preserve all other arguments
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

function run_cmd() {
    local cmd="$@"
    # echo "$cmd"
    if [[ $RUNMODE == "sbatch" ]]; then
        if [ -n "$LASTJOBID" ]; then
            LASTJOBID=$(eval "$cmd --dependency afterok:$LASTJOBID")
        else
            LASTJOBID=$(eval "$cmd")
        fi    
        echo "Submitted slurm job with id $LASTJOBID"
    else
        eval "$cmd"
    fi
}

echo "Using configuration directory $CONFIGDIR"
echo "Using output directory $OUTDIR"

if [[ -d "$OUTDIR/fit" ]]; then
    echo "Skipping fitting upper envelope."
else
    echo "Fitting upper envelope..."
    cmd="$BINDIR/fit $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/fit.json" \
        --in "$INDIR" --out "$OUTDIR/fit" \
        --step fit $PARAMS"
        run_cmd "$cmd"
fi

# if [[ -d "$OUTDIR/fill" ]]; then
#     echo "Skipping filling in holes / smoothing."
# else
#     echo "Filling in holes / smoothing..."
#     cmd="$BINDIR/fit $TYPE $SOURCE \
#         --config "$CONFIGDIR/common.json" \
#         --in "$INDIR" --out "$OUTDIR/fill" \
#         --params "$OUTDIR/fit" \
#         --step fill $PARAMS"
#     run_cmd "$cmd"
# fi

if [[ -d "$OUTDIR/fit-rbf" ]]; then
    echo "Skipping RBF on continuum parameters."
else
    echo "Running RBF on continuum parameters..."
    $cmd="$BINDIR/rbf $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/rbf.json" \
        --in "$INDIR" --out "$OUTDIR/fit-rbf" \
        --params "$OUTDIR/fit" \
        --step fit $PARAMS"
    run_cmd "$cmd"
fi

if [[ -d "$OUTDIR/norm" ]]; then
    echo "Skipping normalizing spectra."
else
    echo "Normalizing spectra..."
    cmd="$BINDIR/fit $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" \
        --in "$INDIR" --out "$OUTDIR/norm" \
        --params "$OUTDIR/fit-rbf" --rbf \
        --step norm $PARAMS"
    run_cmd "$cmd"
fi

if [[ -d "$OUTDIR/pca" ]]; then
    echo "Skipping PCA."
else
    echo "Running PCA..."
    cmd="$BINDIR/pca $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/pca.json" \
        --in "$OUTDIR/norm" --out "$OUTDIR/pca" \
        --params "$OUTDIR/fit-rbf" --rbf $PARAMS"
    run_cmd "$cmd"
fi

if [[ -d "$OUTDIR/pca-rbf" ]]; then
    echo "Skipping RBF on principal components."
else
    echo "Running RBF on principal components..."
    cmd="$BINDIR/rbf $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/rbf.json" \
        --in "$OUTDIR/pca" --out "$OUTDIR/pca-rbf" \
        --params "$OUTDIR/fit-rbf" --rbf --pca \
        --step pca $PARAMS"
    run_cmd "$cmd"
fi