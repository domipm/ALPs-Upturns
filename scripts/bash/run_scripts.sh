#!/opt/homebrew/bin/bash
echo "Running under bash $BASH_VERSION"

set -e # Exit on error
set -u # Exit on undefined

# Define array of all sources names
declare -a ALL_SOURCES
ALL_SOURCES=("1ES0229+200" "1ES0347-121" "1ES0414+009" "1ES1101-232" "1ES1312-423" "1RXSJ101015.9-311909" "3C279" "GRB180720B" "H2356-309" "PG1553+113" "PKS0346-27" "PKS0447-439" "PKS0903-57" "PKS1510-089" "PKS2155-304")

# Define array of blocks
declare -A SOURCE_BLOCKS
# Define array of datasets
declare -A SOURCE_DATASETS

for src in "${ALL_SOURCES[@]}"; do
    # By default, single block
    SOURCE_BLOCKS[$src]="1"
    # By default, H.E.S.S. + Fermi-LAT dataset
    SOURCE_DATASETS[$src]="joint"
done

# Set Bayesian blocks (wherever not just one)
# TODO: Instead of array, just number of blocks
SOURCE_BLOCKS["1ES1101-232"]="2"
SOURCE_BLOCKS["PKS0903-57"]="3"
SOURCE_BLOCKS["PKS2155-304"]="8"

# Set datasets per source (wherever not just joint)
SOURCE_DATASETS["H2356-309"]="hess"
SOURCE_DATASETS["PKS2155-304"]="hess"

# List of EBL models considered (default "all")
ALL_EBL_MODELS=("dominguez" "franceschini" "finke" "saldana-lopez")

# Name of conda environment
CONDA_ENV="alps-upturns"
# Directory location of scripts
SCRIPT_DIR="/path/to/ALPs-Upturns/scripts"

# Default options
SCRIPT="model_upturns.py"
SOURCES=()
EBL_MODELS=()
DATASET_OVERRIDE=""
BLOCK_OVERRIDE=""
DRY_RUN=false
EXTRA_ARGS=()

# Display usage
usage() {
  cat << EOF
Parameterized runner for analysis scripts

Usage:
  ./run_scripts.sh [options] [-- extra kwargs passed straight to the python script]

Options:
  -s, --script SCRIPT       Python script to run (default: model_upturns.py)
  -o, --source SOURCE       Source name, or "all" (default: all). Repeatable.
  -e, --ebl EBL             EBL model, or "all" (default: dominguez). Repeatable.
  -d, --dataset DATASET     Dataset override: hess|joint|all.
                            Default: whatever each source actually has.
  -b, --block BLOCK         Block override, e.g. "block2". Default: all blocks
                            defined for that source.
  -n, --dry-run             Print the commands instead of running them.
  -l, --list                List known sources (and their blocks/datasets) and exit.
  -h, --help                Show this help.
  -k, --kwargs              Keyword arguments for selected script.

Anything after "--kwargs" is appended verbatim as extra kwargs to every python call,
  e.g.:  ./run_scripts.sh -o 3C279 --kwargs --overwrite --verbose

Examples:
  ./run_scripts.sh                                    # everything (old behaviour)
  ./run_scripts.sh -o 3C279                           # one source, all its blocks/datasets
  ./run_scripts.sh -o 3C279 -o PG1553+113             # a couple of sources
  ./run_scripts.sh -e all                             # every source x every EBL model
  ./run_scripts.sh -o PKS2155-304 -b block3 -d hess   # one source, one block, one dataset
  ./run_scripts.sh -s other_script.py -o all -n       # dry-run a different script
EOF
  exit 1
}

# Get a list of all sources, and their time segmentation and datasets
list_sources() {
  local src
  for src in $(printf '%s\n' "${!SOURCE_BLOCKS[@]}" | sort); do
    echo "$src"
    echo "    blocks:   ${SOURCE_BLOCKS[$src]}"
    echo "    datasets: ${SOURCE_DATASETS[$src]}"
  done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--script)   SCRIPT="$2"; shift 2 ;;
    -o|--source)   SOURCES+=("$2"); shift 2 ;;
    -e|--ebl)      EBL_MODELS+=("$2"); shift 2 ;;
    -d|--dataset)  DATASET_OVERRIDE="$2"; shift 2 ;;
    -b|--block)    BLOCK_OVERRIDE="${2#block}"; shift 2 ;;  # strip leading "block" if given
    -n|--dry-run)  DRY_RUN=true; shift ;;
    -l|--list)     list_sources; exit 0 ;;
    -h|--help)     usage; exit 0 ;;
    -k|--kwargs)   shift; EXTRA_ARGS=("$@"); break ;;
    *)             echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Defaults for source/ebl if nothing was passed
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  SOURCES=("all")
fi
if [[ ${#EBL_MODELS[@]} -eq 0 ]]; then
  EBL_MODELS=("dominguez")
fi

# Expand "all"
if [[ "${SOURCES[*]}" == "all" ]]; then
  SOURCES=($(printf '%s\n' "${!SOURCE_BLOCKS[@]}" | sort))
fi
if [[ "${EBL_MODELS[*]}" == "all" ]]; then
  EBL_MODELS=("${ALL_EBL_MODELS[@]}")
fi

# Validate sources
for src in "${SOURCES[@]}"; do
  if [[ -z "${SOURCE_BLOCKS[$src]:-}" ]]; then
    echo "Unknown source: '$src'. Use --list to see known sources." >&2
    exit 1
  fi
done

# Setup conda environment
if ! $DRY_RUN; then
  # conda activate needs conda's shell hook sourced first when run non-interactively
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  cd "$SCRIPT_DIR"
fi

# Run script for all sources, time blocks segmentation, ebl models
for src in "${SOURCES[@]}"; do
  echo "$src"

  # blocks: override if given, else whatever this source has
  if [[ -n "$BLOCK_OVERRIDE" ]]; then
    blocks="$BLOCK_OVERRIDE"
  else
    n_blocks="${SOURCE_BLOCKS[$src]}"
    blocks=$(seq 1 "$n_blocks")
  fi

  # datasets: override if given (and warn if the source doesn't actually have it),
  # else whatever this source has
  if [[ -n "$DATASET_OVERRIDE" ]]; then
    if [[ "$DATASET_OVERRIDE" == "all" ]]; then
      datasets="${SOURCE_DATASETS[$src]}"
    else
      if [[ " ${SOURCE_DATASETS[$src]} " != *" $DATASET_OVERRIDE "* ]]; then
        echo "  [skip] $src has no '$DATASET_OVERRIDE' dataset (has: ${SOURCE_DATASETS[$src]})"
        continue
      fi
      datasets="$DATASET_OVERRIDE"
    fi
  else
    datasets="${SOURCE_DATASETS[$src]}"
  fi
  # ebl models: override if given
  for ebl in "${EBL_MODELS[@]}"; do
    for b in $blocks; do
      bblock="block${b}-${ebl}"
      for dataset in $datasets; do
        cmd=(python "$SCRIPT" --source "$src" --bblock "$bblock" --ebl "$ebl" \
             --plots-only --dataset "$dataset" "${EXTRA_ARGS[@]}")
        if $DRY_RUN; then
          echo "  [dry-run] ${cmd[*]}"
        else
          echo "  ${cmd[*]}"
          "${cmd[@]}"
        fi
      done
    done
  done
done

echo -e "\033[0;31mDone! :)\033[0m"