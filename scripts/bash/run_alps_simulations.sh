#!/opt/homebrew/bin/bash
echo "Running under bash $BASH_VERSION"

# set -e # Exit on error

# Define array of all sources names
readarray -t ALL_SOURCES < <(python -c "from alpsup.utils import get_source_list; print('\n'.join(get_source_list()))")
readarray -t NBLOCKS    < <(python -c "from alpsup.utils import get_source_nblocks; print('\n'.join(map(str, get_source_nblocks())))")

declare -A SOURCE_BLOCKS
for i in "${!ALL_SOURCES[@]}"; do
    SOURCE_BLOCKS["${ALL_SOURCES[$i]}"]="${NBLOCKS[$i]}"
done

# List of EBL models considered (default "ALL")
ALL_EBL_MODELS=("dominguez" "franceschini" "finke2022" "saldana-lopez")

# Name of conda environment
CONDA_ENV="alps-upturns"
# Directory location of scripts
SCRIPT_DIR="scripts/analysis"

# Default options
SCRIPT="alps_simulations.py"
SOURCES=()
EBL_MODELS=()
DATASET_OVERRIDE=""
BLOCK_OVERRIDE=""
DRY_RUN=false
EXTRA_ARGS=()

# Display usage
usage() {
  cat << EOF
Parameterized runner for spectral analysis script

Usage:
  ./run_scripts.sh [options] [-- extra kwargs passed straight to the python script]

Options:
  -o, --source SOURCE       Source name, or "ALL" (default: ALL). Repeatable.
  -e, --ebl EBL             EBL model, or "ALL" (default: dominguez). Repeatable.
  -b, --block BLOCK         Block override, e.g. "block2". Default: all blocks
                            defined for that source.
  -n, --dry-run             Print the commands instead of running them.
  -l, --list                List known sources (and their blocks/datasets) and exit.
  -h, --help                Show this help.
  -k, --kwargs              Keyword arguments for selected script.

Anything after "--kwargs" is appended verbatim as extra kwargs to every python call,
  e.g.:  ./run_spectral_analysis.sh -s 3C279 -e dominguez --kwargs --plots-only

EOF
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source)      SOURCES+=("$2"); shift 2 ;;
    -e|--ebl)         EBL_MODELS+=("$2"); shift 2 ;;
    -d|--dataset)     DATASET_OVERRIDE="$2"; shift 2 ;;
    -b|--block)       BLOCK_OVERRIDE="${2#block}"; shift 2 ;;  # strip leading "block" if given
    -n|--dry-run)     DRY_RUN=true; shift ;;
    -p|--plots-only)  PLOTS_ONLY=false; shift;;
    -h|--help)        usage; exit 0 ;;
    -k|--kwargs)      shift; EXTRA_ARGS=("$@"); break ;;
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

  # ebl models: override if given
  for ebl in "${EBL_MODELS[@]}"; do
    cmd=(python "$SCRIPT" --source "$src" --ebl "$ebl" "${EXTRA_ARGS[@]}")
    if $DRY_RUN; then
      echo "  [dry-run] ${cmd[*]}"
    else
      echo "  ${cmd[*]}"
      "${cmd[@]}"
    fi
  done
done

echo -e "\033[0;32mDone! :)\033[0m"