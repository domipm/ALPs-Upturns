#!/opt/homebrew/bin/bash
echo "Running under bash $BASH_VERSION"

set -e
set -u

declare -a ALL_SOURCES
ALL_SOURCES=$(python -c "from alpsup.utils import get_source_list; print('\n'.join(get_source_list()))")

declare -A SOURCE_BLOCKS
for src in "${ALL_SOURCES[@]}"; do
    SOURCE_BLOCKS[$src]="1"
done

CONDA_ENV="alps-upturns"
SCRIPT_DIR="scripts/analysis/"
SCRIPT="temporal_analysis.py"
SOURCES=()
DRY_RUN=false
PLOTS_ONLY=false
DATASET_OVERRIDE=""
CONFIG_OVERRIDE=""
EXTRA_ARGS=()

usage() {
  cat << EOF
Parameterized runner for analysis scripts

Usage:
  ./run_scripts.sh [options] [-- extra kwargs passed straight to the python script]

Options:
  -o, --source SOURCE       Source name, or "all" (default: all). Repeatable.
  -d, --dataset DATASET     Override HAP dataset for all selected sources (default: per-source from hess_config.yaml)
  -c, --config CONFIG       Override HAP config for all selected sources (default: per-source from hess_config.yaml)
  -p, --plots-only          Only regenerate plots from existing output, don't rerun the analysis.
  -n, --dry-run             Print the commands instead of running them.
  -l, --list                List known sources and exit.
  -h, --help                Show this help.
  -k, --kwargs               Keyword arguments for selected script.
EOF
  exit 1
}

list_sources() {
  local src
  for src in $(printf '%s\n' "${!SOURCE_BLOCKS[@]}" | sort); do
    echo "$src"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--source)       SOURCES+=("$2"); shift 2 ;;
    -d|--dataset)       DATASET_OVERRIDE="$2"; shift 2 ;;
    -c|--config)        CONFIG_OVERRIDE="$2"; shift 2 ;;
    -p|--plots-only)    PLOTS_ONLY=true; shift ;;
    -n|--dry-run)       DRY_RUN=true; shift ;;
    -l|--list)          list_sources; exit 0 ;;
    -h|--help)          usage; exit 0 ;;
    -k|--kwargs)        shift; EXTRA_ARGS=("$@"); break ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  SOURCES=("all")
fi

if [[ "${SOURCES[*]}" == "all" ]]; then
  SOURCES=($(printf '%s\n' "${!SOURCE_BLOCKS[@]}" | sort))
fi

if ! $DRY_RUN; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  cd "$SCRIPT_DIR"
fi

for src in "${SOURCES[@]}"; do
  echo "$src"

  # Look up per-source dataset/config from hess_config.yaml, unless overridden on the command line
  if [[ -n "$DATASET_OVERRIDE" || -n "$CONFIG_OVERRIDE" ]]; then
    DATASET="${DATASET_OVERRIDE:-HAP-HD}"
    CONFIG="${CONFIG_OVERRIDE:-std_ImPACT_hybrid_fullEnclosure_updated}"
  else
    read -r DATASET CONFIG <<< "$(python -c "
from alpsup.utils import get_hess_config
d, c = get_hess_config('$src')
print(d, c)
")"
  fi

  cmd=(python "$SCRIPT" --source "$src" --dataset "$DATASET" --config "$CONFIG")
  if $PLOTS_ONLY; then
    cmd+=(--plots-only)
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  if $DRY_RUN; then
    echo "  [dry-run] ${cmd[*]}"
  else
    echo "  ${cmd[*]}"
    "${cmd[@]}"
  fi
done

echo -e "\033[0;32mDone! :)\033[0m"