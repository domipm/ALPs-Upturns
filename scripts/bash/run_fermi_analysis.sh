#!/opt/homebrew/bin/bash
echo "Running under bash $BASH_VERSION"

set -e

ALL_SOURCES=($(python -c "from alpsup.utils import get_fermi_source_list; print('\n'.join(get_fermi_source_list()))"))

declare -A SOURCE_BLOCKS
for src in "${ALL_SOURCES[@]}"; do
    SOURCE_BLOCKS[$src]="1"
done

CONDA_ENV="alps-upturns-fermipy"
SCRIPT_DIR="scripts/"
SCRIPT="fermi_analysis.py"
SOURCES=()
BLOCK_OVERRIDE=""
DRY_RUN=false
PLOTS_ONLY=false
EXTRA_ARGS=()

usage() {
  cat << EOF
Parameterized runner for the Fermi-LAT analysis script

Usage:
  ./run_fermi_analysis.sh [options] [-- extra kwargs passed straight to the python script]

Options:
  -s, --source SOURCE       Source name, or "all" (default: all, Fermi-LAT sources only). Repeatable.
  -b, --block BLOCK         Run only this block (default: all blocks for each source)
  -p, --plots-only          Only regenerate plots from existing output, don't rerun the analysis.
  -n, --dry-run             Print the commands instead of running them.
  -l, --list                List known (Fermi-LAT) sources and exit.
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
    -s|--source)       SOURCES+=("$2"); shift 2 ;;
    -b|--block)         BLOCK_OVERRIDE="$2"; shift 2 ;;
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

for src in "${SOURCES[@]}"; do
  if [[ -z "${SOURCE_BLOCKS[$src]:-}" ]]; then
    echo "Unknown or non-Fermi source: '$src'. Use --list to see known sources." >&2
    exit 1
  fi
done

if ! $DRY_RUN; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  cd "$SCRIPT_DIR"
fi

for src in "${SOURCES[@]}"; do
  echo "$src"

  if [[ -n "$BLOCK_OVERRIDE" ]]; then
    BLOCKS=("$BLOCK_OVERRIDE")
  else
    BLOCKS=($(python -c "
from alpsup.utils import get_source_blocks
print('\n'.join(get_source_blocks('$src')))
"))
  fi

  if [[ ${#BLOCKS[@]} -eq 0 ]]; then
   echo "  No blocks found for $src — run HESS lightcurve first?" >&2; continue; 
  fi

  for block in "${BLOCKS[@]}"; do
    echo "  $block"
    cmd=(python "$SCRIPT" --source "$src" --bblock "$block")
    if $PLOTS_ONLY; then
      cmd+=(--plots-only)
    fi
    cmd+=("${EXTRA_ARGS[@]}")

    if $DRY_RUN; then
      echo "    [dry-run] ${cmd[*]}"
    else
      echo "    ${cmd[*]}"
      "${cmd[@]}"
    fi
  done
done

echo -e "\033[0;32mDone! :)\033[0m"