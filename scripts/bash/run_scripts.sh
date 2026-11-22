#!/bin/bash
echo "Running under bash $BASH_VERSION"

# Set errors!
set -euo pipefail

# Get directory of this script
BASH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get directory of analysis scripts
SCRIPT_ROOT="${SCRIPT_ROOT:-$BASH_ROOT/../analysis}"
# Get directory for log files
LOG_ROOT="${LOG_ROOT:-$BASH_ROOT/logs}"

# Define conda environment to use
CONDA_ENV="${CONDA_ENV:-alps-upturns}"

# Define list of considered EBL models
EBL_MODELS=(dominguez franceschini finke2022 saldana-lopez)


# Define functions


error()    { echo "Error: $*" >&2; exit 2; }
warn()     { echo "Warning: $*" >&2; }
is_uint()  { [[ "$1" =~ ^[0-9]+$ ]]; }
need_val() { (( $2 >= 2 )) || error "Option $1 requires a value"; }
add_csv()  {
  local -n _arr=$1
  local -a _parts
  IFS=',' read -ra _parts <<< "$2"
  _arr+=("${_parts[@]}")
}
has_all()  {
  local x
  for x in "$@"; do [[ "${x,,}" == "all" ]] && return 0; done
  return 1
}

# Display usage


usage() {
  cat << EOF
  Sample text!
EOF
  exit 1
}


# Parse arguments


# Main configs
SCRIPT=""
SOURCES=()
EBLS=()
BBLOCKS=()
SIM_IDS=""

# Just list the sources and blocks available
LIST_ONLY=false

# Parallel job config
NJOBS=1
THREADS=1
DRY_RUN=false
RESUME_MODE=""

# Additional arguments
KWARGS=()

# Read all arguments
while (( $# )); do
  case "$1" in
    # Script to eXecute
    -x|--script)	need_val "$1" $#; SCRIPT="$2"; shift 2 ;;
    # Common parameters in all scripts
    -s|--source)	need_val "$1" $#; add_csv SOURCES "$2"; shift 2 ;;
    -e|--ebl) 		need_val "$1" $#; add_csv EBLS "$2"; shift 2 ;;
    -b|--bblock)	need_val "$1" $#; add_csv BBLOCKS "$2"; shift 2 ;;
    # Monte Carlo simulations
    -i|--sim-ids)	need_val "$1" $#; SIM_IDS="$2"; shift 2 ;;
    # Parallelization parameters
    -j|--njobs)		need_val "$1" $#; NJOBS="$2"; shift 2 ;;
    -t|--threads)       need_val "$1" $#; THREADS="$2"; shift 2 ;;
    -d|--dry-run)       DRY_RUN=true; shift ;;
    -r|--resume)        RESUME_MODE="resume"; shift ;;
    -R|--resume-failed) RESUME_MODE="resume-failed"; shift ;;
    # Utilities and other arguments
    -l|--list)		LIST_ONLY=true; shift ;;
    -h|--help)		usage; exit 0 ;;
    --kwargs)		shift; KWARGS=("$@"); break ;;
    -*)			error "Unknown option $1" ;;
    *)			[[ -z "$SCRIPT" ]] || error "Unexpected argument: $1"
	    		SCRIPT="$1"; shift ;;
  esac
done


# Check NJOBS and THREADS are integers!
is_uint "$NJOBS" && (( NJOBS >= 1 )) || error "NJOBS must be a positive integer!"
is_uint "$THREADS" && (( THREADS >= 1 )) || error "THREADS must be a positive integer!"


# Check scripts and directory
[[ -f "$SCRIPT_ROOT/$SCRIPT" ]] || error "Script $SCRIPT_ROOT/$SCRIPT not found!"
[[ -d "$SCRIPT_ROOT" ]] || error "Script analysis directory not found!"


# Setup conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Conda environment activated: $CONDA_ENV"
set -u


# Get list of all sources and blocks (only if source and blocks not given)
mapfile -t ALL_SOURCES < <(python -c "from alpsup.utils import get_source_list; print('\n'.join(get_source_list()))")
mapfile -t NBLOCKS < <(python -c "from alpsup.utils import get_source_nblocks; print('\n'.join(map(str, get_source_nblocks())))")

# Ensure source list and block correct
(( ${#ALL_SOURCES[@]} > 0 )) || error "Could not load the source list from alpsup.utils"
(( ${#ALL_SOURCES[@]} == ${#NBLOCKS[@]} )) || error "Source list and block counts differ in length"


# Define source and block list
declare -A SOURCE_BLOCKS
for i in "${!ALL_SOURCES[@]}"; do
  SOURCE_BLOCKS["${ALL_SOURCES[$i]}"]="${NBLOCKS[$i]}"
done


# List all sources and number of blocks
if $LIST_ONLY; then
  for src in "${ALL_SOURCES[@]}"; do
    printf '%24s blocks: %s\n' "$src" "${SOURCE_BLOCKS[$src]}"
  done
  exit 0
fi


# Expand and validate selections


# Expand sources list
if (( ${#SOURCES[@]} == 0 )) || has_all "${SOURCES[@]}"; then SOURCES=("${ALL_SOURCES[@]}"); fi
# Expand EBL list
if (( ${#EBLS[@]} == 0 )); then EBLS=(dominguez); fi
if has_all "${EBLS[@]}"; then EBLS=("${EBL_MODELS[@]}"); fi

# Check given sources
for src in "${SOURCES[@]}"; do
  [[ -n "${SOURCE_BLOCKS[$src]:-}" ]] || error "Unkown source '$src'. Use --list to see known sources."
done

# Check given blocks
for i in "${!BBLOCKS[@]}"; do
  BBLOCKS[$i]="${BBLOCKS[$i]#block}"
  is_uint "${BBLOCKS[$i]}" || error "Invalid block '${BBLOCKS[$i]}'"
done

# Expand simulation ids
SIM_LIST=("")
if [[ -n "$SIM_IDS" ]]; then
  if [[ "$SIM_IDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    mapfile -t SIM_LIST < <(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
  elif is_uint "$SIM_IDS"; then
    SIM_LIST=("$SIM_IDS")
  else
    error "--sim-ids must be an integer N or an integer range A-B"
  fi
fi


# Setup log directories
LOG_DIR="$LOG_ROOT/${SCRIPT%.py}"
JOBFILE="$LOG_DIR/jobs.txt"
JOBLOG="$LOG_DIR/joblog.tsv"
JOBFAILED="$LOG_DIR/failed_runs.txt"


# Define joblist
JOBS=()
# Loop over all sources (or the sources given)
for src in "${SOURCES[@]}"; do
  nb="${SOURCE_BLOCKS[$src]}"
  if (( ${#BBLOCKS[@]} )); then
    blocks=("${BBLOCKS[@]}")
  else
    mapfile -t blocks < <(seq 1 "$nb")
  fi
  # Loop over all EBL models (or those given)
  for ebl in "${EBLS[@]}"; do
    for b in "${blocks[@]}"; do
      if (( b < 1 || b > nb )); then
        warn "$src has blocks 1..$nb, skipping block$b"
	continue
      fi
      # Loop over all simulations
      for sim in "${SIM_LIST[@]}"; do
        tag="${src}_block${b}_${ebl}"
	cmd=(env "OMP_NUM_THREADS=$THREADS" "OPENBLAS_NUM_THREADS=$THREADS" python "$SCRIPT" --source "$src" --bblock "block$b" --ebl "$ebl")
	if [[ -n "$sim" ]]; then
          cmd+=(--sim-id "$sim")
	  tag+="_sim${sim}"
	fi
	cmd+=("${KWARGS[@]}")
	line="$(printf '%q ' "${cmd[@]}")> $(printf '%q' "$LOG_DIR/$tag.log") 2>&1"
	JOBS+=("$line")
      done
    done
  done
done


# Check if any jobs are given!
(( ${#JOBS[@]} > 0 )) || error "No jobs to run!"


# Dry run
if $DRY_RUN; then
  echo "# cd $SCRIPT_ROOT"
  printf '%s\n' "${JOBS[@]}"
  echo "# ${#JOBS[@]} job(s), $NJOBS in parallel, logs in $LOG_DIR"
  exit 0
fi


# Run jobs


command -v parallel > /dev/null || error "GNU parallel not found!"
mkdir -p "$LOG_DIR"
printf '%s\n' "${JOBS[@]}" > "$JOBFILE"
cd "$SCRIPT_ROOT"

export PARALLEL_SHELL=bash

par_opts=(--jobs "$NJOBS" --joblog "$JOBLOG" --bar)
[[ -n "$RESUME_MODE" ]] && par_opts+=("--$RESUME_MODE")


echo "Running ${#JOBS[@]} job(s) of $SCRIPT, $NJOBS in parallel (logs: $LOG_DIR)"


rc=0
parallel "${par_opts[@]}" < "$JOBFILE" || rc=$?

# Check for failed runs!
awk -F'\t' 'NR > 1 { ex[$9] = $7 } END { for (c in ex) if (ex[c] != 0) print c }' "$JOBLOG" | sort > "$JOBFAILED"
nfail=$(wc -l < "$JOBFAILED" | tr -d ' ')

if (( nfail > 0 )); then
  echo -e "\033[0;31m${nfail} job(s) failed\033[0m. Commands in $JOBFAILED"
  echo "Retry them with the same command plus --resume-failed"
  exit 1
fi

(( rc == 0 )) || { echo "parallel exited with status $rc" >&2; exit "$rc"; }
echo -e "\033[0;32mDone! :)\033[0m"


