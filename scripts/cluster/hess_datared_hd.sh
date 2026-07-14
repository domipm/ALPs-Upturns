#!/opt/homebrew/bin/bash

# HESS Data Reduction Script for All Sources!

set -e # Exit on error
set -u # Exit on undefined

# Function to print status message
print_step() {
	echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
# Function to print error message
print_error() {
	echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;31mERROR!\033[0m $1" >&2
}

# Function to display usage
usage() {
    cat << EOF
Run H.E.S.S. data reduction on all sources via hess_datared_hd.py

Usage: $0 [OPTIONS]

Options:
    --sources SOURCE1 [SOURCE2 ...]    Run only specified source(s) (optional)
                                       If not provided, runs all sources
    --help                             Display this help message

Examples:
    # Run all sources
    $0
    
    # Run single or multiple specific sources
    $0 --sources 3C279 PKS2155-304 H2356-309

EOF
    exit 1
}

# Parse command line arguments
OVERRIDE_SOURCES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --sources)
            shift
            # Collect all arguments until we hit another flag or end
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                OVERRIDE_SOURCES+=("$1")
                shift
            done
            ;;
        --help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Source Python environment
# TODO: FIX THIS FOR CLUSTER VERSUS LOCAL
source ./venv/bin/activate

# Define array of all sources names
all_sources=("1ES0229+200" "1ES0347-121" "1ES0414+009" "1ES1101-232" "1ES1312-423" "1RXSJ101015.9-311909" "3C279" "GRB180720B" "H2356-309" "PG1553+113" "PKS0346-27" "PKS0447-439" "PKS0903-57" "PKS1510-089" "PKS2155-304")

# Use override sources if provided, otherwise use all sources
if [ ${#OVERRIDE_SOURCES[@]} -gt 0 ]; then
    sources=("${OVERRIDE_SOURCES[@]}")
    print_step "Running for ${#sources[@]} specified source(s): ${sources[*]}"
else
    sources=("${all_sources[@]}")
    print_step "Running for all ${#sources[@]} sources"
fi

# Declare arrays for datasets, configs, etc
declare -A datasets
declare -A configs
declare -A extra_flags

# Fill with default options (for original sources)
for src in "${sources[@]}"; do
    # Use HAP-HD by default
    datasets[$src]="HAP-HD"
    # Use updated std_ImPACT config
    configs[$src]="std_ImPACT_fullEnclosure_updated"
    # Always include HESS-FC with std_ImPACT_3tel config
    # Also plot significance maps (included in this dataset)
    extra_flags[$src]="--include-hybrid --include-hessfc --plot-map"
done

# Set dataset argument (for new sources)
datasets["3C279"]="HAP-FITS"
datasets["GRB180720B"]="HAP-FITS"
datasets["PKS0346-27"]="HAP-FITS"
datasets["PKS0903-57"]="HAP-HD"
# Set config arguments (for new sources)
configs["3C279"]="loose_ImPACT_fullEnclosure"
configs["GRB180720B"]="loose_ImPACT_mono_fullEnclosure"
configs["PKS0346-27"]="loose_ImPACT_version36_fullEnclosure"
configs["PKS0903-57"]="std_ImPACT_3tel_FullEnclosure"
# Set addiitonal flags (for new sources)
extra_flags["3C279"]="--include-hessfc"
extra_flags["GRB180720B"]="--include-hessfc"
extra_flags["PKS0346-27"]="--include-hessfc"
extra_flags["PKS0903-57"]="--include-hessfc"

# Execute data reduction script for each source, producing theta squared and map plots when possible
for src in ${sources[@]}; do
	print_step "Executing data reduction of ${src}"
	print_step "Using dataset ${datasets[$src]} with config ${configs[${src}]}"
	print_step "Including extra flags: ${extra_flags[$src]}"

	# Create directory if it doesn't exist
	mkdir -p "../${src}/logs"

	python -u ./hess_datared_hd.py --source ${src} --dataset ${datasets[${src}]} --config ${configs[${src}]} ${extra_flags[${src}]} 2>&1 | tee ../${src}/logs/hess_datared_sh.log || print_error "Data reduction for ${src} failed! Skipping."
	print_step "Data reduction of ${src} done!"
done

# Execute analysis script with these configurations (using simple power law with no ebl)
for src in ${sources[@]}; do
	print_step "Executing analysis of ${src}"
	python -u ./hess_analysis_hd.py --source ${src} --dataconf ${datasets[${src}]} ${configs[${src}]} --model PowerLaw --ebl None 2>&1 | tee ../${src}/logs/hess_analysis_sh.log || print_error "Analysis for ${src} failed! Skipping."
	print_step "Initial analysis of ${src} done!"
done

# Print final info message
print_step "\033[0;32mData reduction of sources done! :~)\033[0m"

# Exit with success value
exit 0

