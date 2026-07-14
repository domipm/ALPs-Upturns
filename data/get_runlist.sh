#!/bin/bash

# Generate Runlist Script
# Extracts run numbers from obs_*.fits.gz files and creates a runlist

set -e  # Exit on error
set -u  # Exit on undefined variable

# Function to print status message
print_step() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to print error message
print_error() {
    RED='\033[0;31m'
    NC='\033[0m'
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}ERROR!${NC} $1" >&2
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --target TARGET         Target name (required)
    --dataset DATASET       Dataset type (required): HAP-HD, HAP-FR, or HAP-FITS
    --config CONFIG         Configuration name (optional for HAP-FR): std_ImPACT_fullEnclosure_updated
    --help                  Display this help message

Examples:
    $0 --target 1ES0347-121 --dataset HAP-HD --config std_ImPACT_fullEnclosure

Output:
    Creates runlist_{dataset}_{config}.txt in ./hess-data/{target}/ directory
    
EOF
    exit 1
}

# Parse command line arguments
TARGET=""
DATASET=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
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

# Validate required arguments
if [[ -z "$TARGET" ]] || [[ -z "$DATASET" ]]; then
    print_error "Both --target and --dataset are required"
    usage
fi

# Validate dataset type
if [[ ! "$DATASET" =~ ^(HAP-HD|HAP-FR|HAP-FITS)$ ]]; then
    print_error "Dataset must be HAP-HD, HAP-FR, or HAP-FITS"
    exit 1
fi

# Construct data directory path and runlist filename
if [[ -n "$CONFIG" ]]; then
    DATA_DIR="./${TARGET}/${DATASET}-${CONFIG}"
    RUNLIST_NAME="runlist_${DATASET}_${CONFIG}.txt"
else
    DATA_DIR="./${TARGET}/${DATASET}"
    RUNLIST_NAME="runlist_${DATASET}.txt"
fi

# Output file path (in target directory)
OUTPUT_FILE="./${TARGET}/${RUNLIST_NAME}"

# Original runlist path
ORIGINAL_RUNLIST="./${TARGET}/runlist.txt"

# Define log path file
LOG_FILE="./${TARGET}/get_runlist.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "*** ${TARGET} Runlist Check: $(date '+%Y-%m-%d %H:%M:%S') ***"
echo ""

print_step "Generating runlist for ${TARGET}:"
echo "  Dataset: ${DATASET}"
if [[ -n "$CONFIG" ]]; then
    echo "  Config: ${CONFIG}"
fi
echo ""

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Find all obs_*.fits.gz files and extract run numbers
print_step "Searching for obs_*.fits.gz files in ${DATA_DIR}..."

# Use find to get all matching files, extract run numbers, sort and save
find "$DATA_DIR" -name "obs_*.fits.gz" -type f | \
    sed 's|.*/obs_||; s|\.fits\.gz$||' | \
    sort -n > "$OUTPUT_FILE"

# Check if any files were found
if [[ ! -s "$OUTPUT_FILE" ]]; then
    print_error "No obs_*.fits.gz files found in ${DATA_DIR}"
    rm -f "$OUTPUT_FILE"
    exit 1
fi

# Count runs
NUM_RUNS=$(wc -l < "$OUTPUT_FILE")

echo ""
echo "  Found ${NUM_RUNS} observation file(s)"
echo ""

# Success message
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
print_step "${GREEN}Runlist created successfully!${NC}"
echo ""

# Compare with original runlist if it exists
if [[ -f "$ORIGINAL_RUNLIST" ]]; then
    print_step "Comparing with original runlist..."
    
    # Count runs in original
    # ORIGINAL_COUNT=$(wc -l < "$ORIGINAL_RUNLIST")
    ORIGINAL_COUNT=$(grep -c . "$ORIGINAL_RUNLIST" || wc -l < "$ORIGINAL_RUNLIST")
    echo "  Original runlist: ${ORIGINAL_COUNT} runs"
    echo "  Generated runlist: ${NUM_RUNS} runs"
    echo ""
    
    # Find missing runs (in original but not in generated)
    MISSING_RUNS=$(comm -23 <(sort -n "$ORIGINAL_RUNLIST") <(sort -n "$OUTPUT_FILE"))
    NUM_MISSING=$(echo "$MISSING_RUNS" | grep -c . || echo "0")
    
    if [ "$NUM_MISSING" -gt 0 ]; then
        echo -e "${YELLOW}Warning: ${NUM_MISSING} run(s) from original runlist are missing:${NC}"
        echo "$MISSING_RUNS" | sed 's/^/  /'
        echo ""
        
    else
        echo -e "${GREEN}✓ All runs from original runlist are present!${NC}"
    fi
    echo ""
    
    # Find extra runs (in generated but not in original)
    EXTRA_RUNS=$(comm -13 <(sort -n "$ORIGINAL_RUNLIST") <(sort -n "$OUTPUT_FILE"))
    
    # Count extra runs properly
    if [[ -z "$EXTRA_RUNS" ]]; then
        NUM_EXTRA=0
    else
        NUM_EXTRA=$(echo "$EXTRA_RUNS" | wc -l)
    fi
    
    if [ "$NUM_EXTRA" -gt 0 ]; then
        echo -e "${YELLOW}Warning: ${NUM_EXTRA} additional run(s) found that are not in original runlist:${NC}"
        echo "$EXTRA_RUNS" | sed 's/^/  /'
        echo ""
    fi
    
    # Summary
    print_step "${GREEN}Comparison Summary:${NC}"
    echo "  Original runlist: ${ORIGINAL_COUNT} runs"
    echo "  Generated runlist: ${NUM_RUNS} runs"
    echo "  Missing runs: ${NUM_MISSING}"
    echo "  Extra runs: ${NUM_EXTRA}"
    
else
    echo -e "${YELLOW}Note: Original runlist (${ORIGINAL_RUNLIST}) not found - skipping comparison${NC}"
fi