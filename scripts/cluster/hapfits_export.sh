#!/bin/bash

# HAP FITS Export Script
# Automates HESS data export with configurable parameters
# Based on HAP Tools `hapfits_export.py` script (modified)
# NOTE: Define paths based on local configuration!

set -e # Exit on error
set -u # Exit on undefined

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

# Configuration variables
ENV_FILE="${ENV_FILE:-/path/to/env.sh}"
# HAP Config variable
HAP_CONFIG="${HAP_CONFIG:-std_ImPACT_fullEnclosure}"
# Runlist filename
RUNLIST="${RUNLIST:-runlist.txt}"

# Placeholder directories, replace!

# Base directory for configs
BASE_CONFIG_DIR="${BASE_CONFIG_DIR:-/path/to/dir}"
# Base directory for data
BASE_DATA_DIR="${BASE_DATA_DIR:-/path/to/dir}"
# Base directory for dsts
BASE_DST_DIR="${BASE_DST_DIR:-/path/to/dir}"

# Function to display usage of the script
usage() {

	cat << EOF
Run H.E.S.S. data export / reduction via HAP Tools

Usage: $0 [OPTIONS]

Options:
	--target TARGET			Target name (required)
	--era HESS ERA			Data era (required):
					- hess1 (18000 - 80000)
					- hess2 (80000 - 124680)
					- hess1u (127700 - 154814)
					- hessfc (> 154814)
	--hess-config HESSCONFIG	Dataset version (required):
					- version36_LEGACY      (for RUN ID < 154814 / HESS-FC)
					- version37_PRELIMINARY (for RUN ID > 154814 / HESS-FC)
					- version38_PRELIMINARY (for RUN ID > 154814 / HESS-FC)
	--hess-dst HESSDST		Override HESSDST path from era-based default
	--hess-data HESSDATA		Override HESSDATA path from era-based default
	--runlist RUNLIST		Runlist filename (default: $RUNLIST)
	--hap-config CONFIG		HAP configuration (default: $HAP_CONFIG)
	--env-file FILE			Environment setup file (default: $ENV_FILE)
	--muonentry VALUE		Use local muonentry script with given value (e.g. 302 for version36_LEGACY and HESS-FC data)
	--help				Display this help message
EOF
	exit 1
}

# Parse command line arguments
ERA=""
HESS_CONFIG=""
TARGET=""
OVERRIDE_HESSDATA=""
OVERRIDE_HESSDST=""
MUONENTRY=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--era)
			ERA="$2"
			shift 2
			;;
		--hess-config)
			HESS_CONFIG="$2"
			shift 2
			;;
		--target)
			TARGET="$2"
			shift 2
			;;
		--runlist)
			RUNLIST="$2"
			shift 2
			;;
		--hap-config)
			HAP_CONFIG="$2"
			shift 2
			;;
		--env-file)
			ENV_FILE="$2"
			shift 2
			;;
		--hess-data)
			OVERRIDE_HESSDATA="$2"
			shift 2
			;;
		--hess-dst)
			OVERRIDE_HESSDST="$2"
			shift 2
			;;
		--muonentry)
			MUONENTRY="$2"
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
if [[ -z "$ERA" ]] || [[ -z "$HESS_CONFIG" ]] || [[ -z "$TARGET" ]]; then
	print_error "Options --era, --hess-config, --target are all required"
	usage
fi
# Validate correct HESS era
if [[ ! "$ERA" =~ ^(hess1|hess2|hess1u|hessfc)$ ]]; then
	print_error "Era must be hess1, hess2, hess1u, or hessfc"
	exit 1
fi
# Validate correct analysis directory
if [[ ! "$HESS_CONFIG" =~ ^(version36|version36_LEGACY|version36_PREPRODUCTION|version37_PRELIMINARY|version38_PRELIMINARY)$ ]]; then
	print_error "Configuration directory must be version36, version36_LEGACY, version36_PREPRODUCTION, version37_PRELIMINARY, or version38_PRELIMINARY"
	exit 1
fi

# Display initial information on screen
echo ""
print_step "Starting HESS Data Export Analysis of $TARGET ($ERA era) using $HAP_CONFIG"
if [[ -n "$MUONENTRY" ]]; then
	print_step "Using local muonentry script (muonentry=$MUONENTRY)"
fi
echo ""

# Source environment
print_step "Sourcing environment from $ENV_FILE"
echo ""
if [[ ! -f "$ENV_FILE" ]]; then
	print_error "Environment file not found: $ENV_FILE"
	exit 1
fi
# Temporarily disable unbound variable check
set +u
source "$ENV_FILE" > /dev/null 2>&1
set -u

# Change directory to target folder
cd "$HOME/$TARGET/"

# Set HESSCONFIG env variable
case $HESS_CONFIG in
	version36)
		export HESSCONFIG="$BASE_CONFIG_DIR/version36"
		;;
	version36_LEGACY)
		export HESSCONFIG="$BASE_CONFIG_DIR/version36_LEGACY"
		;;
	version36_PREPRODUCTION)
		export HESSCONFIG="$BASE_CONFIG_DIR/version36_PREPRODUCTION"
		;;
	version37_PRELIMINARY)
		export HESSCONFIG="$BASE_CONFIG_DIR/version37_PRELIMINARY"
		;;
	version38_PRELIMINARY)
		export HESSCONFIG="$BASE_CONFIG_DIR/version38_PRELIMINARY"
		;;
esac
print_step "Using HESSCONFIG path: $HESS_CONFIG"

# Set HESSDATA and HESSDST env variables based on era (unless overridden)
if [[ -n "$OVERRIDE_HESSDATA" ]]; then
	export HESSDATA="$OVERRIDE_HESSDATA"
	print_step "Using custom HESSDATA path: $HESSDATA"
else
	case $ERA in
		hess1)
			export HESSDATA="$BASE_DATA_DIR/hddata_hess1"
			;;
		hess2)
			export HESSDATA="$BASE_DATA_DIR/hddata_hess2"
			;;
		hess1u)
			export HESSDATA="$BASE_DATA_DIR/hddata_hess1u"
			;;
		hessfc)
			export HESSDATA="$BASE_DATA_DIR/hddata_hessfc"
			;;
	esac
	print_step "Using HESSDATA path: $HESSDATA"
fi

if [[ -n "$OVERRIDE_HESSDST" ]]; then
	export HESSDST="$OVERRIDE_HESSDST"
	print_step "Using custom HESSDST path: $HESSDST"
else
	case $ERA in
		hess1)
			export HESSDST="$BASE_DST_DIR/dstdata_hess1"
			;;
		hess2)
			export HESSDST="$BASE_DST_DIR/dst_hess2"
			;;
		hess1u)
			export HESSDST="$BASE_DST_DIR/dst_hess1u"
			;;
		hessfc)
			export HESSDST="$BASE_DST_DIR/dst_hessfc"
			;;
	esac
	print_step "Using HESSDST path: $HESSDST"
fi
echo ""

# Verify HESSROOT is set
if [[ -z "${HESSROOT:-}" ]]; then
	print_error "HESSROOT is not set. Please check your environment file."
	exit 1
fi

BASE_CMD="../scripts/hapfits_export.py"
COMMON_OPTS="--hap-config $HAP_CONFIG --runs-file $RUNLIST"

# Set muon entry flag if provided
MUON_ENTRY_FLAG=""
if [[ -n "$MUONENTRY" ]]; then
	MUON_ENTRY_FLAG="--muon-entry $MUONENTRY"
fi

# Set legacy flags based on dataset version
LEGACY_FLAGS=""
if [[ "$HESS_CONFIG" == "version36_LEGACY" ]] || [[ "$HESS_CONFIG" == "version36" ]]  || [[ "$HESS_CONFIG" == "version36_PREPRODUCTION" ]]  || [[ "$HESS_CONFIG" == "version37_PRELIMINARY" ]]; then
	LEGACY_FLAGS="--use-legacy-irf --use-legacy-training"
	print_step "Using legacy IRF and training flags for $HESS_CONFIG"
else
	print_step "Not using legacy flags for $HESS_CONFIG"
fi
echo ""

# Export events (--overwrite only used with muonentry)
print_step "Exporting events..."
EVENTS_OPTS="$COMMON_OPTS"
if [[ -n "$MUONENTRY" ]]; then
	EVENTS_OPTS="$COMMON_OPTS --overwrite"
fi
if [[ -n "$LEGACY_FLAGS" ]]; then
	$BASE_CMD $EVENTS_OPTS events $LEGACY_FLAGS $MUON_ENTRY_FLAG > events.log 2>&1
else
	$BASE_CMD $EVENTS_OPTS events $MUON_ENTRY_FLAG > events.log 2>&1
fi
echo "  Events exported!"
echo ""

# Check if any fits file produced: events/run*/run*/*.fits
print_step "Verifying events files..."
EVENT_FITS=$(find events/run*/run*/*.fits 2>/dev/null)

if [[ -z "$EVENT_FITS" ]]; then
	print_step "\033[0;31mNo event FITS files found in events/run*/run*/ folder!\033[0m"
	echo ""
	rm -rf events events.log
	exit 1
else
	echo "  Events files verified!"
fi
echo ""

# Export IRFs
print_step "Exporting IRFs..."
if [[ -n "$LEGACY_FLAGS" ]]; then
	$BASE_CMD $COMMON_OPTS irfs --use-legacy-irf $MUON_ENTRY_FLAG > irfs.log 2>&1
else
	$BASE_CMD $COMMON_OPTS irfs $MUON_ENTRY_FLAG > irfs.log 2>&1
fi
echo "  IRFs exported!"
echo ""


# Check if any fits file produced: irfs/run*/run*/*.fits
print_step "Verifying IRFs files..."
IRF_FITS=$(find irfs/run*/run*/*.fits.gz 2>/dev/null)

if [ "$IRF_FITS" -eq 0 ]; then
	print_error "No IRF FITS files were created!"
	exit 1
fi

echo "  IRFs files verified!"
echo ""

# Double-check that IRF files are readable before reformatting
print_step "Verifying IRF files are accessible..."
echo ""

SAMPLE_IRF=$(find irfs/run*/run*/ -name "aeff_*.fits.gz" -type f 2>/dev/null | head -1)
if [[ -n "$SAMPLE_IRF" ]]; then
	# Try to read the file to ensure it's accessible
	if [[ -r "$SAMPLE_IRF" ]]; then
		echo "  Sample IRF file verified: $SAMPLE_IRF"
	else
		print_error "IRF files exist but are not readable. Waiting longer..."
		sleep 10
		sync
	fi
else
	print_error "No IRF files found in irfs/run*/run*/"
	exit 1
fi
echo ""

# Reformatting
print_step "Reformatting files..."
$BASE_CMD $COMMON_OPTS reformat > reformat.log 2>&1
echo "  FITS reformatted!"
echo ""

# Create observation index
print_step "Creating obs-index..."
$BASE_CMD $COMMON_OPTS obs-index > obs-index.log 2>&1
echo "  obs-index created!"
echo ""

# Create HDU index
print_step "Creating hdu-index..."
$BASE_CMD $COMMON_OPTS hdu-index > hdu-index.log 2>&1
echo "  hdu-index created!"
echo ""

# Remove bad runs
print_step "Removing bad runs..."
$BASE_CMD $COMMON_OPTS remove-bad-runs > remove-bad-runs.log 2>&1
echo "  Bad runs removed!"
echo ""

# Success message
print_step "\033[0;32mPipeline completed successfully!\033[0m"
echo ""

# Rename final folder from 'out' to 'out-{hessera}-{config}'
mv ./out ./out-$ERA-$HAP_CONFIG

# Clean-up files
mkdir ./files-$ERA-$HAP_CONFIG
mv *.log events/ irfs/ -t ./files-$ERA-$HAP_CONFIG
