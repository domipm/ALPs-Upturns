# Define the path to all required folders and directories

from    __future__  import  annotations

import  os

from    pathlib     import  Path


# Root folder for the repository
REPO_ROOT = Path(__file__).resolve().parents[2]

# Get directory for HESS and Fermi-LAT data
# Allows for overriding with local environment variables
# Fallback: REPO_ROOT/data/inst-data/

HESS_DATA_DIR = Path(REPO_ROOT / "data" / "hess-data")

FLAT_DATA_DIR = Path(os.environ.get("FLAT_DATA", REPO_ROOT / "data" / "flat-data"))
FLAT_DATA_SPACECRAFT_FILE = Path(FLAT_DATA_DIR / "spacecraft.fits")

# Data directory for GammaPy and FermiPy
FERMIPY_DATA_DIR = Path(os.environ.get("FERMIPY_DATA", REPO_ROOT / "data" / "fermipy-data"))
GAMMAPY_DATA_DIR = Path(os.environ.get("GAMMAPY_DATA", REPO_ROOT / "data" / "gammapy-data"))

# Results folder
RESULTS_DIR = Path(REPO_ROOT / "results")

# Path to sources.yaml file (containing info on all sources)
SOURCES_FILE = Path(REPO_ROOT / "sources" / "sources.yaml")


def gen_dirs(source: str, bblock: str, ebl: str | None = None) -> None:
    """
    Check directory structure for the results folder of a given target
    and generate as needed. Generate also directories for time segmentation blocks.

    Args:
        target (str): Name of target.
        bblock (str): Time segmentation / Bayesian block.
        ebl (str): EBL model.
    """

    # Define subfolders to generate
    subfolders = ["gamma-out", "fermi-out", "plots", "logs"]
    # Generate all subfolders in directory
    for folder in subfolders:
        get_results_dir(source, bblock, ebl, folder).mkdir(parents = True, exist_ok = True)
    
    return


def get_hess_data_dir(source: str,
                      hap_dataset: str | None = None,
                      hap_config: str | None = None, ) -> Path:
    """
    Get directory of HESS data for given source. If not found in `os.environ`, fall-back to `./data/hess-data/`.
    Args:
        Source (str): target source
    Returns:
        $HESS_DATA (pathlib.Path): path to HESS data folder
    """

    # Get directory of HESS data for source
    hess_data_dir = HESS_DATA_DIR / source

    # If hap_dataset and hap_config given, get corresponding subfolder
    if hap_dataset is not None and hap_config is not None:
        hess_data_dir /= f"{hap_dataset.lower()}-{hap_config}"

    return hess_data_dir


def get_flat_data_dir(source: str) -> Path:
    """
    Get directory of Fermi-LAT data for given source. If not found in `os.environ`, fall-back to `./data/flat-data/`.
    Args:
        Source (str): target source.
    Returns:
        $FLAT_DATA (pathlib.Path): path to Fermi-LAT data folder.
    """
    return FLAT_DATA_DIR / source


def get_results_dir(source: str | None = None, 
                    bblock: str | None = None, 
                    ebl: str | None = None, 
                    output: str | None = None) -> Path:
    """
    Get directory of results folder corresponding to given source, time block, and EBL model. 
    If no arguments given, defaults to baseline results directory.
    Args:
        source (str): target source.
        bblock (str): time segmentation block.
        ebl (str): EBL model.
        output (str): Name out output folder to append at the end of the path, if required.
    Returns:
        path (pathlib.Path): path to results directory for given target.
    """

    # Base results directory
    results_path = RESULTS_DIR
    # If source given, move to source subfolder
    if source is not None:
        results_path /= source.upper()
        # If time block given, move to block subfolder
        if bblock is not None:
            results_path /= bblock.lower()
            # If ebl given, move to ebl subfolder
            if ebl is not None:
                results_path /= f"ebl/{ebl.lower()}"
    # Append output suffix
    if output is not None:
        results_path /= output

    # Make sure path exists, if not, create it
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok = True)
        
    # Return final path
    return results_path

