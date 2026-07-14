# Utility functions for FermiPy (standalone, as requires different Python version / compatibility issues otherwise)

import  os
import  yaml
import  glob

import  numpy                   as      np
import  matplotlib.pyplot       as      plt

import  astropy.units           as      u
from    astropy.coordinates     import  SkyCoord
from    astropy.io              import  fits

from    pathlib                 import  Path

from    typing                  import  Optional

from    fermipy.plotting        import  ROIPlotter


# Root folder for the repository
REPO_ROOT = Path(__file__).resolve().parents[1]

# Path to sources.yaml file (containing info on all sources)
SOURCES_FILE = Path(REPO_ROOT / "sources" / "sources.yaml")

# Results folder
RESULTS_DIR = Path(REPO_ROOT / "results")

FLAT_DATA_DIR = Path(REPO_ROOT / "data" / "flat-data")
FERMIPY_DATA_DIR = Path(REPO_ROOT / "data" / "fermipy-data")


def get_source_info(target: str):
    """
    Gather relevant data for a given source from sources.yaml file (as defined by $SOURCES_FILE environment variable).
    Parameters:
        target (str): Name of the target source.
    Returns:
        target_4FGL (str): 4FGL Catalog name of the target source (if available).
        target_position (`astropy.coordinates.SkyCoord`): Position of the target source.
        target_redshift (float): Redshift of the source.
    """

    # Open sources file
    with open(SOURCES_FILE, 'r') as f:

        # Load the yaml file
        data = yaml.full_load(f)

        # Get the position of the source (either from the file or from astropy)
        try:
            target_position = SkyCoord(
                ra = data["sources"][target]["ra"],
                dec = data["sources"][target]["dec"],
                unit = "deg",
                frame = "icrs",)
        except:
            target_position = SkyCoord.from_name(target)

        # Get the redshift of the source
        target_redshift = data["sources"][target]["z"]

        # Get the 4FGL name of source
        target_4FGL = data["sources"][target]["target_4FGL"]

    # Return all parameters
    return target_4FGL, target_position, target_redshift


def parse_kwargs(args):
    """
    Parse all keyword arguments given and return as dictionary.
    Args:
        args (dict): Arguments of any type.
    Returns:
        kwargs (dict): Parsed keyword arguments.
    """

    kwargs = {}
    for kv in args:
        # Obtain key and value
        key, value = kv.split('=', 1)
        # Convert to bool if true or false
        if value.lower() == 'true':
            val = True
        elif value.lower() == 'false':
            val = False
        try:   
            # Convert to int if possible
            val = int(value)
        except ValueError:
            # Convert to float if possible
            try:
                val = float(value)
            # Otherwise, leave as is
            except ValueError:
                val = value
        # Set parsed value to key
        kwargs[key] = val

    # Return dictionary of parsed arguments
    return kwargs


def gen_config(target, target_position, model = None, bblock = "baseline", **kwargs):
    """
    Generate fermi_config.yaml file for given target source with the relevant parameters.

    Args:
        target (str): Name of target source.
        model (str): Model to override default 4FGL model. Choices: 'PowerLaw', 'LogParabola'. Default: None (use 4FGL catalog model or, if unavailable, PowerLaw).
        **kwargs: Additional arguments to modify in the config file.
    """

    # Read default template config yaml file
    with open(REPO_ROOT / "configs/fermi_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Update the fields based on parameters (keep rest default)

    # File IO - set correct output folders (absolute path)
    config["fileio"]["outdir"] = str( RESULTS_DIR.resolve() / f"{target}/{bblock}/fermi-out/" )
    config["fileio"]["logfile"] = str( RESULTS_DIR.resolve() / f"{target}/{bblock}/fermi-out/fermi.log" )

    # Data - set correct events and spacecraft file paths
    config["data"]["evfile"] = str( FLAT_DATA_DIR.resolve() / f"{target}/events_list.txt" )
    # NOTE: Use single, global spacecraft file (make sure it covers full observational period)
    # config["data"]["scfile"] = f"$FLAT_DATA/{target}/spacecraft.fits"
    config["data"]["scfile"] = str( FLAT_DATA_DIR.resolve() / f"spacecraft.fits" )
    # Selection - set correct position of the source
    config["selection"]["ra"] = float( target_position.ra.value )
    config["selection"]["dec"] = float( target_position.dec.value )

    # FermiPy data directory
    config["model"]["extdir"] = str( FERMIPY_DATA_DIR.resolve() )
    config["model"]["isodiff"] = "iso_P8R3_SOURCE_V3_v1.txt"

    # If spectral model given, replace or add it
    if model != None:
        config["model"]["sources"] = [ {
            "name": target,
            "ra": float( target_position.ra.value ),
            "dec": float( target_position.dec.value ),
            "SpectrumType": model, } ]

    # Go over each additional kwarg given and update its value (leave default otherwise)
    for kkey, value in kwargs.items():
        # Loop over all keys in the config file
        for ckey in config:
            # Check if kwarg key is there
            if config[ckey].get(kkey) != None:
                # Set the value of the key to the kwarg value given
                config[ckey][kkey] = value

    # Define and create output directory if doesn't exist
    os.makedirs(name = RESULTS_DIR.resolve() / f"{target}/{bblock}/",
                exist_ok = True)

    # Save modified config template to directory
    with open(RESULTS_DIR.resolve() / f"{target}/{bblock}/fermi_config.yaml", 'w') as f:
        yaml.safe_dump(config, f, sort_keys = False, default_flow_style = False)

    # Return the config dictionary with all the parameters used
    return config


def get_fpul(y, y_ul, y_errn, y_errp, ts, ts_threshold = 4.0, ul_err = 0.2):
    """
    Auxiliary function to get flux points and upper limits values, including their errors
    and maintaining units.
    Parameters:
        y (float, `np.array`, or `astropy.Quantity`): main array of values to process.
        y_ul (float, `np.array`, or `astropy.Quantity`): array of upper limit point values.
        y_errn (float, `np.array` or `astropy.Quantity`): array of negative error of values.
        y_errp (float, `np.array` or `astropy.Quantity`): array of positive error of values.
        ts (float, `np.array` or `astropy.Quantity`): test statistic associated to each point.
        ts_threshold (float): threshold of test statistic at which point considered upper limit.
        ul_err (float): multiplier to assign for each upper limit value, such that final y_ul *= ul_err.
    Returns:
        y_val (`np.array` of `astropy.Quantity` of float): final values for each point.
        y_err (`np.array` of `astropy.Quantity` of float): final error values (positive and negaitve) for each point.
        is_ul ((`np.array` of bool): boolean whether each point is upper limit according to threshold.
    """

    # Generate arrays containing flux points and upper limits
    is_ul = ts < ts_threshold
    y_val = np.where(is_ul, y_ul, y)
    y_err = np.array([ 
        np.where(is_ul, ul_err * y_ul, y_errn),
        np.where(is_ul, ul_err * y_ul, y_errp), ])
    
    # Preserve units in error if given in main value
    if type(y_err) == np.ndarray and type(y) == u.Quantity:
        y_err *= y.unit
        
    return y_val, y_err, is_ul


def plot_sed_fermipy(target: str, bblock: str = "baseline", 
                     ax: Optional[plt.Axes] = None, save_plot: bool = True, **kwargs_plot) -> plt.Axes:
    """
    Plot SED obtained from FermiPy fit
    """

    # Set default plot styling
    kwargs_plot.setdefault("label", "Fermi-LAT")
    kwargs_plot.setdefault("color", "crimson")
    kwargs_plot.setdefault("marker", "o")
    kwargs_plot.setdefault("markersize", 3)
    kwargs_plot.setdefault("capsize", 2)
    kwargs_plot.setdefault("capthick", 1)

    # Load SED data from FITS file
    # TODO: REMOVE GLOB!
    sed_file = glob.glob(
        str( RESULTS_DIR.resolve() / f"{target}/{bblock}/fermi-out/final_sed_*.fits" ) )[0]
    
    # Define required values
    with fits.open(sed_file) as f:
        
        src_sed  = f[1].data
        src_flux = f[2].data

        # Model data (default FermiPy units - MeV)
        energy  = np.array(src_flux['energy']) * u.MeV
        dnde    = np.array(src_flux['dnde']) * u.Unit("1 / (MeV cm2 s)")
        dnde_lo = np.array(src_flux['dnde_lo']) * u.Unit("1 / (MeV cm2 s)")
        dnde_hi = np.array(src_flux['dnde_hi']) * u.Unit("1 / (MeV cm2 s)")

        # Flux point data (default FermiPy units - MeV)
        x_pl  = src_sed['e_ref'] * u.MeV
        x_err = np.array([
            src_sed['e_ref'] - src_sed['e_min'],
            src_sed['e_max'] - src_sed['e_ref'], ]) * u.MeV
        y_pl     = src_sed['e2dnde'] * u.Unit("MeV / (cm2 s)")
        y_err_pl = src_sed['e2dnde_err'] * u.Unit("MeV / (cm2 s)")
        y_pl_ul  = src_sed['e2dnde_ul'] * u.Unit("MeV / (cm2 s)")

        ts = src_sed['ts']

    # Process flux points and upper limits
    y_pl_final, y_err_pl_final, is_uplim = get_fpul(
        y_pl, y_pl_ul, y_err_pl, y_err_pl, ts )

    # Create axes if not given
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Also set units
        ax.xaxis.set_units(energy.unit)
        ax.yaxis.set_units((energy**2 * dnde).unit)

    # Plot flux points
    ax.errorbar(
        x_pl, y_pl_final,
        yerr = y_err_pl_final,
        xerr = x_err,
        uplims = is_uplim,
        linestyle = "",
        **kwargs_plot, )
    # Plot spectral model
    ax.loglog(energy, energy**2 * dnde, color = kwargs_plot["color"], linestyle = "--")
    # Plot uncertainty (pass values without units)
    ax.fill_between(energy.value, (energy**2 * dnde_lo).value, (energy**2 * dnde_hi).value,
                    facecolor = kwargs_plot["color"], alpha = 0.2, )

    # Recompute axis limits
    ax.relim()
    ax.autoscale_view()
    # Fix axis limits
    ax.set_autoscale_on(False)

    # Save plot if requested
    if save_plot:
        plt.ylabel(r"Energy [{}]".format(f"{energy.unit:unicode}"))
        plt.ylabel(r"$\text{E}^2$ $d\text{N}/d\text{E}$ " + r"[{}]".format(f"{(energy**2 * dnde).unit:unicode}"))
        plt.title(f"{target} FermiPy Fermi-LAT SED")
        plt.legend()
        plt.savefig(RESULTS_DIR.resolve() / f"{target}/{bblock}/plots/sed_fermipy_flat.pdf", bbox_inches = "tight")

    # Return axis object
    return ax


def plot_fermi_diagnostics(gta, resid, tsmap, psmap, target, bblock = "baseline"):
    """
    Generate diagnostic plots for FermiPy analysis results.
    """

    # Counts, model counts, excess, and significance
    fig = plt.figure(figsize = (14, 18))

    # Plot the counts map
    ROIPlotter(gta.counts_map(), roi = gta.roi).plot(subplot = 321, cmap = "magma", fraction = 0.046, pad = 0.04)
    plt.gca().set_title("Counts")
    # Plot the model counts map
    ROIPlotter(gta.model_counts_map(), roi = gta.roi).plot(subplot = 322, cmap = "magma", fraction = 0.046, pad = 0.04)
    plt.gca().set_title("Model Counts")
    # Plot residual excess counts plot
    ROIPlotter(resid['excess'], roi = gta.roi).plot(subplot = 323, cmap = 'RdBu_r', fraction = 0.046, pad = 0.04)
    plt.gca().set_title('Excess Counts')
    # Plot residual significance plot
    ROIPlotter(resid['sigma'], roi = gta.roi).plot(levels = [-5, -3, 3, 5, 7, 9], subplot = 324, cmap = 'RdBu_r', fraction = 0.046, pad = 0.04)
    plt.gca().set_title('Significance')
    # Plot sqrt(TS) map
    ROIPlotter(tsmap['sqrt_ts'], roi=gta.roi).plot(
        # vmin = 0, vmax = 5.0,
        subplot = 325, cmap = 'magma', fraction = 0.046, pad = 0.04,
        levels = [1, 2, 3, 4, 5],
        cb_label = r"$\sqrt{TS} \,\,[\sigma]$",)
    plt.gca().set_title(r'$\sqrt{\text{TS}}$',)
    # Plot PS sigma map
    ROIPlotter(psmap["pssigma_map"], roi = gta.roi).plot(
        # vmin = -5, vmax = 5,
        subplot = 326, cmap = "RdBu_r", fraction = 0.046, pad = 0.04,
        interpolation = "bicubic",
        levels = [3, 4, 5],
        cb_label = r"$\text{PS} \,\,[\sigma]$",)
    plt.gca().set_title(r'$\text{PS}$')

    # Save the plots
    plt.tight_layout()
    plt.savefig(RESULTS_DIR.resolve() / f"{target}/{bblock}/fermi-out/final_plots.png", dpi = 300, bbox_inches = "tight")
    plt.close()

