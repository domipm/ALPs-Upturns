# Analysis script for Fermi-LAT data using FermiPy
# Requires conda environment: `env_fermi`
# Create the environment with: conda env create -f envs/env_fermi.yaml

import  os
import  yaml
import  argparse

import  numpy                   as      np
import  matplotlib.pyplot       as      plt

from    fermipy.gtanalysis      import  GTAnalysis
from    fermipy.plotting        import  ROIPlotter

from    alpsup.utils            import  parse_kwargs, get_source_info
from    alpsup.plots            import  plot_sed_fermipy
from    alpsup.paths            import  REPO_ROOT, RESULTS_DIR, FLAT_DATA_DIR, FERMIPY_DATA_DIR, get_results_dir


# TODO: USE get_results_dir() INSTEAD OF RESULST_DIR!


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
    config["model"]["isodiff"] = str( FERMIPY_DATA_DIR.resolve() / "iso_P8R3_SOURCE_V3_v1.txt" )

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

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")

    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121)")
    parser.add_argument("--bblock", default = "baseline", 
                        help = "Which Bayesian block to consider (for analyzing time selection blocks)")
    
    parser.add_argument("--model", choices = ["PowerLaw", "LogParabola"], help = "Override default model with custom choice")

    parser.add_argument("--analysis", choices = ["default", "target-only"], default = "default", help = "What type of analysis to perform. If 'target_only', load ROI from directory specified by '--kwargs load-roi=folder' subdirectory (or default: baseline) and fit target only")

    parser.add_argument("--plots-only", action="store_true", help = "Run only generation of plots from files")
    parser.add_argument("--kwargs", nargs = '*', 
                        help = "Additional keyword arguments ('key=value'). Can be used for, e.g., time selection (tmin, tmax), energy range (emin, emax)")
    
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Load the parameters of the target source from the sources.yaml file (don't need redshift)
    target_4FGL, target_position, _ = get_source_info(target = target)

    # Make sure directory exists
    os.makedirs(RESULTS_DIR / f"{target}/{args.bblock}/fermi-out/", exist_ok = True)
    os.makedirs(RESULTS_DIR / f"{target}/{args.bblock}/plots/", exist_ok = True)

    # Run only the plotting function
    if args.plots_only:
        plot_sed_fermipy(target = target, bblock = args.bblock)
        exit()

    # Run analysis

    # If no 4FGL name available and model not given, default to PowerLaw
    if not target_4FGL and args.model == None:
        args.model = "PowerLaw"

    # Generate the config file (modify with given arguments and, when missing, use default settings)
    gen_config(target, target_position, args.model, bblock = args.bblock, **kwargs)

    # Initialize FermiPy setup
    gta = GTAnalysis( str( RESULTS_DIR / f"{target}/{args.bblock}/fermi_config.yaml" ), logging = {'verbosity': 3})
    gta.setup(overwrite = False)

    # Generate PSF and DRM files for GammaPy analysis
    gta.compute_psf(overwrite = False)
    gta.compute_drm(edisp_bins = 0, overwrite = False)

    # Default analysis methodology
    if (args.analysis == "default"):

        # Free catalog sources within 3 deg of target (included)
        gta.free_sources(free = True, distance = 3)
        # Free galactic and isotropic diffuse model parameters
        gta.free_source(name = 'galdiff', free = True, pars = ['norm', 'index'])
        gta.free_source(name = 'isodiff', free = True, pars = ['norm'])

        # Perform initial optimization
        gta.optimize()

        # Free target source (try if target model available)
        try:
            gta.free_shape(name = target, free = True)

        # Otherwise, add target source with Power-Law model at position where it should be!
        except:

            print("Source not found in 4FGL catalog! Using default Power Law spectral model...")

            # Define the model source as dictionary of parameters
            src_dict = {
                'name': target,                   
                'RA': float(target_position.ra.value), 
                'DEC': float(target_position.dec.value),
                'SpatialModel': 'PointSource',         
                'SpectrumType': 'PowerLaw',  
                'Index': 2,
                'Prefactor': 1e-12,
                'Scale': 1000, }          

            # Add source model to roi
            gta.add_source(name = target, src_dict = src_dict, free = True)
            # Free target source
            gta.free_shape(name = target, free = True)
  
        # Free normalization of sources within 10 deg with TS > 10
        gta.free_sources(free = True, distance = 10.0, minmax_ts = [10, None], pars = 'norm')

        # Perform initial fit
        gta.fit()

        # Delete low significance sources (exclude target, both names just in case, and diffuse backgrounds)
        gta.delete_sources(minmax_ts = [-np.inf, 5], exclude = [target, target_4FGL, 'isodiff', 'galdiff'])

        # Look for any missed sources to include
        model = {'Index': 2.0, 'SpatialModel': 'PointSource'}
        srcs = gta.find_sources(model = model, sqrt_ts_threshold = 5.0, min_separation = 0.5)

        # Perform optimization
        gta.optimize()
        # Run another fit, including found sources
        gta.fit()

    # Fit only target source model
    elif (args.analysis == "target-only"):

        # Load roi from file (baseline subfolder)
        gta.load_roi( get_results_dir(target, kwargs.get('load-roi', 'baseline'), output = "fermi-out").joinpath("final.fits") )

        # Freeze all parameters
        gta.free_sources(free = False)
        # Free target source
        gta.free_source(target, free = True)
        # Perform the fit
        gta.fit()

    # Save final ROI
    gta.write_roi(outfile = "final", 
                  save_model_map = True, save_fits = False, make_plots = True, )

    # Get the SED
    sed = gta.sed(target, prefix = "final_sed",
                  write_fits = True, write_npy = True, make_plots = True, )

    # Compute residual map
    resid = gta.residmap(prefix = "final_resid", model={'SpatialModel': 'PointSource', 'Index': 2.0},
                         write_fits = True, write_npy = True, make_plots = True, )
    # Compute TS map
    tsmap = gta.tsmap(prefix = "final_ts", model = {'SpatialModel': 'PointSource', 'Index': 2.0},
                      write_fits = True, write_npy = True, make_plots = True, )
    # Compute PS map
    psmap = gta.psmap(prefix = "final_ps",
                      cmap = get_results_dir(target, args.bblock, output = "fermi-out").joinpath("ccube_00.fits"),
                      mmap = get_results_dir(target, args.bblock, output = "fermi-out").joinpath("mcube_final_00.fits"),
                      nbinloge = 14,
                      write_fits = True, make_plots = True,
                      emin = 1000, emax = 300000, )
    
    # Generate diagnostic plots
    plot_fermi_diagnostics(gta, resid, tsmap, psmap, target = target, bblock = args.bblock)

    # Generate plots
    plot_sed_fermipy(target = target, bblock = args.bblock)
