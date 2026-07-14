# Analysis script for Fermi-LAT data using GammaPy
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `fermi_analysis.py` to generate required files!

import  os
import  argparse
import  warnings

from    astropy.wcs                 import  FITSFixedWarning

import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path

from    astropy                     import  units   as  u
from    astropy.io                  import  ascii

from    gammapy.irf                 import  EDispKernelMap
from    gammapy.maps                import  Map
from    gammapy.datasets            import  Datasets, FermipyDatasetsReader
from    gammapy.estimators          import  FluxPointsEstimator
from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import  Models

from    utils                       import  parse_kwargs, get_source_info, get_fermipy_models, get_source_list, get_edec, gen_dirs, init_log
from    plots                       import  plot_sed_gammapy, plot_sed_compare


def plot_flat_diagnostics(target, dataset, bblock):

    # fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 10), subplot_kw=dict(projection = dataset.counts.geom.wcs))

    fig = plt.figure(figsize=(16, 10))

    wcs = dataset.counts.geom.wcs

    # Row 0: plots that need WCS projection
    ax00 = fig.add_subplot(2, 3, 1, projection=wcs)
    ax01 = fig.add_subplot(2, 3, 2, projection=wcs)
    ax02 = fig.add_subplot(2, 3, 3, projection=wcs)

    # Row 1: mixed — plot 4 (index [1,0]) and 6 (index [1,2]) need no projection
    ax10 = fig.add_subplot(2, 3, 4)           # Containment radius — no projection
    ax11 = fig.add_subplot(2, 3, 5, projection=wcs)  # PSF kernel — keeps projection
    ax12 = fig.add_subplot(2, 3, 6)           # Edisp matrix — no projection

    # TODO: CLEAN THIS UP!

    # Counts map
    ax00.set_title(f"{target} Fermi-LAT Counts")
    dataset.counts.sum_over_axes().smooth(1).plot(ax00, add_cbar = True, stretch = "sqrt")

    # Model counts map
    ax01.set_title(f"{target} Fermi-LAT Model Counts")
    dataset.npred().sum_over_axes().smooth(1).plot(ax = ax01, add_cbar = True, stretch = "sqrt")
    dataset.mask_fit.sum_over_axes().plot(ax01, alpha = 0.15)

    # Exposure map
    ax02.set_title("Exposure")
    dataset.exposure.sum_over_axes().smooth("0.5 deg").plot(ax = ax02, add_cbar = True, stretch = "sqrt")

    # Containment radius vs energy
    ax10.set_title("Containment Radius vs Energy")
    # ax[1, 0].projection = None
    dataset.psf.plot_containment_radius_vs_energy(ax = ax10, fraction=(0.68, 0.95, 0.99))

    # Point-spread function kernel
    ax11.set_title("PSF Kernel")
    dataset.psf.get_psf_kernel(
        position = dataset.exposure.geom.center_skydir,
        geom = dataset.exposure.geom,
        max_radius = "10 deg"
    ).to_image().psf_kernel_map.plot(
        ax = ax11,
        stretch = "log",
        add_cbar = True,
        cmap = "cividis")

    # Energy dispersion
    ax12.set_title("Edisp Matrix")
    dataset.edisp.get_edisp_kernel().plot_matrix(ax = ax12)

    plt.tight_layout()
    plt.savefig("{}/{}/{}/gamma-out/flat_diagnostic.png".format(os.environ['RESULTS'], target, bblock), dpi = 300, bbox_inches = "tight")
    plt.close()

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (8, 14))

    # Galactic diffuse map
    ax[0].set_title("Galactic Diffuse Background")
    dataset.models["Galactic"].spatial_model.map.sum_over_axes().plot(ax = ax[0], add_cbar = True, cmap = "cividis", stretch = "sqrt")

    # Galactic spectral flux
    ax[1].set_title("Galactic Diffuse Background Spectral Flux")
    dataset.models["Galactic"].spatial_model.map.to_region_nd_map(region = dataset.models[target].position).plot(ax = ax[1])

    # Isotropic diffuse map
    ax[2].set_title("Isotropic Diffuse Background Spectral Flux")
    dataset.models["Isotropic"].spectral_model.plot(ax = ax[2], energy_bounds = [50e-3, 2000] * u.GeV, yunits = u.Unit("1 / (cm2 MeV s)"))

    plt.savefig("{}/{}/{}/gamma-out/flat_diffuse.png".format(os.environ['RESULTS'], target, bblock), dpi = 300, bbox_inches = "tight")
    plt.close()

    return


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")

    parser.add_argument("--source", required = True, choices = get_source_list(), help = "Source name")    
        
    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Parse keyword arguments if given
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get the name of target source
    target = args.source
    # Get target info from config file
    target_4FGL, target_position, target_redshift = get_source_info(target)

    # Generate and check directories
    gen_dirs(target, bblock = args.bblock)

    # Define output directories
    dir_base = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock)}/" )
    dir_gout = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock)}/gamma-out/" )
    dir_pout = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock)}/plots/" )

    # If --plots-only, generate SED and FermiPy/GammaPy SED comparison
    if args.plots_only:
        # Generate SED plot
        plot_sed_gammapy(target = target, bblock = args.bblock, inst = "flat", **kwargs)
        # Generate SED plot comparison between FermiPy and GammaPy
        plot_sed_compare(target = target, bblock = args.bblock)
        exit()

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, fname = "flat_analysis.log", bblock = args.bblock)

    # Ignore fermipy fits date system deprecation warning
    warnings.simplefilter('ignore', FITSFixedWarning)

    # Display info of the target
    log.info("Fermi-LAT analysis of target source:\n{}\n({})\nPosition: {}\nRedshift z = {}".format(target, target_4FGL, target_position.data, target_redshift))

    # ========================= #
    # LOAD DATASET AND GET INFO #
    # ========================= #

    log.info("Loading Fermi-LAT model")
    # Read dataset based on fermipy configuration and output files
    reader = FermipyDatasetsReader(f"{os.environ['RESULTS']}/{target}/{args.bblock}/fermi_config.yaml", edisp_bins = 0)
    # Select only Fermi-LAT dataset
    dataset_flat = reader.read()[0].copy(name = "Fermi-LAT")

    # Extend counts, exposure, and background maps
    dataset_flat.counts = dataset_flat.counts.pad(50)
    dataset_flat.exposure = dataset_flat.exposure.pad(50)
    dataset_flat.background = dataset_flat.background.pad(50)

    dataset_flat.exposure = Map.read(f"{os.environ['RESULTS']}/{target}/{args.bblock}/fermi-out/bexpmap_00.fits").interp_to_geom(dataset_flat.exposure.geom)

    # Define energy dispersion (identity matrix, disable edisp)
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis = dataset_flat.counts.geom.axes["energy"],
        energy_axis_true = dataset_flat.exposure.geom.axes["energy_true"])
    # Add energy dispersion to dataset
    dataset_flat.edisp = edisp

    # Define safe mask fitting models outside roi
    mask_fit = Map.from_geom(dataset_flat.counts.geom, data = True, dtype = bool).binary_erode(width = 5.0 * u.deg, kernel = "disk")
    # Add safe mask to dataset
    dataset_flat.mask_fit = mask_fit
    dataset_flat.mask_safe = mask_fit

    # ========================== #
    # DEFINE TARGET SOURCE MODEL #
    # ========================== #

    # Load models and parameters from fermipy
    models_fermi = get_fermipy_models(target, bblock = args.bblock)
    
    # Select all models except target and diffuse backgrounds
    models = Models([model for model in models_fermi if model.name != "Isotropic" and model.name != "Galactic" and model.name != target])
    # Get all models outside roi
    models_out = models.select_mask(~mask_fit, use_evaluation_region = False)

    # Get all models inside  roi
    models_in  = Models([model for model in models if model not in models_out])
    # Convert models outside roi into template
    models_out = models_out.to_template_sky_model(
        geom = dataset_flat.exposure.geom, name = "Models Background", )

    # Add name for serialization
    models_out.spatial_model.filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_models_background.fits"
    # Add dataset name to model
    models_out.datasets_names = "Fermi-LAT"
    # Write out spatial model to file
    models_out.spatial_model.write(filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_models_background.fits", overwrite = True)

    # Add all models to our dataset
    dataset_flat.models = Models( [
        models_fermi[target], 
        models_fermi["Galactic"], models_fermi["Isotropic"],
        *models_in, models_out ] )
    
    # Generate diagnostic plots
    plot_flat_diagnostics(target = target, dataset = dataset_flat, bblock = args.bblock)

    # Freeze all model parameters
    dataset_flat.models.freeze()
    # Free only target source spectral parameters
    for p in dataset_flat.models[target].spectral_model.parameters:
        # Leave reference fixed
        if p.name != "reference":
            # Free index, amplitude, alpha, beta, ... (depending on model)
            dataset_flat.models[target].spectral_model.parameters[p].frozen = False
    # Free galactic and isotropic parameter normalization
    dataset_flat.models["Galactic"].parameters["norm"].frozen = False
    dataset_flat.models["Isotropic"].parameters["norm"].frozen = False

    # ================================== #
    # RUN FIT AND FLUX POINTS ESTIMATION #
    # ================================== #

    # Run the fit on the dataset
    log.info("Running initial fit...")
    fit_flat = Fit()
    fit_flat_result = fit_flat.run(datasets = [dataset_flat])
    log.info("Initial fit done!")

    # Compute decorrelation energy
    edec = get_edec(dataset_flat.models[target])
    try:
        edec_gp = dataset_flat.models[target].spectral_model.model1.pivot_energy
    except:
        edec_gp = dataset_flat.models[target].spectral_model.pivot_energy
    # If they are not the same, default to gammapy's version
    if edec_gp != edec:
        edec = edec_gp
    # Print decorrelation energy
    log.info(f"Decorrelation energy: {edec:.6f} (GammaPy check: {edec:.6f})")
    dataset_flat.models[target].parameters["reference"].quantity = edec

    # If reference energy given, override it
    if kwargs.get('reference', None) != None:
        dataset_flat.models[target].parameters["reference"].quantity = kwargs["reference"] * u.TeV

    # Re-run fit with new reference energy
    log.info("Running Fit...")
    fit_flat = Fit()
    fit_flat_result = fit_flat.run(datasets = [dataset_flat])
    log.info("Fit done!")


    # Display result of fit
    log.info(fit_flat_result)
    # Display final model of target source
    log.info(dataset_flat.models[target])

    # Define Datasets object containing the Fermi-LAT dataset with best-fit models
    datasets_flat = Datasets( datasets = [dataset_flat] )

    # Compute flux points
    log.info("Running FluxPointsEstimator...")
    fluxp_flat = FluxPointsEstimator(
        source = target,
        energy_edges = dataset_flat.counts.geom.axes["energy"].edges,
        selection_optional = ["all"],
    ).run([dataset_flat])
    log.info("FluxPointsEstimator done!")

    # ================= #
    # SAVE OUTPUT FILES #
    # ================= #

    # Save fit result to file
    fit_flat_result.write(
        path = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_fit.yaml", 
        overwrite = True, overwrite_templates = True, )
    
    # Save final datasets and models
    datasets_flat.write(
        filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_datasets.yaml",
        filename_models = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_models.yaml",
        overwrite = True, )
    
    # Save flux points table as csv file
    ascii.write(
        table = fluxp_flat.to_table(sed_type = "e2dnde"),
        output = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/flat_fluxp.ecsv",
        format = 'ecsv', overwrite = True,)
    # Save flux points as fits as well
    fluxp_flat.write(filename = dir_gout.joinpath("flat_fluxp.fits"), sed_type = "e2dnde", 
                    format = "gadf-sed", overwrite = True, )

    # ============== #
    # GENERATE PLOTS #
    # ============== #

    log.info(f"Saving final plots to {dir_pout}")
    # Generate SED plot
    plot_sed_gammapy(target = target, bblock = args.bblock, inst = "flat")
    # Generate SED plot comparison between FermiPy and GammaPy
    plot_sed_compare(target = target, bblock = args.bblock)

    log.info(f"Fermi-LAT GammaPy Analysis complete! :)")
