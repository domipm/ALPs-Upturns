# Analysis script for HESS data using GammaPy
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `fermi_analysis.py` and `flat_analysis.py` to generate required files!


import  os
import  yaml
import  argparse
import  matplotlib.pyplot           as      plt

import  numpy                       as      np

from    pathlib                     import  Path

from    astropy                     import  units   as  u
from    astropy.io                  import  ascii
from    astropy.time                import  Time

from    gammapy.datasets            import  Datasets, SpectrumDatasetOnOff
from    gammapy.estimators          import  FluxPointsEstimator
from    gammapy.maps                import  MapAxis
from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import (
            Models, SkyModel,
            PowerLawSpectralModel, LogParabolaSpectralModel, PointSpatialModel, TemplateSpectralModel, ConstantSpectralModel, SmoothBrokenPowerLawSpectralModel)

from    ebltable.tau_from_model     import  OptDepth

from    alpsup.utils                import  get_source_info, par_uconv, parse_kwargs, get_edec, get_source_list
from    alpsup.plots                import  plot_sed_gammapy
from    alpsup.models               import  BiasedCompoundSpectralModel, BiasedPriorSpectrumDatasetOnOff
from    alpsup.paths                import  gen_dirs, get_results_dir, get_hess_data_dir
from    alpsup.datasets             import  get_hess_dataset
from    alpsup.logs                 import  init_log


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run HESS analysis for a source using GammaPy", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", required = True, choices = get_source_list(), help = "Source name")    
        
    parser.add_argument("--dataset", default = "HAP-HD", help = "Which dataset to use, choices: HAP-HD, HAP-FR, HAP-FITS")
    parser.add_argument("--config", default = "std_ImPACT_hybrid_fullEnclosure_updated", help = "Which reconstruction configuration to use")
    
    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block / time segmentation block to consider")

    parser.add_argument("--bins", default = 4, type = int, help = "Number of spectral bins per decade")

    parser.add_argument("--reference", type = float, help = "Reference energy in TeV. Default: use computed decorrelation energy")

    parser.add_argument("--model", default = None, choices = ["PowerLaw", "LogParabola", "SmoothBrokenPowerLaw"], help="Spectral model to use for target source (override default read from FermiPy analysis)")

    parser.add_argument("--ebl", default = "dominguez", help = "EBL absorption model to use (loaded from EBLTable)")

    parser.add_argument("--sed_type", default = "e2dnde", choices = ["e2dnde", "dnde"], type = str)

    parser.add_argument("--bias", default = True, choices = [True, False], type = bool)

    parser.add_argument("--include-spatial", action = "store_true", help = "Include spatial model to SkyModel of target")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments if given
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get info of source
    target_4FGL, target_position, target_redshift = get_source_info(target)

    # Generate and check directories (depends on time block and EBL model)
    gen_dirs(target, bblock = args.bblock, ebl = args.ebl)

    # Define output directories
    dir_gout = get_results_dir(source = target, bblock = args.bblock, ebl = args.ebl, output = "gamma-out")

    # Run plotting-only
    if args.plots_only:
        plot_sed_gammapy(target = target, bblock = args.bblock, sed_type = args.sed_type, inst = "hess", **kwargs)
        exit()

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, bblock = args.bblock, ebl = args.ebl, fname = "hess_analysis.log")

    # Display info of the target
    log.info("HESS analysis of target source:\n{}\n({})\nPosition: {}\nRedshift z = {}".format(target, target_4FGL, target_position.data, target_redshift))

    # ========================= #
    # LOAD DATASET AND GET INFO #
    # ========================= #

    log.info(f"Loading HESS data from {args.dataset} dataset using {args.config} configuration...")

    '''
    try:
        # Load time selection from bblock config file in baseline if available
        with open(f"../sources/bblocks.yaml") as f:
            config = yaml.safe_load(f)[target]
        t_min = config[args.bblock.split("-")[0]]["tmin"]
        t_max = config[args.bblock.split("-")[0]]["tmax"]
        # log.info("Loaded time selection from time block config file")

    # Otherwise, use default time selection
    except:        
        t_min = kwargs.get('tmin', sorted(dataset_hess_obs.gti.time_start)[0].mjd)
        t_max = kwargs.get('tmax', sorted(dataset_hess_obs.gti.time_start)[-1].mjd)
        # log.info("Loaded time selection from custom kwargs / default GTIs")
    '''

    dataset_hess_obs, dataset_hess = get_hess_dataset(target)

    # Display summary of dataset
    log.info("HESS dataset summary")
    log.info(dataset_hess)

    # Display info of loaded dataset
    log.info(f"Loaded HESS data with {len(dataset_hess_obs)} observations!")
    log.info("GTI info [MJD]")
    log.info(f"- Duration: {dataset_hess_obs.gti.time_sum.to(u.hr).value} hr")
    log.info(f"- Start: {sorted(dataset_hess_obs.gti.time_start)[0].mjd} MJD ~ {sorted(dataset_hess_obs.gti.time_start)[0].iso} ISO")
    log.info(f"- Stop : {sorted(dataset_hess_obs.gti.time_stop)[-1].mjd} MJD ~ {sorted(dataset_hess_obs.gti.time_start)[-1].iso} ISO")

    # ========================== #
    # DEFINE TARGET SOURCE MODEL #
    # ========================== #

    # Load initial target model from fermi analysis in gammapy
    model_target = None
    try:
        log.info("Loading target source model from Fermi-LAT analysis...")
        model_target = Models.read(
            filename = dir_gout.joinpath("flat_models.yaml"))[target]
        log.info("Model loaded from Fermi-LAT analysis!")
    # If not there, just use a default power law
    except:
        log.info("Fermi-LAT analysis of the source not found!")
    # If model given, overwrite it!
    if model_target == None or args.model != None:
        if args.model == "LogParabola":
            intrinsic = LogParabolaSpectralModel()
        elif args.model == "SmoothBrokenPowerLaw":
            intrinsic = SmoothBrokenPowerLawSpectralModel()
        else:
            intrinsic = PowerLawSpectralModel()
        model_target = SkyModel(
            name = target,
            spectral_model = intrinsic,
            spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec), )
        # Freeze spatial model
        model_target.spatial_model.freeze()
    # Rename dataset associated to model
    model_target.datasets_names = "HESS"

    # Convert the units of the parameters of the model to HESS scale
    model_target = par_uconv("TeV", model_target)

    # Define EBL model
    log.info("Setting EBL model...")
    # Define energy range for EBL
    e_tau = np.logspace(-1, 1.5, 200) * u.TeV
    # Load from EBLTable library
    # (GammaPy's built-in EBL models are not compatible with GammaALPs!)
    if args.ebl != "None":
        # Create model from EBLTable
        tau = OptDepth.readmodel(model = args.ebl)
        att = np.exp(-1. * tau.opt_depth(target_redshift, e_tau.value))
        # Define model from template
        model_ebl = TemplateSpectralModel(energy = e_tau, values = att * u.dimensionless_unscaled)
        log.info(f"Loaded {args.ebl} EBL model using EBLTable!")

    # None model - use constant
    if args.ebl in ["None", None]:
        # Define constant spectral model (unitary, no absorption)
        model_ebl = ConstantSpectralModel(const = 1 * u.Unit("1 / (TeV s cm2)"))
        att = np.ones_like(e_tau.value)
        # Re-define it as template spectral model for consistency
        model_ebl = TemplateSpectralModel(energy = e_tau, values = att * u.dimensionless_unscaled)
        log.info("No EBL model applied!\n")

    # Combine intrinsic model with EBL model
    if args.ebl not in ["None", None]:
        model_spectral = model_target.spectral_model * model_ebl
    else:
        model_spectral = model_target.spectral_model
    # Extract spatial model
    model_spatial = model_target.spatial_model
    # Generate total model for the source
    model_hess = SkyModel(
        name = target,
        datasets_names = "HESS",
        spectral_model = model_spectral,
        spatial_model = model_spatial if args.include_spatial else None, )
    
    # Set parameters bounds if kwargs given (TODO: Move this to utils script?)
    # Set index bounds
    if "index" in model_hess.parameters.names:
        model_hess.parameters["index"].min = kwargs.get('index_min', 0.0)
        model_hess.parameters["index"].max = kwargs.get('index_max', 5.0)
    if "alpha" in model_hess.parameters.names:
        model_hess.parameters["alpha"].min = kwargs.get('alpha_min', -5.0)
        model_hess.parameters["alpha"].max = kwargs.get('alpha_max', +5.0)
    if "beta" in model_hess.parameters.names:
        model_hess.parameters["beta"].min = kwargs.get('beta_min', 0.0)
        model_hess.parameters["beta"].max = kwargs.get('beta_max', 2.0)
    # Set amplitude bounds
    model_hess.parameters["amplitude"].min = kwargs.get('amplitude_min', 1e-14)
    model_hess.parameters["amplitude"].max = kwargs.get('amplitude_max', 1e-06)
    # Set initial reference
    model_hess.parameters["reference"].quantity = kwargs.get('reference', 1.0) * u.TeV

    # Add this model to original dataset
    dataset_hess.models = model_hess

    # ================================== #
    # RUN FIT AND FLUX POINTS ESTIMATION #
    # ================================== #

    # Run initial fit with intrinsic model and ebl
    log.info("Running initial fit...")
    fit_original = Fit().run(datasets = dataset_hess)
    log.info("Initial fit done!")

    # Display results of fit
    log.info(fit_original)
    # Print final model
    log.info(dataset_hess.models[target])

    # Compute decorrelation energy
    edec = get_edec(dataset_hess.models[target])
    try:
        edec_gp = dataset_hess.models[target].spectral_model.model1.pivot_energy
    except:
        edec_gp = dataset_hess.models[target].spectral_model.pivot_energy
    # If they are not the same, default to gammapy's version
    if edec_gp != edec:
        edec = edec_gp
    # Print decorrelation energy
    log.info(f"Decorrelation energy: {edec:.6f} (GammaPy check: {edec_gp:.6f})")
    dataset_hess.models[target].parameters["reference"].quantity = edec

    # If reference energy given, override it
    if args.reference:
        dataset_hess.models[target].parameters["reference"].quantity = args.reference * u.TeV
  
    # Run main fit
    log.info("Running fit...")
    fit_hess = Fit().run(datasets = dataset_hess)
    log.info("Fit done!")

    # Display results of fit
    log.info(fit_hess)
    # Print final model
    log.info(dataset_hess.models[target])

    # Compute flux points
    log.info("Running flux points estimator...")
    fluxp_hess = FluxPointsEstimator(
        energy_edges = dataset_hess.counts.geom.axes["energy"].edges,
        source = target,
        selection_optional = "all",
    ).run([dataset_hess])
    log.info("Flux point estimation done!")

    # Plot diagnostic fit results
    fig, ax = plt.subplots()
    ax.yaxis.set_units(u.Unit("TeV s-1 cm-2"))
    fluxp_hess.plot(ax = ax, sed_type = "e2dnde")
    # fluxp_hess.plot(sed_type = "e2dnde")
    dataset_hess.models[target].spectral_model.plot(ax = ax, energy_bounds = dataset_hess.counts.geom.axes["energy"].bounds, sed_type = "e2dnde")
    dataset_hess.models[target].spectral_model.plot_error(ax = ax, energy_bounds = dataset_hess.counts.geom.axes["energy"].bounds, sed_type = "e2dnde")
    plt.title(f"{target} PowerLaw SED")
    plt.savefig(fname = dir_gout.joinpath("hess_fit.png"), dpi = 300, bbox_inches = "tight",)
    plt.close()

    # ================= #
    # SAVE OUTPUT FILES #
    # ================= #

    log.info(f"Saving output files to {dir_gout}")
    # Save fit results
    fit_hess.write(
        path = dir_gout.joinpath("hess_fit.yaml"),
        overwrite = True, overwrite_templates = True, checksum = True )
    
    # Save final model
    datasets_hess = Datasets( dataset_hess )
    datasets_hess.write(
        filename = dir_gout.joinpath("hess_datasets.yaml"),
        filename_models = dir_gout.joinpath("hess_models.yaml"),
        overwrite = True, )
    
    # Save flux points table to file
    ascii.write(
        table = fluxp_hess.to_table(sed_type = args.sed_type, # sed_type = "e2dnde", 
                                    format = "gadf-sed"), format = 'ecsv',
        output = dir_gout.joinpath("hess_fluxp.ecsv"),
        overwrite = True, )
    # Save flux points as fits as well
    fluxp_hess.write(filename = dir_gout.joinpath("hess_fluxp.fits"), sed_type = args.sed_type, # sed_type = "e2dnde", 
                     format = "gadf-sed", overwrite = True, )
    

    # TODO: CONTINUE FROM HERE!
    # exit()


    # Plot HESS model and flux points        
    plot_sed_gammapy(target = target, bblock = args.bblock, inst = "hess", sed_type = args.sed_type)

    # Run biased fit if required
    if args.bias == False or args.bias == "False":
        exit()

    # ========== #
    # BIASED FIT #
    # ========== #

    # HESS analysis on biased datasets and models
    log.info("Running fit on biased HESS model with bias prior on dataset")

    # Define new spectral model
    spectral_model_hess = BiasedCompoundSpectralModel(
        intrinsic_model = dataset_hess.models[target].spectral_model.model1,
        ebl_model = dataset_hess.models[target].spectral_model.model2,
        bias = 0.0, )

    # Define new sky model
    model_hess_bias = SkyModel(
        datasets_names = "HESS",
        spectral_model = spectral_model_hess,
        spatial_model = dataset_hess.models[target].spatial_model,
        name = target, )

    # Set bias prior to 15%
    sigma_bias = 0.15

    # Wrap dataset with bias prior
    dataset_hess_bias = BiasedPriorSpectrumDatasetOnOff.from_spectrum_dataset(
        dataset_hess, sigma_bias, )
    
    # Add biased model
    dataset_hess_bias.models = Models(model_hess_bias)

    # Define datasets object
    datasets_hess_bias = Datasets(dataset_hess_bias)

    # Run joint biased fit
    log.info("Running biased fit...")
    results_bias = Fit().run(datasets = dataset_hess_bias)
    log.info("Biased fit done!")

    # Display info on fit and best model
    log.info(results_bias)
    log.info(datasets_hess_bias["HESS"].models[target])

    # Compute flux points on biased dataset
    log.info("Running flux points estimator on biased dataset...")
    fluxp_hess = FluxPointsEstimator(
        energy_edges = dataset_hess_bias.counts.geom.axes["energy"].edges,
        source = target,
        selection_optional = "all",
    ).run([dataset_hess_bias])
    log.info("Flux point estimation on biased dataset done!")
    
    # Save final results
    results_bias.write(
        path = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/hess_bias_results.yaml",
        overwrite = True, )
    # Save final datasets and model
    datasets_hess_bias.write(
        filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/hess_bias_datasets.yaml",
        filename_models = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/hess_bias_models.yaml",
        overwrite = True, )
    # Save flux points on biased data
    ascii.write(
        table = fluxp_hess.to_table(sed_type = "e2dnde", format = "gadf-sed"), format = 'ecsv',
        output = dir_gout.joinpath("hess_bias_fluxp.ecsv"),
        overwrite = True, )
    # Save flux points as fits as well
    fluxp_hess.write(filename = dir_gout.joinpath("hess_bias_fluxp.fits"), sed_type = "e2dnde", 
                     format = "gadf-sed", overwrite = True, )

    # ============== #
    # GENERATE PLOTS #
    # ============== #

    log.info(f"Saving final plots...")
    # Plot HESS model and flux points        
    plot_sed_gammapy(target = target, bblock = args.bblock, inst = "hess")
    # Plot HESS bias model and flux points
    plot_sed_gammapy(target = target, bblock = args.bblock, inst = "hess_bias")

    log.info(f"H.E.S.S. GammaPy Analysis complete! :)")
