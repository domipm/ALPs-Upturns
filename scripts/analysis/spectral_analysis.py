# Analysis script for joint Fermi-LAT and HESS analysis with GammaPy
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `fermi_analysis.py`, `flat_analysis.py`, and `hess_analysis.py` to generate required files!

import  os
import  warnings
import  argparse

import  numpy                       as      np

from    pathlib                     import  Path

from    astropy                     import  units as u
from    astropy.io                  import  ascii
from    astropy.wcs                 import  FITSFixedWarning

from    ebltable.tau_from_model     import  OptDepth

from    gammapy.maps                import  MapAxis
from    gammapy.modeling            import  Fit, Parameter, Parameters
from    gammapy.modeling.models     import  (
                                                Models, SkyModel, SpectralModel, PointSpatialModel, TemplateSpectralModel, CompoundSpectralModel, 
                                                PowerLawSpectralModel, LogParabolaSpectralModel, SmoothBrokenPowerLawSpectralModel, BrokenPowerLawSpectralModel
                                            )
from    gammapy.datasets            import  Datasets, SpectrumDatasetOnOff
from    gammapy.estimators          import  FluxPointsEstimator

from    alpsup.models               import  BiasedPriorSpectrumDatasetOnOff, CompositeSpectralModel, EBLTableSpectralModel
from    alpsup.utils                import  get_source_list, get_source_info, par_uconv, tab_uconv, parse_kwargs, get_edec
from    alpsup.logs                 import  init_log
from    alpsup.plots                import  plot_sed_joint
from    alpsup.paths                import  get_results_dir
from    alpsup.datasets             import  get_flat_dataset, get_hess_dataset


def ebl_type(ebl_name):
    """Parse EBL argument to also allow None type"""
    return None if ebl_name.lower() == "none" else ebl_name


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run joint Fermi-LAT and H.E.S.S. spectral analysis for a source using GammaPy",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", "--source", required = True, choices = get_source_list(),
                        help = "Target source name for which analysis is performed.")
    
    parser.add_argument("-b", "--bblock", required = True, 
                        help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs).")
    parser.add_argument("-e", "--ebl", default = "dominguez", type = ebl_type,
                        choices = ["dominguez", "finke2022", "franceschini", "saldana-lopez", None], 
                        help = "EBL absorption model to use (loaded from EBLTable, or None).")

    parser.add_argument("-m", "--model", default = None, choices = ["PowerLaw", "PL", "LogParabola", "LP", "SmoothBrokenPowerLaw", "SBPL", "BrokenPowerLaw", "BPL"],
                        help = "Spectral model to use. By default, load it from HESS analysis, unless overriden.")
    
    parser.add_argument("--bias", action = argparse.BooleanOptionalAction, default = True,
                        help = "Include bias prior on HESS dataset for systematic uncertainty.")

    parser.add_argument("-f", "--force", action = argparse.BooleanOptionalAction, default = True, help = "Force re-running individual analysis steps per-instrument.")
    parser.add_argument("-p", "--plots-only", action = "store_true", help = "Generate output plots, without running the analysis")

    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments formatted as 'key=value'.")

    args = parser.parse_args()

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get the name of target source
    target = args.source

    # Run only plots
    if args.plots_only:
        plot_sed_joint(target = target, bblock = args.bblock, ebl = args.ebl, **kwargs)
        exit()

    # Define output directories for gamma-out for both Fermi-LAT and H.E.S.S. files
    dir_gout = get_results_dir(target, args.bblock, args.ebl, output = "gamma-out")
    dir_fout = get_results_dir(target, args.bblock, output = "gamma-out")

    # Get info on source from data class
    source = get_source_info(target)
    target_4FGL = source.name_4FGL
    target_position = source.position
    target_redshift = source.redshift
    target_instruments = source.inst

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, bblock = args.bblock, ebl = args.ebl, fname = "joint_analysis.log")

    # Display info of the target
    log.info(f"Joint Fermi-LAT and HESS analysis of target source:\n{target}\n({target_4FGL})\nPosition: {target_position.data}\nRedshift z = {target_redshift}\nAvailable instruments: {target_instruments}")

    # ==================================== #
    # LOAD FERMI-LAT AND H.E.S.S. DATASETS #
    # ==================================== #

    # NOTE: For some sources, Fermi-LAT not available, so skip this step!
    # Only load Fermi-LAT if in instruments list of source
    dataset_flat = None
    if "Fermi-LAT" in target_instruments:
        # Generate or load Fermi-LAT datasets and models
        if args.force is False:
            # Attempt to load Fermi-LAT datasets from GammaPy output files, including models
            try:
                dataset_flat = Datasets.read(
                    filename = f"{dir_fout}/flat_datasets.yaml",
                    filename_models = f"{dir_fout}/flat_models.yaml", )[0]
            except:
                raise Exception("GammaPy analysis of Fermi-LAT not found! Run individual analysis, or include --force argument to re-run.")
        # Re-run GammaPy analysis of Fermi-LAT - For this, FermiPy setup and analysis must be carried out first!
        elif args.force is True:
            try:
                # Generate GammaPy datasets from the FermiPy output, including models
                # NOTE: Suppress astropy WCS warnings re: changed FITS naming conventions
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category = FITSFixedWarning)
                    dataset_flat = get_flat_dataset(target = target, bblock = args.bblock)
            except:
                raise Exception("FermiPy analysis of Fermi-LAT not found! Run initial setup / analysis step in FermiPy.")

    # Only load HESS if in instruments list of source
    # NOTE: For now, skip this check, as all sources have HESS data
    # Generate or load H.E.S.S. datasets and models
    # if "HESS" in target_instruments: ...
    if args.force is False:
        # Attempt to load H.E.S.S. datasets from GammaPy output files, including models
        try:
            dataset_hess = Datasets.read(
                filename = f"{dir_gout}/hess_datasets.yaml",
                filename_models = f"{dir_gout}/hess_models.yaml", )[0]
        except:
            raise Exception("GammaPy analysis of H.E.S.S. not found! Run individual analysis, or include --force argument to re-run.")
    # Re-run GammaPy analysis of HESS - For this, GammaPy HESS temporal analysis must be carried out first! (to generate configs)
    else:
        try:
            # Generate GammaPy datasets from the FermiPy output, including models
            # NOTE: THIS NEEDS CONFIG AND HAP DATASET ARGUMENTS! READ FROM CONFIG! - THIS CONFIG CAN BE CREATED BY THE TEMPORAL ANALYSIS (AS THE STEP IS REQUIRED!)
            _, dataset_hess = get_hess_dataset(target = target, bblock = args.bblock)
        except:
            raise Exception("Unable to generate GammaPy H.E.S.S. datasets!")

    # ============================= #
    # DEFINE MODELS FOR ALL SOURCES #
    # ============================= #

    # Get all Fermi-LAT models, removing target (we will define in later)
    # Make sure units converted into HESS scale
    if dataset_flat is not None:
        dataset_flat.models = [
            par_uconv("TeV", model.copy(name = model.name), "energy") 
            for model in dataset_flat.models if model.name != target ]
        # Freeze all parameters of all Fermi-LAT background models
        dataset_flat.models.freeze()
        # If background model found, change filename path for serialization
        if "Models Background" in dataset_flat.models.names:
            dataset_flat.models["Models Background"].spatial_model.filename = f"{dir_fout}/flat_models_background.fits"
        
    # TODO: READ BEST-FIT INITIAL MODEL FROM FILE?
    # TODO: PERFORM INTRINSIC SPECTRAL MODEL TEST?

    # Define target source model 
    # If no specific model given
    if args.model is None:
        
        # Extract target source from HESS dataset
        try:
            target_model = dataset_hess.models[target].copy(name = target, copy_data = True, datasets_names = ["HESS", "Fermi-LAT"])
        # If not found, use default PowerLaw model
        except:
            target_model = SkyModel(
                name = target, datasets_names = ["HESS", "Fermi-LAT"],
                spectral_model = PowerLawSpectralModel(),
                spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )

        # If a compound spectral model (containing EBL), choose intrinsic model only
        if isinstance(target_model, CompoundSpectralModel):
            target_model_intrinsic = target_model.spectral_model.model1
        else:
            # Otherwise, take spectral model directly
            target_model_intrinsic = target_model.spectral_model

    # If model argument given, override
    else:
        # Shorthand for models
        models_equiv = dict.fromkeys(
            ['PowerLaw', 'PL'], PowerLawSpectralModel()) | dict.fromkeys(
            ['LogParabola', 'LP'], LogParabolaSpectralModel()) | dict.fromkeys(
            ['SmoothBrokenPowerLaw', 'SBPL'], SmoothBrokenPowerLawSpectralModel()) | dict.fromkeys(
            ['BrokenPowerLaw', 'BPL'], BrokenPowerLawSpectralModel(), )
        # Define SkyModel for given spectral model
        target_model = SkyModel(
            name = target, datasets_names = ["HESS", "Fermi-LAT"],
            spectral_model = models_equiv[args.model],
            spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )
        # Get intrinsic model
        target_model_intrinsic = target_model.spectral_model

    # In all cases, freeze spatial model
    target_model.spatial_model.freeze()

    # Compute full energy bounds
    dataset_energy_range = [None, dataset_hess.counts.geom.axes["energy"].edges[-1].to(u.TeV)]
    if dataset_flat is not None:
        dataset_energy_range[0] = dataset_flat.counts.geom.axes["energy"].edges[0].to(u.TeV)
    elif dataset_hess is not None:
        dataset_energy_range[0] = dataset_hess.counts.geom.axes["energy"].edges[0].to(u.TeV)

    # Re-define EBL model to cover full energy range, define EBLTableSpectralModel
    model_ebl = None
    if args.ebl not in [None, "none"]:
        model_ebl = EBLTableSpectralModel.read_ebl(
            # TODO: Number of sampling points
            # energy = np.logspace(
            energy = np.geomspace(
                # np.log10(dataset_energy_range[0].value), np.log10(dataset_energy_range[-1].value), 200) * u.TeV,
                dataset_energy_range[0].value, dataset_energy_range[-1].value, 200) * u.TeV,
            ebl_name = args.ebl, redshift = target_redshift, )
    
    # Re-define target model as CompositeSpectralModel (including EBL if given, no upturn, bias if given)
    target_model = SkyModel(
        name = target,
        # Only add not-None datasets
        datasets_names = ["HESS"] + (["Fermi-LAT"] if dataset_flat is not None else []),
        spectral_model = CompositeSpectralModel(intrinsic_model = target_model_intrinsic, ebl_model = model_ebl, upturn_model = None, bias = 0.0),
        spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )
    
    # In all cases, freeze spatial model
    target_model.spatial_model.freeze()
    # TODO: SET UP MODEL LIMITS.GET FROM KWARGS, OTHERWISE, GET FROM HESS CONFIG FILE DEFAULTS?
    target_model.parameters["amplitude"].min = kwargs.get("amplitude_min", 1e-15)
    target_model.parameters["amplitude"].max = kwargs.get("amplitude_max", 1e-06)

    # Define joint datasets object for Fermi-LAT and HESS (including other models)
    dataset_joint = Datasets(
        # Include only not-None datasets
        [dataset for dataset in (dataset_flat, dataset_hess) if dataset is not None] )
        # Add all models to the datasets, including target
    dataset_joint.models = Models( [ *( dataset_flat.models if dataset_flat is not None else [] ), target_model ] )

    # ================================= #
    # SET UP BIASED MODELS AND DATASETS #
    # ================================= #

    # Ensure, if bias included, that datasets include it, otherwise run fit with standard datasets
    if args.bias is True:

        # Define target model for HESS only
        target_model_bias = SkyModel(
            name = target, datasets_names = "HESS",
            spectral_model = CompositeSpectralModel(intrinsic_model = target_model_intrinsic, ebl_model = model_ebl, upturn_model = None, bias = 0.0),
            spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )
        # Freeze spatial model
        target_model_bias.spatial_model.freeze()

        # Wrap HESS dataset with bias prior (15%)
        dataset_hess_bias = BiasedPriorSpectrumDatasetOnOff.from_spectrum_dataset(
            dataset = dataset_hess, sigma_bias = 0.15, )
        
        # Add biased model to dataset
        dataset_hess_bias.models = Models([target_model_bias])
    
        # NOTE: IS THIS STEP ACTUALLY NECESSARY? TO HAVE TWO SEPARATE MODELS, LINKED? 
        # IN PRINCIPLE, ONLY HESS EVALUATES THE BIAS? MAYBE IT'S ENOUGH TO JUST HAVE THE BIASED DATASET ON HESS, AND SHARE MODEL?

        if dataset_flat is not None:
            # Create unbiased target model for Fermi-LAT, with linked parameters
            target_model_unbias = SkyModel(
                name = f"{target} Unbiased",
                datasets_names = "Fermi-LAT",
                spectral_model = target_model_bias.spectral_model.intrinsic_model * target_model_bias.spectral_model.ebl_model, 
                spatial_model = target_model_bias.spatial_model, )
            # Freeze spatial model
            target_model_unbias.spatial_model.freeze()

            # Use usual Fermi-LAT dataset
            dataset_flat_unbias = dataset_flat.copy(name = "Fermi-LAT")
            dataset_flat_unbias.models = Models([ *[model for model in dataset_flat.models if model.name != target], target_model_unbias ])

        # Create final dataset to be used, including models
        dataset_joint = Datasets(
            # Include only not-None datasets
            [dataset for dataset in (dataset_flat, dataset_hess) if dataset is not None] )

        # NOTE: Despite the models' parameters not explicitly showing the "link" property, they are linked together
        # so modifying one version of the model (un/biased) also modifies the other!

    elif args.bias is False:
        target_model.spectral_model.parameters["bias"].frozen = True

    # Display datasets and models to be fit
    log.info(f"Constructed joint Fermi-LAT and H.E.S.S. datasets to fit:\n{dataset_joint}")
    log.info(f"Target source model:\n{target_model}")

    # ================================= #
    # RUN SPECTRAL FIT ON JOINT DATASET #
    # ================================= #

    # Run initial fit
    log.info("Running initial fit...")
    fit_joint = Fit()
    results_joint = fit_joint.run(datasets = dataset_joint)
    log.info("Initial fit done!")

    # Compute decorrelation energy
    edec = get_edec(dataset_joint.models[target])
    edec_gp = dataset_joint.models[target].spectral_model.pivot_energy
    log.info(f"Decorrelation energy: {edec:.6f}")
    if edec_gp is None:
        log.warning("Decorrelation energy computed with GammaPy could not be computed!")
    elif edec_gp != edec:
        log.warning(f"Decorrelation energy computed manually differs from GammaPy check by {np.abs(edec - edec_gp)}")
    else:
        log.info(f"GammaPy check: {edec_gp:.6f}")

    # Set decorrelation energy as reference
    dataset_joint.models[target].parameters["reference"].quantity = edec
    # Set amplitude as that of model evaluated at reference (improves convergence)
    dataset_joint.models[target].parameters["amplitude"].quantity = dataset_joint.models[target].spectral_model(edec)

    # Run the joint fit and print the results
    log.info("Running main fit...")
    fit_joint = Fit()
    results_joint = fit_joint.run(datasets = dataset_joint)
    log.info("Main fit done!")

    # Display info on fit and best-fit model
    log.info(results_joint)
    log.info(dataset_joint.models[target])

    # Compute flux points on Fermi-LAT data
    if dataset_flat is not None:
        log.info("Computing flux points for Fermi-LAT...")
        fluxp_joint_flat = FluxPointsEstimator(
            energy_edges = dataset_joint["Fermi-LAT"].counts.geom.axes["energy"].edges.to(u.TeV),
            source = f"{target} Unbiased" if args.bias else target,
            selection_optional = ["all"]).run([dataset_joint["Fermi-LAT"]])
        log.info("Flux points estimator for Fermi-LAT done!")

    log.info("Computing flux points for HESS...")
    # Compute flux points on HESS data
    fluxp_joint_hess = FluxPointsEstimator(
        energy_edges = dataset_joint["HESS"].counts.geom.axes["energy"].edges.to(u.TeV),
        source = target, selection_optional = ["all"]).run([dataset_joint["HESS"]])
    log.info("Flux points estimator for HESS done!")

    # ============================ #
    # SAVE RESULTS, OUTPUTS, PLOTS #
    # ============================ #

    log.info("Saving result output files...")
    # Save fit results
    results_joint.write(
        path = dir_gout / "joint_fit.yaml",
        overwrite = True, overwrite_templates = True, )
    # Save final datasets and models
    dataset_joint.write(
        filename = dir_gout / "joint_datasets.yaml",
        filename_models = dir_gout / "joint_models.yaml",
        overwrite = True, )

    # Save Fermi-LAT flux points (ensuring TeV scale units)
    if dataset_flat is not None:
        ascii.write(
            table = tab_uconv("TeV", fluxp_joint_flat.to_table(sed_type = "e2dnde")),
            output = dir_gout / "joint_flat_fluxp.ecsv", format = "ecsv", overwrite = True, )
    # Save HESS flux points (ensuring TeV scale units)
    ascii.write(
        table = tab_uconv("TeV", fluxp_joint_hess.to_table(sed_type = "e2dnde")),
        output = dir_gout / "joint_hess_fluxp.ecsv", format = "ecsv", overwrite = True, )

    # Generate plots
    log.info("Generating plots...")

    # TODO: FIX PLOTTING IF FERMI-LAT DATA AVAILABLE!
    plot_sed_joint(target, bblock = args.bblock, ebl = args.ebl)

    log.info(f"Joint Fermi-LAT + H.E.S.S. GammaPy Analysis complete! :)")
