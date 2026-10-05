# Analysis script for joint Fermi-LAT and HESS analysis with GammaPy
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `fermi_analysis.py`, `flat_analysis.py`, and `hess_analysis.py` to generate required files!

import  os
import  yaml
import  logging
import  warnings
import  argparse

import  numpy                       as      np

from    astropy                     import  units as u
from    astropy.io                  import  ascii
from    astropy.wcs                 import  FITSFixedWarning

from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import  ( Models, SkyModel, PointSpatialModel, CompoundSpectralModel, 
                                              PowerLawSpectralModel, LogParabolaSpectralModel, SmoothBrokenPowerLawSpectralModel )
from    gammapy.datasets            import  Datasets, Dataset
from    gammapy.estimators          import  FluxPointsEstimator

from    alpsup.models               import  BiasedPriorSpectrumDatasetOnOff, CompositeSpectralModel, EBLTableSpectralModel, CustomSmoothBrokenPowerLawSpectralModel
from    alpsup.utils                import  get_source_list, get_source_info, par_uconv, tab_uconv, parse_kwargs
from    alpsup.logs                 import  init_log
from    alpsup.plots                import  plot_sed_joint
from    alpsup.paths                import  get_results_dir, CONFIGS_DIR
from    alpsup.datasets             import  get_flat_dataset, get_hess_dataset
from    alpsup.parameters           import  set_param_bounds


# Get logger function
log = logging.getLogger(__name__)


# Default HAP dataset and config
HAP_DATASET = "HAP-HD"
HAP_CONFIG = "std_ImPACT_hybrid_fullEnclosure_updated"

# Shorthand for models
MODELS_EQUIV = dict.fromkeys(
    ['PowerLaw', 'PL'], PowerLawSpectralModel()) | dict.fromkeys(
    ['LogParabola', 'LP'], LogParabolaSpectralModel()) | dict.fromkeys(
    ['CustomSmoothBrokenPowerLaw', 'CSBPL'], CustomSmoothBrokenPowerLawSpectralModel()) | dict.fromkeys(
    # NOTE: SmoothBrokenPowerLaw model, if using index1==index2, throws an (initial) warning about degeneracy!
    # Slight offset in initial parameters solves this issue
    ['SmoothBrokenPowerLaw', 'SBPL'], SmoothBrokenPowerLawSpectralModel(index1 = 2.0, index2 = 1.5), ) 


def ebl_type(ebl_name):
    """Parse EBL argument to also allow None type"""
    return None if ebl_name.lower() == "none" else ebl_name


def run_spectral_fit(dataset: Datasets | Dataset, target: str, use_edec: bool = True):
    """
    Run spectral fit of the dataset, which must include the models to be fit and its parameters.
    """

    # Run initial fit
    log.info("Running initial fit...")
    fit_joint = Fit()
    results_joint = fit_joint.run(datasets = dataset)
    log.info("Initial fit done!")

    # Display info on fit and best-fit model
    log.info(results_joint)
    log.info(dataset.models[target])

    # Compute decorrelation energy
    if use_edec:

        edec = dataset.models[target].spectral_model.pivot_energy
        log.info(f"Decorrelation energy: {edec:.6f}")

        # Get intrinsic spectral model
        intrinsic_model = dataset.models[target].spectral_model
        if intrinsic_model.__class__.__name__ == "CompositeSpectralModel":
            intrinsic_model = dataset.models[target].spectral_model.intrinsic_model

        # Set amplitude as that of model evaluated at reference (improves convergence)
        if intrinsic_model.__class__.__name__ == "SmoothBrokenPowerLawSpectralModel":
            # Get "old" parameter values
            amplitude = dataset.models[target].parameters["amplitude"].quantity
            reference = dataset.models[target].parameters["reference"].quantity
            index1    = dataset.models[target].parameters["index1"].value
            # Compute new amplitude value
            dataset.models[target].parameters["amplitude"].quantity = amplitude * (edec / reference) ** ( - index1)
        else:
            dataset.models[target].parameters["amplitude"].quantity = dataset.models[target].spectral_model(edec)

        # Set decorrelation energy as reference
        dataset.models[target].parameters["reference"].quantity = edec

        # Run the joint fit and print the results
        log.info("Running main fit...")
        fit_joint = Fit()
        results_joint = fit_joint.run(datasets = dataset)
        log.info("Main fit done!")

        # TODO: Run a quick check for any parameters saturating, and maybe re-run with less strict parameter boundaries?
        # Otherwise, just make sure the fits look right!

        # Display info on fit and best-fit model
        log.info(results_joint)
        log.info(dataset.models[target])

    # Return the fit results
    return results_joint


def run_fluxp_estimator(dataset: Datasets | Dataset, target: str,
                        use_bias: bool = True):

    # Compute flux points on Fermi-LAT data
    fluxp_joint_flat = None
    if "Fermi-LAT" in dataset.names:
        log.info("Computing flux points for Fermi-LAT...")
        fluxp_joint_flat = FluxPointsEstimator(
            energy_edges = dataset["Fermi-LAT"].counts.geom.axes["energy"].edges.to(u.TeV),
            source = f"{target} Unbiased" if use_bias else target,
            selection_optional = ["all"]).run([dataset["Fermi-LAT"]])
        log.info("Flux points estimator for Fermi-LAT done!")

    # Compute flux points on HESS data
    fluxp_joint_hess = None
    if "HESS" in dataset.names:
        log.info("Computing flux points for HESS...")
        fluxp_joint_hess = FluxPointsEstimator(
            energy_edges = dataset["HESS"].counts.geom.axes["energy"].edges.to(u.TeV),
            source = target, selection_optional = ["all"]).run([dataset["HESS"]])
    log.info("Flux points estimator for HESS done!")

    return fluxp_joint_flat, fluxp_joint_hess


def main():

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run joint Fermi-LAT and H.E.S.S. spectral analysis for a source using GammaPy",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", required = True, choices = get_source_list(),
                        help = "Target source name for which analysis is performed.")
    
    parser.add_argument("--bblock", required = True, 
                        help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs).")
    parser.add_argument("--ebl", default = "dominguez", type = ebl_type,
                        choices = ["dominguez", "finke2022", "franceschini", "saldana-lopez", None], 
                        help = "EBL absorption model to use (loaded from EBLTable, or None).")

    parser.add_argument("--dataset", default = None, 
                        # default = "HAP-HD", 
                        choices = ["HAP-HD", "HAP-FR", "HAP-FITS"], 
                        help = "Which HAP dataset to use.")
    parser.add_argument("--config", default = None,
                        # default = "std_ImPACT_hybrid_fullEnclosure_updated", 
                        help = "Which HAP reconstruction configuration to use")
    
    parser.add_argument("--model", default = None, choices = [*MODELS_EQUIV.keys()],
                        help = "Spectral model to use. By default, load it from HESS analysis, unless overriden.")
    
    parser.add_argument("--bias", action = argparse.BooleanOptionalAction, default = True,
                        help = "Include bias prior on HESS dataset for systematic uncertainty (--bias, --no-bias).")
    parser.add_argument("--edec", action = argparse.BooleanOptionalAction, default = True,
                        help = "Use decorrelation energy as reference (--edec, --no-edec).")
    
    parser.add_argument("--plots-only", action = "store_true", help = "Generate output plots, without running the analysis")

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

    # Load HESS config data
    with open(CONFIGS_DIR / "hess_config.yaml", "r") as f:
        hess_config = yaml.safe_load(f)

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, bblock = args.bblock, ebl = args.ebl, fname = "spectral_analysis.log")

    # Display info of the target
    log.info(f"Joint Fermi-LAT and HESS analysis of target source:\n{target}\n({target_4FGL})\nPosition: {target_position.data}\nRedshift z = {target_redshift}\nAvailable instruments: {target_instruments}")


    # ==================================== #
    # LOAD FERMI-LAT AND H.E.S.S. DATASETS #
    # ==================================== #


    # NOTE: For some sources, Fermi-LAT not available, so skip this step!
    # Only load Fermi-LAT if in instruments list of source
    dataset_flat = None
    if "Fermi-LAT" in target_instruments:
        # Re-run GammaPy analysis of Fermi-LAT - For this, FermiPy setup and analysis must be carried out first!
        try:
            # Generate GammaPy datasets from the FermiPy output, including models
            # NOTE: Suppress astropy WCS warnings re: changed FITS naming conventions
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = FITSFixedWarning)
                dataset_flat = get_flat_dataset(target = target, bblock = args.bblock)

        except:
            raise Exception("FermiPy analysis of Fermi-LAT not found! Run initial setup / analysis step in FermiPy.")

    # Generate or load H.E.S.S. datasets and models
    # Only load HESS if in instruments list of source
    if "HESS" in target_instruments:
        # Run GammaPy analysis of HESS - For this, GammaPy HESS temporal analysis must be carried out first! (to generate configs)
        try:

            hap_dataset = None
            hap_config = None
            # If dataset and config given by user, prioritize those!
            if args.dataset is not None:
                hap_dataset = args.dataset
            if args.config is not None:
                hap_config = args.config
            # If dataset argument not given, read from file, or use defaults
            if args.dataset is None:
                hap_dataset = hess_config[target].get("hap_dataset", HAP_DATASET)
            if args.config is None:
                hap_config = hess_config[target].get("hap_config", HAP_CONFIG)

            # Generate GammaPy datasets from the HESS data directly
            _, dataset_hess = get_hess_dataset(target = target, bblock = args.bblock, 
                                               dataset = hap_dataset, config = hap_config)

            log.info(f"Loaded H.E.S.S. dataset using {hap_dataset} dataset and {hap_config} configuration.")

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

    # TODO: PERFORM INTRINSIC SPECTRAL MODEL TEST?

    # Read model from HESS config file, if possible, and if model argument not given

    # Define target source model 
    # If no specific model given
    if args.model is None:

        # Read model from HESS config file, if possible
        try:
            model = MODELS_EQUIV[ hess_config[target]["blocks"][args.bblock]["model"] ]
        except:
            # Otherwise, use default PowerLaw model
            model = PowerLawSpectralModel()

        # Define target sky model
        target_model = SkyModel(
            name = target, datasets_names = ["HESS", "Fermi-LAT"],
            spectral_model = model,
            spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )

        # If a compound spectral model (containing EBL), choose intrinsic model only
        if isinstance(target_model, CompoundSpectralModel):
            target_model_intrinsic = target_model.spectral_model.model1
        else:
            # Otherwise, take spectral model directly
            target_model_intrinsic = target_model.spectral_model

    # If model argument given, override
    else:

        # Define SkyModel for given spectral model
        target_model = SkyModel(
            name = target, datasets_names = ["HESS", "Fermi-LAT"],
            spectral_model = MODELS_EQUIV[args.model],
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
            energy = np.geomspace(
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

    # For SmoothBrokenPowerLaw, set Ebreak initial value and bounds based on datasets
    if target_model.spectral_model.intrinsic_model.__class__.__name__ == "SmoothBrokenPowerLawSpectralModel":
        if kwargs.get("ebreak", None) is None:
            kwargs["ebreak"] = float(np.sqrt(dataset_energy_range[0] * dataset_energy_range[1]).value)
        # TODO: Set break energy based on datasets! (same as value)
        if kwargs.get("ebreak_min", None) is None:
            kwargs["ebreak_min"] = float(dataset_energy_range[0].value)
        if kwargs.get("ebreak_max", None) is None:
            kwargs["ebreak_max"] = float(dataset_energy_range[-1].value)
        # NOTE: Free beta parameter? For now, leave fixed
        target_model.spectral_model.parameters["beta"].frozen = True

    # Set all parameter bounds based on kwargs or defaults
    set_param_bounds(target_model, bounds = "fit", **kwargs)

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

        else:
            # Set Fermi-LAT dataset as None
            dataset_flat_unbias = None

        # Create final dataset to be used, including models
        dataset_joint = Datasets(
            # Include only not-None datasets
            [dataset for dataset in (dataset_flat_unbias, dataset_hess_bias) if dataset is not None] )

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


    # Run spectral fitting procedure
    results_joint = run_spectral_fit(dataset = dataset_joint, target = target, use_edec = args.edec)

    # Compute flux points on both Fermi-LAT and HESS
    fluxp_joint_flat, fluxp_joint_hess = run_fluxp_estimator(dataset = dataset_joint, target = target)


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

    # Update model entry (only if EBL dominguez, as that's the reference)
    if args.ebl == "dominguez":
        hess_config[target]["blocks"][args.bblock]["model"] = dataset_joint.models[target].spectral_model.intrinsic_model.__class__.__name__.split("SpectralModel")[0]
        # Save all info to file
        with open(CONFIGS_DIR / "hess_config.yaml", "w") as f:
            yaml.dump(hess_config, f, sort_keys = False)

    # Save Fermi-LAT flux points (ensuring TeV scale units)
    if dataset_flat is not None:
        ascii.write(
            table = tab_uconv("TeV", fluxp_joint_flat.to_table(sed_type = "e2dnde")),
            output = dir_gout / "joint_flat_fluxp.ecsv", format = "ecsv", overwrite = True, )
    # Save HESS flux points (ensuring TeV scale units)
    ascii.write(
        table = tab_uconv("TeV", fluxp_joint_hess.to_table(sed_type = "e2dnde")),
        output = dir_gout.resolve() / "joint_hess_fluxp.ecsv", format = "ecsv", overwrite = True, )

    # Generate plots
    log.info("Generating plots...")
    plot_sed_joint(target, bblock = args.bblock, ebl = args.ebl)

    log.info(f"Joint Fermi-LAT + H.E.S.S. GammaPy Analysis complete! :)")

    return


if __name__ == "__main__":

    main()

