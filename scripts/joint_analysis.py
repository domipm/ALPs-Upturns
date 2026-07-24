# Analysis script for joint Fermi-LAT and HESS analysis with GammaPy
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `fermi_analysis.py`, `flat_analysis.py`, and `hess_analysis.py` to generate required files!

import  os
import  argparse

import  numpy                       as      np

from    pathlib                     import  Path

from    astropy                     import  units as u
from    astropy.io                  import  ascii

from    ebltable.tau_from_model     import  OptDepth

from    gammapy.maps                import  MapAxis
from    gammapy.modeling            import  Fit, Parameter, Parameters
from    gammapy.modeling.models     import  Models, SkyModel, SpectralModel, PointSpatialModel, TemplateSpectralModel
from    gammapy.datasets            import  Datasets, SpectrumDatasetOnOff
from    gammapy.estimators          import  FluxPointsEstimator

from    alpsup.models               import  BiasedPriorSpectrumDatasetOnOff, CompositeSpectralModel, EBLTableSpectralModel
from    alpsup.utils                import  get_source_info, par_uconv, tab_uconv, parse_kwargs, get_edec
from    alpsup.logs                 import  init_log
from    alpsup.plots                import  plot_sed_combine, plot_sed_joint
from    alpsup.paths                import  get_results_dir


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")

    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121, or all)")

    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")

    parser.add_argument("--ebl", default = "dominguez", help = "EBL absorption model to use (loaded from EBLTable). Default: dominguez")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Run only plots
    if args.plots_only:
        plot_sed_combine(target, bblock = args.bblock)
        plot_sed_joint(target = target, bblock = args.bblock, **kwargs)
        exit()

    # Define output directories - they must exist by now!
    dir_gout = get_results_dir(target, args.bblock, args.ebl, output = "gamma-out")
    dir_fout = get_results_dir(target, args.bblock, output = "gamma-out")

    # Get info of source
    target_4FGL, target_position, target_redshift = get_source_info(target)

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, fname = "joint_analysis.log", bblock = args.bblock)

    # Display info of the target
    log.info("Joint Fermi-LAT and HESS analysis of target source:\n{}\n({})\nPosition: {}\nRedshift z = {}".format(target, target_4FGL, target_position.data, target_redshift))

    # =============================== #
    # LOAD DATASETS AND DEFINE MODELS #
    # =============================== #

    # Global models object
    models = Models()

    # Load Fermi-LAT datasets object containing models
    dataset_flat = Datasets.read(
        filename = f"{dir_fout}/flat_datasets.yaml",
        filename_models = f"{dir_fout}/flat_models.yaml", )[0]
    # Remove the target source from the dataset (we define it later based on HESS model including EBL)
    dataset_flat.models = [model for model in dataset_flat.models if model.name != target]
    # Freeze all parameters of all background models
    dataset_flat.models.freeze()

    # If background model found, change filename path for serialization
    if "Models Background" in dataset_flat.models.names:
        dataset_flat.models["Models Background"].spatial_model.filename = f"{dir_fout}/flat_models_background.fits"

    # Append Fermi-LAT models (without target) to global models object
    for model in dataset_flat.models:
        # Convert the units to HESS scale (TeV)
        model_conv = par_uconv("TeV", model.copy(name = model.name), "energy")
        # Append model with converted units to models objects
        models.append(model_conv)
    # Extract Fermi-LAT models
    models_flat = dataset_flat.models

    # Load HESS datasets object containing target model
    dataset_hess = Datasets.read(
        filename = f"{dir_gout}/hess_datasets.yaml",
        filename_models = f"{dir_gout}/hess_models.yaml", )[0]
    # Extract HESS models
    models_hess = dataset_hess.models

    # Extract target source from dataset
    target_model = dataset_hess.models[target].copy(name = target, copy_data = True, datasets_names = ["HESS", "Fermi-LAT"])
    # Convert parameters of the model to HESS scale (TeV)
    target_model_conv = par_uconv("TeV", target_model.copy(name = target_model.name))

    model_intrinsic = target_model_conv.spectral_model.model1
    
    # Re-define EBL model!
    # Define energy range
    e_tau = np.logspace(-3, 1.5, 200) * u.TeV
    # Create model from EBLTable
    tau = OptDepth.readmodel(model = args.ebl)
    att = np.exp(-1. * tau.opt_depth(target_redshift, e_tau.value))
    # Define model from template
    model_ebl = TemplateSpectralModel(energy = e_tau, values = att * u.dimensionless_unscaled)

    # Define target model including new EBL
    target_model_newebl = SkyModel(
        name = target,
        datasets_names = ["HESS", "Fermi-LAT"],
        spectral_model = model_intrinsic * model_ebl,
        spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec, frame = "icrs"), )
    # Freeze spatial compoents
    target_model_newebl.spatial_model.freeze()


    # Append it to models object
    # models.append(target_model_conv)
    models.append(target_model_newebl)

    # Define joint datasets object for Fermi-LAT and HESS (including other models)
    dataset_joint = Datasets( [ dataset_flat, dataset_hess ] )
    # Add all models to the datasets
    dataset_joint.models = models

    # Modify model parameter limits if given
    dataset_joint.models[target].parameters["amplitude"].min = kwargs.get("amplitude_min", 1e-15)
    dataset_joint.models[target].parameters["amplitude"].max = kwargs.get("amplitude_max", 1e-08)
    
    # ================== #
    # RUN MAIN JOINT FIT #
    # ================== #

    # Run the joint fit and print the results
    log.info("Running initial fit...")
    fit_joint = Fit()
    results_joint = fit_joint.run(datasets = dataset_joint)
    log.info("Initial fit done!")

    edec = get_edec(dataset_joint.models[target])
    try:
        edec_gp = dataset_joint.models[target].spectral_model.model1.pivot_energy
    except:
        edec_gp = None
    log.info(f"Decorrelation energy: {edec:.6f} (GammaPy cross-check: {edec_gp:.6f})")
    # If they are not the same, default to gammapy's version
    if edec_gp != edec:
        edec = edec_gp

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

    # Save final datasets and models
    dataset_joint.write(
        filename = f"{dir_gout}/joint_datasets.yaml",
        filename_models = f"{dir_gout}/joint_models.yaml",
        overwrite = True, )
    # Save fit results
    results_joint.write(
        path = dir_gout.joinpath("joint_fit.yaml"),
        overwrite_templates = True,
        overwrite = True, )

    log.info("Computing flux points for Fermi-LAT...")
    # Compute flux points for each dataset separately
    energy_edges_joint_fermi = dataset_joint["Fermi-LAT"].counts.geom.axes["energy"].edges
 
    fluxp_joint_flat = FluxPointsEstimator(
        energy_edges = energy_edges_joint_fermi.to(u.TeV),
        source = target, selection_optional=["all"]
    ).run([dataset_joint["Fermi-LAT"]])
    log.info("Flux points estimator for Fermi-LAT done!")

    # Create flux point table
    fluxp_joint_flat_tab = fluxp_joint_flat.to_table(sed_type = "e2dnde")
    # Ensure correct units in the table (TeV in this case)
    fluxp_joint_flat_tab_conv = tab_uconv("TeV", fluxp_joint_flat_tab)
    # Save flux points table as csv file
    ascii.write(
        table = fluxp_joint_flat_tab_conv,
        output = f"{dir_gout}/joint_flat_fluxp.ecsv",
        format = 'ecsv',
        overwrite = True,)
    
    log.info("Computing flux points for HESS...")
    energy_edges_joint_hess = dataset_joint["HESS"].counts.geom.axes["energy"].edges
    fluxp_joint_hess = FluxPointsEstimator(
        energy_edges = energy_edges_joint_hess.to(u.TeV), 
        source = target, selection_optional=["all"]
    ).run([dataset_joint["HESS"]])
    log.info("Flux points estimator for HESS done!")

    # Create flux point table
    fluxp_joint_hess_tab = fluxp_joint_hess.to_table(sed_type = "e2dnde")
    # Ensure correct units in the table (TeV in this case)
    fluxp_joint_hess_tab_conv = tab_uconv("TeV", fluxp_joint_hess_tab)
    # Save flux points table as csv file
    ascii.write(
        table = fluxp_joint_hess_tab_conv,
        output = f"{dir_gout}/joint_hess_fluxp.ecsv",
        format = 'ecsv',
        overwrite = True,)
    
    # ================================== #
    # BIASED JOINT FIT WITH BIAS ON HESS #
    # ================================== #

    # Joint analysis on Fermi-LAT + HESS biased datasets and models
    log.info("Running joint fit on biased HESS model and unbiased Fermi-LAT model with bias prior on dataset")

    # Joint datasets objects to contain:
    # unbiased Fermi-LAT dataset and models
    # and biased HESS dataset and model
    dataset_joint_bias = Datasets()

    # Load Fermi-LAT dataset and models (without target)
    dataset_flat_bias = dataset_flat.copy(name = "Fermi-LAT")

    # Create new biased version of target model for HESS
    spectral_model_hess = CompositeSpectralModel(
        # Intrinsic spectral model
        intrinsic_model = dataset_hess.models[target].spectral_model.model1,
        # EBL model
        ebl_model = EBLTableSpectralModel.read_ebl(
            energy = np.logspace(-3, 1.5, 200) * u.TeV, ebl_name = args.ebl, redshift = target_redshift),
        # Don't include upturn model
        upturn_model = None,
        # Bias parameter initial value
        bias = 0.0, )

    # Define sky model for hess spectral model
    ps_model_hess = SkyModel(
        datasets_names = "HESS",
        spectral_model = spectral_model_hess,
        spatial_model = dataset_hess.models[target].spatial_model,
        name = f"{target}", )
    models_hess = Models(ps_model_hess)

    # Set bias prior to be 15% energy scale
    sigma_bias = 0.15

    # Wrap dataset with bias prior
    dataset_hess_bias = BiasedPriorSpectrumDatasetOnOff.from_spectrum_dataset(
        dataset_hess, sigma_bias, )
    # Add biased model
    dataset_hess_bias.models = models_hess

    # Create new datasets object
    datasets_hess_bias = Datasets([dataset_hess_bias])

    # Create unbiased version of target model for Fermi-LAT
    target_model_bias = dataset_hess_bias.models[target]
    target_model_unbias = SkyModel(
        name = f"{target} Unbiased",
        datasets_names = "Fermi-LAT",
        # Same spatial model
        spatial_model = target_model_bias.spatial_model,
        # Spectral model being the intrinsic and ebl without bias
        spectral_model = target_model_bias.spectral_model.intrinsic_model * target_model_bias.spectral_model.ebl_model, )

    # Add original Fermi-LAT models and to list
    models_flat = Models(models_flat)
    # Freeze all parameters
    models_flat.freeze(model_type = "spatial")
    models_flat.freeze(model_type = "spectral")
    # Add unbiased model to list
    models_flat.append(target_model_unbias)
    
    # Add these models to the Fermi-LAT dataset
    dataset_flat_bias.models = models_flat

    # Add datasets with their models to the joint dataset
    datasets_joint_bias = Datasets([dataset_flat_bias, dataset_hess_bias])

    # Set reference parameter
    datasets_joint_bias.models[target].parameters["reference"].quantity = edec

    # Run joint biased fit
    log.info("Running biased fit...")
    results_joint_bias = Fit().run(datasets = datasets_joint_bias)
    log.info("Biased fit done!")

    # Display info on fit and best model
    log.info(results_joint_bias)
    log.info(datasets_joint_bias["HESS"].models[0])

    # Save final results
    results_joint_bias.write(
        path = f"{dir_gout}/joint_bias_fit.yaml",
        overwrite_templates = True,
        overwrite = True, )
    # Save final datasets and model
    datasets_joint_bias.write(
        filename = f"{dir_gout}/joint_bias_datasets.yaml",
        filename_models = f"{dir_gout}/joint_bias_models.yaml",
        overwrite = True, )
    
    # ============== #
    # GENERATE PLOTS #
    # ============== #

    log.info("Generating plots...")
    # Generate combined SED plot
    plot_sed_combine(target, bblock = args.bblock)
    # Generate joint SED plot
    plot_sed_joint(target, bblock = args.bblock)

    log.info(f"Joint Fermi-LAT + H.E.S.S. GammaPy Analysis complete! :)")

