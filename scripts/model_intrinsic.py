# Script used to find best model, parameters, and fit statistic for different
# spectral models on the joint dataset
# Requires conda environment: `env_alps`
# Create the environment with: conda env create -f envs/env_alps.yaml
# Run first `joint_analysis.py` to generate required files!


import  os
import  yaml
import  warnings
import  argparse

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    matplotlib.lines            import  Line2D

from    ebltable.tau_from_model     import  OptDepth

from    astropy                     import  units as u

from    gammapy.datasets            import  Datasets
from    gammapy.estimators          import  FluxPointsEstimator
from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import  (
            Models, SkyModel,
            PowerLawSpectralModel, LogParabolaSpectralModel, SmoothBrokenPowerLawSpectralModel, TemplateSpectralModel, )

from    utils                       import  get_source_info, parse_kwargs
from    models                      import  BiasedCompoundSpectralModel


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")
    
    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121, or all)")
    parser.add_argument("--bblock", default = "baseline", help = "Which Bayesian block to consider for loading joint fit results. Default: baseline")
    
    parser.add_argument("--ebl", default = "dominguez", help = "EBL model to use")
    parser.add_argument("--dataset", default = "joint", choices = ["hess", "joint", "hess_bias", "joint_bias"], help = "Which dataset to load from. Default: joint_bias")
    
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

    # Load joint datasets and models
    dataset_joint = Datasets.read(
        filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/{args.dataset}_datasets.yaml",
        filename_models = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/{args.dataset}_models.yaml", )
    
    # Extract best-fit parameters for the target source model (take only intrinsic spectral model)
    bf_params = dataset_joint.models[target].spectral_model.parameters
    model_target_org = dataset_joint.models[target]

    # Obtain value for energy at which optically thick tau < 1
    tau_optdepth = OptDepth.readmodel(model = "dominguez")
    # Define energy grid (fixed at interval 0.1 TeV to 10 TeV)
    energy_grid = np.logspace(-1, 1.5, 200)
    # Compute tau values at different energies for the target's redshift
    tau_values = np.array( [ tau_optdepth.opt_depth(target_redshift, e) for e in energy_grid ] )
    # Obtain value of energy for which tau = 2 (interpolating) - in TeV by default
    tau_limit_e = np.interp(2, tau_values, energy_grid) * u.TeV

    # Limit energy bounds to be within E < E(tau = 2) - convert to TeV
    # Begin at minimum Fermi-LAT energy (1 GeV), and up to E(tau = 2)
    energy_range_hess = [ dataset_joint["HESS"].counts.geom.axes["energy"].bounds[0], tau_limit_e ]
    try:
        energy_range_flat = [ dataset_joint["Fermi-LAT"].counts.geom.axes["energy"].bounds[0], dataset_joint["Fermi-LAT"].counts.geom.axes["energy"].bounds[-1] ]
    except:
        energy_range_flat = energy_range_hess

    # Define over total energy range (limited to E(tau))
    energy_range_total = [ energy_range_flat[0], energy_range_hess[-1] ]

    # Define EBL model from EBLTable
    e_tau = np.logspace(np.log(energy_range_total[0].to("TeV").value), np.log(energy_range_hess[-1].to("TeV").value), 200) * u.TeV
    
    # Create model from EBLTable
    tau = OptDepth.readmodel(model = "dominguez")
    att = np.exp(-1. * tau.opt_depth(target_redshift, e_tau.value))
    # Define model from template
    model_ebl = TemplateSpectralModel(energy = e_tau, values = att)

    # Run joint fit using different models
    models_list = [PowerLawSpectralModel() , LogParabolaSpectralModel()]

    # Remove target model from dataset
    models = Models()
    for model in dataset_joint.models:
        if model.name != target and model.name != f"{target} Unbiased":
            models.append(model)
    # Set new models to dataset without target
    dataset_joint.models = models

    # Initialize plotting
    fig, ax = plt.subplots()
    # Choose as many colors as models to test from colormap
    # cmap = plt.get_cmap('viridis', len(models_list))
    cmap = plt.get_cmap('jet', len(models_list))
    # Get array of RGBA colors  
    colors = cmap(np.linspace(0, 1, len(models_list)))  
    # Set axis units
    ax.xaxis.set_units(u.Unit("TeV"))
    ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))
    # Set labels
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(r"$\text{E}^2 d\text{N}/d\text{E}$ [TeV cm$^{-2}$ s$^{-1}$]")
    # Define sed type
    sed_type = "e2dnde"

    # Keep track of model name, total stat, free parameters, and aic score
    scores = {}

    # Loop over all models
    for k, model_spectral in enumerate(models_list):
        print(f"Processing model: {model_spectral.__class__.__name__}")

        # Try to delete previous dataset, if possible (for memory issues)
        try:
            del dataset
        except:
            pass

        # Re-load joint datasets and models (to clear cached parameters)
        dataset = dataset_joint.copy()

        # Define new target source
        model_target = SkyModel(
            name = target,
            datasets_names = ["Fermi-LAT", "HESS"],
            spatial_model = model_target_org.spatial_model,
            spectral_model = model_spectral * model_ebl, )
        # Add models to dataset
        dataset.models = Models([ *models, model_target ])

        # Make sure model acts on both datasets
        dataset.models[target].datasets_names = ["Fermi-LAT", "HESS"]

        # Set parameter bounds for the model
        # Set index bounds
        if "index" in dataset.models[target].parameters.names:
            dataset.models[target].parameters["index"].min = kwargs.get('index_min', 0.0)
            dataset.models[target].parameters["index"].max = kwargs.get('index_max', 5.0)
        if "alpha" in dataset.models[target].parameters.names:
            dataset.models[target].parameters["alpha"].min = kwargs.get('alpha_min', -5.0)
            dataset.models[target].parameters["alpha"].max = kwargs.get('alpha_max', +5.0)
        if "beta" in dataset.models[target].parameters.names:
            dataset.models[target].parameters["beta"].min = kwargs.get('beta_min', 0.0)
            dataset.models[target].parameters["beta"].max = kwargs.get('beta_max', 2.0)
        # Set amplitude bounds
        dataset.models[target].parameters["amplitude"].min = kwargs.get('amplitude_min', 1e-14)
        dataset.models[target].parameters["amplitude"].max = kwargs.get('amplitude_max', 1e-08)
        # Use by default decorrelation energy for dataset
        dataset.models[target].parameters["reference"].quantity = kwargs.get('reference', bf_params["reference"].value) * u.TeV

        # Set safe-fit mask to run fit only on given energy range (only affects HESS upper bound)
        dataset["HESS"].mask_fit = dataset["HESS"].counts.geom.energy_mask(
            energy_min = energy_range_total[0],
            energy_max = energy_range_total[1], )

        # Run the fit
        fit = Fit(optimize_opts = {"use_cache": False})
        results = fit.run(datasets = dataset)

        # Display result of fit
        print(results)
        # Display best-fit model
        print(dataset.models[target])

        # Add relevant information to dictionary
        scores[model_spectral.__class__.__name__] = {
            "stat": float(results.total_stat),
            "n_params": len(dataset.models[target].spectral_model.parameters.free_parameters),
            "aic": float(results.total_stat) + 2 * len(dataset.models[target].spectral_model.parameters.free_parameters), 
            # TODO: ADD BEST-FITS MODELS!
            "model": dataset.models[target].to_dict(full_output = True), }

        # Compute flux points for each dataset
        try:
            fluxp_fermi = FluxPointsEstimator(
                energy_edges = dataset["Fermi-LAT"].counts.geom.axes["energy"].edges,
                source = target, selection_optional = ["all"],
            ).run([dataset["Fermi-LAT"]])
        except:
            pass
        fluxp_hess = FluxPointsEstimator(
            energy_edges = dataset["HESS"].counts.geom.axes["energy"].edges,
            source = target, selection_optional = ["all"],
        ).run([dataset["HESS"]])
        # Plot flux points
        try:
            fluxp_fermi.plot(
                ax = ax,
                color = colors[k],
                sed_type = "e2dnde",
                marker = "o", )
        except:
            pass
        fluxp_hess.plot(
            ax = ax,
            color = colors[k],
            marker = "s",
            sed_type = "e2dnde", )

        # Plot target source model
        # dataset_joint.models[target].spectral_model.plot(
        dataset.models[target].spectral_model.plot(
            ax = ax,
            energy_bounds = [ energy_range_flat[0], energy_range_hess[1] ],
            sed_type = "e2dnde",
            color = colors[k],
            alpha = 0.75,
            linestyle = "--",
            label = str( models_list[k].__class__.__name__ ))
        # dataset_joint.models[target].spectral_model.plot_error(
        dataset.models[target].spectral_model.plot_error(
            ax = ax,
            energy_bounds = [ energy_range_flat[0], energy_range_hess[1] ],
            sed_type = "e2dnde",
            color = colors[k], )
        
    # Compare log parabola with power law: sigma ~ sqrt stat_pl - stat_lp
    if (scores["PowerLawSpectralModel"]["stat"] > scores["LogParabolaSpectralModel"]["stat"]):
        model_pref = "LogParabolaSpectralModel"
        sigma_pref = float( np.sqrt( scores["PowerLawSpectralModel"]["stat"] - scores["LogParabolaSpectralModel"]["stat"] ) )
    else:
        model_pref = "PowerLawSpectralModel"
        sigma_pref = float(0.0)
    scores["CurvatureTest"] = {"model": model_pref, "sigma": sigma_pref}

    # Save total statistic values and number of free parameters for each model
    with open(f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/intrinsic_scores.yaml", "w") as f:
        yaml.dump(
            scores, f,
            sort_keys = False,
            default_flow_style = False, )

    # Custom legend handles (for black square and circle)
    custom_handles = [
        Line2D([0], [0], marker='s', color='k', linestyle='None', label='H.E.S.S.'),
        Line2D([0], [0], marker='o', color='k', linestyle='None', label='Fermi-LAT'),]

    # Get existing handles/labels from plotted lines
    handles, labels = plt.gca().get_legend_handles_labels()

    # Add custom handles
    handles.extend(custom_handles)
    labels.extend(['H.E.S.S.', 'Fermi-LAT'])

    plt.xlim((energy_range_flat[0].to(u.TeV).value, energy_range_hess[1].to(u.TeV).value))

    plt.title(f"{target} Spectral Model Comparison")
    # Create combined legend
    plt.legend(handles, labels)
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{args.bblock}/plots/intrinsic_model.pdf", bbox_inches = "tight")
    plt.close()

    # TODO: If sigma greater than threshold, update bblocks.yaml file with best-fit model
