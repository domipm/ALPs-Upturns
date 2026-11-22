# ...

import  os
import  re
import  sys
import  logging
import  argparse
import  datetime

from    scipy.interpolate           import  RectBivariateSpline
import  numpy                       as      np
import  matplotlib.pyplot           as      plt
import  matplotlib.colors           as      clr

from    pathlib                     import  Path

from    astropy                     import  units as u
from    astropy.io                  import  ascii
from    astropy.table               import  Table

from    ebltable.tau_from_model     import  OptDepth

from    gammapy.maps                import  MapAxis
from    gammapy.modeling            import  Fit, Parameter, Parameters
from    gammapy.modeling.models     import  Models, SkyModel, SpectralModel, PointSpatialModel, TemplateSpectralModel
from    gammapy.datasets            import  Datasets, SpectrumDatasetOnOff, Dataset
from    gammapy.estimators          import  FluxPointsEstimator

from    alpsup.utils                import  get_source_list, get_source_info, parse_kwargs, get_edec, get_etau, is_converged
from    alpsup.models               import  CompositeSpectralModel, UpturnSpectralModel
from    alpsup.paths                import  get_results_dir
from    alpsup.logs                 import  init_log


# Import required functions from main analysis
from    spectral_analysis           import  run_spectral_fit, run_fluxp_estimator
from    upturn_modeling             import  setup_upturn_model, run_upturn_model
from    likelihood_analysis         import  interp_alp_upturn

# Import parameters from ALP simulations
# from    alps_simulations            import  MALP, B0, N0, L0, GALP_LIST
MALP = 1e-3
B0 = 0.001
N0 = 1e-7
L0 = 1e4
GALP_LIST = np.logspace(-2, +1, 30)


# Get logger function
log = logging.getLogger(__name__)


def simulate_dataset(datasets: Datasets) -> Datasets:
    """
    Description...
    """

    # Define simulated datasets object
    datasets_sim = Datasets()

    # Loop over each dataset loaded
    for dataset in datasets:

        # Simulate fake data for dataset
        if dataset.__class__.__name__ in ["SpectrumDatasetOnOff", "SpectrumDataset", "BiasedPriorSpectrumDatasetOnOff"]:
            log.info(f"Found HESS dataset of type {dataset.__class__.__name__}. Generating fake data!")
            # Simulate fake data (spectrum dataset)
            dataset.fake(npred_background = dataset.background)

        elif dataset.__class__.__name__ in ["MapDatasetOnOff", "MapDataset"]:
            log.info(f"Found Fermi-LAT dataset of type {dataset.__class__.__name__}. Generating fake data!")
            # Simulate fake data (map dataset)
            dataset.fake()

        # Append simulated dataset to datasets object
        datasets_sim.append(dataset)

    # Return simulated datasets object
    return datasets_sim


def main():
    
    parser = argparse.ArgumentParser(description = ("Interpolate ALP simulation best-fit parameters onto upturn C-stat grids and produce detection and constraint curves."))

    # Which source to simulate
    parser.add_argument("--source", required = True, # default = "PKS0447-439", 
                        choices = get_source_list(), help = "Which source to use for Monte Carlo simulations")

    # EBL model to consider
    parser.add_argument("--ebl", required = True, # default = "dominguez",
                        help = "EBL model(s) to process (default: dominguez)")
    # Time segmentation block
    parser.add_argument("--bblock", required = True, 
                        help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs).")

    # Which Monte Carlo simulation to perform (ALP versus No-ALP case)
    parser.add_argument("--mode", choices = ["ALP", "NoALP"], required = True, type = str,
                        help = "Which Monte Carlo simulation to perform: ALP, or NoALP case.")
    # Keep track of simulation ID or number of simulations to perform!
    parser.add_argument("--sim-id", required = True, type = int,
                        help = "Monte Carlo simulation ID (for NoALP case) or number of simulations to perform per coupling (for ALP case).")
    parser.add_argument("--galp-idx", required = False, type = int, default = None, help = "Run only a single ALP coupling")
    # Compute flux points or not (for saving time, once established fit converges)
    parser.add_argument("--fluxp", action = argparse.BooleanOptionalAction, default = True,
                        help = "Compute and plot flux points (--fluxp, --no-fluxp).")

    # TODO: Option to overwrite previous simulations
    # parser.add_argument("--overwrite", action = "store_true", 
    #                     help = "Overwrite previous simulations")
    
    # Other keyword arguments
    parser.add_argument("--kwargs", nargs="*",
                        help="Extra keyword arguments as key=value pairs")
    
    # Parse all arguments
    args = parser.parse_args()
    # Parse keyword arguments if given
    if args.kwargs:
        _ = parse_kwargs(args.kwargs)

    # Get the name of target source
    target = args.source

    # Get info on source from data class
    source = get_source_info(target)
    target_4FGL = source.name_4FGL
    target_position = source.position
    target_redshift = source.redshift
    target_instruments = source.inst


    # Directory of main analysis folder
    dir_gout = get_results_dir(target, args.bblock, args.ebl, output = "gamma-out")


    # ======================= #
    # MONTE CARLO NO-ALP CASE #
    # ======================= #


    if args.mode.lower() == "noalp":

        # Directory of Monte Carlo analysis results
        dir_mcout = get_results_dir(f"MONTECARLO/{target}", args.bblock, args.ebl, output = f"{args.mode}/sim_{args.sim_id:05d}")

        # Initialize logging (initialize after possible plotting, otherwise gets removed!)
        log = init_log(path = dir_mcout, fname = "montecarlo_analysis.log")   

        # Display info of the target
        log.info(f"Monte Carlo {args.mode} analysis of target source: {target})")


        # ========================================= #
        # LOAD DATASETS AND GENERATE SIMULATED DATA #
        # ========================================= #


        # Load dataset
        datasets = Datasets.read(
                    filename = dir_gout / f"joint_datasets.yaml",
                    filename_models = dir_gout / f"joint_models.yaml", )

        # NOTE: IMPORTANT - READING DATASETS DOESN'T ENSURE LINKED PARAMETERS REMAIN LINKED!

        # Define spectral model for Fermi-LAT (unbiased, with the same upturn as HESS)
        if "Fermi-LAT" in datasets.names:
            models = Models()
            for model in datasets.models:
                if model.name != f"{target} Unbiased":
                    models.append(model)
            models.append(
                SkyModel(
                    name = f"{target} Unbiased", datasets_names = "Fermi-LAT",
                    spatial_model = datasets.models[target].spatial_model,
                    spectral_model = CompositeSpectralModel(
                        intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                        ebl_model = datasets.models[target].spectral_model.ebl_model,
                        bias = 0,
                        upturn_model = None, ) ) )
            models[f"{target} Unbiased"].spectral_model.parameters["bias"].frozen = True
            datasets.models = models

        # Generate simulated dataset
        datasets_sim = simulate_dataset(datasets)

        # TODO: SAVE SIMULATED DATASET?

        # Get energy bounds of dataset
        datasets_sim_energy_bounds = [datasets_sim.energy_ranges[0][0].to(u.TeV), datasets_sim.energy_ranges[-1][-1].to(u.TeV)]


        # ====================== #
        # INTRINSIC SPECTRAL FIT #
        # ====================== #


        # Run intrinsic spectral fit on the simulated dataset
        fit_results = run_spectral_fit(datasets_sim, target)
        if args.fluxp:
            fit_fluxp_flat, fit_fluxp_hess = run_fluxp_estimator(datasets_sim, target)

        # Plot intrinsic spectral fit

        fig, ax = plt.subplots()

        ax.xaxis.set_units(u.Unit("TeV"))
        ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

        fit_results.models[target].spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "--")
        fit_results.models[target].spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")
        if args.fluxp:
            if fit_fluxp_flat is not None:
                fit_fluxp_flat.plot(ax = ax, sed_type = "e2dnde", color = "navy", capsize = 2, capthick = 1)
            if fit_fluxp_hess is not None:
                fit_fluxp_hess.plot(ax = ax, sed_type = "e2dnde", color = "crimson", capsize = 2, capthick = 1)

        ax.set_xlim(datasets_sim_energy_bounds)
        ax.set_ylim(1e-15, 1e-10)

        plt.tight_layout()
        plt.savefig(dir_mcout / "intrinsic_fit.png", dpi = 300, bbox_inches = "tight")
        plt.close()


        # =============== #
        # UPTURN MODELING #
        # =============== #


        # Keep track of best-fit intrinsic model
        model_org = datasets_sim.models[target]

        # Prepare dataset for upturn modeling
        setup_upturn_model(datasets = datasets_sim, target = target, 
                           target_redshift = target_redshift, ebl = args.ebl, tau_limit = 1, )

        # Run upturn modeling
        cstat_grid, ibreak_grid, ebreak_grid = run_upturn_model(datasets = datasets_sim, target = target, target_redshift = target_redshift, e_tau =  get_etau(target_redshift, args.ebl, tau_lim = 1), )

        # Construct C-stat grid as a dictionary with all the grids
        cstat_dict = {
            'X': ibreak_grid,
            'Y': ebreak_grid,
            'Z': cstat_grid, }

        # Compute delta upturn grid
        dcstat_grid = cstat_grid - np.nanmin(cstat_grid)
            
        # Save C-stat meshgrid
        np.savez(file = dir_mcout / f"upturns_cstat",
                # Save as X, Y, Z
                X = ibreak_grid, Y = ebreak_grid, Z = cstat_grid)
        # Save DeltaC-stat meshgrid
        np.savez(file = dir_mcout / f"upturns_dcstat",
                # Save as X, Y, Z
                X = ibreak_grid, Y = ebreak_grid, Z = dcstat_grid)

        # Plot DeltaC-stat (as diagnostic)

        # Plot colormesh
        colormesh = plt.pcolormesh(
            ibreak_grid, ebreak_grid, dcstat_grid,
            shading = "auto", cmap = "viridis_r",
            norm = clr.PowerNorm(gamma = 0.5),
            edgecolors = ("white", 0.15),
            linewidth = 0.001, )
        # Plot contour levels        
        contours = plt.contour(
            ibreak_grid, ebreak_grid, dcstat_grid,
            levels = [2.30, 4.61, 5.99], colors = "black", linewidths = 1.5,)
        # Plot contour labels
        plt.clabel(contours, inline = True, fontsize=10, 
                fmt={2.30: r'1$\sigma$', 4.61: r'2$\sigma$', 5.99: r'3$\sigma$'},)
        # Plot colorbar
        plt.colorbar(colormesh, label = r'$\Delta$C-statistic')


        plt.xlabel(r'$\Delta \Gamma$')
        plt.ylabel(r'$E_{\mathrm{break}}$ [TeV]')
        plt.xscale("linear")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(dir_mcout / "upturn_dcstat.png", dpi = 300, bbox_inches = "tight", )
        plt.close()

        # Plot best-fit model including upturn

        # Find index of minimum
        idx = np.unravel_index(np.argmin(dcstat_grid), dcstat_grid.shape)
        # Extract corresponding values
        ibreak_best = ibreak_grid[idx]
        ebreak_best = ebreak_grid[idx] * u.TeV

        # Log what the best-fit parameters are
        log.info(f"Best-fit upturn parameters:\nE_upt = {ebreak_best}\nDeltaGamma_upt = {ibreak_best}")

        # Define upturn model
        model_upturn = model_org.copy(copy_data = True)
        model_upturn.spectral_model.upturn_model = UpturnSpectralModel(indexdelta_upt = ibreak_best, e_upt = ebreak_best)
        # Save the model
        Models([model_upturn]).write(path = dir_mcout / "upturn_bestfit.yaml", overwrite = True, full_output = True, overwrite_templates = True, )

        # Plot best-fit intrinsic model
        fig, ax = plt.subplots()

        ax.xaxis.set_units(u.Unit("TeV"))
        ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

        fit_results.models[target].spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "--")
        fit_results.models[target].spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")
        if args.fluxp:
            fit_fluxp_flat.plot(ax = ax, sed_type = "e2dnde", color = "navy", capsize = 2, capthick = 1, label = r"$\it{Fermi}$-LAT")
            fit_fluxp_hess.plot(ax = ax, sed_type = "e2dnde", color = "crimson", capsize = 2, capthick = 1, label = "H.E.S.S.")

        # Plot best-fit upturn model
        model_upturn.spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "-")
        model_upturn.spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")

        ax.set_xlim(datasets_sim_energy_bounds)
        ax.set_ylim(1e-15, 1e-10)

        plt.tight_layout()
        plt.savefig(dir_mcout / "upturn_fit.png", dpi = 300, bbox_inches = "tight")
        plt.close()


        # =================== #
        # LIKELIHOOD ANALYSIS #
        # =================== #


        # Save results for each time block
        lambda_block = {}
        like_block = {}

        # Loop over all couplings
        for k, galp in enumerate(GALP_LIST):

            # Load ALP upturn fit data
            popt = np.load(
                # Load ALP upturn fit data
                get_results_dir(target, bblock = "alps",
                        output = args.ebl) / f"sim_fit_m{MALP:g}_g{galp:g}_B0{B0:g}.npz" )["popt"]

            # Perform interpolation
            ts, c_noalp, c_alp, cstat_min = interp_alp_upturn(popt, cstat_dict, percentile = 5)

            # Save raw interpolation output (per g_alp)
            # os.makedirs(dir_mcout / "alps/")
            # np.savez(dir_mcout / f"alps/interp_ts_m{MALP}_g{galp}_B0{B0}", 
            #          X = c_noalp, Y = c_alp, Z = ts, )

            # Ensure correct unit scale of g_alp
            galp *= 1e-11

            # Likelihood curve C_ALP - C_NoALP
            lambda_block[galp] = c_alp - c_noalp
            # Raw likelihood C_ALP per block
            like_block[galp] = c_alp

        # Save final likelihood curves and raw likelihoods for all couplings for this source
        with open(dir_mcout / "alp_likecurve.txt", "w") as f:
            for coupling in lambda_block.keys():
                f.write(f"{coupling}, {lambda_block[coupling]}\n")
        with open(dir_mcout / "alp_likelihood.txt", "w") as f:
            for coupling in like_block.keys():
                f.write(f"{coupling}, {like_block[coupling]}\n")

        # Plot resulting likelihood curve
        plt.plot(
            list(like_block.keys()),
            list(like_block.values()) - np.full_like(list(like_block.values()), np.min(list(like_block.values()))),
            marker = ".",
            linestyle = "-", )
        plt.hlines(y = 2.71, xmin = 1e-13, xmax = 1e-10, linestyle = "--", color = "gray")
        plt.xscale("log")
        plt.savefig(dir_mcout / f"alp_likecurve.png", dpi = 300)
        plt.close()


    # ==================== #
    # MONTE CARLO ALP CASE #
    # ==================== #


    if args.mode.lower() == "alp":

        # Directory of ALP results folder
        dir_aout = get_results_dir(target, output = f"alps/{args.ebl}")

        # Directory of Monte Carlo analysis results for current sim-id
        dir_mcout = get_results_dir(f"MONTECARLO/{target}", args.bblock, args.ebl, output = f"{args.mode}/sim_{args.sim_id:05d}")

        # Initialize logging (initialize after possible plotting, otherwise gets removed!)
        log = init_log(path =  dir_mcout / "logs", 
                       fname = "montecarlo_simulations.log")   

        # Display info of the target
        log.info(f"Monte Carlo {args.mode} analysis of target source: {target})")


        # ================================== #
        # LOAD SIMULATED ALP UPTURN FIT DATA #
        # ================================== #


        # Loop over all couplings!
        for k, galp in enumerate(GALP_LIST):

            # If coupling index argument given, run just that one!
            if args.galp_idx is not None:
                if k != args.galp_idx:
                    continue

            log.info(f"Processing coupling g_alp = {galp:g}")
            
            # New subfolder for each coupling!
            dir_mcout_galp = get_results_dir(f"MONTECARLO/{target}", args.bblock, args.ebl, output = f"{args.mode}/sim_{args.sim_id:05d}/g{galp:g}")

            # Load best-fit ALP upturn parameters from simulations
            alp_sim_data = np.load(dir_aout.resolve().joinpath( f"sim_fit_m0.001_g{galp:g}_B00.001.npz") )

            alp_popt = alp_sim_data['popt']
            alp_pcov = alp_sim_data['pcov']

            # Choose randomly within the length of parameter list
            alp_idx = np.random.choice(len(alp_popt))
            log.info(f"Chosen simulation index / magnetic field realization number {alp_idx}")

            # Get ALP upturn DeltaIndex
            indexdelta_upt_alp = alp_popt[alp_idx][1]
            indexdelta_upt_err_alp = np.sqrt(alp_pcov[alp_idx][1][1])

            # Get ALP upturn Ebreak (in TeV)
            e_upt_alp = alp_popt[alp_idx][0] * 1e-3 * u.TeV
            e_upt_err_alp = np.sqrt(alp_pcov[alp_idx][0][0]) * 1e-3 * u.TeV

            log.info("Best-fit ALP upturn parameters chosen:")
            log.info("E_upt_alp = %s +/- %s", e_upt_alp, e_upt_err_alp.value)
            log.info("IndexDelta_upt_alp = %s +/- %s", float(indexdelta_upt_alp), float(indexdelta_upt_err_alp))

            # Load dataset
            datasets = Datasets.read(
                        filename = dir_gout / f"joint_datasets.yaml",
                        filename_models = dir_gout / f"joint_models.yaml", 
                        checksum = False, )

            # NOTE: IMPORTANT - READING DATASETS DOESN'T ENSURE LINKED PARAMETERS REMAIN LINKED!

            # NOTE: Check if Fermi-LAT also needs upturn model?
            model_upturn = UpturnSpectralModel(
                    e_upt = e_upt_alp, indexdelta_upt = indexdelta_upt_alp)

            # Define spectral model for Fermi-LAT (unbiased, with the same upturn as HESS)
            models_upt = Models()
            if "Fermi-LAT" in datasets.names:
                for model in datasets.models:
                    if model.name not in [target, f"{target} Unbiased"]:
                        models_upt.append(model)
                # Append unbiased model to Fermi-LAT dataset
                models_upt.append(
                    SkyModel(
                        name = f"{target} Unbiased", datasets_names = "Fermi-LAT",
                        spatial_model = datasets.models[target].spatial_model,
                        spectral_model = CompositeSpectralModel(
                            intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                            ebl_model = datasets.models[target].spectral_model.ebl_model,
                            bias = 0,
                            upturn_model = model_upturn, ) ) )
                # Freeze bias for Fermi-LAT model
                models_upt[f"{target} Unbiased"].spectral_model.parameters["bias"].frozen = True
            # NOTE: Append upturn model to HESS dataset (parameters should still be linked)
            if "HESS" in datasets.names:
                models_upt.append(
                    SkyModel(
                        name = target, datasets_names = "HESS",
                        spatial_model = datasets.models[target].spatial_model,
                        spectral_model = CompositeSpectralModel(
                            intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                            ebl_model = datasets.models[target].spectral_model.ebl_model,
                            bias = datasets.models[target].spectral_model.bias,
                            upturn_model = model_upturn, ) ) )
                # NOTE: Freeze upturn parameters for HESS dataset
                models_upt[target].spectral_model.parameters["indexdelta_upt"].frozen = True
                models_upt[target].spectral_model.parameters["e_upt"].frozen = True
            datasets.models = models_upt

            log.info(models_upt[target].spectral_model)

            # Generate simulated dataset
            datasets_sim = simulate_dataset(datasets)

            # Get energy bounds of dataset
            datasets_sim_energy_bounds = [datasets_sim.energy_ranges[0][0].to(u.TeV), datasets_sim.energy_ranges[-1][-1].to(u.TeV)]

            # Save simulated dataset
            datasets_sim.write(
                filename = dir_mcout_galp / "joint_datasets.yaml",
                filename_models = dir_mcout_galp / "joint_models.yaml",
                overwrite = True, )

            # NOTE: Before continuing with the rest of the analysis
            # remove the upturn model from the dataset! The simulated data will already contained the effects of the ALP-simualted upturn.
            models = Models()
            # Keep all Fermi-LAT models if present
            if "Fermi-LAT" in datasets_sim.names:
                for model in datasets_sim["Fermi-LAT"].models:
                    if model.name != f"{target} Unbiased":
                        models.append(model)
                models.append(
                        SkyModel(
                            name = f"{target} Unbiased", datasets_names = "Fermi-LAT",
                            spatial_model = datasets.models[target].spatial_model,
                            spectral_model = CompositeSpectralModel(
                                intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                                ebl_model = datasets.models[target].spectral_model.ebl_model,
                                bias = 0.00,
                                upturn_model = None), ) )
                # Freeze the bias
                models[f"{target} Unbiased"].spectral_model.parameters["bias"].frozen = True
            # Use model without an upturn for the intrinsic modeling
            if "HESS" in datasets_sim.names:
                models.append(
                    SkyModel(
                        name = target, datasets_names = "HESS",
                        spatial_model = datasets.models[target].spatial_model,
                        spectral_model = CompositeSpectralModel(
                            intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                            ebl_model = datasets.models[target].spectral_model.ebl_model,
                            bias = datasets.models[target].spectral_model.bias,
                            upturn_model = None), ) )
            datasets_sim.models = models

            log.info(datasets_sim.models[f"{target}"])
            log.info(datasets_sim.models[f"{target} Unbiased"])

            # ====================== #
            # INTRINSIC SPECTRAL FIT #
            # ====================== #


            # Run intrinsic spectral fit on the simulated dataset
            fit_results = run_spectral_fit(datasets_sim, target)
            if args.fluxp:
                fit_fluxp_flat, fit_fluxp_hess = run_fluxp_estimator(datasets_sim, target)

            # Plot intrinsic spectral fit

            fig, ax = plt.subplots()

            ax.xaxis.set_units(u.Unit("TeV"))
            ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

            fit_results.models[target].spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "--")
            fit_results.models[target].spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")
            if args.fluxp:
                if fit_fluxp_flat is not None:
                    fit_fluxp_flat.plot(ax = ax, sed_type = "e2dnde", color = "navy", capsize = 2, capthick = 1)
                if fit_fluxp_hess is not None:
                    fit_fluxp_hess.plot(ax = ax, sed_type = "e2dnde", color = "crimson", capsize = 2, capthick = 1)

            ax.set_xlim(datasets_sim_energy_bounds)
            ax.set_ylim(1e-15, 1e-10)

            plt.tight_layout()
            plt.savefig(dir_mcout_galp / "intrinsic_fit.png", dpi = 300, bbox_inches = "tight")
            plt.close()
 
            # Save results of the fit to file
            fit_results.write(path = dir_mcout_galp / "intrinsic_fit.yaml", overwrite = True, full_output = True, overwrite_templates = True, write_covariance = True, checksum = False)


            # =============== #
            # UPTURN MODELING #
            # =============== #


            # Keep track of best-fit intrinsic model
            model_org = datasets_sim.models[target]

            # Prepare dataset for upturn modeling
            setup_upturn_model(datasets = datasets_sim, target = target, 
                            target_redshift = target_redshift, ebl = args.ebl, tau_limit = 1, )

            # Run upturn modeling
            cstat_grid, ibreak_grid, ebreak_grid = run_upturn_model(datasets = datasets_sim, target = target, target_redshift = target_redshift, e_tau =  get_etau(target_redshift, args.ebl, tau_lim = 1), )

            # Construct C-stat grid as a dictionary with all the grids
            cstat_dict = {
                'X': ibreak_grid,
                'Y': ebreak_grid,
                'Z': cstat_grid, }

            # Compute delta upturn grid
            dcstat_grid = cstat_grid - np.nanmin(cstat_grid)
                
            # Save C-stat meshgrid
            np.savez(file = dir_mcout_galp / f"upturns_cstat",
                    # Save as X, Y, Z
                    X = ibreak_grid, Y = ebreak_grid, Z = cstat_grid)
            # Save DeltaC-stat meshgrid
            np.savez(file = dir_mcout_galp / f"upturns_dcstat",
                    # Save as X, Y, Z
                    X = ibreak_grid, Y = ebreak_grid, Z = dcstat_grid)

            # Plot DeltaC-stat (as diagnostic)

            # Plot colormesh
            colormesh = plt.pcolormesh(
                ibreak_grid, ebreak_grid, dcstat_grid,
                shading = "auto", cmap = "viridis_r",
                norm = clr.PowerNorm(gamma = 0.5),
                edgecolors = ("white", 0.15),
                linewidth = 0.001, )
            # Plot contour levels        
            contours = plt.contour(
                ibreak_grid, ebreak_grid, dcstat_grid,
                levels = [2.30, 4.61, 5.99], colors = "black", linewidths = 1.5,)
            # Plot contour labels
            plt.clabel(contours, inline = True, fontsize=10, 
                    fmt={2.30: r'1$\sigma$', 4.61: r'2$\sigma$', 5.99: r'3$\sigma$'},)
            # Plot colorbar
            plt.colorbar(colormesh, label = r'$\Delta$C-statistic')


            plt.xlabel(r'$\Delta \Gamma$')
            plt.ylabel(r'$E_{\mathrm{break}}$ [TeV]')
            plt.xscale("linear")
            plt.yscale("log")

            plt.tight_layout()
            plt.savefig(dir_mcout_galp / "upturn_dcstat.png", dpi = 300, bbox_inches = "tight", )
            plt.close()

            # Plot best-fit model including upturn

            # Find index of minimum
            idx = np.unravel_index(np.argmin(dcstat_grid), dcstat_grid.shape)
            # Extract corresponding values
            ibreak_best = ibreak_grid[idx]
            ebreak_best = ebreak_grid[idx] * u.TeV

            # Log what the best-fit parameters are
            log.info(f"Best-fit upturn parameters:\nE_upt = {ebreak_best}\nDeltaGamma_upt = {ibreak_best}")

            # Define upturn model
            model_upturn = model_org.copy(copy_data = True)
            model_upturn.spectral_model.upturn_model = UpturnSpectralModel(indexdelta_upt = ibreak_best, e_upt = ebreak_best)
            # Save the model
            Models([model_upturn]).write(path = dir_mcout_galp / "upturn_bestfit.yaml", overwrite = True, full_output = True, overwrite_templates = True, )

            # Plot best-fit intrinsic model
            fig, ax = plt.subplots()

            ax.xaxis.set_units(u.Unit("TeV"))
            ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

            fit_results.models[target].spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "--")
            fit_results.models[target].spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")
            if args.fluxp:
                fit_fluxp_flat.plot(ax = ax, sed_type = "e2dnde", color = "navy", capsize = 2, capthick = 1, label = r"$\it{Fermi}$-LAT")
                fit_fluxp_hess.plot(ax = ax, sed_type = "e2dnde", color = "crimson", capsize = 2, capthick = 1, label = "H.E.S.S.")

            # Plot best-fit upturn model
            model_upturn.spectral_model.plot(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde", color = "black", linestyle = "-")
            model_upturn.spectral_model.plot_error(ax = ax, energy_bounds = datasets_sim_energy_bounds, sed_type = "e2dnde")

            ax.set_xlim(datasets_sim_energy_bounds)
            ax.set_ylim(1e-15, 1e-10)

            plt.tight_layout()
            plt.savefig(dir_mcout_galp / "upturn_fit.png", dpi = 300, bbox_inches = "tight")
            plt.close()


            # =================== #
            # LIKELIHOOD ANALYSIS #
            # =================== #


            # Save results for each time block
            lambda_block = {}
            like_block = {}

            # Loop over all couplings
            for k, galp in enumerate(GALP_LIST):

                # Load ALP upturn fit data
                popt = np.load(
                    # Load ALP upturn fit data
                    get_results_dir(target, bblock = "alps",
                            output = args.ebl) / f"sim_fit_m{MALP:g}_g{galp:g}_B0{B0:g}.npz" )["popt"]

                # Perform interpolation
                ts, c_noalp, c_alp, cstat_min = interp_alp_upturn(popt, cstat_dict, percentile = 5)

                # Ensure correct unit scale of g_alp
                galp *= 1e-11

                # Likelihood curve C_ALP - C_NoALP
                lambda_block[galp] = c_alp - c_noalp
                # Raw likelihood C_ALP per block
                like_block[galp] = c_alp

            # Save final likelihood curves and raw likelihoods for all couplings for this source
            with open(dir_mcout_galp / "alp_likecurve.txt", "w") as f:
                for coupling in lambda_block.keys():
                    f.write(f"{coupling}, {lambda_block[coupling]}\n")
            with open(dir_mcout_galp / "alp_likelihood.txt", "w") as f:
                for coupling in like_block.keys():
                    f.write(f"{coupling}, {like_block[coupling]}\n")

            # Plot resulting likelihood curve
            plt.plot(
                list(like_block.keys()),
                list(like_block.values()) - np.full_like(list(like_block.values()), np.min(list(like_block.values()))),
                marker = ".",
                linestyle = "-", )
            plt.hlines(y = 2.71, xmin = 1e-13, xmax = 1e-10, linestyle = "--", color = "gray")
            plt.xscale("log")
            plt.savefig(dir_mcout_galp / f"alp_likecurve.png", dpi = 300)
            plt.close()


        return


if __name__ == "__main__":

    main()

