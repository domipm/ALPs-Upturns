# Script used for modeling upturns onto our data
# Generates C-stat grid by fitting upturn model for each pair of E break and Delta Gamma
# using as baseline the best-fit model of the given dataset (including any EBL or bias)
# Requires main fit to be performed for desired dataset

import  os
import  gc
import  argparse

from    pathlib                     import  Path

import  matplotlib.pyplot           as      plt
import  matplotlib.colors           as      clr

import  numpy                       as      np
import  astropy.units               as      u

from    gammapy.modeling            import  Fit, Parameter
from    gammapy.modeling.models     import  Models, SpectralModel, SkyModel
from    gammapy.datasets            import  Datasets, FluxPointsDataset
from    gammapy.estimators          import  FluxPointsEstimator

from    models                      import  SmoothBrokenSpectralModel
from    utils                       import  get_source_info, get_edec, get_etau, init_log, parse_kwargs
from    plots                       import  plot_sed_gammapy


def is_converged(params):
    """
    Script to check for convergence of all parameters for a model
    """

    # Loop over free parameters
    for param in params:
        if param.frozen == False:
            # Check if parameter saturates
            if np.round(param.value, 2) == param.min or np.round(param.value, 2) == param.max:
                return False, f"Parameter {param.name} saturated (value: {param.value})!"
            # Check if parameter has very low relative error
            if param.error / param.value > 10.0:
                return False, f"Parameter {param.name} has too low relative error!"
    # If passes these tests, converged
    return True, "Converged"


def plot_cstatgrid(target, bblock, dataset, which, rcolors = True, cmap = "viridis", plot_best = False):

    # NOTE: Other nice looking colormaps: OrRd + White, Viridis_r + Black

    # Define fname and labels
    if which == "cstat":
        fname = f"upturns_cstat_{dataset}.npz"
        ptitle = r'{} C-stat Surface'.format(target)
        pcblab = r'C-statistic'
    if which == "dcstat":
        fname = f"upturns_dcstat_{dataset}.npz"
        # ptitle = r'{} $\Delta$C-stat Surface'.format(target)
        ptitle = r'{} Upturn Grid'.format(target)
        pcblab = r'$\Delta$C-statistic'
    # Load data file
    data = np.load(file = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/" + fname)
    # Select corresponding values
    ibreak_grid = data['X']
    ebreak_grid = data['Y']
    dcstat_grid = data['Z']

    # Define colors
    colors = [cmap, "white"]
    if rcolors == True:
        colors = [cmap + "_r", "black"]
    
    # Plot grid surface
    colormesh = plt.pcolormesh(
        ibreak_grid, ebreak_grid, dcstat_grid, 
        shading = 'auto',
        cmap = colors[0],
        # Use sqrt normalization for color
        norm = clr.PowerNorm(gamma = 0.5),
        edgecolors = (colors[1], 0.15),
        linewidth = 0.001,)

    if which == "dcstat":

        # Plot contour map with confidence levels (68% - 1sigma, 90% - 2sigma, 95% - 3 sigma)
        contours = plt.contour(
            ibreak_grid, ebreak_grid, dcstat_grid,
            levels = [2.30, 4.61, 5.99], colors = colors[1], linewidths = 1.5,)

        # Get best-fit point (where DeltaC-stat = 0)
        idx_bf = np.where(dcstat_grid == 0.0)
        ibreak_bf = ibreak_grid[idx_bf]
        ebreak_bf = ebreak_grid[idx_bf]

        # Plot best fit point
        if plot_best:
            plt.scatter(ibreak_bf, ebreak_bf, label = "Best-fit parameters",
                        s = 100, marker = "*", color = colors[1], zorder = 5)

        # Plot contour labels
        plt.clabel(contours, inline = True, fontsize=10, 
                fmt={2.30: r'1$\sigma$', 4.61: r'2$\sigma$', 5.99: r'3$\sigma$'},)
    
    # Plot colorbar
    plt.colorbar(colormesh, label = pcblab)

    # Add label for dataset
    label_dataset = {
        "joint": r"$\it{Fermi}$-LAT and H.E.S.S.",
        "joint_bias": r"$\it{Fermi}$-LAT and H.E.S.S.",
        "hess": r"H.E.S.S.",
        "hess_bias": r"H.E.S.S.", }

    plt.title(label_dataset[dataset] + f"\nBlock {bblock.split("-")[0].split("block")[-1]}", loc = "left", x = 0.035, y = 0.875, bbox = dict(facecolor = 'white', edgecolor = 'gray', boxstyle = 'round', alpha = 0.75), fontsize = 10)

    # Axis labels and formatting
    plt.xlabel(r'$\Delta \Gamma$')
    plt.ylabel(r'$E_{\mathrm{break}}$ [TeV]')

    # if "block1" not in bblock:
    #     ptitle = ptitle.split(" ")[0] + f" (Block {bblock.split("-")[0].split("block")[-1]}) Upturn Grid"
    plt.title(ptitle)
    
    plt.xscale('linear')
    plt.yscale('log')
    # Save plot
    if which == "dcstat" and plot_best:
        plt.legend()
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/" + fname.split(".")[0] + ".pdf", bbox_inches = "tight")
    plt.close()

    return


def plot_bestfit(target, bblock, dataset):

    # Load data file
    data = np.load(file = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/upturns_cstat_{dataset}.npz")

    # Select corresponding values
    ibreak_grid = data['X']
    ebreak_grid = data['Y']
    dcstat_grid = data['Z']

    # Get ebreak, ibreak for lowest cstat

    # Find index of minimum
    idx = np.unravel_index(np.argmin(dcstat_grid), dcstat_grid.shape)

    # Extract corresponding values
    ibreak_best = ibreak_grid[idx]
    ebreak_best = ebreak_grid[idx] * u.TeV
    dcstat_min = dcstat_grid[idx]

    # Load best-fit intrinsic model for this block
    model = Models.read(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/{dataset}_models.yaml")[target]

    # Add smooth break and plot the model
    model_upturn_spectral = model.spectral_model * SmoothBrokenSpectralModel(
        indexdelta_brk = ibreak_best,
        e_brk = ebreak_best, )
    
    # Save this compound model
    model_upturn = SkyModel(name = f"{target} Upturn", spectral_model = model_upturn_spectral)
    Models([model_upturn]).write(path = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/upturn_bestfit.yaml",
                                 overwrite = True)
    
    # Define energy range
    if dataset == "joint" or dataset == "joint_bias":
        energy_range = [1e-3, 31.6] * u.TeV
    else:
        energy_range = [0.1, 31.6] * u.TeV

    # Plot best-fit upturn model
    ax = model_upturn_spectral.plot(energy_range, sed_type = "e2dnde",
                                    color = "tab:orange", label = "Upturn Model")
    model_upturn_spectral.plot_error(energy_range, ax = ax, color = "tab:orange", sed_type = "e2dnde")


    # Plot non-upturned model to compare
    model.spectral_model.plot(energy_range, ax = ax, sed_type = "e2dnde", 
                              color = "tab:blue", label = "Best-fit Model")
    model.spectral_model.plot_error(energy_range, ax = ax, sed_type = "e2dnde", color = "tab:blue")

    # Plot non-upturned flux points
    plot_sed_gammapy(target, bblock, inst = dataset, ax = ax,
                     plot_model = False, plot_fluxp = True, 
                     color = "tab:blue", label = None,
                     save_plot = False)

    ax.set_ylim(1e-16, 1e-10)

    plt.legend()
    # Save plot
    plt.tight_layout()
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/upturn_bestfit.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    return


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")

    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121, or all)")

    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")

    parser.add_argument("--ebl", default = "dominguez", help = "EBL absorption model to use (loaded from EBLTable). Default: dominguez")

    parser.add_argument("--dataset", choices = ["hess", "hess_bias", "joint", "joint_bias"], default = "hess", help = "Which dataset and model to use for upturn search. Options: {'hess', 'joint'}. Default: 'hess'")

    parser.add_argument("--npoints", type = int, default = 20, help = "Number of point to take for each axis in grid search (E_break and DeltaIndex_break)")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--tau-limit", default = 1, help = "Minimum energy limit for grid search corresponding to optical depth value. Default: 1")

    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments (if given)
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get info of source
    target_4FGL, target_position, target_redshift = get_source_info(target)

    if args.plots_only:
        plot_cstatgrid(target, args.bblock, dataset = args.dataset, which = "cstat")
        plot_cstatgrid(target, args.bblock, dataset = args.dataset, which = "dcstat")
        plot_bestfit(target, args.bblock, dataset = args.dataset)
        exit()

    log = init_log(target, fname = f"model_upturns_{args.dataset}.log", bblock = args.bblock)

    # Define output directory for current block (TODO: Separate into subfolder?)
    # dir_gout = Path( f"{os.environ['RESULTS']}/{target}/{args.bblock}/upturns/" )
    dir_gout = Path( f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/" )
    dir_pout = Path( f"{os.environ['RESULTS']}/{target}/{args.bblock}/plots/" )

    # Load dataset and best-fit intrinsic model (by default, take hess data, choice for joint)
    log.info("Loading datasets and models...")
    datasets = Datasets.read(
        filename = dir_gout.joinpath(f"{args.dataset}_datasets.yaml"),
        filename_models = dir_gout.joinpath(f"{args.dataset}_models.yaml"), )
    
    # Get spectral model for target from dataset (including EBL)
    model_dataset = datasets.models[target].spectral_model
    # Check if loading biased model, if so, load only intrinsic and ebl
    if "bias" in datasets.models[target].parameters.names:
        # Ignore bias parameter
        # model_dataset = datasets.models[target].spectral_model.intrinsic_model * datasets.models[target].spectral_model.ebl_model
        # Otherwise, also include bias parameter
        model_dataset = datasets.models[target].spectral_model

    # Combine model with additional smooth break in the spectra
    model_spectral = model_dataset * SmoothBrokenSpectralModel()
    # Define spatial model (same as that of datasets)
    model_spatial = datasets.models[target].spatial_model
    # Define sky model for target source
    model_target = SkyModel(
        spectral_model = model_spectral,
        spatial_model = model_spatial,
        name = target,
        datasets_names = ["Fermi-LAT", "HESS"],)

    # Remove target model from Fermi-LAT dataset
    models = Models()
    for model in datasets.models:
        # Also remove unbiased target if present
        if model.name != target and model.name != f"{target} Unbiased":
            models.append(model)

    # Add custom joint model to models list
    models.append(model_target)
    # Add models to dataset
    datasets.models = models

    # Compute decorrelation energy for model
    e_dec = get_edec(datasets.models[target])
    # Compute energy at tau optical depth limit
    e_tau = get_etau(target_redshift, args.ebl, tau_lim = args.tau_limit)

    # Set reference to decorrelation energy (this should already be set! only change if not equal)
    # if e_dec != datasets.models[target].parameters["reference"].quantity:
    #     datasets.models[target].parameters["reference"].quantity = e_dec
    # Set break energy to E(tau = tau_lim)
    datasets.models[target].parameters["e_brk"].quantity = e_tau
    # Set index delta value to -0.1 by default
    datasets.models[target].parameters["indexdelta_brk"].value = - 0.1
    log.info("Datasets and models loaded!")

    # Set amplitude bounds if required (sometimes needed to relax maximum)
    if "index" in datasets.models[target].parameters.names:
        datasets.models[target].parameters["index"].max = kwargs.get("index_max", 7.5)
    if "alpha" in datasets.models[target].parameters.names:
        datasets.models[target].parameters["alpha"].max = kwargs.get("alpha_max", 7.5)
    if "beta" in datasets.models[target].parameters.names:
        datasets.models[target].parameters["beta"].max = kwargs.get("beta_max", 2.5)
        datasets.models[target].parameters["beta"].min = kwargs.get("beta_min", 0.0)
    datasets.models[target].parameters["amplitude"].max = kwargs.get("amplitude_max", 1e-5)

    # Define grid for Index2_brk = gamma2 and E_brk = ebreak2
    ibreak_vals = np.linspace(kwargs.get("ibreak_min", -0.01), -5.00, args.npoints) # from 0 (no break) to -5 (sharp break) linearly spaced
    ebreak_vals = np.logspace(np.log10( e_tau.value ), np.log10( 31.6 ), args.npoints) # from E(tau) to 31.6 TeV

    # Generate meshgrid
    ibreak_grid, ebreak_grid = np.meshgrid(ibreak_vals, ebreak_vals, indexing = "ij")
    cstat_grid = np.zeros_like(ibreak_grid)

    # Create a copy of the dataset to restore to
    dataset_org = datasets.copy()

    # Initialize new fitting object
    fit = Fit()

    # Loop over both axes in grid
    for i, ibreak in enumerate(ibreak_vals):
        log.info(f"*** deltaindex_brk = {ibreak:.6f} ***")
        for j, ebreak in enumerate(ebreak_vals):
            log.info(f"··· e_brk = {ebreak:6f} ···")

            # Restore dataset to original state
            dataset = dataset_org

            # Set break parameters from loop
            dataset.models[target].parameters["e_brk"].value = ebreak
            dataset.models[target].parameters["indexdelta_brk"].value = ibreak
            # Freeze these parameters
            dataset.models[target].parameters["e_brk"].frozen = True
            dataset.models[target].parameters["indexdelta_brk"].frozen = True

            # Rest of the parameters (index, amplitude, etc. depend on intrinsic model)
            # are left free to vary within their default bounds
            if "index" in datasets.models[target].parameters.names:
                dataset.models[target].parameters["index"].frozen = False
            elif "alpha" in datasets.models[target].parameters.names:
                dataset.models[target].parameters["alpha"].frozen = False
                dataset.models[target].parameters["beta"].frozen = False
                # If beta parameter effectively zero - use equivalent power law
                if (0 <= dataset.models[target].parameters["beta"].value <= 1e-5):
                    dataset.models[target].parameters["beta"].frozen = True
            dataset.models[target].parameters["amplitude"].frozen = False
            # Make sure reference is frozen just in case
            dataset.models[target].parameters["reference"].frozen = True

            if "bias" in datasets.models[target].parameters.names:
                # Freeze bias if present
                dataset.models[target].parameters["bias"].frozen = True
                # Otherwise, leave it free
                # dataset.models[target].parameters["bias"].frozen = False

            # Attempt fit with retry
            attempts = 10
            success = False
            # Restart result value
            result = None
            # Loop over all attempts
            for attempt in range(attempts):
                # Try to run fit
                try:

                    # On retries, perturb free parameters randomly
                    if attempt > 0:
                        for par in dataset.models[target].parameters:
                            if not par.frozen:
                                par.value = par.min + np.random.uniform(0.1, 0.9) * (par.max - par.min)

                    result = fit.run(dataset)
                    # Check convergence of parameters for baseline model
                    converged, reason = is_converged(result.models[target].parameters)
                    if converged == True:
                        log.info(f"Fit passed convergence test!")
                    else:
                        log.warning(f"Fit failed convergence test! Reason: {reason}. Retrying...")
                        continue
                    # If fit successful
                    if result.success:
                        # Print info on screen
                        log.info(f"Fit success! Gamma2 = {ibreak:.2f}, Ebreak2 = {ebreak:.2e}, C-stat = {result.total_stat:.2f}, Beta = {dataset.models[target].parameters["beta_brk"].value:.2f}")
                        # Print also intrinsic parameters
                        for par in dataset.models[target].parameters:
                            if par.frozen == False:
                                log.info(f"\t{par.name} = {par.value:.4e} +/- {par.error:.4e}")
                        # Add result to grid
                        cstat_grid[i, j] = result.total_stat
                        # Set success flag
                        success = True
                        # Break from for loop!
                        break
                    # If fit unsucessful
                    else:
                        log.warning(f"\tFit failed at attempt {attempt}! Retrying...")
                # Catch exceptions
                except Exception as e:
                    log.error(f"\tException at fit attempt! Reason: {e}. Continuing...")

            # If fit has not converged, return NaN
            if not success:
                cstat_grid[i, j] = np.nan
                log.warning("Max attempts reached!")

        # Force python to collect garbage (memory usage)
        gc.collect()

    # Get minimum C-stat and compute DeltaC-stat grid
    min_cstat = np.nanmin(cstat_grid)
    delta_cstat_grid = cstat_grid - min_cstat

    log.info(f"Saving files to {dir_gout}")
    np.savez(file = dir_gout.joinpath(f"upturns_cstat_{args.dataset}"),
             # Save as X, Y, Z
             X = ibreak_grid, Y = ebreak_grid, Z = cstat_grid)
    # Save DeltaC-stat meshgrid
    np.savez( file = dir_gout.joinpath(f"upturns_dcstat_{args.dataset}"),
             # Save as X, Y, Z
             X = ibreak_grid, Y = ebreak_grid, Z = delta_cstat_grid)

    # Generate plots
    log.info(f"Generating and saving plots to {dir_pout}")
    plot_cstatgrid(target, args.bblock, dataset = args.dataset, which = "cstat")
    plot_cstatgrid(target, args.bblock, dataset = args.dataset, which = "dcstat")
    plot_bestfit(target, args.bblock, dataset = args.dataset)
