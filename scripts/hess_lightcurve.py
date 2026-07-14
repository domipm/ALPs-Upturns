# Script to generate nightly-binned light curves for H.E.S.S. observations data

import  os
import  argparse

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path

from    astropy                     import  units   as  u
from    astropy.io                  import  ascii
from    astropy.time                import  Time
from    astropy.table               import  Table
from    astropy.stats               import  bayesian_blocks

from    gammapy.stats               import  WStatCountsStatistic
from    gammapy.datasets            import  Datasets, SpectrumDatasetOnOff
from    gammapy.estimators          import  LightCurveEstimator, FluxPointsEstimator
from    gammapy.modeling            import  Fit, FitResult, OptimizeResult, CovarianceResult
from    gammapy.modeling.models     import  PowerLawSpectralModel, SkyModel
from    gammapy.maps                import  MapAxis

from    alpsup.utils                import  parse_kwargs, get_source_info, get_source_list, init_log, gen_dirs, get_edec, is_converged
from    alpsup.plots                import  plot_ellipses, plot_lightcurve
from    alpsup.paths                import  get_hess_data_dir


# Define tolerance for time selection (~ 20 mins / avg HESS run)
ATOL = 0.015 * u.d


def get_ethresh(datasets):
    """Compute threshold energy for all datasets"""

    # Get threshold energy for stacked dataset
    emin_stack, emax_stack = datasets.stack_reduce().energy_range_total

    # Get threshold energy for per-observation datasets
    emin = max(dataset.energy_range_safe[0].data[0][0] * u.TeV for dataset in datasets)
    emax = min(dataset.energy_range_safe[-1].data[0][0] * u.TeV for dataset in datasets)    

    # Get maximum lower and minimum higher bounds
    emin = max(emin, emin_stack)
    emax = min(emax, emax_stack)

    # Get average threshold energy over all observations
    emin_avg = np.sum( [ dataset.energy_range_safe[0].data[0][0] for dataset in datasets ] ) / len(datasets) * u.TeV

    return emin, emax, emin_avg


def gen_tintervals(datasets, method = "night_times", **kwargs):

    # Get GTIs from datasets
    gti_starts = np.array( sorted( [ds.gti.time_start.mjd[0] for ds in datasets]) )
    gti_stops = np.array(  sorted( [ds.gti.time_stop.mjd[-1] for ds in datasets]) )
    # Array for final time intervals
    t_intervals = []

    # Nightly method based on binning over fixed times
    if method == "night_times":
    
        # Calculate midpoint of each observation
        midpoints = 0.5 * (gti_starts + gti_stops)
        # Generate night ids based on time of night start and midpoints
        night_start_hour = 12.0
        shift = night_start_hour / 24.0
        # Obs with midpoint between day N 12:00 and day N+1 12:00 get night_id = N
        night_ids = np.floor(midpoints - shift).astype(int)

        # Build intervals: for each unique night, span from earliest start to latest stop
        for night in np.unique(night_ids):
            t_intervals.append((
                Time(np.min(gti_starts[night_ids == night]), format="mjd", scale="utc"),
                Time(np.max(gti_stops[night_ids  == night]), format="mjd", scale="utc"), ))

    # Nightly simple method based on number of bins
    if method == "night_bins":

        t0 = Time(sorted(datasets.gti.time_start)[0].mjd, format = "mjd")
        bin_size = kwargs.get("bin_size", 24.) * u.hour
        tf = Time(sorted(datasets.gti.time_stop)[-1].mjd, format = "mjd")
        # Compute difference between final and start times, giving total days
        n_bins = (tf - t0).value + 1
        times = t0 + np.arange(n_bins) * bin_size
        t_intervals = [Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])]
            
    # Per-observation binning (return None for LightCurveEstimator)
    if method == "obs_times":
        return None

    return t_intervals


def time_resolved_spectroscopy(datasets: Datasets, table: Table, use_edec: bool = False, check_convergence: bool = False) -> Table:
    """
    Auxiliary function to perform time-resolved spectroscopy, which runs independent fit for each time interval.
    Based on GammaPy's 2.0 tutorial implementation. Runs fit of model on dataset for the given time intervals.
    Args:
        datasets (`gammapy.datasets.Datasets`): object containing all the data to fit, including model.
        time_intervals: time intervals over which to run the fit.
        set_edec (`bool`): compute and set decorrelation energy as reference for each fit. Default: False.
    Returns:
        table (`astropy.Table`): updated table for each flux point with fit results (nan if not converged)
    """

    # Get time intervals for each entry
    time_intervals = [ ( Time(row["time_min"], format = "mjd"), Time(row["time_max"], format = "mjd") ) for row in table ]

    # Copy table
    table = table.copy()

    # Add empty columns for ts, index + err, amplitude + err (fill with NaN initially)
    table.add_columns(
        names = ["fit_ts", "index", "index_err", "amplitude", "amplitude_err"],
        cols = [np.full(shape = len(table["time_ref"]), fill_value = np.nan)] * 5, )

    # Initialize fit and results array
    fit = Fit()

    # Keep all fit results
    results = []

    # For each time edge in intervals list
    for k, time in enumerate(time_intervals):

        t_min, t_max = time[0], time[1]

        # Select dataset to time interval, stack dataset
        datasets_to_fit = datasets.copy().select_time(time_min = t_min, time_max = t_max, atol = ATOL)

        # If time interval empty, skip
        if len(datasets_to_fit) == 0:    
            print(f"No Dataset for the time interval {t_min} to {t_max}. Skipping interval.")            
            # Keep nan values in this case
            continue

        # Stack-reduce the dataset, maintaining the PowerLaw model
        dataset_to_fit = datasets_to_fit.stack_reduce()
        dataset_to_fit.models = datasets_to_fit.models[0]
        # Make sure index is free
        dataset_to_fit.models.parameters["index"].frozen = False
        # Make sure amplitude is free
        dataset_to_fit.models.parameters["amplitude"].frozen = False

        # Run fit on stacked dataset and obtain decorrelation energy for this block
        result = fit.run(dataset_to_fit)
        # Append to array
        results.append(result)

        # If required, set decorrelation as reference and re-fit
        if use_edec:

            dataset_to_fit.models[0].parameters["reference"].quantity = result.models[0].spectral_model.pivot_energy
            result = fit.run(dataset_to_fit)

        # Check if fit has converged according to quality cuts
        converged, reason = is_converged(result.models[0].parameters)
        # If not converged and checking required, skip!
        if not converged and check_convergence:
            print(f"Fit not converged for the time interval {t_min} to {t_max}. Reason: {reason} Skipping interval.")
            continue
        
        # If convergence successful, append
        print(f"Fit converged for the time interval {t_min} to {t_max}.")

        # Update table row with values
        table["fit_ts"][k] = result.total_stat
        table["index"][k] = result.models[0].parameters["index"].value
        table["index_err"][k] = result.models[0].parameters["index"].error
        table["amplitude"][k] = result.models[0].parameters["amplitude"].value
        table["amplitude_err"][k] = result.models[0].parameters["amplitude"].error

        # Add further constraints if possible
        # (for flux points, we have ts, for index we don't)
        try:
            # If ts < 4 - upper limit on flux point as well
            table["is_ul"] = table["ts"] < 4
            # If flux is upper limit, index flux also upper limit
            table["index_is_ul"] = table["is_ul"]
        except:
            pass

    # Return updated light curve table
    return table, results


if __name__ == "__main__":

    # ========================= #
    # INITIALIZATION AND SET-UP #
    # ========================= #

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run Fermi-LAT analysis for a source using FermiPy")
    parser.add_argument("--source", required = True, choices = get_source_list(), help = "Source name")

    parser.add_argument("--dataset", default = "HAP-HD", help = "Which dataset to use (HAP-HD, HAP-FR, HAP-FITS)")
    parser.add_argument("--config", default = "std_ImPACT_fullEnclosure_updated", help = "Which reconstruction configuration to use (default: std_ImPACT_fullEnclosure)")

    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")

    parser.add_argument("--binning", default = "night_times", choices = ["night_times", "night_bins", "obs_times"], help = "Which time-binning to use. Default: nightly binning from GTI times")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--annotate", action = "store_true", help = "Annotate ellipses for identification. Default: False")
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

    # Generate and check directories
    gen_dirs(target, bblock = args.bblock)

    # Define output directories
    dir_base = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock or '')}/" )
    dir_gout = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock or '')}/gamma-out/" )
    dir_pout = Path( f"{os.environ["RESULTS"]}/{target}/{str(args.bblock or '')}/plots/" )

    # Run plotting-only
    if args.plots_only:
        plot_lightcurve(target, args.bblock)
        plot_ellipses(target, args.bblock, args.annotate, **kwargs)
        exit()

    # Initialize logging (initialize after possible plotting, otherwise gets removed!)
    log = init_log(target = target, fname = "hess_lightcurve.log", bblock = args.bblock)

    # ========================= #
    # LOAD DATASET AND GET INFO #
    # ========================= #

    # Load per-observation datasets
    log.info(f"Loading HESS data from {args.dataset} dataset using {args.config} configuration...")
    # Path to data directory
    dir_data = get_hess_data_dir(target, args.dataset.lower(), args.config)
    # Get all fits files
    obs_rmfs  = list(dir_data.glob('*rmf.fits'))
    obs_files = [ Path(str(x).replace("_rmf", "")) for x in obs_rmfs ]
    if not obs_files:
        raise FileNotFoundError(f"No observation files found in {dir_data}")
    # Empty dataset object
    dataset_obs = Datasets()
    # Append each observation to datasets
    for obs in obs_files:
        dataset_obs.append(SpectrumDatasetOnOff.read(obs))
    log.info(f"Loaded HESS data with {len(dataset_obs)} observations!")

    # Perform time selection (if given)
    dataset_obs = dataset_obs.select_time(
        time_min = Time( [ kwargs.get('tmin', sorted(dataset_obs.gti.time_start)[0].mjd ) ], format = "mjd" ),
        time_max = Time( [ kwargs.get('tmax', sorted(dataset_obs.gti.time_stop)[-1].mjd ) ], format = "mjd" ), )

    # Run initial fit on stacked dataset
    dataset = dataset_obs.stack_reduce()

    # Define HESS energy axis
    energy_axis_hess = MapAxis.from_energy_bounds(
        energy_min = dataset.counts.geom.axes["energy"].bounds[0],
        energy_max = dataset.counts.geom.axes["energy"].bounds[1],
        unit = u.TeV, nbin = 4, per_decade = True, )
    
    # Resample HESS energy axis for stacked dataset
    dataset = dataset.resample_energy_axis(energy_axis_hess)

    # Get Good Time Intervals (GTIs) and display total observation time
    log.info(f"GTI Info:")
    log.info(f"- Time start: {sorted(dataset_obs.gti.time_start)[0].mjd} MJD ~ {sorted(dataset_obs.gti.time_start)[0].iso} ISO")
    log.info(f"- Time stop : {sorted(dataset_obs.gti.time_stop)[-1].mjd} MJD ~ {sorted(dataset_obs.gti.time_stop)[-1].iso} ISO")

    # Get ON and OFF counts
    n_on = dataset.counts.data.sum()
    n_off = dataset.counts_off.data.sum() if dataset.counts_off is not None else 0
    alpha = dataset.alpha.data.mean() if dataset.alpha is not None else 1.0

    # Get information on available data
    log.info(f"Data Info:")
    log.info(f"- Total observation time: {dataset_obs.gti.time_sum:.6f} ~ {dataset_obs.gti.time_sum.to(u.h):.6f}")
    # Compute detection significance (Li & Ma approach)
    if n_off > 0:
        stat = WStatCountsStatistic(n_on = n_on, n_off = n_off, alpha = alpha)
        significance_lima = stat.sqrt_ts
        log.info(f"- Detection significance: {significance_lima:.4f} σ")
    else:
        log.error(f"No OFF counts available! Unable to estimate significance")

    # Get information available on energy
    log.info(f"Energy Info:")
    log.info(f"- Total energy range: {np.min(dataset_obs.energy_ranges[0].value):.3f} TeV - {np.max(dataset_obs.energy_ranges[1].value):.3f} TeV")
    # Define energy bounds
    energy_bounds = [ np.min(dataset_obs.energy_ranges[0].value), np.max(dataset_obs.energy_ranges[1].value)] * u.TeV

    # ========================= #
    # RUN INITIAL POWER LAW FIT #
    # ========================= #

    # Define spectral model
    model_target = SkyModel(
        name = target,
        spectral_model = PowerLawSpectralModel(
                            index = 2.0,
                            amplitude = 1e-12 * u.Unit("1 / (TeV s cm2)"),
                            reference = kwargs.get("reference", 1) * u.TeV, ), )
    # Make sure index and amplitude are free
    model_target.spectral_model.parameters["index"].frozen = False
    model_target.spectral_model.parameters["amplitude"].frozen = False
    # Set parameter range
    model_target.spectral_model.parameters["index"].min = 0.0
    model_target.spectral_model.parameters["index"].max = kwargs.get("index_max", 6.5)
    model_target.spectral_model.parameters["amplitude"].min = kwargs.get("amplitude_min", 1e-15)
    model_target.spectral_model.parameters["amplitude"].max = kwargs.get("amplitude_max", 1e-08)
    # Make sure reference is frozen
    model_target.spectral_model.parameters["reference"].frozen = True

    # Add model to datasets
    dataset_obs.models = model_target
    dataset.models = model_target

    # Run global fit on stacked dataset
    log.info("Running initial PowerLaw fit on stacked dataset")
    fit = Fit()
    fit_result = fit.run(dataset)
    log.info("Initial PowerLaw fit completed!")

    # Display fit results and best fit model
    log.info(fit_result)
    log.info(fit_result.models[target])

    # Compute decorrelation energy
    edec = get_edec(dataset.models[target])
    # Can also be computed directly with GammaPy as cross-check
    edec_gp = dataset.models[target].spectral_model.pivot_energy
    log.info(f"Decorrelation energy: {edec:.6f} (GammaPy check: {edec_gp:.6f})")

    # Set new reference energy
    dataset_obs.models[target].parameters["reference"].quantity = edec
    dataset.models[target].parameters["reference"].quantity = edec

    # Run the fit again if reference has been changed
    # (Don't re-compute if specific reference given!)
    if (fit_result.models[target].parameters["reference"].quantity != dataset.models[target].parameters["reference"].quantity):
        log.info(f"Updated reference PowerLaw fit on stacked dataset...")
        # Set amplitude as that of model evaluated at reference (improves convergence)
        dataset.models[target].parameters["amplitude"].quantity = dataset.models[target].spectral_model(edec)
        fit_result = fit.run(dataset)
        log.info("Updated PowerLaw fit completed!")

    # Display fit results and best fit model
    log.info(fit_result)
    log.info(fit_result.models[target])

    # Add model to per-observation dataset
    dataset_obs.models = fit_result.models

    # Run flux point estimator
    fluxp = FluxPointsEstimator(
        energy_edges = dataset.counts.geom.axes["energy"].edges, 
        source = target,
        selection_optional = "all")
    fluxp_points = fluxp.run(dataset)

    # Plot best-fit model and flux points
    fluxp_points.plot(sed_type = "e2dnde")
    dataset.models[target].spectral_model.plot(energy_bounds = energy_bounds, sed_type = "e2dnde")
    dataset.models[target].spectral_model.plot_error(energy_bounds = energy_bounds, sed_type = "e2dnde")
    plt.title(f"{target} PowerLaw SED")
    plt.savefig(fname = dir_gout.joinpath("lc_fitpl.png"), dpi = 300, bbox_inches = "tight",)
    plt.close()

    # ================================ #
    # GENERATE LIGHT CURVE FLUX POINTS #
    # ================================ #

    emin, emax, emin_avg = get_ethresh(dataset_obs)
    log.info(f"Threshold energy and total energy ranges:")
    log.info(f"- Maximal energy threshold: {emin.to(u.TeV).value:.3f} [TeV]")
    log.info(f"- Average energy threshold: {emin_avg.to(u.TeV).value:.3f} [TeV]")
    log.info(f"- Stacked energy threshold: {dataset.energy_range_total[0].to(u.TeV).value:.3f} [TeV]")
    log.info(f"- Maximal energy: {emax.to(u.TeV).value:.3f} [TeV]")

    # Construct time intervals
    t_intervals = gen_tintervals(dataset_obs, method = args.binning)
    if t_intervals != None:
        log.info(f"Built {len(t_intervals)} time intervals!")

    # Initialize LightCurveEstimator
    lc_estimator = LightCurveEstimator(
        source = target,
        # Use threshold energy of stacked dataset
        # energy_edges = dataset.energy_range_total,
        # Use average threshold energy
        # energy_edges = [emin_avg, emax],
        # Use energy edges of each dataset
        energy_edges = dataset_obs.energy_ranges,
        # Tolerance for choosing time intervals
        atol = ATOL,
        time_intervals = t_intervals,
        selection_optional = "all",
        stack_over_time_interval = True, )

    # Freeze index for light curve estimation
    dataset_obs.models[target].parameters["index"].frozen = True
    # Run light curve estimator
    log.info("Running light curve estimator...")
    lc_flux = lc_estimator.run(dataset_obs)
    log.info("Light curve estimator done!")

    # Create diagnostic plot
    lc_flux.plot(sed_type = "flux", time_format = "mjd")
    # Save diagnostic light curve plot
    plt.title(f"{target} Light Curve")
    plt.savefig(fname = dir_gout.joinpath("lc_flux.png"), dpi = 300, bbox_inches = "tight")
    plt.close()

    # Save initial light curve as fits file
    lc_flux.write(
        filename = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/lc_flux.fits",
        sed_type = "dnde", format = "lightcurve", overwrite = True, )

    # Generate light curve table - main results table
    lc_flux_tab = lc_flux.to_table(sed_type = "dnde", format = "lightcurve")

    # Add column for reference time
    lc_flux_tab.add_column(
        col = ( lc_flux_tab["time_min"] + lc_flux_tab["time_max"] ) / 2.0,
        index = 2,
        name = "time_ref", )
    
    # Add column if point is upper limit
    lc_flux_tab["is_ul"] = lc_flux_tab["ts"] < 4

    # ======================================= #
    # BAYESIAN BLOCK ALGORITHM ON FLUX POINTS #
    # ======================================= #

    # Define mask for non-upper-limit points
    mask_notul = ~lc_flux_tab["is_ul"].squeeze()

    # Run Bayesian block algorithm on flux points (ignoring upper limits!)
    # Make sure length of t values is greater than 1
    if len(lc_flux_tab["time_ref"].flatten()[mask_notul]) > 1:
        bblocks_edges_flux = bayesian_blocks(
            t = lc_flux_tab["time_ref"].flatten()[mask_notul],
            x = lc_flux_tab["dnde"].flatten()[mask_notul],
            sigma = lc_flux_tab["dnde_err"].flatten()[mask_notul],
            fitness = "measures", )
    # If not enough points, consider full time range as single block
    else:
        bblocks_edges_flux = [ lc_flux_tab["time_min"][0], lc_flux_tab["time_max"][-1] ]
    
    log.info(f"Bayesian blocks on flux done! Found {len(bblocks_edges_flux)-1} block(s)")
    log.info(f"Bayesian block edges [MJD]:\n{bblocks_edges_flux}")

    # Extend first and last blocks to include time error bars
    bblocks_edges_flux[0] = bblocks_edges_flux[0] - ( lc_flux_tab["time_ref"][0] - lc_flux_tab["time_min"][0] )
    bblocks_edges_flux[-1] = bblocks_edges_flux[-1] + ( lc_flux_tab["time_max"][-1] - lc_flux_tab["time_ref"][-1] )

    # Define time intervals for Bayesian blocks
    t_edges_bb_flux = Time(bblocks_edges_flux, format = "mjd")
    t_intervals_bb_flux = [Time([tstart, tstop]) for tstart, tstop in zip(t_edges_bb_flux[:-1], t_edges_bb_flux[1:])]

    # ==================================== #
    # RE-GENERATE LIGHT CURVE OVER BBLOCKS #
    # ==================================== #

    # Reconstruct light curve to get flux points per Bayesian block
    log.info("Reconstructing light curve...")

    # Re-define time intervals
    lc_estimator.time_intervals = t_intervals_bb_flux
    # Generate new light curve
    lc_flux_bblock = lc_estimator.run(dataset_obs)
    log.info("Light curve reconstruction done!")

    # Convert light curve to table for flux points per Bayesian block
    lc_flux_bblock_tab = lc_flux_bblock.to_table(sed_type = "dnde", format = "lightcurve")

    # If new flux points are upper limits, use that value!
    mask = lc_flux_bblock_tab["is_ul"] == True
    lc_flux_bblock_tab["dnde"][mask] = lc_flux_bblock_tab["dnde_ul"][mask]

    # Add column for reference time
    lc_flux_bblock_tab.add_column(
        col = ( lc_flux_bblock_tab["time_min"] + lc_flux_bblock_tab["time_max"] ) / 2.0,
        index = 2, 
        name = "time_ref", )
    
    # Plot original light curve
    lc_flux.plot(sed_type = "dnde", time_format = "mjd", axis_name = "time")
    # Plot reconstructed light curve
    lc_flux_bblock.plot(sed_type = "dnde", time_format = "mjd", axis_name = "time")
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/lc_flux_bb.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    # ====================================== #
    # RUN FIT FOR EACH FLUX POINT SEPARATELY #
    # ====================================== #

    # Run time-resolved spectroscopy for full time interval - get index and norm
    log.info("Running time resolved spectroscopy on flux points...")
    lc_flux_tab, _ = time_resolved_spectroscopy(dataset_obs, lc_flux_tab)
    log.info("Time resolved spectroscopy on flux points done!")

    # ======================================= #
    # RUN FIT FOR EACH BAYESIAN BLOCK ON FLUX #
    # ======================================= #

    log.info("Running time resolved spectroscopy on Bayesian-blocked flux points...")
    lc_flux_bblock_tab, lc_flux_bblock_results = time_resolved_spectroscopy(dataset_obs, lc_flux_bblock_tab, check_convergence = True, use_edec = kwargs.get("use_edec", False))
    log.info("Time resolved spectroscopy on Bayesian-blocked flux points done!")

    # For each fit result, save covariance matrix
    log.info(f"Computing and saving covariance matrix to {dir_gout}")
    lc_flux_bblock_cov = []
    for r in lc_flux_bblock_results:
        # Get sub-covariance for index and ampltiude
        cov = r.models[target].covariance.get_subcovariance(
            parameters = [r.models[target].parameters["index"], 
                        r.models[target].parameters["amplitude"],] ).data
        lc_flux_bblock_cov.append(cov)
    np.save(file = dir_gout.joinpath("lc_flux_bblock_cov.npy"), arr = np.array(lc_flux_bblock_cov))
 
    # ================================= #
    # BAYESIAN BLOCK ALGORITHM ON INDEX #
    # ================================= #

    log.info("Running Bayesian blocks on index values...")
    # Define mask for non-upper-limit points
    mask_notul = ~lc_flux_tab["index_is_ul"].squeeze()

    # Run Bayesian block algorithm on index values
    # Make sure length of t values is greater than 1
    if len(lc_flux_tab["time_ref"].flatten()[mask_notul]) > 1:
        bblocks_index = bayesian_blocks(
            t = lc_flux_tab["time_ref"].flatten()[mask_notul],
            x = lc_flux_tab["dnde"].flatten()[mask_notul],
            sigma = lc_flux_tab["dnde_err"].flatten()[mask_notul],
            fitness = "measures", )
    # If not enough points, consider full time range as single block
    else:
        bblocks_index = [ lc_flux_tab["time_min"][0], lc_flux_tab["time_max"][-1] ]
    log.info("Bayesian blocks on index done!")
    
    # Define time intervals for Bayesian blocks on index
    t_edges_bb_index = Time(bblocks_index, format = "mjd")
    t_intervals_bb_index = [Time([tstart, tstop]) for tstart, tstop in zip(t_edges_bb_index[:-1], t_edges_bb_index[1:])]
    
    # ======================================== #
    # RUN FIT FOR EACH BAYESIAN BLOCK ON INDEX #
    # ======================================== #

    # Create index value table
    lc_index_bblock_tab = Table()
    time_min = t_edges_bb_index[:-1]
    time_max = t_edges_bb_index[1:]
    # Define times based on block segments
    lc_index_bblock_tab.add_columns(
        cols = [time_min, time_max],
        names = ["time_min", "time_max"], )
    # Add reference time column
    lc_index_bblock_tab.add_column(
        col = Time( ( lc_index_bblock_tab["time_min"].mjd + lc_index_bblock_tab["time_max"].mjd ) / 2.0, format = "mjd" ),
        name = "time_ref", )
    
    # Run time resolved spectroscopy for each Bayesian block on index values (gives index per Bayesian block value)
    lc_index_bblock_tab, lc_index_bblock_results = time_resolved_spectroscopy(dataset_obs, lc_index_bblock_tab, check_convergence = True)

    # ============================================ #
    # SAVE LIGHT CURVE TO FILES AND GENERATE PLOTS #
    # ============================================ #

    log.info(f"Writing output files to {dir_gout}")
    # Save main light curve flux points and index values
    ascii.write(
        table = lc_flux_tab, overwrite = True, format = "ecsv",
        output = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/lc_flux_tab.ecsv", )
    # Save Bayesian blocks on flux
    ascii.write(
        table = lc_flux_bblock_tab, overwrite = True, format = "ecsv",
        output = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/lc_flux_bblock_tab.ecsv", )
    # Save bayesian blocked flux light curve table
    ascii.write(
        table = lc_flux_bblock_tab, format = "ecsv", overwrite = True,
        output = dir_gout.joinpath("lc_flux_bblock_tab.ecsv"), )
    # Save Bayesian blocks on index
    ascii.write(
        table = lc_index_bblock_tab, overwrite = True, format = "ecsv",
        output = f"{os.environ['RESULTS']}/{target}/{args.bblock}/gamma-out/lc_index_bblock_tab.ecsv", )

    # ================================================================ #
    # PLOT FINAL LIGHT CURVE AND BEST-FIT ELLIPSES FOR BAYESIAN BLOCKS #
    # ================================================================ #

    log.info(f"Saving plots to {dir_pout}")
    plot_lightcurve(target, args.bblock)
    plot_ellipses(target, args.bblock, args.annotate, loc = kwargs.get("loc", "default"))

    # TODO: Generate bblocks.yaml file containing final time segmentation
    # Compute which best-fit ellipses fall within each other at significance level
    # Combine the ellipses, and get the time intervals
