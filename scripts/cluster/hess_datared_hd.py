import  os
import  glob
import  yaml
import  argparse
import  warnings

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path

from    regions                     import  CircleSkyRegion

from    astropy                     import  units   as  u
from    astropy.coordinates         import  Angle, SkyCoord

from    gammapy.visualization       import  plot_distribution, plot_spectrum_datasets_off_regions
from    gammapy.data                import  DataStore, Observations
from    gammapy.datasets            import  Datasets, SpectrumDataset, MapDataset
from    gammapy.maps                import  MapAxis, RegionGeom, WcsGeom
from    gammapy.makers              import  ReflectedRegionsBackgroundMaker, SafeMaskMaker, SpectrumDatasetMaker, MapDatasetMaker, FoVBackgroundMaker, RingBackgroundMaker
from    gammapy.makers.utils        import  make_theta_squared_table
from    gammapy.visualization       import  plot_theta_squared_table
from    gammapy.modeling.models     import  PointSpatialModel, PowerLawSpectralModel, SkyModel
from    gammapy.estimators          import  TSMapEstimator, ExcessMapEstimator


# HESS era definitions
HESS_ERAS = {
    'hess1':  {'start': 18000,  'end': 80000 },
    'hess2':  {'start': 80000,  'end': 124680},
    'hess1u': {'start': 127700, 'end': 154814},
    'hessfc': {'start': 154814, 'end': 999999}, }

# Dataset paths
DATASET_PATHS = {
    'HAP-HD-NEW': '/path/to/hap-hd-new/',
    'HAP-HD': '/path/to/hap-hd/',
    'HAP-FR': '/path/to/hap-fr/', }

# ON region radius optimal for each config
ONREG_CONFIG = {
    'std_ImPACT_fullEnclosure': np.sqrt(0.005),
    'std_ImPACT_fullEnclosure_updated': np.sqrt(0.005),
    'std_ImPACT_3tel_fullEnclosure': np.sqrt(0.005),
    'std_ImPACT_hybrid_fullEnclosure': np.sqrt(0.007),
    'loose_ImPACT_fullEnclosure': np.sqrt(0.01),
    'loose_ImPACT_fullEnclosure_version36': np.sqrt(0.01),
    'loose_ImPACT_mono_fullEnclosure': np.sqrt(0.02),
    'std_zeta_hybrid_fullEnclosure': np.sqrt(0.01),
    'std_zeta_mono_fullEnclosure': np.sqrt(0.016),
    'safe_zeta_mono_fullEnclosure': np.sqrt(0.016), }

def get_source_info(target):
    """
    Gather relevant data for a given source from sources.yaml file (as defined by $SOURCES_FILE environment variable).
    Args:
        name (str): Name of the target source.
    Returns:
        target_4FGL (str): 4FGL Catalog name of the target source (if available).
        target_position (`astropy.coordinates.SkyCoord`): Position of the target source.
        target_redshift (float): Redshift of the source.
        exclusions (dict): Exclusion regions.
    """

    # Open sources file
    with open(Path('./sources.yaml'), 'r') as f:
        # Load the yaml file
        data = yaml.full_load(f)

        # Get the position of the source (either from the file or from astropy)
        try:
            target_position = SkyCoord(
                ra = data["sources"][target]["ra"],
                dec = data["sources"][target]["dec"],
                unit = "deg",
                frame = "icrs",)
        except:
            target_position = SkyCoord.from_name(target)

        # Get the redshift of the source
        target_redshift = data["sources"][target]["z"]

        # Get the 4FGL name of source
        target_4FGL = data["sources"][target]["target_4FGL"]

        # Get exclusion regions, if any
        exclusions = data["sources"][target]["xcl"]
    
    # Return all parameters
    return target_4FGL, target_position, target_redshift, exclusions


def parse_kwargs(args):
    """
    Parse all keyword arguments given and return as dictionary.
    """

    kwargs = {}
    for kv in args:
        # Obtain key and value
        key, value = kv.split('=', 1)
        # Convert to bool if true or false
        if value.lower() == 'true':
            val = True
        elif value.lower() == 'false':
            val = False
        try:   
            # Convert to int if possible
            val = int(value)
        except ValueError:
            # Convert to float if possible
            try:
                val = float(value)
            # Otherwise, leave as is
            except ValueError:
                val = value
        # Set parsed value to key
        kwargs[key] = val

    # Return dictionary of parsed arguments
    return kwargs


def classify_runs_by_era(runs, excluded_eras = None):
    """
    Classify each run in runlist by HESS eras.
    """

    # Create dictionary of empty lists for each era
    classified = {era: [] for era in HESS_ERAS.keys()}
    
    # For each run in runlist
    for run in runs:
        # For each era in HESS eras
        for era_key, era_info in HESS_ERAS.items():
            # If run within era boundaries, append to dict
            if era_info['start'] <= run < era_info['end']:
                classified[era_key].append(run)
                break

    # Exclude given eras
    for era in excluded_eras:
        classified[era] = []
    
    # Return dictionary
    return classified


def get_datastore(dataset_type, era = None, config = None, target = None):
    """
    Load DataStore for the specified dataset type.
    Parameters:
        dataset_type (str): Dataset type: 'HAP-HD', 'HAP-FR', or 'HAP-FITS'.
        era (str): HESS era name (for HAP-HD).
        config (str): Configuration for HAP-HD.
        target (str): Target name for HAP-FITS.
    Returns:
        data_store (`gammapy.data.DataStore`): Loaded datastore object.
    """

    # HDU and OBS table paths
    hdu_table = "hdu-index.fits.gz"
    obs_table = "obs-index.fits.gz"

    # HAP-FITS-Export
    if dataset_type == "HAP-FITS":
        path = Path(f"../{target}/out-{era}-{config}")
    # HAP-FR-Prod04
    elif dataset_type == "HAP-FR":
        path = Path(DATASET_PATHS['HAP-FR'])
    # HAP-HD-Prod05
    elif dataset_type == "HAP-HD":
        path = Path(DATASET_PATHS['HAP-HD']) / era / config
    # HAP-HD-Prod06
    elif dataset_type == "HAP-HD-NEW":
        path = Path(DATASET_PATHS['HAP-HD-NEW']) / config
        hdu_table = path / "hdu-index-bkg-3d-v01a-fov-radec.fits.gz"
        obs_table = path / "obs-index-bkg-3d-v01a-fov-radec.fits.gz"

    # Check if path exists
    if not path.exists():
        print(f"Warning: Path {path} does not exist!")
        return None
    
    try:
        # Load dataset into data store
        data_store = DataStore.from_dir(
            path,
            hdu_table_filename = hdu_table,
            obs_table_filename = obs_table, )
        return data_store
    except Exception as e:
        print(f"Error loading datastore from {path}: {e}")
        return None


def gen_datasets(observations, target_position, on_region_radius = np.sqrt(0.005), exclusions = None, max_offset = 2.5 * u.deg, ebias_percent = 10, emax = 31.6):
    """
    Create spectrum datasets from observations using reflected regions background.
    Parameters:
        observations (`gammapy.data.Observations`): GammaPy observations object.
        target_position (`astropy.SkyCoord`): Source position.
        on_region_radius (float): on region radius in degrees. Default: based on best radius for config, if found, otherwise std_ImPACT cut of sqrt(0.005) ~ 0.07071067811.
    Returns:
        datasets (`gammapy.data.Datasets`): Processed datasets object.
    """
   
    print(f"Generating datasets for main analysis. Safe mask with max offset: {max_offset} and ebias percent: {ebias_percent}")
 
    # Define energy axes
    energy_axis = MapAxis.from_energy_bounds(
        0.1, emax, nbin = 10, per_decade = True, unit = "TeV", name = "energy", )
    energy_axis_true = MapAxis.from_energy_bounds(
        0.1, emax, nbin = 20, per_decade = True, unit = "TeV", name = "energy_true")

    # Define geometry for full roi
    geom = WcsGeom.create(
        skydir = target_position,
        binsz = 0.02, width = (2.5, 2.5),
        frame = "icrs", proj = "TAN", )
        # axes = [energy_axis], )

    # Define on-region
    on_region = CircleSkyRegion(
        center = target_position,
        radius = Angle(on_region_radius, unit = "deg"), )
    # Create geometry for on region
    geom_on = RegionGeom.create(region = on_region, axes = [energy_axis])

    # Add target to excluded regions
    excl_target = {"name": target, "ra": target_position.ra.value, "dec": target_position.dec.value, "size": 0.4}
    if exclusions == None:
        exclusions = [excl_target] 
    # If we have exclusion regions, append target
    else:
        exclusions.append(excl_target)
    
    # Define exclusion regions, if any
    excl_mask = None
    if len(exclusions) > 0:
        excl_regs = []
        for exclusion in exclusions:

            print(f"Exclusion region: {exclusion['name']} at {exclusion['ra']} RA, {exclusion['dec']} DEC, Size: {exclusion['size']}")

            # Define sky region
            excl_regs.append(
                CircleSkyRegion(center = SkyCoord(exclusion["ra"], exclusion["dec"], unit = "deg", frame = "icrs"), radius = exclusion["size"] * u.deg))
        
        # Define exclusion mask from all exclusion regions
        excl_mask = ~geom.region_mask(excl_regs)

    # Define empty dataset
    dataset_empty = SpectrumDataset.create(
        geom = geom_on, energy_axis_true = energy_axis_true, )
    
    # Define dataset makers
    dataset_maker = SpectrumDatasetMaker(
        containment_correction = True,
        selection = ["counts", "exposure", "edisp"], )
    bkg_maker = ReflectedRegionsBackgroundMaker(
        exclusion_mask = excl_mask, )
    safe_mask_maker = SafeMaskMaker(
            methods = ["edisp-bias"],
            bias_percent = ebias_percent, 
            offset_max = max_offset, )
    
    # Define datasets object
    datasets = Datasets()
 
    # Process each observation
    for k, observation in enumerate(observations):
         
        print(f"Processing observation {k+1} / {len(observations)}", end = "\r", flush=True)
 
        try:
            # Run makers in sequence
            dataset = dataset_maker.run(
                dataset_empty.copy( name = str(observation.obs_id) ),
                observation, )
            dataset_onoff = bkg_maker.run(dataset, observation)
            dataset_onoff_safe = safe_mask_maker.run(dataset_onoff, observation)

            # Append observation dataset
            datasets.append(dataset_onoff_safe)
        # Skip problematic observations
        except Exception as e:
            print(f"Error processing observation {observation.obs_id}: {e}")
            continue
    
    # Plot exclusion region alongside all on-off regions (at least for a subset of datasets)
    plt.figure()
    ax = excl_mask.plot()
    on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
    plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
    plt.title(f"{target} Exclusions and On-Off Regions")
    plt.savefig(f"../{target}/plots/reg_onoff.png", dpi = 300)
    plt.close()

    print("\n")
    # Return datasets object containing a dataset per each observation
    return datasets


def gen_mapdatasets(observations, target_position, on_region_radius = np.sqrt(0.005), hap_dataset = None, hap_config = None, exclusions = None, emax = 31.6):
    """
    Create map spectrum datasets from observations
    """
 
    print("Generating map dataset plots")

    # Define energy axes
    energy_axis = MapAxis.from_energy_bounds(
        0.1, emax, nbin = 10, per_decade = True, unit = "TeV", name = "energy", )
    energy_axis_true = MapAxis.from_energy_bounds(
        0.1, emax, nbin = 20, per_decade = True, unit = "TeV", name = "energy_true", )

    # Define geometry
    geom = WcsGeom.create(
        skydir = target_position,
        binsz = 0.02, width = (1.25, 1.25),
        frame = "icrs", proj = "TAN",
        axes = [energy_axis], )

    # List of exclusion regions
    excl_regs = []

    # Define exclusion source at target
    excl_regs.append(CircleSkyRegion(target_position, radius = 0.4 * u.deg))

    # Define other exclusion regions, if any
    if len(exclusions) > 0:
        for exclusion in exclusions:
            print(f"Exclusion region: {exclusion['name']} at {exclusion['ra']} RA, {exclusion['dec']} DEC")
            # Define sky region
            excl_regs.append(
                CircleSkyRegion(center = SkyCoord(exclusion["ra"], exclusion["dec"], unit = "deg", frame = "icrs"), radius = exclusion["size"] * u.deg))
    
    excl_mask = ~geom.to_image().to_cube([energy_axis.squash()]).region_mask(excl_regs)

    # Define dataset maker
    dataset_maker = MapDatasetMaker()

    # Define background maker
    backg_maker = RingBackgroundMaker(r_in = "0.5 deg", width = "0.3 deg", exclusion_mask = excl_mask)

    # Define datasets object
    datasets = Datasets()
    
    # Process each observation
    for k, observation in enumerate(observations):

        print(f"Processing observation {k+1} / {len(observations)}", end = "\r", flush=True)

        try:
            # Define empty dataset
            dataset_empty = MapDataset.create(geom = geom, energy_axis_true = energy_axis_true)
            # Run makers in sequence
            dataset = dataset_maker.run(
                dataset_empty.copy( name = str(observation.obs_id) ),
                observation, )
            # Run background maker
            try:
                dataset = backg_maker.run(dataset, observation)
            except:
                print("Unable to estimate background!")
                pass
            # Append observation dataset
            datasets.append(dataset)
        # Skip problematic observations
        except Exception as e:
            print(f"Error processing observation {observation.obs_id}: {e}")
            continue
    
    print() 
    # Stack all datasets
    dataset = datasets.stack_reduce()
   
    # Make sure directory for fits files and plots exists
    os.makedirs(f"../{target}/mapfits/", exist_ok = True)

    # Save map dataset as fits file (contains counts, background, excess, etc.)
    dataset.write(f"../{target}/plots/mapfits/mapdata_{hap_dataset}_{hap_config}.fits", overwrite = True)
 
    # Plot counts
    print("Generating map dataset counts plot")
    ax = dataset.counts.sum_over_axes().smooth(0.01 * u.deg).plot(add_cbar = True)
    ax.set_title(f"{target} Counts Map")
    plt.savefig(f"../{target}/plots/mapdata_counts_{hap_dataset}_{hap_config}.png", dpi = 300)
    plt.close() 

    # Generate TS map
    print("Generating TS map")
    try:
        model = SkyModel(spatial_model = PointSpatialModel(), spectral_model = PowerLawSpectralModel())
        # Initialize estimator
        ts_estimator = TSMapEstimator(
            model = model,
            kernel_width = "0.1 deg",
            energy_edges = [0.1, args.emax] * u.TeV,
            n_sigma = 1, n_sigma_ul = 2,
            selection_optional = None,
            n_jobs = 1, sum_over_energy_groups = True, )
        # Run estimator
        ts_maps = ts_estimator.run(dataset.to_map_dataset())

        # Save resulting flux map as fits file
        ts_maps.write(filename = f"../{target}/plots/mapfits/mapdata_ts_{hap_dataset}_{hap_config}.fits", overwrite = True)       
        # Visualize excess map
        ax = ts_maps["npred_excess"].plot(add_cbar = True)
        ax.set_title(f"{target} Excess")
        plt.savefig(f"../{target}/plots/mapdata_ts_excess_{hap_dataset}_{hap_config}.png", dpi = 300)
        plt.close()

        # Visualize significance map
        ax = ts_maps["sqrt_ts"].plot(add_cbar = True) 
        ax.set_title(r"{} √TS [σ]".format(target))
        plt.savefig(f"../{target}/plots/mapdata_ts_significance_{hap_dataset}_{hap_config}.png", dpi = 300)
        plt.close()
    except Exception as e:
        print(f"Unable to generate TS map! Exception: {e}. Skipping")

    # Can also be computed using an ExcessMapEstimator
    excess_estimator = ExcessMapEstimator(0.1 * u.deg, correlate_off = False)
    # Run estimator on stacked on-off map dataset
    lima_maps = excess_estimator.run(dataset)

    # Save resulting flux map as fits file
    lima_maps.write(filename = f"../{target}/plots/mapfits/mapdata_lima_{hap_dataset}_{hap_config}.fits", overwrite = True)   

    # Plot excess and significance map
    fig, ax = plt.subplots(subplot_kw = {'projection': lima_maps.geom.wcs}, ncols = 2, figsize = (8, 4))
    # Plot significance map
    ax[0].set_title(f"{target} Significance Map")
    lima_maps["sqrt_ts"].plot(ax = ax[0], add_cbar = True)
    # Plot excess map
    ax[1].set_title(f"{target} Excess Map")
    lima_maps["npred_excess"].plot(ax = ax[1], add_cbar = True)
    # Save figure
    plt.tight_layout()
    plt.savefig(f"../{target}/plots/mapdata_lima_{hap_dataset}_{hap_config}.png", bbox_inches = "tight", dpi = 300) 
    plt.close()

    # Generate histogram plot
    dataset.mask_fit = excl_mask
    lima_maps2 = excess_estimator.run(dataset)
    significance_map_off = lima_maps2["sqrt_ts"]

    # Save resulting flux map as fits file
    lima_maps2.write(filename = f"../{target}/plots/mapfits/mapdata_lima_excl_{hap_dataset}_{hap_config}.fits", overwrite = True)       
    fig, ax = plt.subplots(figsize = (8, 6))

    kwargs_axes = {"xlabel": "Significance", "yscale": "log", "ylim": (1e-4, 1)}
    plot_distribution(
        lima_maps["sqrt_ts"],
        ax = ax,
        kwargs_hist={
            "density": True,
            "alpha": 0.5, "color": "red",
            "label": "all bins", "bins": 51, },
        kwargs_axes = kwargs_axes, )

    plot_distribution(
        significance_map_off,
        ax = ax,
        func = "norm",
        kwargs_hist = {
            "density": True,
            "alpha": 0.5, "color": "blue",
            "label": "off bins", "bins": 51, },
        kwargs_axes = kwargs_axes, )
    
    # Plot significance histogram
    plt.title(f"{target} Significance Histogram")
    plt.savefig(f"../{target}/plots/mapdata_hist_{hap_dataset}_{hap_config}.png", dpi = 300)
    plt.close()


# Function to generate per-observation diagnostics
def gen_obsdiag(observations, target, dataset, config):

    print("Generating per-observation diagnostic plots")

    # Make sure plot directory exists
    os.makedirs(f"../{target}/plots/obs_{dataset}_{config}/", exist_ok = True)
    
    # Loop over all observations
    for k, observation in enumerate(observations):

        print(f"Processing observation {k+1} / {len(observations)}", end='\r', flush=True)
    
        # Peek into the events (without any offset)
        observation.events.select_offset([0, 2.5] * u.deg).peek()
        plt.savefig(f"../{target}/plots/obs_{dataset}_{config}/events_obs_{observation.obs_id}.png", dpi = 300)
        plt.close()
        
        # Peek into the effective area
        # NOTE: Produces RuntimeWarnings due to missing data outside bounds (not a big problem) - We can safely ignore these!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            observation.aeff.peek()
            plt.savefig(f"../{target}/plots/obs_{dataset}_{config}/aeff_obs_{observation.obs_id}.png", dpi = 300)
            plt.close()
        
        # Peek into the energy dispersion
        observation.edisp.peek()
        plt.savefig(f"../{target}/plots/obs_{dataset}_{config}/edisp_obs_{observation.obs_id}.png", dpi = 300)
        plt.close()
        
        # Peek into point spread function
        observation.psf.peek()
        plt.savefig(f"../{target}/plots/obs_{dataset}_{config}/psf_obs_{observation.obs_id}.png", dpi = 300)
        plt.close()

    return


# Function to generate theta squared plot
def gen_theta2(observations, target, target_position, dataset, config):

    print("Generating theta square plot")

    # Create theta^2 table            
    theta2_axis  = MapAxis.from_bounds(0.0, 0.1, nbin = 15, interp = "lin", unit = "deg2")
    theta2_table = make_theta_squared_table(
        observations = observations,
        position = target_position,
        theta_squared_axis = theta2_axis, ) 
    
    # Compute center of bins 
    x = ( theta2_table["theta2_min"] + theta2_table["theta2_max"] ) / 2.0

    # Plot counts, counts off, and excess
    plt.errorbar(x, theta2_table["counts"], yerr = None, xerr = x - theta2_table["theta2_min"], linestyle = "", label = "Counts ON", color = "tab:blue", marker = ".")
    plt.errorbar(x, theta2_table["counts_off"], yerr = None, xerr = x - theta2_table["theta2_min"], linestyle = "", label = "Counts OFF", color = "tab:orange", marker = ".")
    plt.errorbar(x, theta2_table["excess"], yerr = None, xerr = x - theta2_table["theta2_min"], linestyle = "", label = "Excess", color = "tab:red", marker = ".")

    # Setup plot
    plt.xlabel(r"$\theta^2$ [deg$^{2}$]")
    plt.ylabel("Counts")    
    plt.legend()

    plt.title(f"{target}")

    # Check if directory exists, if not, create it
    dir_plots = f"../{target}/plots/"
    os.makedirs(dir_plots, exist_ok=True)

    # Save plot   
    plt.savefig(f"../{target}/plots/theta2_{dataset}_{config}.png", dpi = 300, bbox_inches = "tight")

    plt.close()
    print("Theta squared plot generated!")

    # Save theta plot as fits file
    dir_fits = f"../{target}/plots/mapfits/"
    os.makedirs(dir_fits, exist_ok = True)
    theta2_table.write(f"{dir_fits}/theta2_{dataset}_{config}.fits", format = "fits", overwrite = True)

    return


if __name__ == "__main__":


    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run HESS data reduction for a source using GammaPy within cluster")
    
    # Source configuration and settings
    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121)")

    parser.add_argument("--dataset", choices = ["HAP-HD", "HAP-HD-NEW", "HAP-FR", "HAP-FITS", "ALL"], required = True, help = "Dataset to use for analysis (HAP-HD, HAP-FR, HAP-FITS from custom export)")
    parser.add_argument("--config", default = "std_ImPACT_fullEnclosure_updated", help = "Reconstruction configuration for HAP-HD dataset. Default: std_ImPACT_fullEnclosure_updated")

    parser.add_argument("--exclude-eras", nargs = "+", choices = ["hess1", "hess2", "hess1u", "hessfc"], default = [], help = "HESS eras to exclude from analysis. Choices: ['hess1', 'hess2', 'hess1u', 'hessfc']. Default: None")
    # Force HESS-FC inclusion (mixing configs!)
    parser.add_argument("--include-hessfc", action = "store_true", help = "Force the inclusion of HESS-FC era data. Warning: this includes data generated using the std_ImPACT_3tel_fullEnclosure configuration, which may be different from the rest!")

    # Force std_ImPACT_hybrid_fullEnclosure config (only for HESS 2 and 1U era!
    parser.add_argument("--include-hybrid", action = "store_true", help = "Force inclusion of std_ImPACT_hybrid_fullEnclosure configuration on HESS 2 and 1U era data. Warning: this mixes configurations, be careful! Default: False")

    # Maximum energy
    parser.add_argument("--emax", type = int, default = 31.6, help = "Maximum energy")

    # On region radius
    parser.add_argument("--on-radius", default = None, help = "ON region radius. Default: based on best radius for config, if found, otherwise std_ImPACT cut of sqrt(0.005) ~ 0.07071067811.")

    # Dataset generation arguments
    parser.add_argument("--max-offset", default = 2.5, type = float, help = "Maximum offset [deg]")
    parser.add_argument("--bias-percent", default = 10, type = float,  help = "Bias percent")

    # Runlist name
    parser.add_argument("--runlist", default = "runlist.txt", help = "Name of runlist to use in target's folder. Default: runlist.txt")

    # Run plots only (theta squared, per-observation diagnostics, counts map, TS map, if possible)
    parser.add_argument("--plots-only", action = "store_true", help = "Run only plots: theta squared, and optionally additional plots")
    parser.add_argument("--plot-obs", action = "store_true", help = "Plot per-observation diagnostic plots")
    parser.add_argument("--plot-map", action = "store_true", help = "Plot map dataset counts and TS plots")
    # Other keyword arguments
    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of the target source
    target_name = args.source

    # Allow for names such as {target}_{info} for different runlists, settings, etc
    # Split by "_" string, first part being the target full name: if target_name == target, everything works fine, if target_name contains suffix, data saved and loaded from that suffixed directory
    # TODO: Finish implementing this!
    target = target_name.split("_")[0]

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get info of source
    target_4FGL, target_position, target_redshift, exclusions = get_source_info(target)

    # Get best ON region radius if not given
    on_radius = args.on_radius
    if args.on_radius == None:
        on_radius = ONREG_CONFIG.get(args.config, np.sqrt(0.005))
    print(f"Using ON region radius of {on_radius}\n")
   
    # Make sure max offset has units
    max_offset = args.max_offset * u.deg
 
    # Load runlist (telescope id not required, since cuts performed using HAP tools)
    runlist_path = Path(f"../{target}/{args.runlist}")
    if not runlist_path.exists():
        raise FileNotFoundError(f"Runlist not found: {runlist_path}")
    runlist = np.loadtxt(runlist_path, usecols = 0, dtype = int)
    print(f"Loaded {len(runlist)} runs from runlist")
    
    # Classify runs by era
    runs_by_era = classify_runs_by_era(runlist, args.exclude_eras)
    
    # Determine which datasets to process
    datasets_to_process = ['HAP-HD', 'HAP-HD-NEW', 'HAP-FR', 'HAP-FITS'] if args.dataset == 'ALL' else [args.dataset]
    
    # Keep track of the length of generated datasets
    dataset_len = {
        "HAP-HD-NEW": 0,
        "HAP-HD": 0,
        "HAP-FR": 0,
        "HAP-FITS": 0, }

    # Go over each dataset to process
    for dataset_type in datasets_to_process:

        info_str = f"Processing {dataset_type} dataset"
        print("\n")
        print("*"*len(info_str))
        print(info_str)
        print("*"*len(info_str))
        print("\n")

        # If HAP-FR, load directly
        if dataset_type == "HAP-FR":

            # Load data store
            data_store = get_datastore(dataset_type)
            
            # Get observations
            try:
                observations = data_store.get_observations(
                    # Use runlist directly, no need to sort by eras
                    obs_id = runlist,
                    required_irf = "point-like",
                    skip_missing = True, )
                print(f"Loaded {len(observations)} observations from {dataset_type}!")
            # If unable, skip to next dataset type
            except Exception as e:
                print(f"Error loading observations: {e}")
                continue
            # If no observations found, also skip
            if len(observations) == 0:
                print(f"No valid observations found! Skipping")
                continue
           
            # Generate theta^2 plot
            gen_theta2(observations, target, target_position, dataset = args.dataset, config = args.config)

            # Generate per-observation diagnostics
            if args.plot_obs:
                gen_obsdiag(observations, target, dataset = args.dataset, config = args.config)
  
            # NOTE: Map dataset and TS plot estimation seems to be incompatible with HAP-FR data, treat with caution!
 
            # Generate map dataset and plots
            if args.plot_map and observations[0].bkg != None:
                gen_mapdatasets(observations = observations, target_position = target_position, hap_dataset = args.dataset, hap_config = args.config, exclusions = exclusions, emax = args.emax)

           # Create datasets (only if not plotting)
            if not args.plots_only:
                datasets = gen_datasets(observations, target_position, on_region_radius = on_radius, exclusions = exclusions, ebias_percent = args.bias_percent, max_offset = max_offset, emax = args.emax)
                print(f"Created {len(datasets)} datasets!")

                # Append length of created dataset to dict
                dataset_len["HAP-FR"] = len(datasets)

                # Save datasets
                for dataset in datasets:
                    datasets.write(
                        filename = Path(f"../{target}/{dataset_type.lower()}/obs_{dataset.name}.fits.gz"), overwrite = True, )

        # Else HAP-HD or HAP-FITS, load per-era - also keep track of which configuration is used (as to not mix!)
        else:

            # Total observations arrays
            observations = Observations()

            # Go over each HESS era
            for era_key, era_runs in runs_by_era.items():

                # Skip if no runs in era
                if not era_runs:
                    continue

                print(f"Processing {era_key} ({len(era_runs)} runs)...")

                config = args.config
                # If HESS 1 era, no hybrid config available
                if era_key == "hess1" and "hybrid" in config and dataset_type in ["HAP-HD", "HAP-HD-NEW"]:
                    print("No hybrid configuration available for HESS 1. Using default: std_ImPACT_fullEnclosure_updated.")
                    config = "std_ImPACT_fullEnclosure_updated"    

                # If HESS-FC era, only config available "std_ImPACT_3tel_fullEnclosure"
                if era_key == "hessfc" and config != "std_ImPACT_3tel_fullEnclosure" and dataset_type == "HAP-HD":
                    print("Only available configuration for HESS-FC era data: std_ImPACT_3tel_fullEnclosure.")
                    # If given, force inclusion of HESS-FC data even if configuration is mis-matched!
                    if args.include_hessfc:
                        print("Including HESS-FC. Warning: this may produce configuration mis-match!")
                        config = "std_ImPACT_3tel_fullEnclosure"
                    else:
                        print("Skipping HESS-FC...")
                        continue
                
                # If HESS 2 or HESS 1U eras, and hybrid config enabled, append it to list of configurations to process
                configs = [config]
                if era_key == "hess2" or era_key == "hess1u":
                    if args.include_hybrid == True:
                        print("Including Hybrid configuration. Warning: this may produce configuration mis-match!")
                        configs.insert(0, "std_ImPACT_hybrid_fullEnclosure") # Process hybrid config first
                observations_era_total = Observations()         

                # Loop over all configurations (including hybrid if given!)
                for hap_config in configs: 
                    # Debug config info
                    print(f"Processing {era_key} config {hap_config}")
                    # Load datastore
                    data_store = get_datastore(dataset_type, era = era_key, config = hap_config, target = target)

                    # If data store empty, skip
                    if data_store is None:
                        print(f"Skipping {era_key} and config {hap_config}, no data available!")
                        continue

                    try:
                        # Get observations for this era from data store
                        observations_era = data_store.get_observations(
                            obs_id = runs_by_era[era_key],
                            required_irf = "point-like",
                            skip_missing = True, )
                        
                        # Append observations found for era-config pair
                        for obs in observations_era:
                            # If observation already in list, skip it
                            if str(obs.obs_id) in observations_era_total.ids:
                                print(f"Observation {obs.obs_id} already in dataset (with hybrid config), skipping!")
                                continue
                            observations_era_total.append(obs)                     
                    except Exception as e:
                        print(f"Error loading observations for era {era_key}: {e}")
                        continue

                # If no observations found, skip
                if len(observations_era_total) == 0:
                    print(f"No valid observations found for era {era_key}! Skipping")
                    continue

                # Append observations found for this era to global observations
                for obs_era in observations_era_total:
                    observations.append(obs_era)

            print(f"Loaded {len(observations)} observations from {dataset_type}!")
            
            # Generate theta squared plot
            gen_theta2(observations, target, target_position, dataset = args.dataset, config = args.config)
            
            # Generate per-observation diagnostics
            if args.plot_obs:
                gen_obsdiag(observations, target, dataset = args.dataset, config = args.config)

            # Generate map dataset and plots
            if args.plot_map:
                if observations[0].bkg != None:
                    gen_mapdatasets(observations = observations, target_position = target_position, hap_dataset = args.dataset, hap_config = args.config, exclusions = exclusions, emax = args.emax)
                else:
                    print("No background found in observations! Unable to create map datasets!")            

            # Generate datasets (if not plots only)
            if not args.plots_only:
                try:
                    datasets = gen_datasets(observations, target_position, on_region_radius = on_radius, exclusions = exclusions, ebias_percent = args.bias_percent, max_offset = max_offset, emax = args.emax)          
                    print(f"Created {len(datasets)} datasets!")
                except Exception as e:
                    print(f"Unable to create datasets! Exception: {e}")

                # Append length of dataset created to dict
                dataset_len[dataset_type] = len(datasets)

                # Filename for dataset
                fname_data = f"{dataset_type.lower()}-{args.config}"
                if args.include_hybrid == True:
                    fname_data += "-hybrid"

                # After all eras processed save datasets
                for dataset in datasets:
                    datasets.write(
                        filename = Path(f"../{target}/{fname_data}/obs_{dataset.name}.fits.gz"), overwrite = True) 

    # Print final results
    if args.plots_only:
        info_str = "Plots generated!"
    else:
        info_str = "Final datasets generated!"
    print("\n")
    print("*"*len(info_str))
    print(info_str)
    print("*"*len(info_str))
    print("\n")
    if not args.plots_only:
        for k, v in dataset_len.items():
            print(f"{k} dataset containing {v} observations")
        print("\n")

