import  os
import  sys
import  glob
import  yaml
import  logging
import  argparse
import  datetime

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path

from    regions                     import  CircleSkyRegion

from    astropy                     import  units   as  u
from    astropy.coordinates         import  Angle, SkyCoord

from    gammapy.data                import  DataStore, Observations
from    gammapy.datasets            import  Datasets, SpectrumDataset, SpectrumDatasetOnOff, FluxPointsDataset
from    gammapy.maps                import  MapAxis, RegionGeom, WcsGeom

from    gammapy.estimators          import  FluxPointsEstimator
from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import  Models, EBLAbsorptionNormSpectralModel, SkyModel
from    gammapy.modeling.models     import  PowerLawSpectralModel, LogParabolaSpectralModel, PointSpatialModel, TemplateSpectralModel, ConstantSpectralModel, CompoundSpectralModel

from    ebltable.tau_from_model     import  OptDepth


def get_source_info(target):
    """
    Gather relevant data for a given source from sources.yaml file (as defined by $SOURCES_FILE environment variable).
    Args:
        name (str): Name of the target source.
    Returns:
        target_4FGL (str): 4FGL Catalog name of the target source (if available).
        target_position (`astropy.coordinates.SkyCoord`): Position of the target source.
        target_redshift (float): Redshift of the source.
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

    # Return all parameters
    return target_4FGL, target_position, target_redshift


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


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run HESS analysis for a source using GammaPy within HD cluster")
    # Source configuration and settings
    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121, or all)")

    # Data reduction configuration and dataset
    parser.add_argument("--dataconf", nargs = "*", action = "append", metavar = ("DATASET", "CONFIG"), help = "Dataset to use for analysis followed by none, one, or more reconstruction configurations (e.g. --dataconf HAP-HD std_ImPACT --dataconf HAP-FR --dataconf HAP-FITS std_ImPACT loose_ImPACT )")

    # Analysis arguments
    parser.add_argument("--bins", default = 4, type = int, help = "Number of bins per decade. Default: 4")
    parser.add_argument("--emask", nargs = 2, type=float, default = [0.1, 10], metavar=('EMIN', 'EMAX'), help="Energy mask values: EMIN EMAX [TeV] (e.g. --emask 0.1 0.6)")
    parser.add_argument("--reference", default = 1.0, type = float, help = "Reference energy in TeV. Default: 1.0 TeV")
    parser.add_argument("--model", default = "PowerLaw", choices = ["PowerLaw", "LogParabola"], help="Spectral model to use for target source")

    # Other keyword arguments
    parser.add_argument("--ebl", default = "dominguez", help = "EBL absorption model to use (from GammaPy or, if not found, EBLTable)")
    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of the target source
    target = args.source

    # Parse keyword arguments
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)
    
    # Parse dataconf arguments (old version)
    # dataconf = {
    #     group[0]: group[1:] if len(group) > 1 else [] for group in args.dataconf }
    # Improved dataconf parser
    dataconf = {}
    for group in args.dataconf:
        dataset = group[0]
        configs = group[1:] if len(group) > 1 else []
        
        # If dataset already exists, extend its config list
        if dataset in dataconf:
            dataconf[dataset].extend(configs)
        else:
            dataconf[dataset] = configs
    
    # Get info of source
    target_4FGL, target_position, target_redshift = get_source_info(target)

    # Configure logging
    dir_log = Path(f"../{target}/logs/")
    os.makedirs(dir_log, exist_ok = True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename = dir_log.joinpath("hess_analysis.log"), mode = "a"),
            logging.StreamHandler(sys.stdout)] )
    log = logging.getLogger(__name__)

    # Print header with info on analysis
    log.info(f"\n### HESS Analysis of {target} [{datetime.datetime.now()}] ###\n")

    # Define colors for each dataset
    dataset_cmaps = {
        "HAP-FR": plt.cm.Blues,
        "HAP-HD-NEW": plt.cm.Purples,
        "HAP-HD": plt.cm.Reds,
        "HAP-FITS": plt.cm.Greens, }
    colors = {}

    # Set up plotting
    fig, ax = plt.subplots()
    ax.xaxis.set_units(u.Unit("TeV"))
    ax.yaxis.set_units(u.Unit("TeV s-1 cm-2"))
    # Limit y axis (if given)
    if kwargs.get('ymin', None) != None and kwargs.get('ymax', None) != None:
        ax.set_ylim( kwargs.get('ymin', None), kwargs.get('ymax', None)  )
    # Limit x axis (if given)
    if kwargs.get('xmin', None) != None and kwargs.get('xmax', None) != None:
        ax.set_xlim( kwargs.get('xmin', None), kwargs.get('xmax', None) )

    # Loop over each dataset and its configs
    for dataset_type, config_list in dataconf.items():

        # Get color map for dataset
        cmap = dataset_cmaps[dataset_type]
        n = max(1, len(config_list))
        vals = np.linspace(0.3, 0.9, n)
        colors[dataset_type] = {}

        info_str = f"Processing {dataset_type} dataset"

        log.info(
            "*"*len(info_str) + "\n" + info_str + "\n" + "*"*len(info_str) + "\n")

        # Loop over each config in config_list
        for config, v in zip( config_list or [None], vals ):

            # Get color for config
            colors[dataset_type][config] = cmap(v)

            log.info(f"- Processing config {config}...\n")

            # If config_list = [] and dataset_type = HAP-FR, load directly
            if config == None and dataset_type == "HAP-FR":
                path = Path(f"../{target}/{dataset_type.lower()}/")
            # Otherwise, if HAP-HD or HAP-FITS, load data using correct configuration
            elif dataset_type in ["HAP-HD", "HAP-HD-NEW", "HAP-FITS"]:
                path = Path(f"../{target}/{dataset_type.lower()}-{config}/")

            try:

                # Get all fits files
                obs_rmfs  = list(path.glob('*rmf.fits'))
                obs_files = []

                for obs in obs_rmfs:
                    obs_files.append(Path(str(obs).replace('_rmf', '')))
                
                # Empty dataset object
                datasets = Datasets()

                # Append each observation to datasets
                for obs in obs_files:
                    datasets.append(SpectrumDatasetOnOff.read(obs))
                
                # Stack-reduce the dataset
                dataset_hess = datasets.stack_reduce()
                # dataset_hess = datasets                

                # Define EBL model from EBLTable
                ebl_model = None
                if args.ebl != "None":
                    e_tau = np.logspace(-1, 1.5, 200) * u.TeV
                    tau = OptDepth.readmodel(model = args.ebl)
                    att = np.exp( - 1. * tau.opt_depth(target_redshift, e_tau.value ) )
                    ebl_model = TemplateSpectralModel(energy = e_tau, values = att)

                # Define intrinsic model
                if args.model == "LogParabola":
                    intrinsic = LogParabolaSpectralModel()
                else:
                    intrinsic = PowerLawSpectralModel()
                if ebl_model != None:
                    spectral_model = intrinsic * ebl_model
                else:
                    spectral_model = intrinsic

                # Define target model
                model = SkyModel(
                    name = target,
                    spectral_model = spectral_model,
                    spatial_model = PointSpatialModel(lon_0 = target_position.ra, lat_0 = target_position.dec),
                )               

                # Freeze spatial model
                model.spatial_model.parameters.freeze_all()
 
                # Set reference to given value (default 1 TeV)
                model.parameters["reference"].quantity = args.reference * u.TeV # kwargs.get('reference', 1.0) * u.TeV

                # Set parameter bounds
                if "index" in model.parameters.names:
                    model.parameters["index"].min = kwargs.get('index_min', 0.0)
                    model.parameters["index"].max = kwargs.get('index_max', 6.5)
                if "alpha" in model.parameters.names:
                    model.parameters["alpha"].min = kwargs.get('alpha_min', -5.0)
                    model.parameters["alpha"].max = kwargs.get('alpha_max', +5.0)
                model.parameters["amplitude"].min = kwargs.get('amplitude_min', 1e-15)
                model.parameters["amplitude"].max = kwargs.get('amplitude_max', 1e-08)

                # Add model to dataset
                dataset_hess.models = [model]

                # Run the fit
                fit = Fit()
                fit_result = fit.run(dataset_hess)
                
                # Print fit result and best-fit model
                log.info(fit_result)
                log.info(fit_result.models[target])
                
                # Set decorrelation as reference energy
                spec = dataset_hess.models[target].spectral_model
                if type(spec) == CompoundSpectralModel:
                    # Select intrinsic model
                    spec = spec.model1
                # Compute decorrelation
                edec = spec.pivot_energy
                # Compute amplitude evaluated at this energy
                aref = spec(edec)
                # Set reference and amplitude (to stabilize numeric convergence)
                try:
                    dataset_hess.models[target].spectral_model.parameters["reference"].quantity = edec
                    dataset_hess.models[target].spectral_model.parameters["amplitude"].quantity = aref
                except:
                    dataset_hess.models[target].spectral_model.model1.parameters["reference"].quantity = edec
                    dataset_hess.models[target].spectral_model.model1.parameters["amplitude"].quantity = aref

                # Run another fit
                fit_result = fit.run(dataset_hess)

                # Print fit result and best-fit model again!
                log.info(fit_result)
                log.info(fit_result.models[target])
                
                # Compute flux points
                fluxp_hess = FluxPointsEstimator(
                   energy_edges = dataset_hess.counts.geom.axes["energy"].edges,
                    source = target,
                    selection_optional = "all", ).run([dataset_hess])
                
                # Define fluxpoints dataset for easier plotting
                fluxp_dataset = FluxPointsDataset(
                    models = fit_result.models[target],
                    data = fluxp_hess, )
                
                # Set label
                if dataset_type == "HAP-FR":
                    label = dataset_type
                else:
                    label = dataset_type + "-" + config

                # Plot spectrum including flux points, fit, and error bars
                fluxp_dataset.plot_spectrum(
                    ax = ax,
                    kwargs_fp = {"color": colors[dataset_type][config], "label": label},
                    kwargs_model = {"color": colors[dataset_type][config], "label": None}, )

                # Save flux points dataset as fits

            except Exception as e:

                log.warning(f"\033[93mWARNING\033[0m: Dataset {dataset_type} with config {config} could not be processed! {e}\n")
                # If dataset couldn't be loaded, just skip it!
                continue
    log.info("Generating plots...\n")
    # Set title
    plt.title(f"{target} SED Datasets")
    # Set plot axes labels
    plt.xlabel("Energy [TeV]")
    plt.ylabel(r"$\text{E}^2d\text{N}/d\text{E}$ [TeV s$^{-1}$ cm$^{-2}$]")
    # Once all datasets have been processed, save plot
    plt.legend()
    # If path doesn't exist, create it
    out_path = f"../{target}/plots/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(f"{out_path}/{target}_sed_{dataset_type}_{config}.png", dpi = 300, bbox_inches = "tight")

    log.info("\033[92mAnalysis complete! :)\n\033[0m")

