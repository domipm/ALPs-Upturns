import  os
import  astropy.units       as      u

from    pathlib                 import  Path
from    astropy.time            import  Time
from    gammapy.maps            import  Map, MapAxis
from    gammapy.datasets        import  SpectrumDatasetOnOff, Datasets, FermipyDatasetsReader
from    gammapy.modeling.models import  Models
from    gammapy.irf             import  EDispKernelMap

from    alpsup.paths            import  get_hess_data_dir, get_results_dir
from    alpsup.utils            import  get_fermipy_models


ATOL = 0.015 * u.d


def get_hess_dataset(target: str, 
                     dataset: str = "HAP-HD", 
                     config: str = "std_ImPACT_hybrid_fullEnclosure_updated",
                     dataset_name: str = "HESS",
                     time_select_min: Time = None,
                     time_select_max: Time = None, 
                     time_select_atol: u.Quantity = ATOL,
                     time_select_fmt: str = "mjd",
                     nbins: int = 4,
                     nbins_per_dec: bool = True,
                     bblock: str | None = None):

    # Load dataset from folder
    path = get_hess_data_dir(target, hap_dataset = dataset, hap_config = config)

    # Get all fits files
    obs_rmfs  = list(path.glob('*rmf.fits'))
    obs_files = []
    for obs in obs_rmfs:
        obs_files.append(Path(str(obs).replace('_rmf', '')))

    # Empty dataset object
    dataset_obs = Datasets()

    # Append each observation to datasets
    for obs in obs_files:
        dataset_obs.append(SpectrumDatasetOnOff.read(obs))

    # Perform time selection
    # If values given (in MJD by default)
    if time_select_min is not None and time_select_max is not None:
        dataset_obs.select_time(
            atol = time_select_atol,
            time_min = Time( [time_select_min], format = time_select_fmt ),
            time_max = Time( [time_select_max], format = time_select_fmt, ) )
    # TODO: INCORPORATE TIME SELECTION HERE?
    # If values not given...
    else:
        # If bblock specified, attempt to read from file
        if bblock is not None:
            pass
        # Otherwise, default to GTIs of the dataset
        else: 
            dataset_obs.select_time(
                atol = time_select_atol,
                time_min = Time( [sorted(dataset_obs.gti.time_start)[0].mjd], format = "mjd" ),
                time_max = Time( [sorted(dataset_obs.gti.time_start)[-1].mjd], format = "mjd", ) )

    # Stack reduce the dataset
    dataset = dataset_obs.stack_reduce(name = dataset_name)

    # Define energy axis
    energy_axis = MapAxis.from_energy_bounds(
        energy_min = dataset.counts.geom.axes["energy"].bounds[0],
        energy_max = dataset.counts.geom.axes["energy"].bounds[1],
        unit = u.TeV, nbin = nbins, per_decade = nbins_per_dec, )

    # Resample energy axis
    dataset = dataset.resample_energy_axis(energy_axis, name = dataset_name)

    return dataset_obs, dataset


def get_flat_dataset(target: str,
                     bblock: str,
                     dataset_name: str = "Fermi-LAT",
                     include_models: bool = True,
                     combine_bkg_models: bool = True,
                     pad = 50):

    # Read dataset based on fermipy configuration and output files
    reader = FermipyDatasetsReader(f"{get_results_dir(target, bblock)}/fermi_config.yaml", edisp_bins = 0)
    # Select only Fermi-LAT dataset
    dataset_flat = reader.read()[0].copy(name = "Fermi-LAT")

    # Extend counts, exposure, and background maps
    dataset_flat.counts = dataset_flat.counts.pad(pad)
    dataset_flat.exposure = dataset_flat.exposure.pad(pad)
    dataset_flat.background = dataset_flat.background.pad(pad)

    dataset_flat.exposure = Map.read(f"{get_results_dir(target, bblock, ebl = None, output = "fermi-out")}/bexpmap_00.fits").interp_to_geom(dataset_flat.exposure.geom)

    # Define energy dispersion (identity matrix, disable edisp)
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis = dataset_flat.counts.geom.axes["energy"],
        energy_axis_true = dataset_flat.exposure.geom.axes["energy_true"])
    # Add energy dispersion to dataset
    dataset_flat.edisp = edisp

    # Define safe mask fitting models outside roi
    mask_fit = Map.from_geom(dataset_flat.counts.geom, data = True, dtype = bool).binary_erode(width = 5.0 * u.deg, kernel = "disk")
    # Add safe mask to dataset
    dataset_flat.mask_fit = mask_fit
    dataset_flat.mask_safe = mask_fit

    # Load models
    if include_models:

        # Load models and parameters from fermipy
        models_fermi = get_fermipy_models(target, bblock = bblock)

        # Select all models except target and diffuse backgrounds
        models = Models([model for model in models_fermi if model.name != "Isotropic" and model.name != "Galactic" and model.name != target])
        # Get all models outside roi
        models_out = models.select_mask(~mask_fit, use_evaluation_region = False)

        # Get all models inside  roi
        models_in  = Models([model for model in models if model not in models_out])
        # Convert models outside roi into template
        if combine_bkg_models:
            models_out = models_out.to_template_sky_model(
                geom = dataset_flat.exposure.geom, name = "Models Background", )

        # Add name for serialization
        models_out.spatial_model.filename = f"{get_results_dir(target, bblock, output = "gamma-out")}/flat_models_background.fits"
        # Add dataset name to model
        models_out.datasets_names = "Fermi-LAT"
        # Write out spatial model to file
        models_out.spatial_model.write(filename = f"{get_results_dir(target, bblock, output = "gamma-out")}/flat_models_background.fits", overwrite = True)

        # Add all models to our dataset
        dataset_flat.models = Models( [
            models_fermi[target], 
            models_fermi["Galactic"], models_fermi["Isotropic"],
            *models_in, models_out ] )
        
        # Freeze all model parameters
        dataset_flat.models.freeze()
        # Free only target source spectral parameters
        for p in dataset_flat.models[target].spectral_model.parameters:
            # Leave reference fixed
            if p.name != "reference":
                # Free index, amplitude, alpha, beta, ... (depending on model)
                dataset_flat.models[target].spectral_model.parameters[p].frozen = False
        # Free galactic and isotropic parameter normalization
        dataset_flat.models["Galactic"].parameters["norm"].frozen = False
        dataset_flat.models["Isotropic"].parameters["norm"].frozen = False
            
    return dataset_flat


def set_dataset_model(dataset, target):

    model_target = None

    # If no model given, read from Fermi-LAT analysis

    # If model given, select

    # If spatial model given, set

    # Freeze parameters

    # Convert parameters if required

    return

