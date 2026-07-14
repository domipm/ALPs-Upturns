import  os
import  astropy.units       as      u

from    pathlib                 import  Path
from    astropy.time            import  Time
from    gammapy.maps            import  MapAxis
from    gammapy.datasets        import  SpectrumDatasetOnOff, Datasets
from    gammapy.modeling.models import  Models

from    alpsup.paths            import  get_hess_data_dir


def get_hess_dataset(target: str, 
                     dataset: str = "HAP-HD", 
                     config: str = "std_ImPACT_hybrid_fullEnclosure_updated",
                     dataset_name: str = "HESS",
                     time_select_min: Time = None,
                     time_select_max: Time = None, 
                     time_select_atol: u.Quantity = 0.015 * u.d,
                     time_select_fmt: str = "mjd",
                     nbins: int = 4,
                     nbins_per_dec: bool = True):

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
    # If values not given, default to GTIs
    else:
        dataset_obs.select_time(
            atol = time_select_atol,
            time_min = Time( [sorted(dataset_obs.gti.time_start)[0].mjd], format = "mjd" ),
            time_max = Time( [sorted(dataset_obs.gti.time_start)[-1].mjd], format = "mjd", ) )

    # Stack reduce the dataset
    dataset = dataset_obs.stack_reduce(name = dataset_name)

    # Define energy axis (from dataset bounds)
    energy_axis = MapAxis.from_energy_bounds(
        energy_min = dataset.counts.geom.axes["energy"].bounds[0],
        energy_max = dataset.counts.geom.axes["energy"].bounds[1],
        unit = u.TeV, nbin = nbins, per_decade = nbins_per_dec, )

    # Resample energy axis
    dataset.resample_energy_axis(energy_axis, name = dataset_name)

    return dataset_obs, dataset


def set_dataset_model(dataset, target):

    model_target = None

    # If no model given, read from Fermi-LAT analysis

    # If model given, select

    # If spatial model given, set

    # Freeze parameters

    # Convert parameters if required

    return

