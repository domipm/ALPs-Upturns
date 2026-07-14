# Various utilities functions that may be used in multiple scripts at different stages of the analysis

from    __future__              import  annotations

import  os
import  sys
import  yaml

import  numpy                   as      np
import  astropy.units           as      u

from    xml.etree               import  ElementTree     as  ET
from    astropy.coordinates     import  SkyCoord

from    gammapy.modeling        import  Parameters
from    gammapy.modeling.models import (PowerLawSpectralModel, 
                                        PowerLawNormSpectralModel,
                                        ExpCutoffPowerLawSpectralModel,
                                        SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
                                        LogParabolaSpectralModel,
                                        PointSpatialModel,
                                        GaussianSpatialModel,
                                        SkyModel,
                                        GaussianSpatialModel,
                                        TemplateSpatialModel,
                                        Models, Model,
                                        create_fermi_isotropic_diffuse_model,)

from    alpsup.paths            import  SOURCES_FILE, get_results_dir


def get_source_list() -> list:
    """
    Get list of all sources available for analysis (as defined in `sources.yaml` file)
    Returns:
        sources (list): List of all sources names.
    """

    # Open sources file
    with open(SOURCES_FILE, 'r') as f:
        # Load the yaml file
        data = yaml.full_load(f)
        # Get the name of all sources
        sources = list(data["sources"].keys())

    # Return all parameters (sorted alphabetically)
    return sorted(sources)


def get_source_info(target: str):
    """
    Gather relevant data for a given source from sources.yaml file (as defined by $SOURCES_FILE environment variable).
    Parameters:
        target (str): Name of the target source.
    Returns:
        target_4FGL (str): 4FGL Catalog name of the target source (if available).
        target_position (`astropy.coordinates.SkyCoord`): Position of the target source.
        target_redshift (float): Redshift of the source.
    """

    # Open sources file
    with open(SOURCES_FILE, 'r') as f:

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


def set_params(model, entry):
    """
    Auxiliary function to set all parameters of a model
    taking into account equivalencies between GammaPy and FermiPy naming conventions.
    Parameters:
        model (`gammapy.modeling.models.*Model`): GammaPy model to use (either spectral or spatial)
        entry (`xml.etree.ElementTree.Element`): Entry of the XML file that describes a model (either spatial or spectral).
    Returns:
        model (`gammapy.modeling.models.*Model`): GammaPy model with all parameters and values set.
    """

    # Parameter equivalencies between FermiPy and GammaPy including the default units associated
    par_equiv = {
        # SPATIAL MODELS
        "PointSpatialModel": {
            "RA": ["lon_0", u.Unit("deg")],
            "DEC": ["lat_0", u.Unit("deg")],
        },
        "GaussianSpatialModel": {
            "RA": ["lon_0", u.Unit("deg")],
            "DEC": ["lat_0", u.Unit("deg")],
            "Sigma": ["sigma", u.dimensionless_unscaled],
        },
        # SPECTRAL MODELS
        "PowerLawSpectralModel": {
            "Prefactor": ["amplitude", u.Unit("1 / (MeV cm2 s)")],
            "Scale": ["reference", u.Unit("MeV")],
            "Index": ["index", u.dimensionless_unscaled],
        },
        "PowerLawNormSpectralModel": {
            "Prefactor": ["norm", u.dimensionless_unscaled],
            "Scale": ["reference", u.Unit("MeV")],
            "Index": ["tilt", u.dimensionless_unscaled],
            "Normalization": ["norm", u.dimensionless_unscaled],
        },
        "LogParabolaSpectralModel": {
            "norm": ["amplitude", u.Unit("1 / (MeV cm2 s)")],
            "alpha": ["alpha", u.dimensionless_unscaled],
            "beta": ["beta", u.dimensionless_unscaled],
            "Eb": ["reference", u.Unit("MeV")],
        },
        "CompoundSpectralModel": { # 'galdiff' model
            "Normalization": ["norm", u.dimensionless_unscaled]
        },
        "ExpCutoffPowerLawSpectralModel": {
            "Prefactor": ["amplitude", u.Unit("1 / (MeV cm2 s)")],
            "Scale": ["reference", u.Unit("MeV")],
            "Index1": ["index", u.dimensionless_unscaled],
            "Index2": ["alpha", u.dimensionless_unscaled],
            "Expfactor": ["lambda_", u.Unit("1 / MeV")]
        },
        "SuperExpCutoffPowerLaw4FGLDR3SpectralModel" : {
            "Prefactor": ["amplitude", u.Unit("1 / (MeV cm2 s)")],
            "IndexS": ["index_1", u.dimensionless_unscaled],
            "Scale": ["reference", u.Unit("MeV")],
            "ExpfactorS": ["expfactor", u.dimensionless_unscaled],
            "Index2": ["index_2", u.dimensionless_unscaled],
        },
    }

    # Select the correct parameter mapping
    param_map = par_equiv.get(model.__class__.__name__)

    # Loop over all parameters in spatial component
    for p in entry.findall("parameter"):

        # Get GammaPy name of parameter
        p_name = param_map[ p.attrib["name"] ][0]

        # Check if name is within model's parameters
        if p_name in model.parameters.names:
            # Modify the parameter based on the values from file (value_gammapy = value_fermipy * scale_fermipy)
            model.parameters[p_name].value = float( p.attrib["value"] ) * abs( float(p.attrib["scale"]) )
            model.parameters[p_name].min = float( p.attrib["min"] ) * abs( float(p.attrib["scale"]) )
            model.parameters[p_name].max = float( p.attrib["max"] ) * abs( float(p.attrib["scale"]) )
            model.parameters[p_name].frozen = not bool(p.attrib["free"])
            # Try to add error if available
            try:
                model.parameters[p_name].error = float(p.attrib["error"]) * float(p.attrib["scale"])
            except:
                model.parameters[p_name].error = 0.0
            # Set correct units
            model.parameters[p_name].unit = param_map[p.attrib["name"]][1]

    return model


def get_fermipy_models(target, models = None, bblock = None):
    """
    Load the best-fit models obtained from FermiPy in order to use them within GammaPy. 
    Parameters:
        target (str): Name of target source (used to define path to xml file as well as renaming convention).
        ignore_diffuse (bool): Ignore diffuse background models (galactic + isotropic). Default: False
        ignore_target (bool): Ignore the target source (to be defined elsewhere). Default: False
    Returns:
        models_out (`gammapy.modeling.models.Models`): List of models with their best-fit parameters obtained from FermiPy.
    """

    # Get 4FGL name of target (name used in FermiPy)
    target_4FGL, _, _ = get_source_info(target = target)

    # Define the directory containing the XML file
    dir = ("{}/{}/{}/fermi-out/final_00.xml").format(os.environ['RESULTS'], target, bblock)
    # dir = get_results_dir(source = target, bblock = bblock)

    # Load the XML file containing best-fit final models from FermiPy
    tree = ET.parse(source = dir)
    root = tree.getroot()

    # Equivalencies of spatial models
    spatial_equiv = {"SkyDirFunction": PointSpatialModel(frame = "icrs"), 
                     "RadialGaussian": GaussianSpatialModel(),}
    # Equivalences of spectral models
    spectral_equiv = {"PowerLaw": PowerLawSpectralModel(), 
                      "LogParabola": LogParabolaSpectralModel(),
                      "PLSuperExpCutoff2": ExpCutoffPowerLawSpectralModel(),
                      "PLSuperExpCutoff4": SuperExpCutoffPowerLaw4FGLDR3SpectralModel()}

    # Define empty list of models
    models = Models()

    # Loop over all models in file
    for model in root:

        # Get the name of the model
        name = model.attrib["name"]
        # Get the spectral and spatial components
        spectral = model.find("spectrum")
        spatial  = model.find("spatialModel")
        # Get type of source (PointSource or DiffuseSource)
        src_type = model.attrib["type"]

        # Point sources
        if src_type == "PointSource":
            
            # Spatial model - point source
            spatial_model = spatial_equiv[spatial.attrib["type"]].copy()
            # Find and set the equivalent parameters in the model
            set_params(spatial_model, spatial)

            # Initialize the spectral model
            spectral_model = spectral_equiv[spectral.attrib["type"]].copy()
            # Find and set the equivalent parameters in the model
            set_params(spectral_model, spectral)

        # Diffuse models
        elif src_type == "DiffuseSource":

            # Galactic diffuse background
            if name == 'galdiff':

                # Spatial model from template - MapCubeFunction (params: Normalization)
                spatial_model = TemplateSpatialModel.read(
                    # filename = "$FERMIPY_DATA/gll_iem_v07.fits",
                    filename = f"{os.environ['FERMIPY_DATA']}/gll_iem_v07.fits",
                    normalize = False, ).copy()
                # Spectral model - PowerLawNorm
                spectral_model = PowerLawNormSpectralModel().copy()
                # Get best-fit parameters from FermiPy output
                set_params(spectral_model, spectral)

            # Isotropic diffuse background
            if name == 'isodiff':

                # Load isotropic model directly with built-in function as a dataset object
                iso_model = create_fermi_isotropic_diffuse_model(
                    filename = ("{}/iso_P8R3_SOURCE_V3_v1.txt").format(os.environ['FERMIPY_DATA']),
                    datasets_names = "Fermi-LAT"
                ).copy("name = Isotropic")
                spatial_model = iso_model.spatial_model.copy()
                # Spectral model - FileFunction in FermiPy (params: Normalization), 
                # CompoundSpectralModel in GammaPy (TemplateSpectralModel + PowerLawNormSpectralModel)
                spectral_model = iso_model.spectral_model.copy()

                # Read parameters values (in this case, just normalization)
                set_params(spectral_model, spectral)

        # Renaming of models (just for consistency)
        name_equiv = {
            target_4FGL: target,
            "galdiff": "Galactic",
            "isodiff": "Isotropic", }

        # Define the model for each source
        model_model = SkyModel(
            spectral_model = spectral_model,
            spatial_model = spatial_model,
            name = name_equiv.get(name, name),
            datasets_names = "Fermi-LAT", )

        # Append current model to models list
        models.append(model_model)

    return models


def get_fpul(y, y_ul, y_errn, y_errp, ts, ts_threshold = 4.0, ul_err = 0.2):
    """
    Auxiliary function to get flux points and upper limits values, including their errors
    and maintaining units.
    Parameters:
        y (float, `np.array`, or `astropy.Quantity`): main array of values to process.
        y_ul (float, `np.array`, or `astropy.Quantity`): array of upper limit point values.
        y_errn (float, `np.array` or `astropy.Quantity`): array of negative error of values.
        y_errp (float, `np.array` or `astropy.Quantity`): array of positive error of values.
        ts (float, `np.array` or `astropy.Quantity`): test statistic associated to each point.
        ts_threshold (float): threshold of test statistic at which point considered upper limit.
        ul_err (float): multiplier to assign for each upper limit value, such that final y_ul *= ul_err.
    Returns:
        y_val (`np.array` of `astropy.Quantity` of float): final values for each point.
        y_err (`np.array` of `astropy.Quantity` of float): final error values (positive and negaitve) for each point.
        is_ul ((`np.array` of bool): boolean whether each point is upper limit according to threshold.
    """

    # Generate arrays containing flux points and upper limits
    is_ul = ts < ts_threshold
    y_val = np.where(is_ul, y_ul, y)
    y_err = np.array([ 
        np.where(is_ul, ul_err * y_ul, y_errn),
        np.where(is_ul, ul_err * y_ul, y_errp), ])
    
    # Preserve units in error if given in main value
    if type(y_err) == np.ndarray and type(y) == u.Quantity:
        y_err *= y.unit
        
    return y_val, y_err, is_ul


def par_uconv(e_unit, model, p_type = "energy"):
    """
    Auxiliary function used to convert the energy unit of all spectral
    parameters of a model, while keeping the other units the same.
    Args:
        e_unit (str): Desired final unit for the energy.
        model (`gammapy.modeling.models`): model object to convert units.
        p_type (str): Physical type of the unit we wish to convert. Default: 'energy'.
    Returns:
        model (`gammapy.modeling.models`): final model with converted units.
    """

    # Loop over all parameters in the spectral model
    for par in model.spectral_model.parameters:
        # Check if unit has energy component
        if p_type in [ u.get_physical_type(base) for base in par.unit.bases ]:
            # Track original units or parameter
            units_og = par.unit
            # Define final units object (dimensionless initially)
            units = u.dimensionless_unscaled
            # Loop over all unit bases and their powers found in parameter
            for b, p in zip( par.unit.bases, par.unit.powers ):
                # If it corresponds to energy
                if b.physical_type == p_type:
                    # Convert to correct units
                    units *= np.power( b, p ).to( np.power( u.Unit(e_unit), p ) ) * np.power( u.Unit(e_unit), p )
                # Keep remaining units the same
                else:
                    units *= np.power( b, p )
            # Convert parameter value, min, max, to final units
            par.value = (par.value * units_og).to(units.unit).value
            par.error = (par.error * units_og).to(units.unit).value
            par.max = (par.max * units_og).to(units.unit).value
            par.min = (par.min * units_og).to(units.unit).value
            par.unit = units.unit

    return model


def tab_uconv(e_unit, table):
    """
    Auxiliary function used to convert the energy unit of all columns
    within an `astropy.table` object containing flux points for a
    'e2dnde' spectral energy distribution.
    Args:
        e_unit (str): Desired final unit for the energy.
        table (`astropy.table`): model object to convert units.
    Returns:
        table (`astropy.table`): final table with converted units.
    """

    # TODO: Make this general (same as par_uconv function)

    for col in table.colnames:
        if table[col].unit is None:
            continue
        elif table[col].unit.is_equivalent(u.Unit(e_unit) / (u.cm**2 * u.s)):
            table[col] = table[col].quantity.to(u.Unit(e_unit) / (u.cm**2 * u.s))
        elif table[col].unit.is_equivalent(u.Unit(e_unit)):
            table[col] = table[col].quantity.to(u.Unit(e_unit))
    return table


def met_to_mjd(time_met):
    """
    Convert between Fermi-LAT Mission Elapsed Time (MET)
    to Modified Julian Date (MJD).
    Args:
        time_met (`astropy.time.Time`): Time in Fermi-LAT MET
    Returns:
        time_mjd (`astropy.time.Time`): Time in MJD
    """

    MJDREFF = 51910
    MJDREFFI = 7.428703703703703 * (10**-4)

    # Convert from elapsed seconds to elapsed days
    elapsed_days = time_met / 86400.

    # Add the elapsed days to the refrence epoch
    time_mjd = MJDREFF + MJDREFFI + elapsed_days

    return time_mjd


def parse_kwargs(args):
    """
    Parse all keyword arguments given and return as dictionary.
    Args:
        args (dict): Arguments of any type.
    Returns:
        kwargs (dict): Parsed keyword arguments.
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


def get_edec(model: Model) -> u.Quantity:
    """
    Compute decorrelation energy for a model from covariance of parameters.

    Parameters
    ----------
    model (`gammapy.modeling.models.Model`): target source model object.

    Returns
    -------
    e_dec (`astropy.Quantity`): Decorrelation energy with units.
    """

    e0 = model.parameters['reference'].quantity
    f0 = model.parameters['amplitude'].value

    cov_gamma = model.covariance.data

    e_dec = e0 * np.exp(cov_gamma[0, 1] / f0 / cov_gamma[0, 0])
    return e_dec


def get_etau(redshift: float, ebl_model: str = "dominguez", tau_lim: float = 2.0) -> u.Quantity:
    """
    Compute energy E(tau = tau_lim) at which, for a given redshift and EBL model, the optical depth is tau_lim.

    Parameters
    ----------
    redshift (float)    : Redshift of the source.
    ebl_model (str)     : Name of EBL model to use. Default: 'dominguez'.
    tau_lim (float)     : Optical depth limit value. Default: 2.0.

    Returns
    -------
    e_tau (`astropy.Quantity`)  : Energy for limit optical depth.
    """

    from    ebltable.tau_from_model import  OptDepth

    # Create tau values for given redshift
    tau_optdepth = OptDepth.readmodel(model = ebl_model)

    # Define energy grid for interpolation (fixed at interval 0.1 to 100 TeV)
    e_grid = np.logspace(-1, 2, 100)
    # Compute tau values at different energies for the given redshift
    tau_values = np.array( [ tau_optdepth.opt_depth(redshift, e) for e in e_grid ] )
    # Obtain value of energy for which tau = 1 or 2 (interpolating)
    tau_limit_e = np.interp(tau_lim, tau_values, e_grid)

    # Return final value (given in TeV by default)
    e_tau = tau_limit_e * u.TeV
    return e_tau


def is_converged(params: Parameters, exclude_params: list = []) -> tuple[bool, str]:
    """
    Script to check for convergence of all parameters for a model
    
    Parameters
    ----------
    params (gammapy.modeling.Parameters) : Parameters of a given model
    
    Returns
    -------
    is_converged (bool) : Status of convergence (True/False)
    info (str)          : Information on convergence or error
    """
    # Loop over free parameters
    for param in params:
        if param.frozen == False and param.name not in exclude_params:
            # Check if parameter saturates
            if np.round(param.value, 3) in [param.min, param.max]:
                return False, f"Parameter {param.name} saturated (value: {param.value})!"
            # Check if parameter has very low relative error
            if param.error / param.value > 10.0:
                return False, f"Parameter {param.name} has too low relative error!"
            # Check if parameter is 
    # If passes these tests, converged
    return True, "Converged"


# TODO: SAVE OUTPUT FILES!

'''
def save_output(
        dataset: Datasets = None,   # Datasets object
        model: Models = None,       # Models object (redundant if datasets given)
        fluxp: None = None,         # Flux points object or table
        fit: None = None,           # Fit results object
):

    # TODO: DEPENDING ON WHAT'S GIVEN, SAVE OUTPUT FILES
    # TODO: SAVE AS GENERAL AS POSSIBLE (ECSV, FITS) ALL FORMATS

    return
'''

