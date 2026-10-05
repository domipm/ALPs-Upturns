from gammapy.modeling.models import SkyModel, SpectralModel


# Get current logging system
import logging
log = logging.getLogger(__name__)


PARAMETERS_FIT_DEFAULT_BOUNDS = {
    # Amplitude, global parameter (in TeV cm-2 s-1)
    "amplitude": {
        "amplitude_min": 1e-20,
        "amplitude_max": 1e-04, },
    # Reference, global parameter (in TeV)
    "reference": {
        "reference": 1.00,
        "reference_min": None,
        "reference_max": None, },
    # PowerLaw
    "index": {
        "index_min": 0.00,
        "index_max": 5.00, },
    # LogParabola
    "alpha": {
        "alpha_min": 0.00,
        "alpha_max": 5.00, },
    "beta_lp": {
        "beta": 0.50,
        "beta_min": 0.00,
        "beta_max": 2.50, },
    # SmoothBrokenPowerLaw
    "index1": {
        "index1": 2.00,
        "index1_min": 0.00,
        "index1_max": 5.00, },
    "index2": {
        "index2": 2.50,
        "index2_min": 0.00,
        "index2_max": 5.00, },
    "indexdelta": {
        "indexdelta": 2.00,
        "indexdelta_min": 0.00,
        "indexdelta_max": 5.00, },
    "beta_sbpl": {
        "beta": 1.00,
        "beta_min": 0.10,
        "beta_max": 2.50, },
    # NOTE / TODO: Ebreak parameter bounds based on energy range of datasets! (in TeV)
    "ebreak": {
        "ebreak_min": 0.01,
        "ebreak_max": 1.00, },
}


# TODO: Not sure if this is used?
PARAMETERS_UPT_DEFAULT_BOUNDS = {
    # Amplitude, global parameter (in TeV cm-2 s-1)
    "amplitude": {
        "amplitude_min": 1e-17,
        "amplitude_max": 1e-04, },
    # PowerLaw
    "index": {
        "index_min": 0.00,
        "index_max": 7.50, },
    # LogParabola
    "alpha": {
        "alpha_min": 0.00,
        "alpha_max": 7.50, },
    "beta": {
        "beta_min": 0.00,
        "beta_max": 2.50, },
    # SmoothBrokenPowerLaw
    "index1": {
        "index1_min": 0.00,
        "index1_max": 6.50, },
    "index2": {
        "index2_min": 0.00,
        "index2_max": 6.50, },
    # NOTE: Ebreak bounds based on energy range of datasets! (in TeV)
    "ebreak": {
        "ebreak_min": 0.01,
        "ebreak_max": 1.00, },
    # TODO: UpturnSpectralModel parameters?
}


def set_param_bounds(model: SpectralModel | SkyModel, 
                     bounds: str, **kwargs):

    # Set which parameter bounds to use as defaults
    if bounds == "fit":
        default_bounds = PARAMETERS_FIT_DEFAULT_BOUNDS
    elif bounds == "upt":
        default_bounds = PARAMETERS_UPT_DEFAULT_BOUNDS
    else:
        raise KeyError(f"Unknown type of bounds: {bounds}!")

    # Loop over all parameters in the spectral model
    for par in model.parameters:

        # Get name of parameter
        parname = par.name
        # Get type of intrinsic spectral model
        spectral_model = model.spectral_model
        if spectral_model.__class__.__name__ is "CompositeSpectralModel":
            intrinsic_model = spectral_model.intrinsic_model
        else:
            intrinsic_model = spectral_model

        # NOTE: Leave beta of SmoothBrokenPowerLaw default always!
        # TODO: This could have been done by renaming keys in parameters default into class name of model
        if par.name == "beta" and intrinsic_model.__class__.__name__ == "SmoothBrokenPowerLawSpectralModel":
            parname = par.name + "_sbpl"

        elif par.name == "beta" and intrinsic_model.__class__.__name__ == "LogParabolaSpectralModel":
            parname = par.name + "_lp"
        
        # Check if parameter name in list of expected parameters
        if parname not in default_bounds:
            log.warning(f"Parameter {par.name} not found in default bound list! Skipping.")
            continue

        # Get default parameters for current parameter
        par_defaults = default_bounds[parname]

        # Set default parameter values (if not given and not found, ignore)
        par_value = kwargs.get(f"{par.name}", par_defaults.get(f"{par.name}", None))
        if par_value is not None:
            par.value = par_value
        # Set parameters based on given kwargs, or if not given, default bounds
        par.min = kwargs.get(f"{par.name}_min", par_defaults[f"{par.name}_min"])
        par.max = kwargs.get(f"{par.name}_max", par_defaults[f"{par.name}_max"])

    # NOTE: No need to return model, as params modified in-place!
    return

