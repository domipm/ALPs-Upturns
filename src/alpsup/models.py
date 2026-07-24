# Contains all custom models used for ALP upturn search
# As well as compound model and on-off dataset including bias prior

from    __future__              import  annotations

import  yaml

import  numpy                       as      np
import  astropy.units               as      u
import  matplotlib.pyplot           as      plt

from    gammapy.datasets            import  SpectrumDatasetOnOff, DATASET_REGISTRY
from    gammapy.maps                import  MapAxis
from    gammapy.estimators.map.core import  DEFAULT_UNIT
from    gammapy.modeling            import  Parameter, Parameters
from    gammapy.modeling.models     import  SpectralModel, EBLAbsorptionNormSpectralModel, TemplateSpectralModel, SPECTRAL_MODEL_REGISTRY, scale_plot_flux
from    gammapy.utils.scripts       import  make_path

from    astropy.visualization       import  quantity_support


# Custom model used for simulating upturns (smooth break)
class SmoothBrokenSpectralModel(SpectralModel):
    """
    Custom spectral model for upturn searches by introducing smooth break in spectrum,
    with parameters: 
    indexdelta_brk (strictly < 0 for upturns), beta_brk (frozen at 1 by default),
    and break energy e_brk.
    """

    # Define tag for model
    tag = ["SmoothBrokenSpectralModel"]

    # Define parameters
    indexdelta_brk = Parameter(
        name = "indexdelta_brk",
        value = -2.0,
        min = -5.0,
        max = 0.0,
        frozen = False, )
    beta_brk = Parameter(
        name = "beta_brk",
        value = 1.0,
        min = 0.01,
        max = 1.00,
        frozen = True, )
    e_brk = Parameter(
        name = "e_brk",
        value = 1.0,
        unit = "TeV",
        min = 0.1,
        max =  31.6,
        frozen = False, )
    
    # Evaluate method - return function value
    @staticmethod
    def evaluate(energy, indexdelta_brk, beta_brk, e_brk):
        # Multiply beta by sign of delta index (!)
        beta_brk *= np.sign(indexdelta_brk)
        # Return spectral model evaluated at energy and indices
        return (1 + (energy / e_brk) ** ( (indexdelta_brk) / beta_brk ) ) ** (- beta_brk)


# Custom spectrum dataset for HESS including a Gaussian prior on a spectral bias parameter
class BiasedPriorSpectrumDatasetOnOff(SpectrumDatasetOnOff):
    """
    Spectrum ON-OFF Dataset with a Gaussian prior on a spectral bias parameter
    Extra term added to -2 log L: (bias / sigma_bias)^2
        where bias is taken from the HESS model containing bias parameter

    Used only for HESS dataset to encode the ~15% energy-scale calibration uncertainty
    as a nuisance parameter
    """

    # Define tag for dataset
    tag = "BiasedPriorSpectrumDatasetOnOff"

    def __init__(self, *args, sigma_bias = None, **kwargs):
        self.sigma_bias = sigma_bias
        super().__init__(*args, **kwargs)

    @property
    def sigma_bias(self):
        return self._sigma_bias
    
    @sigma_bias.setter
    def sigma_bias(self, value):
        self._sigma_bias = value

    @classmethod
    def from_spectrum_dataset(cls, dataset, sigma_bias):
        """
        Clone an existing SpectrumDatasetOnOff and attach a bias prior

        Parameters:
            dataset (`gammapy.datasets.SpectrumDatasetOnOff`): Original HESS spectral dataset.
            sigma_bias (float): Gaussian prior width for the bias parameter (fractional).
        """
        
        # Return instance of this class with all required parameters, including bias
        return cls(
            models=dataset.models,
            counts=dataset.counts,
            counts_off=dataset.counts_off,
            exposure=dataset.exposure,
            edisp=dataset.edisp,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=dataset.acceptance,
            acceptance_off=dataset.acceptance_off,
            gti=dataset.gti,
            name=dataset.name,
            meta_table=dataset.meta_table,
            sigma_bias=sigma_bias, )
    
    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create spectrum dataset from dictionary.

        Reads file from the disk as specified in the dict.

        Parameters
        ----------
        data : dict
            Dictionary containing data to create dataset from.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.
        """

        filename = make_path(data["filename"])

        # Call default read class
        dataset_def = cls.read(filename = filename)
        dataset_def.mask_fit = None

        # Convert dataset to biased prior dataset
        dataset = cls.from_spectrum_dataset( dataset = dataset_def, sigma_bias = data["sigma_bias"])

        return dataset

    def to_dict(self):
        """Convert to dict for YAML serialization."""

        filename = f"pha_obs{self.name}.fits"

        data = {"name": self.name, "type": self.tag, "filename": filename}

        # Append bias parameter
        data.update( { "sigma_bias": self._sigma_bias } )

        return data
    
    def stat_sum(self):
        """
        Standard -2 log L from SpectrumDatasetOnOff plus Gaussian prior on HESS bias parameter
        """

        # Compute stat
        stat = super().stat_sum()

        # If no bias, return stat
        if self._sigma_bias is None:
            return stat

        # Load model that presents a bias parameter (which should be the target)
        bias_par = None
        for m in self.models:
            if "bias" in m.parameters.names:
                bias_par = m.parameters["bias"]
                break
        
        # If bias parameter found and not frozen
        if bias_par is not None and not bias_par.frozen:
            # Add to statistic
            stat += (bias_par.value / self._sigma_bias) ** 2

        # Return final stat
        return stat


# Custom class for wrapping together compound intrinsic model with EBL
# including a Gaussian prior on a spectral bias parameter
class BiasedCompoundSpectralModel(SpectralModel):
    """
    HESS wrapper: PowerLaw * EBL with single fractional energy bian E' = E * (1 + bias)
    applied to both intrinsic spectrum and tau(E)
    """

    tag = "BiasedCompositeSpectralModel"
    bias = Parameter(name = "bias",
                     value = 0.00,
                     min = - 0.25,
                     max = + 0.25, )
    
    def __init__(self, intrinsic_model, ebl_model, bias = bias.quantity, **kwargs):
        self._intrinsic_model = intrinsic_model
        self._ebl_model = ebl_model

        super().__init__(bias = bias, **kwargs)

    @property
    def intrinsic_model(self):
        return self._intrinsic_model
    
    @property
    def ebl_model(self):
        return self._ebl_model
    
    @property
    def parameters(self):
        return(
            Parameters([self.bias]) + self._intrinsic_model.parameters + self._ebl_model.parameters )

    # TODO: WHEN PLOTTING WITH BIAS AND UPTURN, EVALUATION IN "PLOT_ERROR" PASSES ALL ARGUMENTS
    # THIS RESULTS IN ERROR! "evaluate() takes 2 positional arguments, but 8 were given"
    # NEEDS TO OVERRIDE EVALUATE IN THIS CASE!

    def evaluate(self, energy, *args):

        args1 = args[: len(self.model1.parameters)]
        args2 = args[len(self.model1.parameters) :]

        val1 = self.model1.evaluate(energy, *args1)
        val2 = self.model2.evaluate(energy, *args2)

        return self.operator(val1, val2)
    

    def evaluate_error(self, energy, n_samples=3500, random_state=42, samples=None):
        """Evaluate spectral model error from parameter distribution sampling."""

        m = self.copy()
        pars_names = [p.name for p in m.parameters]

        def fct(*args):
            kwargs = dict(zip(pars_names, args))
            return m.evaluate(energy, **kwargs)

        propagated_samples = self._samples(
            fct,
            n_samples = len(pars_names) * n_samples,
            random_state = random_state,
            samples = samples, )

        return self._get_errors(propagated_samples)
    
    # Override pivot energy to be evaluated at intrinsic model only
    def pivot_energy(self):
        return self.intrinsic_model.pivot_energy
        
    # Override plot error function
    def plot_error(self, energy_bounds, ax = None, sed_type = "dnde", energy_power = 0, n_points = 100, n_samples = 3500, random_state = 42, samples = None, facecolor = "black", **kwargs,):
        """Plot spectral model error band"""

        if isinstance(energy_bounds, (tuple, list, u.Quantity)):
            energy_min, energy_max = energy_bounds
            energy = MapAxis.from_energy_bounds(
                energy_min,
                energy_max,
                n_points, )
        elif isinstance(energy_bounds, MapAxis):
            energy = energy_bounds

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)
        if ax.yaxis.units is None:
            ax.yaxis.set_units(DEFAULT_UNIT[sed_type] * energy.unit**energy_power)

        flux, flux_errn, flux_errp = self._get_plot_flux_error(
            sed_type=sed_type,
            energy=energy,
            n_samples=n_samples,
            random_state=random_state,
            samples=samples, )
        y_lo = scale_plot_flux(flux - flux_errn, energy_power).quantity[:, 0, 0]
        y_hi = scale_plot_flux(flux + flux_errp, energy_power).quantity[:, 0, 0]
        
        # Append facecolor to fillbetween
        kwargs["facecolor"] = facecolor
        with quantity_support():
            ax.fill_between(energy.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax, energy_power, sed_type)
        return ax
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize model from dictionary"""

        # Reconstruct intrinsic spectral model
        intrinsic_data = data["spectral"].pop('intrinsic_model')["spectral"]
        intrinsic_tag = intrinsic_data.get('type')
        intrinsic_model_cls = SPECTRAL_MODEL_REGISTRY.get_cls(intrinsic_tag)
        intrinsic_model = intrinsic_model_cls.from_dict(intrinsic_data)

        # Reconstruct EBL model
        ebl_data = data["spectral"].pop('ebl_model')
        ebl_tag = ebl_data["spectral"].get('type')
        ebl_model_cls = SPECTRAL_MODEL_REGISTRY.get_cls(ebl_tag)
        ebl_model = ebl_model_cls.from_dict(ebl_data)

        # Load parameters of biased composite model
        biascomp_data = data["spectral"]["parameters"]
        # NOTE: Remove "scale_transform" (problematic?)
        for biascomp_data_dict in biascomp_data:
            try:
                biascomp_data_dict.pop("scale_transform")
            except:
                pass

        biascomp_params = Parameters.from_dict(   
           data = biascomp_data, )

        # Create model instance
        model = cls(
            intrinsic_model = intrinsic_model,
            ebl_model = ebl_model,
            bias = biascomp_params["bias"].value, )
        
        # Upadate bias error
        model.parameters["bias"].error = biascomp_params["bias"].error

        # Return final version of model with all parameters
        return model
    
    def to_dict(self, full_output = False):
        """Serialize model to dictionary with full structure"""

        # Call original function to generate dictionary
        data = super().to_dict(full_output = full_output)

        # Add intrinsic and EBL model information (must be within "spectral" info in model)
        data["spectral"]["intrinsic_model"] = self._intrinsic_model.to_dict(full_output=full_output)
        data["spectral"]['ebl_model'] = self._ebl_model.to_dict(full_output=full_output)

        return data
    
    @classmethod
    def read(cls, filename):
        """Read model from YAML file"""

        # Open yaml file for the model
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)

        # Extract model from dictionary
        return cls.from_dict(data["components"][0])
        
    # Override evaluate function (fixed)
    def evaluate(self, energy, **kwargs):
        """Evaluate the model with variable parameters"""

        bias = kwargs.pop("bias")

        kwargs_ebl = {}
        for name in self._ebl_model.parameters.names:
            kwargs_ebl[name] = kwargs.pop(name)

        kwargs_intr = {}
        for name in self._intrinsic_model.parameters.names:
            kwargs_intr[name] = kwargs.pop(name)

        # Compute re-scaled energy
        energy_scaled = energy * (1.0 + bias)
        
        # Evaluate absorption and intrinsic models
        absorption = self._ebl_model.evaluate(energy_scaled, **kwargs_ebl)
        flux_intr = self._intrinsic_model.evaluate(energy_scaled, **kwargs_intr)
        
        # Return final value of absorbed intrinsic model
        return absorption * flux_intr


# Add custom models to registry for serialization
# (this is done every time script imported)
SPECTRAL_MODEL_REGISTRY.append(SmoothBrokenSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(BiasedCompoundSpectralModel)

# Add custom dataset to registry for serizalization
DATASET_REGISTRY.append(BiasedPriorSpectrumDatasetOnOff)


# TODO: SIMPLE EBLTABLE MODEL WRAPPER


class EBLTableSpectralModel(TemplateSpectralModel):

    tag = ["EBLTableSpectralModel"]

    def __init__(self, energy: u.Quantity, values: np.ndarray, 
                 ebl_name: str | None = None, redshift: float | np.floating | None = None, 
                 **kwargs, ):
        
        self._ebl_name = ebl_name
        self._redshift = redshift

        super().__init__(energy = energy, values = values, 
                         meta = {"name": self._ebl_name, "redshift": self._redshift},
                         **kwargs, )
        
    @property
    def ebl_name(self):
        return self._ebl_name
    
    @property
    def redshift(self):
        return self._redshift
        
    @classmethod
    def read_ebl(cls, energy = None,
                 ebl_name = "dominguez", redshift = 0.1, ):
        """Build directly from EBLTable, given a model name and redshift"""

        if energy is None:
            # Define energy array to evaluate
            energy = np.logspace(-1, 1.5, 200) * u.TeV

        from ebltable.tau_from_model import OptDepth
        # Construct optical depth from model data
        tau = OptDepth.readmodel(model = ebl_name)
        att = np.exp(-1.0 * tau.opt_depth(redshift, energy.value))

        return cls(energy = energy, values = att * u.dimensionless_unscaled,
                   ebl_name = ebl_name, redshift = redshift, )
    
    def to_dict(self, full_output = False):

        data = super().to_dict(full_output=full_output)["spectral"]

        items = list(data.items())
        idx = next(i for i, (k, _) in enumerate(items) if k == "parameters")

        items[idx:idx] = [
            ("ebl_name", self.ebl_name),
            ("redshift", self.redshift),
        ]

        return {"spectral": dict(items)}

    @classmethod
    def from_dict(cls, data):
        
        data = data.copy()

        ebl_name = data["spectral"].pop("ebl_name", None)
        redshift = data["spectral"].pop("redshift", None)

        template = TemplateSpectralModel.from_dict(data)

        return cls(energy = template.energy, values = template.values,
                   ebl_name = ebl_name, redshift = redshift, )


# TODO: SIMPLE SPECTRAL MODEL WRAPPER FOR UPTURN MODEL


class UpturnSpectralModel(SpectralModel):

    # Define tag for model
    tag = ["UpturnSpectralModel"]

    # Define parameters
    indexdelta_upt = Parameter(
        name = "indexdelta_upt",
        value = -2.0,
        min = -5.0,
        max = 0.0,
        frozen = False, )
    beta_upt = Parameter(
        name = "beta_upt",
        value = 1.00,
        min = 0.01,
        max = 1.00,
        frozen = True, )
    e_upt = Parameter(
        name = "e_upt",
        value = 1.0,
        unit = "TeV",
        min = 0.1,
        max =  31.6,
        frozen = False, )
    
    # Evaluate method - return function value
    @staticmethod
    def evaluate(energy, indexdelta_upt, beta_upt, e_upt):
        # Multiply beta by sign of delta index (!)
        beta_upt *= np.sign(indexdelta_upt)
        # Return spectral model evaluated at energy and indices
        return (1 + (energy / e_upt) ** ( (indexdelta_upt) / beta_upt ) ) ** (- beta_upt)


# TODO: GLOBAL WRAPPER THAT CONTAINS INTRINISC MODEL, EBL ABSORPTION, BIAS, AND UPTURN TERMS


class CompositeSpectralModel(SpectralModel):


    tag = "CompositeSpectralModel"

    # Bias parameter
    bias = Parameter(name = "bias",
                     value = 0.00,
                     min = -0.25,
                     max = +0.25,
                     frozen = False, )

    def __init__(self, 
                 intrinsic_model: SpectralModel, 
                 # TODO: Implement GammaPy built-in EBL model as well
                 ebl_model: EBLTableSpectralModel | TemplateSpectralModel | None, 
                 upturn_model: UpturnSpectralModel | None = UpturnSpectralModel(),
                 bias: float | u.Quantity = bias.quantity, 
                 **kwargs):

        self._intrinsic_model = intrinsic_model
        self._ebl_model = ebl_model
        self._upturn_model = upturn_model
        self._bias = bias

        super().__init__(bias = bias, **kwargs)

    def _get_kwargs(model, kwargs):

        # Construct dictionary for the kwargs of each model component
        return {name: kwargs.pop(name) for name in model.parameters.names}

    @property
    def intrinsic_model(self):
        return self._intrinsic_model
    
    @property
    def ebl_model(self):
        return self._ebl_model
    
    @property
    def upturn_model(self):
        return self._upturn_model
    
    @property
    def parameters(self):

        # Define total parameters
        pars = Parameters([])

        # Intrinsic parameters
        pars_int = self._intrinsic_model.parameters

        # Add intrinsic parameters to total
        pars += pars_int

        # EBL parameters, if given
        if self._ebl_model is not None:
            pars += self._ebl_model.parameters
        # Upturn parameters, if given
        if self._upturn_model is not None:
            pars += self._upturn_model.parameters
        # Bias parameter, if given
        if self.bias is not None:
            pars += Parameters([self.bias])

        # Return all existing parameters
        return pars
    
    @property
    def pivot_energy(self):
        # Ensure pivot / decorrelation energy computed on intrinsic model
        return self._intrinsic_model.pivot_energy

    def evaluate(self, energy, **kwargs):

        # Get bias parameter from keyword arguments
        bias = kwargs.pop("bias")
        # Rescale energy with bias (no rescaling if zero)
        energy_re = energy * (1.0 + bias)

        # Get intrinsic model parameters and evaluate it
        kwargs_int = {}
        for name in self._intrinsic_model.parameters.names:
            kwargs_int[name] = kwargs.pop(name)
        eval_int = self._intrinsic_model.evaluate(energy_re, **kwargs_int)

        # Get EBL parameters and evaluate EBL model, if given
        if self._ebl_model is not None:
            kwargs_ebl = {}
            for name in self._ebl_model.parameters.names:
                kwargs_ebl[name] = kwargs.pop(name)
            eval_ebl = self._ebl_model.evaluate(energy_re, **kwargs_ebl)
        else:
            eval_ebl = 1.0

        # Get upturn parameters and evaluate, if given
        if self._upturn_model is not None:
            kwargs_upt = {}
            for name in self._upturn_model.parameters.names:
                kwargs_upt[name] = kwargs.pop(name)
            eval_upt = self._upturn_model.evaluate(energy_re, **kwargs_upt)
        else:
            eval_upt = 1.0

        # Final evaluation result
        return eval_int * eval_ebl * eval_upt
    
    # Override evaluation of error to ensure keyword arguments (for error plotting)
    def evaluate_error(self, energy, n_samples = 3500, random_state = 42, samples = None):

        m = self.copy()
        pars_names = [p.name for p in m.parameters]
        def fct(*args):
            kwargs = dict(zip(pars_names, args))
            return m.evaluate(energy, **kwargs)
        
        propagated_samples = self._samples(
            fct,
            n_samples = len(pars_names) * n_samples,
            random_state = random_state,
            samples = samples, )

        return self._get_errors(propagated_samples)
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize model from dictionary"""

        # Reconstruct intrinsic spectral model
        intrinsic_data = data["spectral"].pop('intrinsic_model')["spectral"]
        intrinsic_tag = intrinsic_data.get('type')
        intrinsic_model_cls = SPECTRAL_MODEL_REGISTRY.get_cls(intrinsic_tag)
        intrinsic_model = intrinsic_model_cls.from_dict(intrinsic_data)

        # Reconstruct EBL model
        ebl_data = data["spectral"].pop('ebl_model')
        ebl_tag = ebl_data["spectral"].get('type')
        ebl_model_cls = SPECTRAL_MODEL_REGISTRY.get_cls(ebl_tag)
        ebl_model = ebl_model_cls.from_dict(ebl_data)

        # Load parameters of biased composite model
        biascomp_data = data["spectral"]["parameters"]

        biascomp_params = Parameters.from_dict(   
           data = biascomp_data, )

        # Create model instance
        model = cls(
            intrinsic_model = intrinsic_model,
            ebl_model = ebl_model,
            bias = biascomp_params["bias"].value, )
        
        # Upadate bias error
        model.parameters["bias"].error = biascomp_params["bias"].error

        # Return final version of model with all parameters
        return model

    def to_dict(self, full_output = False):
        """Serialize model to dictionary with all components"""

        # Call original function to generate dictionary
        data = super().to_dict(full_output = full_output)

        # Add intrinsic model information
        data["spectral"]["intrinsic_model"] = self._intrinsic_model.to_dict(full_output = full_output)
        # Add EBL model information
        data["spectral"]['ebl_model'] = self._ebl_model.to_dict(full_output = full_output)
        # Add upturn model, if given
        if self._upturn_model is not None:
            data["spectral"]["upturn_model"] = self._upturn_model.to_dict(full_output = full_output)

        return data


# Add new custom models to registry for serialization
SPECTRAL_MODEL_REGISTRY.append(EBLTableSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(UpturnSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(CompositeSpectralModel)
