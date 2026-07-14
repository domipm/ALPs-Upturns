import  os
import  glob
import  argparse

import  numpy                       as      np

from    pathlib                     import  Path
from    typing                      import  Optional, Tuple

from    scipy.special               import  gammaincinv

import  astropy.units               as      u
from    astropy.io                  import  fits, ascii
from    astropy.time                import  Time

import  matplotlib.pyplot           as      plt
from    matplotlib.patches          import  Ellipse
from    matplotlib.ticker           import  MaxNLocator

from    gammapy.modeling.models     import  Models, PowerLawSpectralModel
from    gammapy.estimators          import  FluxPoints

from    alpsup.utils                       import  get_fpul, get_source_list, parse_kwargs
from    alpsup.models                      import  BiasedCompoundSpectralModel


def _get_results_path(*parts: str) -> Path:
    """Construct path relative to RESULTS folder from environment variable,
    including `parts` subfolders."""
    return Path(os.environ['RESULTS']).joinpath(*parts)


def plot_sed_fermipy(target: str, bblock: str = "baseline", 
                     ax: Optional[plt.Axes] = None, save_plot: bool = True, **kwargs_plot) -> plt.Axes:
    """
    Plot SED obtained from FermiPy fit
    """

    # Set default plot styling
    kwargs_plot.setdefault("label", "Fermi-LAT")
    kwargs_plot.setdefault("color", "crimson")
    kwargs_plot.setdefault("marker", "o")
    kwargs_plot.setdefault("markersize", 3)
    kwargs_plot.setdefault("capsize", 2)
    kwargs_plot.setdefault("capthick", 1)

    # Load SED data from FITS file
    sed_file = glob.glob(
        str(_get_results_path(target, bblock, "fermi-out", "final_sed_*.fits")) )[0]
    
    # Define required values
    with fits.open(sed_file) as f:
        
        src_sed  = f[1].data
        src_flux = f[2].data

        # Model data (default FermiPy units - MeV)
        energy  = np.array(src_flux['energy']) * u.MeV
        dnde    = np.array(src_flux['dnde']) * u.Unit("1 / (MeV cm2 s)")
        dnde_lo = np.array(src_flux['dnde_lo']) * u.Unit("1 / (MeV cm2 s)")
        dnde_hi = np.array(src_flux['dnde_hi']) * u.Unit("1 / (MeV cm2 s)")

        # Flux point data (default FermiPy units - MeV)
        x_pl  = src_sed['e_ref'] * u.MeV
        x_err = np.array([
            src_sed['e_ref'] - src_sed['e_min'],
            src_sed['e_max'] - src_sed['e_ref'], ]) * u.MeV
        y_pl     = src_sed['e2dnde'] * u.Unit("MeV / (cm2 s)")
        y_err_pl = src_sed['e2dnde_err'] * u.Unit("MeV / (cm2 s)")
        y_pl_ul  = src_sed['e2dnde_ul'] * u.Unit("MeV / (cm2 s)")

        ts = src_sed['ts']

    # Process flux points and upper limits
    y_pl_final, y_err_pl_final, is_uplim = get_fpul(
        y_pl, y_pl_ul, y_err_pl, y_err_pl, ts )

    # Create axes if not given
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Also set units
        ax.xaxis.set_units(energy.unit)
        ax.yaxis.set_units((energy**2 * dnde).unit)

    # Plot flux points
    ax.errorbar(
        x_pl, y_pl_final,
        yerr = y_err_pl_final,
        xerr = x_err,
        uplims = is_uplim,
        linestyle = "",
        **kwargs_plot, )
    # Plot spectral model
    ax.loglog(energy, energy**2 * dnde, color = kwargs_plot["color"], linestyle = "--")
    # Plot uncertainty (pass values without units)
    ax.fill_between(energy.value, (energy**2 * dnde_lo).value, (energy**2 * dnde_hi).value,
                    facecolor = kwargs_plot["color"], alpha = 0.2, )

    # Recompute axis limits
    ax.relim()
    ax.autoscale_view()
    # Fix axis limits
    ax.set_autoscale_on(False)

    # Save plot if requested
    if save_plot:
        plt.ylabel(r"Energy [{}]".format(f"{energy.unit:unicode}"))
        plt.ylabel(r"$\text{E}^2$ $d\text{N}/d\text{E}$ " + r"[{}]".format(f"{(energy**2 * dnde).unit:unicode}"))
        plt.title(f"{target} FermiPy Fermi-LAT SED")
        plt.legend()
        plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/sed_fermipy_flat.pdf", bbox_inches = "tight")

    # Return axis object
    return ax


def plot_sed_gammapy(target: str, bblock: str = "baseline", inst: Optional[str] = None, ax: Optional[plt.Axes] = None,
                     plot_model: bool = True, plot_error: bool = True, plot_fluxp: bool = True,
                     model_ebounds: Tuple = None,
                     save_plot: bool = True, 
                     sed_type: str = "e2dnde", **kwargs_plot) -> plt.Axes:
    
    # Set default plot styling
    kwargs_plot.setdefault("markersize", 3)
    kwargs_plot.setdefault("capsize", 2)
    kwargs_plot.setdefault("capthick", 1)
    # Set default plot limits
    if inst == "hess":
        kwargs_plot.setdefault("ymin", 1e-16)
        kwargs_plot.setdefault("ymax", 1e-10)
    elif inst == "flat":
        kwargs_plot.setdefault("ymin", None)
        kwargs_plot.setdefault("ymax", None)
    else:
        kwargs_plot.setdefault("ymin", None)
        kwargs_plot.setdefault("ymax", None)

    # Set certain kwargs based on instrument
    if inst in ["hess", "hess_bias", "joint_hess", "joint_hess_bias"]:
        # kwargs_plot.setdefault("label", "H.E.S.S.")
        # kwargs_plot.setdefault("color", "navy")
        kwargs_plot.setdefault("label", "H.E.S.S.")
        kwargs_plot.setdefault("color", "crimson")
        kwargs_plot.setdefault("marker", "s")
    elif inst in ["flat", "flat_bias", "joint_flat"]:
        # kwargs_plot.setdefault("label", "Fermi-LAT")
        # kwargs_plot.setdefault("color", "crimson")
        kwargs_plot.setdefault("label", r"$\it{Fermi}$-LAT")
        kwargs_plot.setdefault("color", "navy")
        kwargs_plot.setdefault("marker", "o")
    # Otherwise, use default kwargs
    else:
        kwargs_plot.setdefault("label", "Flux Points")
        kwargs_plot.setdefault("color", "teal")
        kwargs_plot.setdefault("marker", ".")

    # Load data
    base_path = _get_results_path(target, bblock, "gamma-out")
    # Define name of models file
    if inst in ["joint_flat", "joint_hess"]:
        models_file = base_path / "joint_models.yaml"
    else:
        models_file = base_path / f"{inst}_models.yaml"

    # Load model and flux points
    model = Models.read(models_file)[target]

    # Define instruments
    instruments = {
        "joint": ["hess", "flat"],
        "joint_bias": ["hess_bias", "flat"],
        # "hess": "hess",
        "joint_hess": ["hess"],
        # "flat": "flat",
        "joint_flat": ["flat"],
        # "hess_bias": "hess_bias", 
    }.get(inst, [inst])

    # Loop over each instrument
    for instrument in instruments:

        # Load flux points for instrument
        fluxp = ascii.read(base_path / f"{instrument}_fluxp.ecsv")

        # Extract flux point data
        fp_data = {
            'eref': fluxp['e_ref'].quantity,
            'e2dnde': fluxp[sed_type].quantity,
            'e2dnde_ul': fluxp[sed_type + '_ul'].quantity,
            'xerr': np.array([
                (fluxp['e_ref'] - fluxp['e_min']).value,
                (fluxp['e_max'] - fluxp['e_ref']).value, ]) * fluxp['e_ref'].unit,
            'yerrn': fluxp['e2dnde_errn'].quantity,
            'yerrp': fluxp['e2dnde_errp'].quantity,
            'ts': fluxp['ts'], 
            'ebounds': [ np.min(fluxp['e_min']), np.max(fluxp['e_max']) ] * fluxp['e_ref'].unit, }
        
        # Set up axes if not present
        if ax is None:
            fig, ax = plt.subplots()
            # Make sure correct units on axes
            ax.xaxis.set_units(fp_data['eref'].unit)
            ax.yaxis.set_units(fp_data[sed_type].unit)
        # Set logarithmic scale
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Process flux points and upper limits
        e2dnde_val, e2dnde_err, is_ul = get_fpul(
            fp_data[sed_type], fp_data[sed_type + '_ul'], fp_data['yerrn'], fp_data['yerrp'], fp_data['ts'], )

        # Convert to axis units (important when dealing with different units!)
        eref = fp_data['eref'].to(ax.xaxis.get_units())
        e2dnde_val = e2dnde_val.to(ax.yaxis.get_units())
        xerr = fp_data['xerr'].to(ax.xaxis.get_units())
        e2dnde_err = e2dnde_err.to(ax.yaxis.get_units())

        # Plot flux points
        if plot_fluxp:
            ax.errorbar(
                x = eref, y = e2dnde_val,
                xerr = xerr, yerr = e2dnde_err,
                uplims = is_ul,
                linestyle = "",
                zorder = 3,
                color = kwargs_plot["color"],
                label = kwargs_plot["label"],
                marker = kwargs_plot["marker"],
                markersize = kwargs_plot["markersize"],
                capsize = kwargs_plot["capsize"],
                capthick = kwargs_plot["capthick"], )
        
    # Plot model if requested (true by default)
    if plot_model:
        # Set energy bounds
        if model_ebounds == None:
            ebounds = fp_data['ebounds']
        else:
            ebounds = model_ebounds
        # Plot model
        model.spectral_model.plot(
            ax = ax, 
            energy_bounds = ebounds,
            sed_type = sed_type, 
            zorder = 2,
            color = kwargs_plot["color"] if kwargs_plot.get("color_model", None) == None else kwargs_plot["color_model"],
            linestyle = "--", )
        # Plot uncertainty if requested (true by default)
        if plot_error:
            model.spectral_model.plot_error(
                ax = ax,
                energy_bounds = ebounds,
                sed_type = sed_type, 
                alpha = 0.2,
                zorder = 1,
                facecolor = kwargs_plot["color"] if kwargs_plot.get("color_model", None) == None else kwargs_plot["color_model"], )

    # For Fermi-LAT only plot, set x-axis range
    # if inst == "flat":
    #     ax.set_xlim(ebounds[0].value - 150, ebounds[-1].value + 150)

    # Save plots
    if save_plot:
        plt.xlabel(r"Energy [{}]".format(f"{eref.unit:unicode}"))
        if sed_type == "e2dnde":
            plt.ylabel(r"$\text{E}^2$ $d\text{N}/d\text{E}$ " + r"[{}]".format(f"{e2dnde_val.unit:unicode}"))
        if sed_type == "dnde":
            plt.ylabel(r"$d\text{N}/d\text{E}$ " + r"[{}]".format(f"{e2dnde_val.unit:unicode}"))
        # Set yaxis limits
        plt.ylim(kwargs_plot["ymin"], kwargs_plot["ymax"])
        plt.title(f"{target} GammaPy SED")
        plt.legend()
        plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/sed_gammapy_{inst}.pdf", bbox_inches = "tight")

    return ax


def plot_sed_compare(target: str, bblock: str = "baseline") -> None:
    """
    Plot comparison of Fermi-LAT fit data from FermiPy and GammaPy
    """

    # Plot FermiPy Fermi-LAT SED
    ax = plot_sed_fermipy(target, bblock, save_plot = False, label = "FermiPy", color = "teal")
    # Plot GammaPy Fermi-LAT SED
    plot_sed_gammapy(target, bblock, "flat", ax = ax, save_plot = False, label = "GammaPy", color = "navy")

    # Set titles and labels
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(r"$\text{E}^2d\text{N}/d\text{E}$ " + r"[$\text{MeV cm}^{-2}\text{ s}^{-1}$]")
    ax.set_title(f"{target} Fermi-LAT SED Comparison")
    # Save plot
    plt.legend()
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/sed_gammapy_compare_flat.pdf", bbox_inches = "tight")
    # Save also png copy
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/{target.lower()}_sed_gammapy_compare_flat.png", bbox_inches = "tight", dpi = 300)

    return


def plot_sed_combine(target: str, bblock: str = "baseline") -> None:
    """
    Plot combined (individual) Fermi-LAT + H.E.S.S. SED fits from GammaPy
    """

    # Initialize plotting
    fig, ax = plt.subplots()
    # Define units for axex
    ax.yaxis.set_units(u.Unit("TeV / (s cm2)"))
    ax.xaxis.set_units(u.Unit("TeV"))
    ax.set_xscale('log')
    ax.set_yscale('log') 

    # First, plot models for Fermi-LAT and HESS
    plot_sed_gammapy(target = target, bblock = bblock, inst = "flat", ax = ax, plot_model = True, plot_fluxp = False, save_plot = False)
    plot_sed_gammapy(target = target, bblock = bblock, inst = "hess", ax = ax, plot_model = True, plot_fluxp = False, save_plot = False)

    # Next, plot flux points
    # Plot gammapy sed, no save (always for Fermi-LAT, since no HESS comparison)
    plot_sed_gammapy(target = target, bblock = bblock, inst = "flat", ax = ax, plot_model = False, plot_fluxp = True, save_plot = False)
    # Plot gammapy sed, no save (always for Fermi-LAT, since no HESS comparison)
    plot_sed_gammapy(target = target, bblock = bblock, inst = "hess", ax = ax, plot_model = False, plot_fluxp = True, save_plot = False)

    # Set titles and labels
    plt.title("{} Combined SED".format(target))
    plt.xlabel("Energy [TeV]")
    plt.ylabel(r"$\text{E}^2 d\text{N}/d\text{E}$ [TeV cm$^{-2}$ s$^{-1}$]")
    # Save plot
    plt.legend()
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/sed_gammapy_combined.pdf", bbox_inches = "tight")

    return


def plot_sed_joint(target: str, bblock: str = "baseline", save_plot = True, ax = None, bias = False, **kwargs) -> None:
    """
    Plot joint Fermi-LAT + H.E.S.S. SED fit
    """

    # Initialize plotting
    if ax is None:
        fig, ax = plt.subplots()
        # Set limits on x-axis (based on energy range [1 GeV, 31.6 TeV])
        ax.set_xlim(kwargs.get("xmin", 1e-03 - 0.25 * 1e-03), kwargs.get("xmax", 3.16e+01))
        ax.set_ylim(kwargs.get("ymin", 1e-16), kwargs.get("ymax", 1e-10))
    # Define scale of axis
    ax.set_xscale('log')
    ax.set_yscale('log') 
    # Ensure units of axis are correct
    ax.xaxis.set_units(u.Unit("TeV"))
    ax.yaxis.set_units(u.Unit("TeV / (s cm2)"))

    # Get energy range from flux points of each instrument
    emin = ascii.read(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/joint_flat_fluxp.ecsv")['e_min'].quantity[0]
    emax = ascii.read(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/joint_hess_fluxp.ecsv")['e_max'].quantity[-1]

    # Plot joint model
    plot_sed_gammapy(target = target, bblock = bblock, inst = "joint", ax = ax, plot_model = True, plot_fluxp = False, save_plot = False, 
                     model_ebounds = [emin, emax], color = "black")
    # Plot gammapy sed, no save (always for Fermi-LAT, since no HESS comparison)
    plot_sed_gammapy(target = target, bblock = bblock, inst = "joint_flat", ax = ax, plot_model= False, save_plot = False)
    # Plot gammapy sed, no save (always for Fermi-LAT, since no HESS comparison)
    plot_sed_gammapy(target = target, bblock = bblock, inst = "joint_hess", ax = ax, plot_model = False, save_plot = False)

    # Save plot
    if save_plot == True:
        plt.title("{} Joint SED".format(target))
        plt.xlabel("Energy [TeV]")
        plt.ylabel(r"$\text{E}^2 d\text{N}/d\text{E}$ [TeV cm$^{-2}$ s$^{-1}$]")
        plt.legend()
        plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/sed_gammapy_joint.pdf", bbox_inches = "tight")
        return
    # Otherwise return axis
    else:
        return ax


def plot_sed_gammapy_bias(target: str, bblock: str, inst: str = "hess_bias", ax = None, save_plot: bool = False, **kwargs_plot):
    """
    Plot H.E.S.S. biased SED and comparison between the systematic and statistic uncertainties
    """

    # Plot biased model fit (without errors!) and flux points
    if ax == None:
        ax = plot_sed_gammapy(target, bblock, inst.split("_bias")[0], plot_error = False, save_plot = False, **kwargs_plot)
        # Set title and axes labels
        ax.set_title(f"{target} GammaPy H.E.S.S. Biased SED")
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel(r"E$^2d$N/$d$E [TeV s$^{-1}$ cm$^{-2}$]")
    # Unless axes given
    else:
        # Make sure units are correct
        ax.xaxis.set_units(u.TeV)
        ax.yaxis.set_units(u.Unit("TeV s-1 cm-2"))
        plot_sed_gammapy(target, bblock, inst.split("_bias")[0], plot_error = False, save_plot = False, ax = ax, **kwargs_plot)

    # Load biased model
    model_bias = Models.read(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/{inst}_models.yaml")[target].spectral_model
    # Load unbiased model
    model_orig = Models.read(f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/{inst.split('_bias')[0]}_models.yaml")[target].spectral_model

    # Statistical error - obtained by error plotting unbiased spectral model
    model_orig.plot_error(
        ax = ax,
        energy_bounds = [0.1, 31.6] * u.TeV,
        sed_type = "e2dnde",
        linewidth = 0.5,
        edgecolor = "black",
        label = "Stat. error",
        facecolor = "tab:blue" if kwargs_plot.get("color_model", None) == None else kwargs_plot["color_model"],
        alpha = 0.3, )
    
    # Define new model with systematic errors
    model_bias_syst = model_bias
    for par in model_bias.parameters.free_parameters:
        # Ignore bias parameter
        if par.name != "bias":
            model_bias_syst.parameters[par.name].error = np.sqrt( np.abs( 
                model_bias.parameters[par.name].error**2 - model_orig.parameters[par.name].error**2 ) )

    # Plot systematic error 
    model_bias_syst.plot_error(
        ax = ax,
        energy_bounds = [0.1, 31.6] * u.TeV,
        sed_type = "e2dnde", 
        label = "Syst. error",
        edgecolor = "black",
        linewidth = 0.5,
        facecolor = "tab:blue" if kwargs_plot.get("color_model", None) == None else kwargs_plot["color_model"],
        alpha = 0.1, )


    return


def plot_xchecks(target: str, xcheck_dir: Optional[Path] = None, **kwargs) -> None:
    """
    Generate cross check plots (lightcurves and SEDs) comparing obtained results from baseline blocks
    with results from cross check using alternative methodology from data in corresponding subfolder
    """

    # Directory of cross check data (use default unless specified)
    if xcheck_dir == None:
        dir_xchecks = f"{os.environ['HESS_DATA']}/XCHECKs/{target}/"
    else:
        dir_xchecks = xcheck_dir

    # Plot baseline fit and get its axes
    if ax == None:
        ax = plot_sed_gammapy(target = target, bblock = "baseline", inst = "hess", save_plot = False, label = "HAP Analysis", color = "tab:orange")

    # Load cross check flux points data
    try:
        # Load fits file
        fp_xc = FluxPoints.read(dir_xchecks + f"spectrum_{target.lower()}.fits", format = "gadf-sed")
        fp_xc.plot(ax = ax, sed_type = "e2dnde", label = "PA Analysis")
    except:
        # If not found, load txt file and plot manually
        fp_xc = np.loadtxt(dir_xchecks + f"spectrum_PA_{target.lower()}.txt", skiprows = 1)
        
        # Upper limit mask
        mask = np.array( [not int(ul) for ul in fp_xc[:, -1]] )

        # Flux points
        ax.errorbar(
            x = fp_xc[:, 0][mask] * u.TeV,
            y = fp_xc[:, 3][mask] * u.Unit("TeV / (cm2 s)"),
            xerr = [fp_xc[:, 0][mask] - fp_xc[:, 1][mask], fp_xc[:, 2][mask] - fp_xc[:, 0][mask]] * u.TeV,
            yerr = fp_xc[:, 4][mask] * u.Unit("TeV / (cm2 s)"),
            marker = "s",
            markersize = 3,
            linestyle = "",
            color = "tab:blue",
            label = "PA Analysis" )
        # Upper limits
        ax.errorbar(
            x = fp_xc[:, 0][~mask],
            y = fp_xc[:, 4][~mask],
            yerr = [fp_xc[:, 4][~mask] - 0.2 * fp_xc[:, 4][~mask], np.full_like(fp_xc[:, 4][~mask], 0)],
            xerr = [fp_xc[:, 0][~mask] - fp_xc[:, 1][~mask], fp_xc[:, 2][~mask] - fp_xc[:, 0][~mask]],
            marker = "s",
            markersize = 3,
            linestyle = "",
            color = "tab:blue",
            uplims = np.full_like(fp_xc[:, 4][~mask], True), )

    # Load cross check butterfly data
    try:
        bf_xc = np.loadtxt(dir_xchecks + f"spectrum_PA_{target.lower()}_butterfly.txt", skiprows = 1)
        e_bin_butterfly_pa = bf_xc[:, 0] * u.Unit("TeV")
        butterfly_low = bf_xc[:, 2] * u.Unit("TeV / (cm2 s)")
        butterfly_high = bf_xc[:, 3] * u.Unit("TeV / (cm2 s)")

        # Plot model fit
        ax.plot(
            e_bin_butterfly_pa,
            bf_xc[:, 1] * u.Unit("TeV / (cm2 s)"),
            color = "tab:blue",
            linestyle = "--", )
        # Plot butterfly sed
        ax.fill_between(
            x = e_bin_butterfly_pa,
            y1 = butterfly_low,
            y2 = butterfly_high,
            facecolor = 'tab:blue',
            alpha = 0.2, )
    except:
        pass
    
    ax.set_xlabel("E [TeV]")
    ax.set_ylabel(r"$\text{E}^2 d\text{N}/d\text{E}$" + r" [TeV s$^{-1}$ cm$^{-2}$]")
    ax.set_title(f"{target} H.E.S.S. Cross Check")

    # Set axis limits, if given
    ax.set_xlim(kwargs.get('xmin', 0.1), kwargs.get('xmax', 31.6))
    ax.set_ylim(kwargs.get('ymin', 1e-16), kwargs.get('ymax', 5e-10))
    
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.legend()
    # Save figure to baseline plots folder
    plt.tight_layout()
    plt.savefig(f"{os.environ['RESULTS']}/XCHECKs/{target.lower()}_sed_xcheck.png", bbox_inches = "tight", dpi = 300)
    plt.close()

    # Plot light curve cross check

    # Plot baseline light curve
    lc = FluxPoints.read(f"{os.environ['RESULTS']}/{target}/baseline/gamma-out/lc_flux.fits",
                            format = "lightcurve", sed_type = "dnde", )
                            # reference_model = PowerLawSpectralModel())
    ax = lc.plot(sed_type = "dnde", time_format = "mjd", label = "HAP Analysis", color = "tab:orange")

    # Load light curve table from fits
    try:
        # Load fits file from gammapy
        lc_xc = FluxPoints.read(dir_xchecks + f"lc_{target.lower()}.fits",
                                format = "lightcurve", sed_type = "dnde", )
        # Plot loaded light curve
        lc_xc.plot(sed_type = "dnde", time_format = "mjd", label = "PA Analysis", color = "tab:blue")
    except:
        try:
            # Load manually from numpy txt file
            lc_xc = np.loadtxt(dir_xchecks + f"lightcurve_PA_{target.lower()}.txt", skiprows = 1)
            # Get upper limit mask
            mask = np.array( [bool(ul) for ul in lc_xc[:, -1]] )

            # Plot flux points
            ax.errorbar(
                x = lc_xc[:, 0][~mask],
                y = lc_xc[:, 2][~mask] * u.Unit("TeV-1 cm-2 s-1"),
                xerr = lc_xc[:, 1][~mask],
                yerr = lc_xc[:, 3][~mask] * u.Unit("TeV-1 cm-2 s-1"),
                marker = "o",
                markersize = 4,
                linestyle = "",
                label = "PA Analysis",
                color = "tab:blue", )
            # Plot upper limits
            ax.errorbar(
                x = lc_xc[:, 0][mask],
                y = lc_xc[:, 4][mask] * u.Unit("TeV-1 cm-2 s-1"),
                xerr = lc_xc[:, 1][mask],
                yerr = lc_xc[:, 3][mask] * u.Unit("TeV-1 cm-2 s-1"),
                # yerr = 0.2 * lc_xc[:, 4][mask] * u.Unit("TeV-1 cm-2 s-1"),
                uplims = np.full_like(lc_xc[:, 4][mask], True),
                marker = ".",
                linestyle = "",
                color = "tab:blue")
        except:
            print("No cross-check lightcurve file found!")
            pass
    
    ax.set_title(f"{target} H.E.S.S. Light Curve Cross Check")
    ax.set_ylabel(r"$d\text{N}/d\text{E}$" + r" [TeV$^{-1}$ s$^{-1}$ cm$^{-2}$]")

    # Set x ticks to integers, no rotation
    plt.xticks(ticks = plt.xticks()[0], labels = plt.xticks()[0].astype(int), 
               rotation = 0)
    # Center labels
    for label in ax.get_xticklabels(minor = False):
        label.set_horizontalalignment('center')

    plt.legend()

    # Save figure to baseline plots folder
    plt.tight_layout()
    plt.savefig(f"{os.environ['RESULTS']}/XCHECKs/{target.lower()}_lightcurve_xcheck.png", bbox_inches = "tight", dpi = 300)
    plt.close()

    return


def plot_lightcurve(target: str, bblock: str = "baseline", **kwargs) -> None:
    """
    Plot light curve including flux and spectral index evolution over time.
    """

    # Load flux point table
    lc_table = ascii.read(
        table = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/lc_flux_tab.ecsv", )
    # Load flux Bayesian block table
    lc_flux_bblock_table = ascii.read(
        table = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/lc_flux_bblock_tab.ecsv", )
    # Load index Bayesian block table
    lc_index_bblock_table = ascii.read(
        table = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/lc_index_bblock_tab.ecsv", )

    # Define figure
    fig, ax = plt.subplots(nrows = 2, ncols = 1, height_ratios = (2, 1), sharex = True)
    fig.subplots_adjust(hspace = 0)

    # Setup axes
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"Flux $d\text{N}/d\text{E}$ [TeV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
    ax[1].set_ylabel(r"Index $\Gamma$")
    ax[1].set_xlabel(r"Time [MJD]")
    ax[0].set_title(f"{target} Light Curve")

    # Set value of upper limit points to be correct ones!
    mask_ul = [ True if is_ul == True else False for is_ul in lc_table["is_ul"] ]
    mask_notul = [not elem for elem in mask_ul]
    lc_table["dnde"][mask_ul] = lc_table["dnde_ul"][mask_ul]
    # Set errors to zero
    lc_table["dnde_err"][mask_ul] = 0
    lc_table["dnde_errn"][mask_ul] = 0
    lc_table["dnde_errp"][mask_ul] = 0

    # Plot flux points - NON-UL
    ax[0].errorbar(
        x = lc_table["time_ref"][mask_notul].value.squeeze(),
        y = lc_table["dnde"][mask_notul].value.squeeze(),
        xerr = [lc_table["time_ref"][mask_notul] - lc_table["time_min"][mask_notul], lc_table["time_max"][mask_notul] - lc_table["time_ref"][mask_notul]],
        yerr = [lc_table["dnde_errn"][mask_notul].value.squeeze(), lc_table["dnde_errp"][mask_notul].value.squeeze()],
        linestyle = "", marker = ".", color = "crimson",
        capsize = 2, zorder = 4, )
    # Plot flux points - UL
    ax[0].errorbar(
        x = lc_table["time_ref"][mask_ul].value.squeeze(),
        y = lc_table["dnde"][mask_ul].value.squeeze(),
        xerr = [lc_table["time_ref"][mask_ul] - lc_table["time_min"][mask_ul], lc_table["time_max"][mask_ul] - lc_table["time_ref"][mask_ul]],
        yerr = np.vstack([lc_table["dnde_errn"][mask_ul].value.squeeze(), lc_table["dnde_errp"][mask_ul].value.squeeze()]),
        linestyle = "",
        marker = "v",
        capsize = 0,
        alpha = 0.25,
        color = "tomato",
        zorder = 3, )
    
    # Plot Bayesian block values for flux points as horizontal lines
    ax[0].hlines(
        y = lc_flux_bblock_table["dnde"],
        xmin = lc_flux_bblock_table["time_min"],
        xmax = lc_flux_bblock_table["time_max"],
        color = "black",
        linestyle = "-.",
        linewidth = 1,
        alpha = 1,
        zorder = 5, )
    
    # NOTE: Set errors to be 0.2 * value on index upper limits
    for k, row in enumerate(lc_table["index"]):
        if lc_table["index_is_ul"][k] == True:
            lc_table["index_err"][k] = 0 * lc_table["index"][k]

    # Plot index values for each flux point
    ax[1].errorbar(
        x = lc_table["time_ref"][mask_notul].value.squeeze(),
        y = lc_table["index"][mask_notul].value.squeeze(),
        xerr = [ lc_table["time_ref"][mask_notul] - lc_table["time_min"][mask_notul], lc_table["time_max"][mask_notul] - lc_table["time_ref"][mask_notul] ],
        yerr = lc_table["index_err"][mask_notul].value.squeeze(),
        marker = ".",
        linestyle = "", 
        capsize = 2,
        zorder = 4, 
        color = "navy", )
    # Plot index values for each flux point - UPLIMS
    ax[1].errorbar(
        x = lc_table["time_ref"][mask_ul].value.squeeze(),
        y = lc_table["index"][mask_ul].value.squeeze(),
        xerr = [ lc_table["time_ref"][mask_ul] - lc_table["time_min"][mask_ul], lc_table["time_max"][mask_ul] - lc_table["time_ref"][mask_ul] ],
        yerr = lc_table["index_err"][mask_ul].value.squeeze(),
        marker = "v",
        linestyle = "", 
        alpha = 0.25,
        capsize = 0,
        zorder = 3,
        color = "royalblue", )
        # uplims = lc_table["index_is_ul"].squeeze())
    
    # Plot Bayesian block values for index flux points as horizontal lines
    ax[1].hlines(
        y = lc_index_bblock_table["index"].value.squeeze(),
        xmin = lc_index_bblock_table["time_min"].value.squeeze(),
        xmax = lc_index_bblock_table["time_max"].value.squeeze(),
        color = "black",
        linestyle = "-.",
        linewidth = 1,
        alpha = 1,
        zorder = 5, )
    
    # Ensure x-axis integer ticks
    ax[0].set_xticklabels(ax[0].get_xticks().astype(int))
    
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/lightcurve_hess.pdf", bbox_inches = "tight")
    plt.close()

    return


def plot_ellipses(target: str, bblock: str = "baseline", print_labels: bool = False, **kwargs) -> None:
    """
    Plot best-fit index versus amplitude / normalization ellipses
    """

    # Load Bayesian block table
    lc_bblock_table = ascii.read(
        table = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/lc_flux_bblock_tab.ecsv", )
    # Extract Bayesian block edges for flux
    bblocks_edges_flux = [lc_bblock_table["time_min"][0]]
    bblocks_edges_flux.extend(lc_bblock_table["time_max"])

    fit_times_bb_flux = [Time(val = list(p), format = "mjd") for p in zip(bblocks_edges_flux, bblocks_edges_flux[1:])]

    # Load the covariance matrix of each block
    cov_bblock = np.load(file = f"{os.environ['RESULTS']}/{target}/{bblock}/gamma-out/lc_flux_bblock_cov.npy")
    # Compute best-fit ellipses (axes swapped, so x -> norm, y -> index)
    fig, ax = plt.subplots()

    # Generate list containing Bayesian block index for each flux point
    fp_block = np.digitize(lc_bblock_table["time_ref"].value, bblocks_edges_flux)
    # Choose a colormap with discrete colors
    cmap = plt.get_cmap("coolwarm", np.max(fp_block))
    # Convert block indices into colors
    colors = cmap(fp_block - 1)

    for k, result in enumerate(lc_bblock_table):

        # Get value of index and amplitude
        index_val = result["index"]
        index_err = result["index_err"]
        amplitude_val = result["amplitude"]
        amplitude_err = result["amplitude_err"]

        # Get sub-covariance matrix for index-amplitude for block
        cov = cov_bblock[k]

        # Compute eigenvalues of covariance matrix
        w, v = np.linalg.eig(cov)

        # Compute alpha angle to rotate ellipses (in degrees)
        # choosing the eigenvector corresponding to the largest eigenvalue
        vec = v[:, np.argmax(w)]
        alpha = np.rad2deg( np.arctan2( vec[1], vec[0] ) )

        # Iterate over confidence levels
        for ic, l in enumerate( [0.68, 0.95, 0.99] ):

            # Compute ellipse axes
            a = 2. * np.sqrt( gammaincinv(1, l) * w[0] )
            b = 2. * np.sqrt( gammaincinv(1, l) * w[1] )

            # Compute the ellipse (plot only if height and width are physically correct)
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 and amplitude_val > 0 and index_val > 0:
                ellipse = Ellipse(
                    # Center ellipse at point
                    xy = (amplitude_val,
                        index_val),
                    # Define axes
                    width = b, height = a,
                    # Define transparency
                    alpha = [0.75, 0.5, 0.25][ic],
                    # Rotate ellipse
                    angle = alpha,
                    # Use color for edges, otherwise facecolor
                    facecolor = colors[k], )
                # Plot the ellipse
                ax.add_patch(ellipse)

        # Plot index and normalization points
        ax.errorbar(
            x = amplitude_val,
            y = index_val,
            xerr = np.sqrt(cov[1, 1]),
            # xerr = amplitude_err,
            yerr = np.sqrt(cov[0, 0]),
            # yerr = index_err,
            # Label edges of Bayesian blocks
            label = "[{:.3f} - {:.3f}] MJD".format( 
                fit_times_bb_flux[k][0].mjd, fit_times_bb_flux[k][1].mjd ),
            color = colors[k],
            markerfacecolor = colors[k],
            markeredgecolor = 'black',
            ecolor = 'black',
            marker = ".", )
        
        # Print label for each point for identification
        if print_labels:
            ax.annotate(
                text = k + 1,
                xy = (amplitude_val + 0.05 * amplitude_val, index_val), )
        
    # Set axes
    ax.set_title(f"{target} Best-fit Ellipses")
    ax.set_xlabel(r"Normalization $N_0$ [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]")
    ax.set_ylabel(r"Index $\Gamma$")
    # Set logarithmic scale
    ax.set_xscale("log")

    # Also allow fontsize and lcols options
    # Move legend outside plot - right center
    if kwargs.get("loc", "default") == "default":
        ax.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = kwargs.get("fontsize", None), ncols = kwargs.get("ncols", 1) ) # Right center
    # If legend explicitly set to None location, remove it!
    elif kwargs.get("loc", "default") == "None" or kwargs.get("loc", "default") == None:
        pass
    # If position for legend given, move it there!
    else:
        ax.legend(loc = kwargs["loc"], fontsize = kwargs.get("fontsize", None), ncols = kwargs.get("ncols", 1))
    # ax.legend( loc = 'upper center', bbox_to_anchor = (0.5, -0.15) ) # Bottom center
    # Save figure
    plt.savefig(f"{os.environ['RESULTS']}/{target}/{bblock}/plots/lightcurve_hess_ellipses.pdf", bbox_inches = "tight")
    plt.close()

    return


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Generate individual plots for each target and time block")

    parser.add_argument("--source", required = True, choices = get_source_list(), help = "Source name")    
        
    parser.add_argument("--bblock", default = "baseline", 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")
    
    parser.add_argument("--plot", default = None, choices = ["xcheck", "lightcurve", "sed", "sed_bias", "sed_compare"], help = "Which plots to generate")

    parser.add_argument("--inst", choices = ["hess", "hess_bias", "fermi", "flat", "joint", "joint_bias"])

    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Parse keyword arguments if given
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Get target name
    target = args.source
    # Get bblock
    bblock = args.bblock

    # Plot HESS light curve and best-fit ellipses
    if args.plot == "lightcurve":
        plot_lightcurve(target, bblock = "baseline", **kwargs)
        plot_ellipses(target, bblock = "baseline", **kwargs)
    # Plot HESS cross checks using baseline analysis
    if args.plot == "xcheck":
        plot_xchecks(target, **kwargs)
    # Plot GammaPy SED
    if args.plot == "sed":
        if args.inst != "joint":
            plot_sed_gammapy(target, args.bblock, args.inst, save_plot = True)
        elif args.inst == "joint":
            plot_sed_joint(target, args.bblock)
    # Plot GammaPy HESS bias (+ Fermi-LAT) systematic errors
    if args.plot == "sed_bias":
        # Ensure using biased instrument
        if len(args.inst.split("_bias")) == 1:
            print("Biased SED plotting requires using biased instrument!")
            exit()
        else:
            plot_sed_gammapy_bias(target, args.bblock, "hess_bias", save_plot = True)
    # Plot FermiPy GammaPy Comparison
    if args.plot == "sed_compare":
        plot_sed_compare(target, args.bblock)
