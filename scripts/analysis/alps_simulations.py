# Run photon-ALP propagation simulations across sources and EBL models with GammaALPs

import  os
import  yaml
import  glob
import  argparse

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path
from    datetime                    import  datetime

from    scipy.optimize              import  curve_fit

from    gammaALPs.core              import  Source, ALP, ModuleList
from    gammaALPs.base              import  environs, transfer

from    alpsup.utils                import  get_source_info, parse_kwargs, get_source_list
from    alpsup.logs                 import  init_log
from    alpsup.paths                import  get_results_dir


# Define default mass for ALP [neV]
MALP = 1e-3
# Default B0 parameter
B0 = 0.001
# Default n0 parameter
N0 = 1e-07
# Default L0 parameter
L0 = 1e+04

# Define default grid for ALP couplings [10^-11 GeV^-1]
GALP_LIST = np.logspace(-2, +1, 30)


# Functions for curve fitting - smooth break
def f_inner(E, E_break, delta_gamma, beta):
        return (1 + (E/E_break)**(delta_gamma / beta))**(-beta)
def f_curve(E, E_break, delta_gamma, beta, N):
        return N * f_inner(E, E_break, delta_gamma, beta)


def plot_alp_fit(target, ebl_model, 
                 # NOTE: For now, treat m_alp, B0, and other params as fixed!
                 # m_alp, g_alp, b0, 
                 n_iter = 0, n_galp = 0):
    """
    Plot ALP simulation fit results, for a given target source, EBL model, ALP parameters, and iteration.
    """

    # TODO: Allow for multiple couplings to be plotted at the same time
    # (This might be much easier to just do separately, depending on the need)

    # Define output directory with ALP simulation result files
    dir_aout_ebl = get_results_dir(target, output = "alps") / ebl_model

    # Get g_alp corresponding to n_coupling
    sim_iter_files = sorted( glob.glob( str( dir_aout_ebl.resolve() / f"sim_iter{n_iter}_*" ) ) )
    sim_iter_files_sorted = sorted(
        sim_iter_files,
        key = lambda f: float(f.split("_g")[1].split("_B")[0]), )
    sim_iter_file = sim_iter_files_sorted[n_galp]

    # Get g_alp value
    g_alp = float(sim_iter_file.split("_g")[1].split("_")[0])

    # Load simulation results
    sim_data = np.load(sim_iter_file)

    # Load parameters of fit
    popt = np.load(dir_aout_ebl / f"sim_fit_{ sim_iter_file.split("sim_iter0_")[-1] }")["popt"][n_galp]

    # TODO: These plots can be made nicer-looking!
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

    # Plot photon survival probability
    ax[0].loglog(
        sim_data["EGeV"], sim_data["p"],
        label = "ALP Simulation (" + r"$g_{a\gamma} = $" + f"${g_alp}$" + r" [$\cdot 10^{-11}$ GeV$^{-1}$])" )

    # Plot simulated upturn and fit
    ax[1].loglog(
        sim_data["EGeV"], sim_data["p"] * sim_data["exp_tau"],
        label = "ALP Simulation (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [$\cdot 10^{-11}$ GeV$^{-1}$])")
    ax[1].loglog(sim_data["EGeV"], f_curve(sim_data["EGeV"], *popt), 
                 label = "Best-fit upturn model",
                 color = "black")

    ax[0].set_title("Simulated photon survival probability")
    ax[0].set_xlabel("E [GeV]")
    ax[0].set_ylabel("Photon survival probability")

    ax[1].set_title("Simulated Photon-ALP upturn")
    ax[1].set_xlabel("E [GeV]")
    ax[1].set_ylabel("PSP x e(+tau)")
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    ax[1].set_xlim(1e3, 3*1e4)
    ax[1].set_ylim(0.75, 10)

    ax[0].legend(fontsize = 8)
    ax[1].legend(fontsize = 8)

    plt.savefig(dir_aout_ebl / f"sim_plots_iter{n_iter}_galp{g_alp}.pdf", dpi = 300)
    plt.close()

    return


def save_alp_metadata(dir_aout_ebl, target, ebl_model, args):
    """
    Save metadata for the ALP simulation parameters.
    """

    # Define metadata dictionary with relevant parameters
    meta = {"target": target, 
            "date": str(datetime.now()),
            "ebl_model": ebl_model, 
            "nsim": int(args.nsim), 
            "seed": int(args.seed),
            "malp_neV": np.atleast_1d(args.malp).tolist(), 
            "galp_1e-11GeV-1": args.galp.tolist(),
            "b0": args.B0, 
            "gmf_model": args.gmf, 
            "n0": args.n0, 
            "L0": args.L0}

    # Save metadata dictionary as file
    with open(dir_aout_ebl / "sim_metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys = False)

    return


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run HESS analysis for a source using GammaPy",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices = [*get_source_list(), "ALL"], help = "Source name. If 'ALL', run for all sources.")    
    
    # Simulation parameters
    parser.add_argument("--nsim", default = 100, type = int, help = "Number of simulations to perform.")
    parser.add_argument("--ebl", choices = ["dominguez", "finke2022", "franceschini", "saldana-lopez", "ALL"], default = "dominguez", type = str, help = "EBL absorption model to use (loaded from EBLTable). If 'ALL', run for all EBL models.")
    parser.add_argument("--seed", default = 42, type = int, help = "Random seed")
    # ALP parameters
    parser.add_argument("--malp", default = MALP, type = float, help = "ALP mass in neV.")
    parser.add_argument("--galp", default = GALP_LIST, nargs = "+", type = float, help = "ALP couplings in 10^-11 GeV-1")

    # IGMF/GMF parameters
    parser.add_argument("--B0", default = B0, type = float, help = "IGMF magnetic field strength.")
    parser.add_argument("--gmf", default = "jansson12", type = str, help = "GMF model to use.")
    parser.add_argument("--n0", default = N0, type = float, help = "IGMF n_0 parameter.")
    parser.add_argument("--L0", default = L0, type = float, help = "IGMF L_0 parameter.")

    parser.add_argument("--save-iter", default = 0, type = int, help = "Save all results for given iteration.")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--plot-diag", action = "store_true", help = "Generate diagnostic plots for each simulation")
    parser.add_argument("--plot-iter", default = 0, type = int, help = "Which iteration to plot.")
    parser.add_argument("--plot-galp", default = 0,  type = int, help = "Index of coupling to plot, if --plot-iter.")

    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments if given
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # Run single target or all
    if target == "ALL" or target == None:
        sources = get_source_list()
    else:
        sources = [target]

    # Run plots only
    if args.plots_only:
        if target == "ALL" or args.ebl == "ALL":
            raise Exception("For plotting only, a single source / EBL model must be specified!")
        plot_alp_fit(target, args.ebl, n_iter = args.plot_iter, n_galp = args.plot_galp)
        exit()
  
    # Loop over all sources
    for target in sources:

        # Initialize logging, per target source
        log = init_log(target = target, fname = "alps_sim.log", bblock = "alps")
        log.info(f"ALP Simulation for {target}...")

        # Get info of source
        source = get_source_info(target)
        target_4FGL, target_position, target_redshift = source.name_4FGL, source.position, source.redshift

        # Define output directories saved to "alps" subfolder of target
        dir_aout = get_results_dir(target, output = "alps")
        # Create directories if not found
        os.makedirs(name = dir_aout, exist_ok = True)

        # ALP mass [neV]
        m_alp = args.malp
        # Magnetic field
        B0 = args.B0

        # Define EBL models to loop over
        if args.ebl == "ALL":
             ebls = ["dominguez", "finke2022", "franceschini", "saldana-lopez"]
        else:
             ebls = [args.ebl]
        # Loop over all EBL models (or just one)
        for ebl in ebls:

            log.info(f'EBL model: {ebl}')

            # Set up simulation

            # Define source for gammaALPs
            src = Source(z = target_redshift, 
                        ra = target_position.ra.value, dec = target_position.dec.value, )

            # EBL Model
            ebl_model = ebl

            # Define output directory and ensure it exists
            dir_aout_ebl = dir_aout.resolve() / f"{ebl_model}/"
            os.makedirs(dir_aout_ebl, exist_ok = True)
            # Save simulation metadata for reference
            save_alp_metadata(dir_aout_ebl, target, ebl_model, args)

            # Energy range
            EGeV = np.logspace(kwargs.get('emin', 1.0), kwargs.get('emax', 4.5), kwargs.get('enum', 200))
            pin  = np.diag( (1., 1., 0.) ) * 0.5

            # Loop over all given ALP couplings [GeV-1]
            for j, g_alp in enumerate(args.galp):

                log.info(f"Running simulation for m_a = {m_alp} [neV], g_ag = {g_alp} [GeV-1]")

                # Initialize ALP parameters
                ml = ModuleList(ALP(m = m_alp, g = g_alp), src, pin = pin, EGeV = EGeV, seed = args.seed)

                # Define propagation simulation

                # IGMF - Photon -> ALP at Extragalactic
                ml.add_propagation(
                    environ = "IGMF",
                    order = 0,
                    nsim = args.nsim,
                    B0 = args.B0,
                    n0 = args.n0,
                    L0 = args.L0,
                    ebl_model = ebl_model, )
                
                # GMF - ALP -> Photon at Milky Way
                ml.add_propagation(
                    environ = "GMF",
                    order = 1,
                    model = args.gmf, )
                
                # Define optical depth from EBL absorption
                tau = ml.modules["IGMFCell"].t.opt_depth(ml.source.z, ml.EGeV / 1e3)

                # Run ALPs Case
                px, py, pa = ml.run()

                # Create Pgg array
                pgg = px + py

                # Perform a fit on the Pgg vs E curve
                E_b_min = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 1)
                # E_b_max = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 5)
                E_b_max = 31.6 * 1e3

                p0 = [ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 2), -2, -1, 1]
                f_in = f_inner(ml.EGeV[0], *p0[:-1])

                # Define initial parameters
                p0[-1] = 1 / f_in
                # Define bounds on parameters
                bounds = np.array([[E_b_min, E_b_max], [-5, 0], [-1.001, -0.999], [0, 0]])

                # Define energy mask for fit (tau = 10)
                E_max_fit = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 10)

                # Define mask below E(tau=6)
                mask = ml.EGeV < E_max_fit

                # Run simulations

                # Initialize arrays for results
                popt_list = []
                pcov_list = []
                chisq_list = []

                for i, p in enumerate(pgg):
                    
                    N = p[0] * np.exp(tau[0])
                    p0[-1] = N
                    bounds[-1] = [N * 0.99, N * 1.01]

                    popt, pcov = curve_fit(
                        f = f_curve,
                        xdata = ml.EGeV[mask],
                        ydata = p[mask] * np.exp(tau[mask]),
                        p0 = p0,
                        bounds = bounds.T, 
                        # Maximum number of iterations
                        maxfev = 5000, )
                    
                    # Append to array
                    popt_list.append(popt)
                    pcov_list.append(pcov)

                    # Print best fit models
                    log.info(f"Best fit parameters (Iter. {i+1}):\n- E_brk = {popt_list[i][0] * 1e-3:.3f} TeV\n- DGamma = {popt_list[i][1]:.3f}")
    
                    # Calculate chi square of the fit
                    chi_sq = np.sum( (f_curve( ml.EGeV[mask], *popt ) - p[mask] * np.exp(tau[mask]) ) ** 2 )
                    dof = ml.EGeV.size - 4
                    # NOTE: Should be mask.sum() - len(popt) ?
                    chisq_list.append(chi_sq)
                    # Check if chi squared is too large
                    if chi_sq / dof > 2:
                            log.info(f"chi_sq/dof too large! i = {i+1}, chi_sq = {chi_sq}, chi_sq/dof = {chi_sq / dof}")

                    # Save simulation results to file
                    # NOTE: Save results of all couplings for only one iteration!
                    if i == args.save_iter:
                        log.info(f"Saving simulation results of iteration {args.save_iter} into files...")
                        np.savez(file = dir_aout_ebl / f"sim_iter{i}_m{m_alp:g}_g{g_alp:g}_B0{B0:g}.npz",
                                EGeV = ml.EGeV, p = p, exp_tau = np.exp(tau))
                        # TODO: Generate diagnostic plots

                log.info(f"Saving simulation fit results into files...")
                # Save as a single numpy file
                np.savez(file = dir_aout_ebl / f"sim_fit_m{m_alp:g}_g{g_alp:g}_B0{B0:g}.npz", 
                         popt = popt_list, pcov = pcov_list, chisq = chisq_list,)
                log.info(f"Results saved!")
