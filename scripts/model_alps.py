# ...

import  os
import  argparse

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    pathlib                     import  Path

from    scipy.optimize              import  curve_fit

from    gammaALPs.core              import  Source, ALP, ModuleList
from    gammaALPs.base              import  environs, transfer

from    alpsup.utils                import  get_source_info, parse_kwargs, get_source_list
from    alpsup.paths                import  get_results_dir


# Define default grid for ALP masses [neV]
malp_list = [1e-3]
# Define default grid for ALP couplings [10^-11 GeV^-1]
galp_list = np.logspace(-2, +1, 30)


if __name__ == "__main__":

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run HESS analysis for a source using GammaPy")

    parser.add_argument("--source", choices = [*get_source_list(), "ALL"], help = "Source name. If 'ALL', run for all sources")    
    
    # Simulation parameters
    parser.add_argument("--nsim", default = 100, type = int, help = "Number of simulations to perform. Default: 100")
    parser.add_argument("--ebl", default = "dominguez", type = str, help = "EBL absorption model to use (loaded from EBLTable). Default: dominguez")
    parser.add_argument("--seed", default = 42, type = int, help = "Random seed")
    # ALP parameters
    parser.add_argument("--malp", default = 1e-3, nargs = "+", type = float, help = "ALP mass in neV. Default: 1e-3")
    parser.add_argument("--galp", default = galp_list, nargs = "+", type = float, help = "ALP couplings in 10^-11 GeV-1")

    # IGMF/GMF parameters
    parser.add_argument("--b0", default = 1e-3, type = float, help = "IGMF magnetic field strength. Default: 1e-3")
    parser.add_argument("--gmf", default = "jansson12", type = str, help = "GMF model to use. Default: jansson12")
    parser.add_argument("--n0", default = 1e-07, type = float, help = "IGMF n_0 parameter")
    parser.add_argument("--L0", default = 1e+04, type = float, help = "IGMF L_0 parameter")

    parser.add_argument("--plot-iter", default = -1, type = int, help = "Which iteration to plot. Default: -1 (None)")
    parser.add_argument("--plot-galp", default = 0,  type = int, help = "Index of coupling to plot, if --plot-iter. If -1, plot all. Default: 0 (First coupling)")

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
  
    # Loop over all sources
    for target in sources:
        print(f"### ALP Simulation for {target} ###")

        # Get info of source
        target_4FGL, target_position, target_redshift = get_source_info(target)

        # Define output directories saved to "alps" subfolder of target
        dir_aout = get_results_dir(target, output = "alps")
        # Create directories if not found
        os.makedirs(name = dir_aout, exist_ok = True)

        # Set plotting colors for each coupling
        cmap = plt.get_cmap("tab20b")
        colors = []
        for i in range( len( args.galp ) ):
            colors.append( cmap( i / (len(args.galp)) ) )

        # Set up plotting
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        # Set up simulation

        # Define source for gammaALPs
        src = Source(z = target_redshift, 
                    ra = target_position.ra.value, dec = target_position.dec.value, )

        # Initial parameters setup
        
        # ALP mass [neV]
        m_alp = args.malp
        # Magnetic field
        B0 = args.b0
        # EBL Model
        ebl_model = args.ebl

        # Energy range
        EGeV = np.logspace(kwargs.get('emin', 1.0), kwargs.get('emax', 4.5), kwargs.get('enum', 200))
        pin  = np.diag((1., 1., 0.)) * 0.5

        # Loop over all given ALP couplings [GeV-1]
        for j, g_alp in enumerate(args.galp):

            print(f"*** Running simulation for m_a = {m_alp} [neV], g_ag = {g_alp} [GeV-1], and EBL {ebl_model} ***")

            # Initialize ALP parameters
            ml = ModuleList(ALP(m = m_alp, g = g_alp), src, pin = pin, EGeV = EGeV, seed = args.seed)

            # Define propagation simulation

            # IGMF - Photon -> ALP at Extragalactic
            ml.add_propagation(
                environ = "IGMF",
                order = 0,
                nsim = args.nsim,
                B0 = B0,
                n0 = 1e-07,
                L0 = 1e04,
                ebl_model = ebl_model, )
            
            # GMF - ALP -> Photon at Milky Way
            ml.add_propagation(
                environ = "GMF",
                order = 1,
                model = "jansson12", )
            
            # Define optical depth from EBL absorption
            tau = ml.modules["IGMFCell"].t.opt_depth(ml.source.z, ml.EGeV / 1e3)

            # Run ALPs Case
            px, py, pa = ml.run()

            # Create Pgg array
            pgg = px + py

            # Define function for curve fitting - smooth break
            def f_inner(E, E_break, delta_gamma, beta):
                    return (1 + (E/E_break)**(delta_gamma / beta))**(-beta)
            def f_curve(E, E_break, delta_gamma, beta, N):
                    return N * f_inner(E, E_break, delta_gamma, beta)

            # Perform a fit on the Pgg vs E curve
            E_b_min = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 1)
            # E_b_max = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 5)
            E_b_max = 31.6 * 1e3

            p0 = [ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 2), -2, -1, 1]
            f_in = f_inner(ml.EGeV[0], *p0[:-1])

            # Define initial parameters
            p0[-1] = 1 / f_in
            # Define bounds on parameters
            bounds = np.array([[E_b_min, E_b_max], [-5, 0], [-1.001, -0.999], [p0[-1]/10, p0[-1]*10]])
            bounds = np.array([[E_b_min, E_b_max], [-5, 0], [-1.001, -0.999], [p0[-1] + 0.001, p0[-1] - 0.001]])

            # Define energy mask for fit
            E_max_fit = ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 10)
            # E_max_fit = 31.6 * 1e3

            # Define mask below E(tau=6)
            mask = ml.EGeV < E_max_fit

            # Run simulations

            # Initialize arrays for results
            popt_list = []
            pcov_list = []
            chisq_list = []

            for i, p in enumerate(pgg):
                
                N = p[0] * np.exp(tau[0])
                bounds[-1] = [0.5, 1.]
                bounds[-1] = [N*0.99, N*1.01]
                p0[-1] = N

                popt, pcov = curve_fit(
                    f = f_curve,
                    xdata = ml.EGeV[mask],
                    ydata = p[mask] * np.exp(tau[mask]),
                    p0 = p0,
                    bounds = bounds.T, 
                    # Increase maximum iterations
                    maxfev = 5000, )

                '''
                # Attempt fit with retry
                attempts = 10
                success  = False

                # Loop over attempts
                for attempt in range(attempts):

                    # Define decreasing energy mask over attempts
                    # mask = ml.EGeV < ml.modules["IGMFCell"].t.opt_depth_inverse(ml.source.z, 20 - attempt)

                    # Run the fit
                    popt, pcov = curve_fit(
                        f = f_curve,
                        xdata = ml.EGeV[mask],
                        ydata = p[mask] * np.exp(tau[mask]),
                        p0 = p0,
                        bounds = bounds.T, )
                    
                    # Check convergence of Ebreak - lower limit
                    if np.round( popt[0], 3) == np.round( E_b_min, 3 ):
                        print(f"Fit failed convergence test! Ebreak {popt[0]:.3f} =  {E_b_min:.3f} (Ebreak_min)")
                        success = False
                        # Retry
                        continue
                    # Check convergence of Ebreak - upper limit
                    if np.round( popt[0], 3 ) == np.round( E_b_max, 3 ):
                        print(f"Fit failed convergence test! Ebreak {popt[0]:.3f} = {E_b_max:.3f} (Ebreak_max)")
                        success = False
                        # Retry
                        continue

                    # Otherwise, fit successful
                    success = True
                    break

                # If fit could not converge
                if not success:
                    print("Fit failed to converge after all attempts!")
                    # Set all parameters to zero (won't contribute)
                    popt = [0, 0, 0, 0]
                else:
                    # Otherwise, fit successful
                    success = True
                '''
                
                # Append to array
                popt_list.append(popt)
                pcov_list.append(pcov)

                # Print best fit models
                print(f"Best fit parameters (Iter. {i+1}):\n- E_brk = {popt_list[i][0] * 1e-3:.3f} TeV\n- DGamma = {popt_list[i][1]:.3f}")
 
                # Calculate chi square
                # chi_sq = np.sum((f_curve(ml.EGeV[mask], *popt) - p[mask]*np.exp(tau[mask]))**2)
                # dof = ml.EGeV.size - 4
                # chisq_list.append(chi_sq)

                # if chi_sq / dof > 2:
                #         print(f"chi_sq/dof too large! i = {i+1}, chi_sq = {chi_sq}, chi_sq/dof = {chi_sq / dof}")

                # Plot iteration if required
                if i == args.plot_iter:
                    # If given specific coupling to plot
                    if args.plot_galp != -1 and g_alp == args.galp[args.plot_galp]:
                        ax1.loglog(ml.EGeV, p * np.exp(tau), color = colors[j], label = "Simulation (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])")
                        ax1.loglog(ml.EGeV[mask], f_curve(ml.EGeV[mask], *popt), color = "black", ls="--", label = "Best fit model (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])" + "\n" + r"$E_{brk}$" + f" = {popt_list[i][0] * 1e-3:.3f} TeV" + "\n" + r"$\Delta\Gamma$" + f"= {popt_list[i][1]:.3f}")
                        # Plot also photon survival probability
                        ax2.loglog(ml.EGeV, p, color = colors[j], label = "ALP Simulation (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])")
                    # Otherwise, plot all couplings
                    if args.plot_galp == -1:
                        ax1.loglog(ml.EGeV, p * np.exp(tau), color = colors[j], label = "Simulation (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])")
                        ax1.loglog(ml.EGeV[mask], f_curve(ml.EGeV[mask], *popt), color = colors[j], ls="--", label = "Best fit model (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])")
                        # Plot also photon survival probability
                        ax2.loglog(ml.EGeV, p, color = colors[j], label = "ALP Simulation (" + r"$g_{a\gamma} = $" + f"{g_alp}" + r" [GeV$^{-1}$])")

            # Save all files as numpy objects
            np.save(file = dir_aout.joinpath(f"popt_{ebl_model}_m{m_alp}_g{g_alp}_B0{B0}.npy"), arr = popt_list)
            np.save(file = dir_aout.joinpath(f"pcov_{ebl_model}_m{m_alp}_g{g_alp}_B0{B0}.npy"), arr = pcov_list)
            np.save(file = dir_aout.joinpath(f"chisq_{ebl_model}_m{m_alp}_g{g_alp}_B0{B0}.npy"), arr = chisq_list)

        # If plot iter not none, generate plot
        if args.plot_iter != -1:

            # Plot EBL just once (independent on ALP parameters!)
            ax2.loglog(ml.EGeV, np.exp(-tau), linestyle = "--", color = "black", label = f"EBL Absorption ({ebl_model})")

            # Save simulated upturn plot

            ax1.set_xlim(kwargs.get('xmin', 10.0), kwargs.get('xmax', 1e6))
            ax1.set_ylim(kwargs.get('ymin', 0.10), kwargs.get('ymax', 2e5))

            ax1.set_title(f"{target} ALPs Simulated Upturn")
            ax1.set_xlabel("Energy [GeV]")
            ax1.set_ylabel("De-Absorbed Photon Survival Probability " + r"$P_{\gamma\gamma}\,e^{\tau}$")

            # ax1.legend(loc = 'center left', bbox_to_anchor = (1.015, 0.5))
            ax1.legend(loc = 'upper left')
            fig1.savefig(dir_aout.joinpath(f"alps_{ebl_model}_m{m_alp}_g{args.galp}_B0{B0}.pdf"), bbox_inches = "tight")
            
            # Save photon survival probability plot
            ax2.set_xlim(1e1, 3e4)
            ax2.set_ylim(1e-6, 1e1)
            ax2.set_xlabel("Energy [GeV]")
            ax2.set_ylabel("Photon Survival Probability " + r"$P_{\gamma\gamma}$")
            ax2.set_title(f"{target} Photon Survival Probability")
            
            ax2.legend()
            fig2.savefig(dir_aout.joinpath(f"psp_{ebl_model}_m{m_alp}_g{g_alp}_B0{B0}.pdf"), bbox_inches = "tight")
