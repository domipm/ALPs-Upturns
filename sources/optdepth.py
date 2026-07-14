# SCRIPT FOR COMPUTING THE ENERGY AT WHICH THE OPTICAL DEPTH REACHES A CERTAIN VALUE FOR A GIVEN SOURCE / REDSHIFT


import  yaml
import  argparse

import  numpy                       as      np
import  matplotlib.pyplot           as      plt
from    matplotlib.lines            import  Line2D

from    astropy                     import  units    as  u

from    ebltable.ebl_from_model     import  EBL
from    ebltable.tau_from_model     import  OptDepth

import sys
sys.path.append("../scripts")
from utils import get_source_info


# Clean EBL name labels with reference
EBL_LABELS = {"dominguez": "Domínguez et al. (2011)", "finke2022": "Finke et al. (2022)", "franceschini": "Franceschini et al. (2008)", "saldana-lopez": "Saldana-López et al. (2021)"}


if __name__ == "__main__":


    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run optical depth analysis for a source.")
    parser.add_argument("--sources", default = "ALL", nargs = "+", help = "Source name (e.g. 1ES0347-121, or ALL). If ALL, plots optical depth for all sources in file.")
    parser.add_argument("--ebl-models", default = ["dominguez"], nargs = "+", help = "EBL models to consider.")
    parser.add_argument("--suffix", default = None, type = str, help = "Suffix to add to plots for identification. Default: None")
    args = parser.parse_args()

    # Load sources file
    with open('./sources.yaml', 'r') as f:
        # Load the yaml file
        data = yaml.full_load(f)

    # Define the target
    targets = {}

    # Initialize plot
    fig, ax1 = plt.subplots()
    ax1.set_xlim(1e-1, 1e2)
    ax1.set_xlabel("Energy [TeV]")
    ax1.set_ylabel(r"Optical Depth $\tau(E, z)$")

    # Load name and redshift of given target sources
    if "ALL" in args.sources:

        # Load all sources from file
        for target in list( data["sources"].keys() ):
            # Append to dictionary pairs (name, redshift)
            targets[target] = data["sources"][target]["z"]

    else:

        # Load only given sources
        for target in list( args.sources ):
            # Append to dictionary pairs (name, redshift)
            targets[target] = data["sources"][target]["z"]


    # Energy values to consider
    energy = np.logspace(-1, 2, 50)

    # Dictionary for results
    optdepth = dict()

    # Keep track of the colors used for each model for the first iter
    colors = []

    # Loop over given EBL models
    for k, ebl_model in enumerate(args.ebl_models):

        print(f"EBL Model: {ebl_model}")
        optdepth[ebl_model] = {}

        # Read the model from EBL Table
        tau = OptDepth.readmodel(model = ebl_model)

        # Loop over sources
        for j, target in enumerate(targets):

            # Skip certain targets!
            if target == "PKS0625-354" or target == "GRB180720B":
                continue

            print(f"Source: {target}")
            
            # Compute tau values at different energies for the target's redshift
            tau_values = np.array( [ tau.opt_depth(z = targets[target], ETeV = e) for e in energy ] )

            # Find value where tau = 1
            tau_limit_1 = np.interp(1, tau_values, energy)
            print("E(tau = 1) = {:.4f} TeV".format( tau_limit_1 ))
            # Find value where tau = 2
            tau_limit_2 = np.interp(2, tau_values, energy)
            print("E(tau = 2) = {:.4f} TeV".format( tau_limit_2 ))

            # Update dictionary
            optdepth[ebl_model].update({target: [float(tau_limit_1), float(tau_limit_2)]})

            # Label only dominguez ebl model sources
            if ebl_model == "dominguez":
                target_4FGL, target_position, target_redshift = get_source_info(target)
                label_target = target + f" (z = {target_redshift})"
            else:
                label_target = None

            # Define alpha values (dominguez opaque, rest reducing in opacity)
            alpha_target = ( len(args.ebl_models) - k ) / len(args.ebl_models)

            # Plot optical depth curve
            if k == 0:
                odc = ax1.loglog(
                    energy,
                    tau_values,
                    # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                    alpha = alpha_target,
                    label = label_target,
                    linewidth = alpha_target * 2, )
            else:
                odc = ax1.loglog(
                energy,
                tau_values,
                # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                alpha = alpha_target,
                label = label_target,
                linewidth = alpha_target * 2,
                color = colors[j] )
            
            # Append color for each target
            if k == 0:
                colors.append(odc[0].get_color()) 
            
            # Plot vertical line where tau = 1 and 2
            if ebl_model == "dominguez":
                ax1.vlines(x = tau_limit_1, ymin = 1e-3, ymax = 1,
                        color = odc[0].get_color(), alpha = 0.25,
                        linewidth = 1, linestyle = "dotted")
                ax1.vlines(x = tau_limit_2, ymin = 1e-3, ymax = 2,
                        color = odc[0].get_color(), alpha = 0.5,
                        linewidth = 1, linestyle = (0, (5, 1)))
            
    # Plot horizontal lines where tau = 1 and 2
    ax1.hlines(y = [1, 2], xmin = 1e-1, xmax = 1e2, color = "black", 
              linestyle = "--", alpha = 0.5, linewidth = 1)

    # Save optical depths and energies to yaml file
    yaml.safe_dump(data = optdepth, stream = open("optdepth.yaml", "w"))

    # Create custom handles with alpha values
    ebl_handles = []
    for i, ebl in enumerate(args.ebl_models):
        alpha_value = (len(args.ebl_models) - i) / len(args.ebl_models)
        handle = Line2D( [0], [0], color = 'black', alpha = alpha_value, linewidth = alpha_value * 2,
                    label = EBL_LABELS.get(ebl, None))
        ebl_handles.append(handle)

    # Get the first legend (your existing one)
    legend1 = plt.legend(loc='center left', bbox_to_anchor = (1, 0.5),
                         title = "Sources")

    # Add the first legend back to the plot
    plt.gca().add_artist(legend1)

    # Create second legend
    legend2 = plt.legend(handles = ebl_handles, loc = 'upper left', 
                        title = 'EBL Models')

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Save figure
    plt.title("Optical Depth")
    plt.savefig(f"optdepth_{args.suffix}.pdf", bbox_inches = "tight")
    plt.close()


    # Plot also the EBL attenuation we obtain


    # Initialize plot
    fig, ax2 = plt.subplots()
    ax2.set_xlim(1e-1, 1e2)
    ax2.set_xlabel("Energy [TeV]")
    ax2.set_ylabel(r"Attenuation $e^{-\tau(E, z)}$")

    # Loop over given EBL models
    for k, ebl_model in enumerate(args.ebl_models):

        print(f"EBL Model: {ebl_model}")
        optdepth[ebl_model] = {}

        # Read the model from EBL Table
        tau = OptDepth.readmodel(model = ebl_model)

        # Loop over sources
        for j, target in enumerate(targets):

            # Skip certain targets!
            if target == "PKS0625-354" or target == "GRB180720B":
                continue

            print(f"Source: {target}")
            
            # Compute tau values at different energies for the target's redshift
            tau_values = np.array( [ tau.opt_depth(z = targets[target], ETeV = e) for e in energy ] )

            # Find value where tau = 1
            tau_limit_1 = np.interp(1, tau_values, energy)
            print("E(tau = 1) = {:.4f} TeV".format( tau_limit_1 ))
            # Find value where tau = 2
            tau_limit_2 = np.interp(2, tau_values, energy)
            print("E(tau = 2) = {:.4f} TeV".format( tau_limit_2 ))

            # Update dictionary
            optdepth[ebl_model].update({target: [float(tau_limit_1), float(tau_limit_2)]})

            # Label only dominguez ebl model sources
            if ebl_model == "dominguez":
                target_4FGL, target_position, target_redshift = get_source_info(target)
                label_target = target + f" (z = {target_redshift})"
            else:
                label_target = None

            # Define alpha values (dominguez opaque, rest reducing in opacity)
            alpha_target = ( len(args.ebl_models) - k ) / len(args.ebl_models)

            # Plot optical depth curve
            if k == 0:
                odc = ax2.loglog(
                    energy,
                    np.exp(-tau_values),
                    # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                    alpha = alpha_target,
                    label = label_target,
                    linewidth = alpha_target * 2, )
            else:
                odc = ax2.loglog(
                energy,
                np.exp(-tau_values),
                # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                alpha = alpha_target,
                label = label_target,
                linewidth = alpha_target * 2,
                color = colors[j] )
            
            # Append color for each target
            if k == 0:
                colors.append(odc[0].get_color()) 
            
    # Save optical depths and energies to yaml file
    yaml.safe_dump(data = optdepth, stream = open("optdepth.yaml", "w"))

    # Create custom handles with alpha values
    ebl_handles = []
    for i, ebl in enumerate(args.ebl_models):
        alpha_value = (len(args.ebl_models) - i) / len(args.ebl_models)
        handle = Line2D( [0], [0], color = 'black', alpha = alpha_value, linewidth = alpha_value * 2,
                    label = EBL_LABELS.get(ebl, None))
        ebl_handles.append(handle)

    # Get the first legend (your existing one)
    legend1 = plt.legend(loc='center left', bbox_to_anchor = (1, 0.5),
                         title = "Sources")

    # Add the first legend back to the plot
    plt.gca().add_artist(legend1)

    # Create second legend
    legend2 = plt.legend(handles = ebl_handles, loc = 'upper left', 
                        title = 'EBL Models')

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Set y-limits
    ax2.set_ylim(1e-10, 1e2)
    
    # Save figure
    plt.title("EBL Attenuation")
    plt.savefig("optdepth_ebl.pdf", bbox_inches = "tight")
    plt.close()


    # TODO: Create joint plot

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 10))
    for axis in ax:
        axis.set_box_aspect(1)

    # Loop over given EBL models
    for k, ebl_model in enumerate(args.ebl_models):

        optdepth[ebl_model] = {}
        # Read the model from EBL Table
        tau = OptDepth.readmodel(model = ebl_model)

        # Loop over sources
        for j, target in enumerate(targets):

            # Skip certain targets!
            if target == "PKS0625-354" or target == "GRB180720B":
                continue

            # Compute tau values at different energies for the target's redshift
            tau_values = np.array( [ tau.opt_depth(z = targets[target], ETeV = e) for e in energy ] )

            # Find value where tau = 1
            tau_limit_1 = np.interp(1, tau_values, energy)
            # Find value where tau = 2
            tau_limit_2 = np.interp(2, tau_values, energy)

            # Update dictionary
            optdepth[ebl_model].update({target: [float(tau_limit_1), float(tau_limit_2)]})

            # Label only dominguez ebl model sources
            if ebl_model == "dominguez":
                target_4FGL, target_position, target_redshift = get_source_info(target)
                label_target = target + f" (z = {target_redshift})"
            else:
                label_target = None

            # Define alpha values (dominguez opaque, rest reducing in opacity)
            alpha_target = ( len(args.ebl_models) - k ) / len(args.ebl_models)

            # Plot optical depth curve
            if k == 0:
                odc = ax[0].loglog(
                    energy,
                    tau_values,
                    # np.exp(-tau_values),
                    # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                    alpha = alpha_target,
                    label = label_target,
                    linewidth = alpha_target * 2, )
            else:
                odc = ax[0].loglog(
                energy,
                tau_values,
                # np.exp(-tau_values),
                # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                alpha = alpha_target,
                label = label_target,
                linewidth = alpha_target * 2,
                color = colors[j] )

            # Plot EBL curve
            if k == 0:
                odd = ax[1].loglog(
                    energy,
                    np.exp(-tau_values),
                    # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                    alpha = alpha_target,
                    label = label_target,
                    linewidth = alpha_target * 2, )
            else:
                odd = ax[1].loglog(
                energy,
                np.exp(-tau_values),
                # label = f"{target} [ E$(\\tau = 1)$ = {tau_limit_1:.4f} TeV, E$(\\tau = 2)$ = {tau_limit_2:.4f} TeV]", )
                alpha = alpha_target,
                label = label_target,
                linewidth = alpha_target * 2,
                color = colors[j] )
            
            # Append color for each target
            if k == 0:
                colors.append(odc[0].get_color()) 

    # Plot horizontal lines where tau = 1 and 2
    ax[0].hlines(y = [1], xmin = 1e-1, xmax = 1e2, color = "black", 
              linestyle = "--", alpha = 0.5, linewidth = 1) 
    # TODO: FIX THIS LABEL! 
    # ax[0].text(1, 0.5, r'$\tau(E, z) = 1$',
    #     transform = ax[0].get_xaxis_transform(),
    #     ha = 'center', va = 'bottom')

    ax[1].set_ylim(1e-10, 1e1)

    ax[0].set_title("Optical Depth")
    ax[0].set_xlabel(r"Energy [TeV]")
    ax[0].set_ylabel(r"Optical Depth $\tau(E, z)$")

    ax[1].set_title("EBL Absorption")
    ax[1].set_xlabel(r"Energy [TeV]")
    ax[1].set_ylabel(r"EBL Absorption $e^{-\tau(E, z)}$")


    # Create legend!
    handles_plot, labels_plot = ax[0].get_legend_handles_labels()

    # Create "header" entries
    header1 = Line2D([], [], linestyle='none', label='Sources')
    header2 = Line2D([], [], linestyle='none', label='EBL Models')

    # (Optional) spacer for visual separation
    spacer = Line2D([], [], linestyle='none', label='')

    # Combine handles
    handles = (
        [header1] +
        handles_plot +
        [spacer] + 
        [header2] +
        ebl_handles
    )

    labels = [h.get_label() for h in handles]

    # Generate legend
    legend = plt.legend(handles, labels,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig("optdepth_comb.pdf", bbox_inches = "tight")
            