# Script to perform final interpolation step between ALP upturn best-fit model parameters obtained 
# from simulations with the C-stat grid obtained from upturn modeling of our data
# Generates ALP upturn likelihood curves, as well as comparison across given EBL models
# Requires ALP simulation and upturn modeling steps to be completed (with best-fit models and C-stat grid)


import  os
import  re
import  argparse

import  numpy               as      np
import  matplotlib.pyplot   as      plt
from    mpl_toolkits.axes_grid1.inset_locator import inset_axes
from    mpl_toolkits.axes_grid1.inset_locator import mark_inset

from    pathlib             import  Path

from    astropy.io          import  ascii
from    astropy.table       import  Table

from    scipy.interpolate   import  RectBivariateSpline

from    alpsup.utils    import  get_source_list, parse_kwargs
from    alpsup.paths    import  get_results_dir


# Define default ALP couplings values
GALP_DEFAULT  = np.logspace(-2, +1, 30)
# Define default pattern for block naming
BLOCK_PATTERN = re.compile(r"^block(\d+)")
# Define labels for each dataset
DATASET_LABEL = {"hess": "H.E.S.S.", "hess_bias": "H.E.S.S.",
                 "joint": r"$\it{Fermi}$-LAT + H.E.S.S.", "joint_bias": r"$\it{Fermi}$-LAT + H.E.S.S."}
# Define markers for each EBL
EBL_MARKERS = {"dominguez": "o", "finke2022": "s", "franceschini": "^", "saldana-lopez": "v"}
# Define labels for each EBL
EBL_LABELS = {"dominguez": "Domínguez et al. (2011)", "finke2022": "Finke et al. (2022)", "franceschini": "Franceschini et al. (2008)", "saldana-lopez": "Saldana-López et al. (2021)"}


# Utilities for astropy.table.Table objects
def _empty_table(galp_list: np.ndarray) -> Table:
    """Define empty astropy table object for each coupling value per target/block"""
    colnames = ["source_block"] + [f"g_ag = {g}" for g in galp_list]
    dtypes = [str] + [float] * len(galp_list)
    return Table(names=colnames, dtype=dtypes)
def _add_total_row(table: Table) -> None:
    """Append a row with the column-wise nansum (in-place)"""
    table.add_row( ["total_sum"] + [ float(np.nansum(table[col].data)) for col in table.colnames[1:] ] )
    return


def get_block_dict(ebl: str) -> dict[str, list[str]]:
    """
    Return dict{target: [block_folder_name, ...]} for all valid sources,
    filtered to blocks belonging to the given EBL model, sorted numerically by index.
    
    Parameters
    ----------
    results_dir (pathlib.Path)  : Directory of results
    ebl (str)                   : name of EBL model considered

    Returns
    -------
    all_blocks (dict)           : Dictionary of all available blocks for given EBL model per target
    """

    # Get list of all available targets
    targets = get_source_list()
    all_blocks: dict[str, list[str]] = {}
    # Loop over all sources
    for target in targets:
        # Check if folder exists
        target_path = get_results_dir(target)
        if not target_path.is_dir():
            print(f"Warning: {target} folder not found — skipping!")
            continue
        # Obtain valid folders sorted by index
        valid = sorted(
            (p.name for p in target_path.iterdir()
             if p.is_dir()
             and BLOCK_PATTERN.match(p.name)
             and BLOCK_PATTERN.match(p.name)),
            key = lambda name: int(BLOCK_PATTERN.match(name).group(1)), )
        # Add valid blocks to list
        if valid:
            all_blocks[target] = valid
    # Return list of all valid blocks found for given EBL model
    return all_blocks


def interp_alp_upturn(popt: np.ndarray, cstat: dict, percentile: int = 5) -> tuple[np.ndarray, float, float, float]:
    """
    Interpolate best-fit ALP upturn parameters onto upturn C-stat grid.

    Parameters
    ----------
    popt (np.ndarray)   : (N, 2) array of [ebreak, dgamma] best-fit values from ALP upturn simulations.
    cstat (dict)        : dictionary with keys X: ebreak, Y: dgamma, Z: C-stat grid.
    percentile (int)    : lower percentile for choosing ALP C-stat values (default: 5 ~ 95% confidence level).

    Returns
    -------
    ts         : interpolated C-stat values at each of the N simulation points
    c_noalp    : C-stat at the no-ALP point (top-right corner of grid)
    c_alp      : `percentile`-th percentile of ts
    cstat_min  : global minimum of the C-stat grid (best-fit upturn)
    """

    # Get ALP upturn best-fit parameters
    # (ignore N, beta parameters fixed)
    ebreak_alp = popt[:, 0]
    dgamma_alp = popt[:, 1]

    # Obtain C-stat grid from upturn modeling
    # (reverse dgamma so strictly increasing)
    dgamma_grid = cstat['X'][:, 0][::-1]
    ebreak_grid = cstat['Y'][0, :]
    # (also reverse x axis on grid!)
    cstat_grid  = cstat['Z'][::-1, :]

    # Define interpolation function
    # (use logarithm of energy for evaluation)
    spline = RectBivariateSpline(
        dgamma_grid, np.log10(ebreak_grid), cstat_grid,
        kx = 1, ky = 1, s = 0, )
    # Evaluate interpolation on ALP parameters
    # (converting GeV -> TeV for consistency)
    ts = spline.ev(dgamma_alp, np.log10(ebreak_alp) - 3)

    # Take percentile on the ALP C-stat values
    c_alp = np.percentile(ts, percentile)
    # Take no-ALP case from top right corner
    # (Ebreak ~ 31.6 TeV, DGamma ~ 0)
    c_noalp = cstat_grid[-1, -1]
    # Obtain best-fit point on grid
    c_min = cstat_grid.min()

    # Return interpolated values, C-stat for no-ALP and ALP cases, minimum of grid
    return ts, c_noalp, c_alp, c_min


def plot_interp_histogram(ts: np.ndarray, c_alp: float, c_noalp: float, c_min: float,
                          target: str, bblock: str, ebl: str, out_path: Path, ) -> None:
    """
    Plot diagnostic histogram of interpolated C-stat values for the target-block pair
    """

    # Initialize plots
    fig, ax = plt.subplots()
    # Plot hisogram
    ax.hist(ts, bins = 30, alpha = 0.7, edgecolor = "black")
    # Plot vertical line for C-stat ALP
    ax.axvline(c_alp, color = "red", linestyle = "--", linewidth = 2,
               label = rf"$C_{{\mathrm{{ALP,\,5%}}}} = {c_alp:.3f}$")
    # Plot vertical line for C-stat No-ALP
    ax.axvline(c_noalp, color = "orange", linestyle = "--", linewidth = 2,
               label = rf"$C_{{\mathrm{{No-ALP,\,5%}}}} = {c_noalp:.3f}$")
    # Plot vertical line for C-stat minimum
    ax.axvline(c_min, color = "green", linestyle = "--", linewidth = 2,
            label = rf"$C_{{\min}} = {c_min:.3f}$")
    # Set labels and titles
    ax.set_xlabel("C-stat")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{target} — {bblock} — ALP interpolation ({ebl})")
    # ax.legend()
    ax.grid(alpha = 0.3)
    # Save figure
    os.makedirs(get_results_dir(target, bblock, ebl, output = "alps"), exist_ok = True)
    fig.savefig(get_results_dir(target, bblock, ebl, output = "alps").joinpath("interp_ts_hist.png"),
                dpi = 300, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_curve(couplings: np.ndarray, constraint_table: Table, ebl: str, dataset: str,
                          malp: float, b0: float, logscale: bool, out_path: Path, **kwargs) -> None:
    """
    Plot ALPs likelihood curve over considered coupling range:
        lambda(g) = sum(C_ALP(g)) - min_g[sum(C_ALP(g))]
    lambda(g) > 2.71 gives 95% CL upper limit considering Wilk's theorem (initial approximation!)
    Interpolated crossig point gives coupling upper limit
    """

    # Load constraint table from file if found
    constraint_table = Table.read(get_results_dir("ALPs", ebl) / f"alp_constraint_{dataset}_{ebl}_m{malp}_B0{b0}.ecsv")
    # Load the detection significance table
    sigma_table = Table.read(get_results_dir("ALPs", ebl) / f"alp_significance_{dataset}_{ebl}_m{malp}_B0{b0}.ecsv")

    # Initialize plots
    fig, ax = plt.subplots()
    # Set labels and titles
    ax.set_xlabel(r"$g_{a\gamma}$ [GeV$^{-1}$]")
    # ax.set_ylabel(r"$\lambda(g_{a\gamma}) \sim C_{\text{ALP}} - C_{\text{NoALP}}$")
    ax.set_ylabel(r"$C_{\text{ALP}} - \text{min}(C_{\text{ALP}})$")
    ax.set_title(f"ALPs Likelihood Curve {DATASET_LABEL.get(dataset, dataset)}")

    # ax.set_xlim(kwargs.get("xmin", 1e-14), kwargs.get("xmax", 1e-9))

    # Per-block contributions
    for row in constraint_table[:-1]:
        parts = row[0].split("_")
        label = f"{parts[0]} Block {parts[1][5:]}"
        values = np.array(list(row[1:]))
        # Plot contribution of each source (shifted by their minimum value)
        ax.plot(couplings, values - np.nanmin(values),
                marker = ".", linestyle = "-", alpha = 0.15, label = label)

    # Total shifted by value at minimum
    total = np.array(list(constraint_table[-1][1:]))
    total -= np.nanmin(total)
    # Plot total curve
    ax.plot(couplings, total, marker=".", linestyle="-",
            color="black", linewidth=1.5, label="Total")

    # 95% CL threshold
    ax.axhline(2.71, linestyle="--", color="gray",
               label = r"Wilks' 95% CL ($\Delta C = 2.71$)")

    # Add text annotation
    text = (f"EBL {EBL_LABELS.get(ebl, "ModelName")}" + "\n" 
            + rf"$TS = {float(sigma_table["ts_alp"].value[0]):.3f} \, (\sigma \sim {float(sigma_table["sigma"].value[0]):.2f})$" + "\n"
            + rf"$g_{{a\gamma, \text{{min}}}} = {float(sigma_table["g_min"].value[0]):.3f} \cdot 10^{{-11}} \, \text{{GeV}}^{{-1}}$" )
    ax.text(0.025, 0.975, text,
            fontdict = {"fontsize": 8},
            transform = ax.transAxes, 
            verticalalignment = "top",
            bbox = dict(boxstyle = "round, pad = 0.5", fc = "white", alpha = 0.15), )
    
    ax.legend(loc = "center left", bbox_to_anchor = (1.0, 0.5), ncol = 2)

    # Set logarithmic scale
    if logscale:
        ax.set_xscale("log")

    # fig.tight_layout()
    fig.savefig(out_path, bbox_inches = "tight")
    plt.close(fig)

    return


def plot_ebl_comparison(couplings: np.ndarray, ebl_models: list[str], odir: Path, dataset: str,
                        malp: float, b0: float, logscale: bool, **kwargs) -> None:
    """
    Overlay the total summed curve for each EBL model on one plot.
    Applies the same min-subtraction as the individual plots.
    """

    # Initialize plots
    fig, ax = plt.subplots()
    # Set labels and titles
    ax.set_xlabel(r"$g_{a\gamma}$ [GeV$^{-1}$]")
    ax.set_ylabel(r"$\lambda(g_{a\gamma}) \sim C_{\text{ALP}} - C_{\text{NoALP}}$")
    ax.set_title(f"ALPs Likelihood Curve {DATASET_LABEL.get(dataset, dataset)} EBL Comparison")

    # Loop over each EBL model
    for k, ebl in enumerate(ebl_models):
        fpath = odir / f"{ebl}/alp_upturns_{dataset}_{ebl}_m{malp}_B0{b0}.ecsv"
        try:
            table = ascii.read(fpath, format="ecsv")
        except FileNotFoundError:
            print(f"Table not found for EBL {ebl}, skipping! (Path: {fpath})")
            continue

        total = np.array(list(table[-1][1:]))
        total -= np.nanmin(total)
        # Plot total curve for each coupling
        ax.plot(couplings, total, 
                marker = EBL_MARKERS.get(ebl, "."), markersize = 3,
                linestyle = "-",
                label = EBL_LABELS.get(ebl, "EBL Model"))

    ax.axhline(2.71, linestyle="--", color="gray",
               label=r"Wilks' 95% CL ($\Delta C = 2.71$)")
    ax.legend(fontsize = 8)

    if logscale:
        ax.set_xscale("log")

    # Set x-axis limits if given
    if kwargs.get("xmax", None) != None and kwargs.get("xmin", None) != None:
        ax.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))

    # Add in-line plot to include zoom-in where crossing occurs!
    if kwargs.get("add_inset", False):
        axins = inset_axes(ax, width="40%", height="40%", loc="center left", bbox_to_anchor=(0.05, 0.025, 1, 1), bbox_transform=ax.transAxes)
        for k, ebl in enumerate(ebl_models):
            fpath = odir / f"{ebl}/alp_upturns_{dataset}_{ebl}_m{malp}_B0{b0}.ecsv"
            try:
                table = ascii.read(fpath, format="ecsv")
            except FileNotFoundError:
                print(f"Table not found for EBL {ebl}, skipping! (Path: {fpath})")
                continue

            total = np.array(list(table[-1][1:]))
            total -= np.nanmin(total)
            # Plot total curve for each coupling
            axins.plot(couplings, total, 
                    marker = EBL_MARKERS.get(ebl, "."), markersize = 3,
                    linestyle = "-",
                    label = EBL_LABELS.get(ebl, "EBL Model"))
            axins.hlines(2.71, 1e-12, 2e-11, color = "gray", linestyle = "--")
        axins.set_xlim(1e-12, 2e-11)
        axins.set_ylim(0, 10)
        axins.tick_params(axis='both', which='major', labelsize=6)

        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

        if logscale:
            axins.set_xscale("log")

    fig.tight_layout()
    fig.savefig(odir / f"alps_curve_{dataset}_ebls_m{malp}_B0{b0}.pdf",
                bbox_inches="tight")
    plt.close(fig)


def process_ebl(ebl: str, results_dir: Path, odir: Path, args: argparse.Namespace, **kwargs) -> None:
    """Run the full interpolation pipeline for one EBL model."""

    # Create sub-folder for intermediate results
    odir = get_results_dir("ALPs") / ebl
    odir.mkdir(parents = True, exist_ok = True)

    # Get available blocks for EBL
    block_dict = get_block_dict(ebl)
    if not block_dict:
        # Skip if none found
        print(f"No blocks found for EBL={ebl}, skipping")
        return
    else:
        # Otherwise, display them
        # for block in block_dict:
        #     print(block, "- Blocks found: ", len(block_dict[block]))
        pass

    # If source given, run analysis on just this
    if args.source:
        # Get all blocks for this source, or empty list if not found
        block_dict = {args.source: block_dict.get(args.source, [])}
    # Run analysis on single given bayesian block if required
    if args.bblock:
        # Require source argument!
        if not args.source:
            raise ValueError("--bblock requires --source to also be specified")
        block_dict = {args.source: [args.bblock]}

    # lambda = C_ALP - C_noALP per block/coupling
    lambda_table = _empty_table(args.galp)
    # C_ALP per block/coupling
    calp_table = _empty_table(args.galp)
    # Running sum of C_noALP across all blocks
    noalp_sums = np.zeros(len(args.galp))

    # Loop over all pairs of target and block
    for target, blocks in block_dict.items():
        print(f"\nTarget: {target}")

        # If PKS2155-304 or H2356-309, process HESS-only always
        if target in ["PKS2155-304", "H2356-309"]:
            if args.dataset == "joint":
                dataset = "hess"
            elif args.dataset == "joint_bias":
                dataset = "hess_bias"
        # Otherwise, use same dataset as arguments
        else:
            dataset = args.dataset

        # Per-block results (rows in table)
        lambda_block = {}
        calp_block = {}

        # Loop over all blocks
        for block in blocks:
            print(f"  Block: {block}")

            cstat_path = (results_dir / target / block / "ebl" / ebl
                          / "gamma-out" / f"upturns_cstat_{dataset}.npz")
            try:
                cstat = np.load(cstat_path)
            except Exception as e:
                print(f"    Cannot load C-stat grid: {e} — skipping block")
                continue

            for k, galp in enumerate(args.galp):
                popt_path = (results_dir / target / "alps"  
                            / f"popt_{ebl}_m{args.malp}_g{galp}_B0{args.b0}.npy")
                try:
                    popt = np.load(popt_path)
                except Exception as e:
                    print(f"    Cannot load ALP popt (g={galp}): {e} — skipping")
                    continue

                ts, c_noalp, c_alp, cstat_min = interp_alp_upturn(popt, cstat, percentile = 5)

                # Save raw interpolation output
                np.savez(
                    results_dir / target / "alps"
                    / f"interp_ts_{ebl}_m{args.malp}_g{galp}_B0{args.b0}",
                    X = c_noalp, Y = c_alp, Z = ts, )

                # Lambda C_ALP - C_NoALP
                lambda_block[galp] = c_alp - c_noalp
                # Raw C_ALP per block
                calp_block[galp] = c_alp
            
                # Accumulate C_noALP (same value for all couplings
                # within a block, so only add once per block)
                if k == 0:
                    noalp_sums += c_noalp

                # Plot diagnostic histogram of fit results
                hist_path = (results_dir / target / block / "alps"
                             / f"interp_ts_hist_{ebl}.pdf")
                hist_path.parent.mkdir(parents = True, exist_ok = True)
                plot_interp_histogram(
                    ts, c_alp, c_noalp, cstat_min,
                    target, block, ebl, hist_path, )

            block_base = BLOCK_PATTERN.match(block).group(0).rsplit("-", 1)[0]
            row_label  = f"{target}_{block_base}"

            # Add block row to each table
            for table, store in (
                (lambda_table, lambda_block),
                (calp_table, calp_block), ):
                vals = [row_label] + [store.get(g, np.nan) for g in args.galp]
                table.add_row(vals)

    # If empty table, skip EBL
    if len(lambda_table) == 0:
        print(f"EBL '{ebl}': no results produced, skipping!")
        return

    # Add total sum row to each table
    _add_total_row(lambda_table)
    _add_total_row(calp_table)

    # Detection significance
    # sum C_ALP(g) across all sources (last row of constraint table)
    calp_total  = np.array(list(calp_table[-1][1:]))
    # C_noALP is coupling-independent; noalp_sums should be uniform across g
    noalp_total = float(noalp_sums[0])

    # Compute detection significance
    # Get index where C_ALP is minimum
    idx_best = int(np.nanargmin(calp_total))
    # Evaluate total C_NoALP - minimum C_ALP
    ts_det = noalp_total - calp_total[idx_best]
    # Compute significance level
    sigma = np.sign(ts_det) * np.sqrt(np.abs(ts_det))

    # Print detection significance (value at min) with Wilks' approximation and coupling value
    print(f"Best-fit coupling g_ag = {args.galp[idx_best] * 1e-11} GeV-1")
    print(f"TS_det = {ts_det:+3f} (~ sigma = {sigma:+.2f})")

    # Save tables
    lambda_table.write(
        odir / f"alp_upturns_{args.dataset}_{ebl}_m{args.malp}_B0{args.b0}.ecsv",
        format = "ascii.ecsv", overwrite = True)
    calp_table.write(
        odir / f"alp_constraint_{args.dataset}_{ebl}_m{args.malp}_B0{args.b0}.ecsv",
        format = "ascii.ecsv", overwrite = True)
    # Save computed detection significance as table
    # TODO: For now save single-row table with best-fit coupling, ts value, and sigma value?
    det_signif = Table(names = ["g_min", "ts_alp", "sigma"], data = [[args.galp[idx_best]], [ts_det], [sigma]]).write(
        odir / f"alp_significance_{args.dataset}_{ebl}_m{args.malp}_B0{args.b0}.ecsv",
        format = "ascii.ecsv", overwrite = True)

    # Plot the coupling constraint curve
    plot_constraint_curve(
        args.galp * 1e-11, calp_table,
        ebl, args.dataset, args.malp, args.b0, args.logscale,
        # odir / f"{ebl}/alps_constraint_{args.dataset}_{ebl}_m{args.malp}_B0{args.b0}.pdf", )
        odir / f"alps_constraint_{args.dataset}_{ebl}_m{args.malp}_B0{args.b0}.pdf", )

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = ("Interpolate ALP simulation best-fit parameters onto upturn C-stat grids and produce ALP likelihood curves."))
    # Source to analyze (single source or all)
    parser.add_argument("--source", type = str, default = None, choices = [*get_source_list(), "ALL"],
                        help = "Run only for this source (default: all)")
    # Bayesian block to consider (single block or all by default)
    parser.add_argument("--bblock", type = str, default = None,
                        help = "Run only for this block (requires --source)")
    # Dataset to use
    parser.add_argument("--dataset", required = True, 
                        choices = ["hess", "joint", "joint_bias", "hess_bias"],
                        help = "Dataset used for upturn searches")
    # EBL model
    parser.add_argument("--ebl", nargs = "+", required = True, # default = ["dominguez"],
                        choices = ["dominguez", "finke2022", "franceschini", "saldana-lopez"],
                        help = "EBL model(s) to process (default: dominguez)")
    # ALP parameters
    parser.add_argument("--malp", default = 1e-3, type = float,
                        help = "ALP mass [neV] (default: 1e-3)")
    parser.add_argument("--galp", nargs = "+", type = float, default = GALP_DEFAULT,
                        help = "ALP coupling grid [GeV-1]")
    # ALP simulation arguments
    parser.add_argument("--b0",  default = 1e-3, type = float, help = "IGMF B0 [G]")
    parser.add_argument("--gmf", default = "jansson12", help = "GMF model")
    parser.add_argument("--n0",  default = 1e-7, type = float, help = "IGMF n0")
    parser.add_argument("--L0",  default = 1e+4, type = float, help = "IGMF L0")
    # Plotting arguments
    parser.add_argument("--plots-only", nargs = "+", choices = ["alp_curve", "ebl_compare", "all"], help = "Generate plots only")
    parser.add_argument("--logscale", type = bool, default = True,
                        help = "Log scale on x-axis for coupling plots")
    # Additional keyword arguments
    parser.add_argument("--kwargs", nargs = "*",
                        help="Extra keyword arguments as key=value pairs")
    # Parse arguments
    args = parser.parse_args()
    args.galp = np.array(args.galp)
    # Parse keyword arguments
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)
    else:
        kwargs = {}

    # Define and create directory for ALP final results
    results_dir = get_results_dir("ALPs")
    results_dir.mkdir(parents = True, exist_ok = True)

    # Plots only
    if args.plots_only:
        for plots in args.plots_only:
            # Plot ALP curve for given EBL
            if plots == "alp_curve":
                if len(args.ebl) > 1:
                    print("Single EBL model required!")
                    exit()
                else:
                    plot_constraint_curve(args.galp * 1e-11, None, args.ebl[0], args.dataset, args.malp, args.b0, True, 
                                          results_dir / f"{args.ebl[0]}/alps_constraint_{args.dataset}_{args.ebl[0]}_m{args.malp}_B0{args.b0}.pdf",
                                          **kwargs, )
            elif plots == "ebl_compare":
                plot_ebl_comparison(
                    args.galp * 1e-11, args.ebl, results_dir,
                    args.dataset, args.malp, args.b0, args.logscale, **kwargs, )
        exit()

    # Loop over all given EBL models
    for ebl in args.ebl:
        print(f"Processing EBL: {ebl}...")
        # Process for current EBL model for all available pairs of target/block
        # (unless single target/block specified via args parameter)
        process_ebl(ebl, get_results_dir(), results_dir, args, **kwargs, )

    # If more than one EBL model given, generate comparison plots
    if len(args.ebl) > 1:
        plot_ebl_comparison(
            args.galp * 1e-11, args.ebl, results_dir,
            args.dataset, args.malp, args.b0, args.logscale, **kwargs, )
