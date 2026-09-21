import  yaml

import  numpy           as      np
import  pandas          as      pd

from    gammapy.modeling    import  Parameter

from    astropy.table   import  Table
from    astropy.units   import  Quantity

from    alpsup.utils    import  get_source_list, get_source_blocks
from    alpsup.paths    import  get_results_dir, CONFIGS_DIR


# LaTeX standalone document header
LATEX_DOC_HEADER = r"""
\documentclass{standalone}
\usepackage{adjustbox}
\usepackage{booktabs}
\begin{document}
\renewcommand{\arraystretch}{1.25}
"""
# LaTeX standalone document footer
LATEX_DOC_FOOTER = r"""
\bottomrule
\end{tabular}
\end{document}
"""

# LaTeX table header
LATEX_TAB_HEADER = r"""
% \begin{table*}[htbp]
% \centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l c c c c c c c}
\hline
\textbf{Source Name} &
\textbf{Observation Range} &
$N_0$ &
% $\alpha$ &
$\Gamma$ &
$\beta$ &
$E_0$ &
$s_{\mathrm{H.E.S.S.}}$ \\
&
{[DD.MM.YYYY]} &$z$
{[cm$^{-2}$ s$^{-1}$ TeV$^{-1}$]} &
&
&
{[TeV]} &
\\
\hline
"""

# LaTeX table footer
LATEX_TAB_FOOTER = r"""
\hline
\end{tabular}
\end{adjustbox}
% \caption{Caption...}
% \label{tab:combined_blazar_summary}
% \end{table*}
\end{document}
"""


def export_latex_table(rows,
                       standalone = True):

    doc = str()

    for k, row in enumerate(rows):
        print(row)

        # Add table header (only before first row)
        if k == 0:
            # If required, add standalone document LaTeX header
            if standalone is True:
                doc += LATEX_DOC_HEADER
            doc += LATEX_TAB_HEADER

        # TODO: FORMAT THE ROW AND ADD WRITE IT NICELY
        row_tex = str()

        row_tex += row["source"] + " & "

        # Format time nicely into DD.MM.YYYY
        from astropy.time import Time
        tmin = Time(row["time"][0], format = "mjd").ymdhms
        tmax = Time(row["time"][1], format = "mjd").ymdhms

        row_tex += f"{tmin['day']:02d}.{tmin['month']:02d}.{tmin['year']} -- {tmax['day']:02d}.{tmax['month']:02d}.{tmax['year']}" + " & "

        # Format amplitude
        def format_scientific(value, error, decimals=2):
            exponent = int(f"{value:.0e}".split("e")[1])
            coefficient = value / 10**exponent
            error_coefficient = error / 10**exponent

            exponent_str = f"{exponent:+03d}".replace("+", "")
            if exponent < 0:
                exponent_str = f"-{abs(exponent):02d}"
            else:
                exponent_str = f"{exponent:02d}"

            return f"$({coefficient:.{decimals}f} \\pm {error_coefficient:.{decimals}f})\\cdot10^{{{exponent_str}}}$"
        # row_tex += f"${row['amplitude']:g} \pm {row['amplitude_err']:g}$" + " & "
        row_tex += format_scientific(row['amplitude'], row['amplitude_err']) + " & "

        # Format index
        row_tex += f"${row['index']:.2f} \pm {row['index_err']:.2f}$" + " & "

        # Format beta
        if row['beta'] is not None:
            row_tex += f"${row['beta']:.2f} \pm {row['beta_err']:.2f}$" + " & "
        else:
            row_tex += f"--" + " & "

        # Format reference
        row_tex += f"${row['reference']:.5f}$" + " & "

        # Format bias
        if row['bias'] > 0:
            row_tex += "$\phantom{+}$" + f"${row['bias']:.3f} \pm {row['bias_err']:.3f}$" + " \\\\ " + "\n"
        else:
            row_tex += f"${row['bias']:.3f} \pm {row['bias_err']:.3f}$" + " \\\\ " + "\n"

        # Add row to document
        doc += row_tex

    # Add table footer at the end
    doc += LATEX_TAB_FOOTER
    # Add document footer at the end
    doc += LATEX_DOC_FOOTER

    # Save table to file
    with open(get_results_dir("RESULTS") / "tab_results.tex", "w") as f:
        f.write(doc)

    return


def get_spectral_results_table(
        target: str = None,
        ebl: str = "dominguez",
    ):

    """
    Aggregate spectral analysis results of one or all sources, across all blocks, for
    a given EBL model. Return format can be specified by the user (either pandas table,
    formatted LaTeX table, astropy table, etc.)
    """

    # Get all sources or use the one specified
    # NOTE: Check first if NoneType ;)
    if target is None or target.lower() == "all":
        sources = get_source_list()
    # Otherwise, use just the specified source
    else:
        sources = [target]

    rows = []
    for source in sources:
        for block in get_source_blocks(source):

            # Open spectral analysis fit results
            with open(get_results_dir(source, block, ebl, "gamma-out") / "joint_models.yaml", "r") as f:
                # NOTE: Assuming the target model is the last one!
                data_spec = yaml.safe_load(f)["components"][-1] # [source]
                # Check if name is correct
                if data_spec["name"] != source:
                    raise Exception("Target source is not last in the list of components!")

            # TODO: Also load time range selection for the block!
            with open(CONFIGS_DIR / "hess_config.yaml", "r") as f:
                data_time = yaml.safe_load(f)[source]["blocks"][block]

            # Load the time range of the block (NOTE: in MJD)
            time = [ data_time["tmin"], data_time["tmax"] ]

            # Load the spectral model
            pars = data_spec["spectral"]["parameters"]

            # Display info
            print(source, block)
            print(f"Time: {time[0]:.3f} - {time[1]:.3f} MJD")
            print("Spectral parameters: ")
            for par in pars:
                print(f"  {par["name"]}")
                print(f"  {par['value']:g}")
                print(f"  {par['error']:g}")
                print(f"  {par['unit']}")

            # Construct the row dictionary with useful information
            row = {
                "source": source,
                "block": block,
                "time": time,
                "amplitude": None,
                "amplitude_err": None,
                "amplitude_unit": None,
                "index": None, # Either PL index or alpha LP or index1 SBPL
                "index_err": None,
                "index_unit": None,
                "beta": None, # Either beta LP or index2 SBPL
                "beta_err": None,
                "beta_unit": None,
                "reference": None,
                "reference_err": None,
                "reference_unit": None,
                "bias": None,
                "bias_err": None,
                "bias_unit": None,
            }

            # Load data into the dictionary
            for par in pars:
                if par["name"] in row.keys():
                    row[par["name"]] = par["value"]
                    row[f"{par['name']}_err"] = par["error"]
                # TODO: WHAT IF SBPL OR LP?
                elif par["name"] in ["alpha", "index1"]:
                    row["index"] = par["value"]
                    row["index_err"] = par["error"]
                elif par["name"] in ["index2"]:
                    row["beta"] = par["value"]
                    row["beta_err"] = par["error"]

            # Append row of the table
            rows.append(row)

    print(f"Constructed spectral results table for {len(sources)} sources, with a total of {len(rows)} blocks!\nEBL model: {ebl}")

    # TODO: CONVERT THIS INTO ASTROPY TABLE? - QUANTITY DOESN'T HAVE UNCERTAINTY...

    # Return all rows
    return rows


if __name__ == "__main__":

    rows = get_spectral_results_table(
        # target = "1ES0229+200",
        ebl = "dominguez", )

    # Format table into LaTeX
    export_latex_table(rows)