import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

import numpy    as np
import pandas   as pd

from astropy.time import Time

from alpsup.utils import get_source_list, get_source_blocks
from alpsup.paths import get_results_dir, CONFIGS_DIR


# ---------------------------------------------------------------------------
# LaTeX templates
# ---------------------------------------------------------------------------

# The table body is a self-contained group {...}, so the \arraystretch change
# does not leak into the host document when the file is \input{}-ed.
# Requires only `booktabs` in the host document.
def latex_tab_header(show_model = True):
    """Table header; the model column is optional."""

    colspec = "l l c c c c c c c" if show_model else "l c c c c c c c"
    model_head = "\\textbf{Model} &\n" if show_model else ""
    model_unit = "&\n" if show_model else ""

    return (
        r"{\renewcommand{\arraystretch}{1.25}" + "\n"
        + rf"\begin{{tabular}}{{{colspec}}}" + "\n"
        + r"\toprule" + "\n"
        + r"\textbf{Source Name} &" + "\n"
        + model_head
        + r"""\textbf{Observation Range} &
$N_0$ &
$\Gamma$ &
$\Delta\Gamma$ &
$E_{\mathrm{break}}$ &
$E_0$ &
$s_{\mathrm{H.E.S.S.}}$ \\
&
""" + model_unit
        + r"""{[DD.MM.YYYY]} &
{[cm$^{-2}$ s$^{-1}$ TeV$^{-1}$]} &
&
&
{[TeV]} &
% {[TeV]} &
{[GeV]} &
\\
\midrule
"""
    )


LATEX_TAB_FOOTER = r"""\bottomrule
\end{tabular}}
"""

# Only used to build the standalone PDF
LATEX_STANDALONE_HEADER = r"""\documentclass[border=5pt]{standalone}
\usepackage{booktabs}
\begin{document}
"""
LATEX_STANDALONE_FOOTER = r"""\end{document}
"""

# Short labels for the model column
MODEL_LABELS = {
    "PowerLawSpectralModel":                    "PL",
    "LogParabolaSpectralModel":                 "LP",
    "SmoothBrokenPowerLawSpectralModel":        "SBPL",
    "CustomSmoothBrokenPowerLawSpectralModel":  "CSBPL",
}

# Map of fit-parameter names -> row keys.
# NOTE: "beta" is only kept for LogParabola (for SBPL it is the fixed smoothness).
PARAM_MAP = {
    "amplitude":    "amplitude",
    "index":        "index",        # PL
    "alpha":        "index",        # LP
    "index1":       "index",        # SBPL
    "index2":       "beta",         # SBPL
    "indexdelta":   "beta",         # CSBPL
    "beta":         "beta",         # LP
    "ebreak":       "ebreak",       # (C)SBPL
    "reference":    "reference",
    "bias":         "bias",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _valid(x):
    return x is not None and np.isfinite(x)


def tex_escape(s):
    return str(s).replace("_", r"\_")


def fmt_pm(value, error, decimals = 2, phantom_sign = False):
    """'value ± error' in math mode. Just 'value' if the error is missing/zero
    (frozen parameter), '--' if there is no value (parameter not in this model).
    phantom_sign pads positive numbers so they align with negative ones."""

    if not _valid(value):
        return "--"

    sign = r"\phantom{-}" if (phantom_sign and value >= 0) else ""
    body = f"{value:.{decimals}f}"
    if _valid(error) and error != 0:
        body += rf" \pm {error:.{decimals}f}"

    return f"${sign}{body}$"


def fmt_sci(value, error, decimals = 2):
    """(c ± e)·10^n with a common exponent."""

    if not _valid(value) or value == 0:
        return "--"

    exp = int(np.floor(np.log10(abs(value))))
    # Guard against rounding up to 10.00
    if round(abs(value) / 10**exp, decimals) >= 10:
        exp += 1

    coef = value / 10**exp
    if _valid(error) and error != 0:
        return rf"$({coef:.{decimals}f} \pm {error / 10**exp:.{decimals}f})\cdot10^{{{exp}}}$"
    return rf"${coef:.{decimals}f}\cdot10^{{{exp}}}$"


def format_row(row, show_source = True, show_model = True):

    t0, t1 = (Time(t, format = "mjd").strftime("%d.%m.%Y") for t in row["time"])

    cells = [
        tex_escape(row["source"]) if show_source else "",
        MODEL_LABELS.get(row["model"], tex_escape(row["model"])),
        f"{t0} -- {t1}",
        fmt_sci(row["amplitude"], row["amplitude_err"]),
        fmt_pm(row["index"], row["index_err"]),
        fmt_pm(row["beta"], row["beta_err"]),
        fmt_pm(row["ebreak"], row["ebreak_err"]),    # "--" for PL / LP
        f"${row['reference']*1000:.2f}$" if _valid(row["reference"]) else "--",
        fmt_pm(row["bias"], row["bias_err"], decimals = 3, phantom_sign = True),
    ]

    if not show_model:
        del cells[1]

    return " & ".join(cells) + r" \\" + "\n"


# ---------------------------------------------------------------------------
# Table building / export
# ---------------------------------------------------------------------------

def build_latex_table(rows, show_model = True):
    """Return the bare tabular (wrapped in a group) as a string."""

    body = latex_tab_header(show_model)
    prev_source = None

    for row in rows:
        new_source = row["source"] != prev_source
        # Separate sources with a rule, print the name only on its first block
        if new_source and prev_source is not None:
            body += "\\midrule\n"
        body += format_row(row, show_source = new_source, show_model = show_model)
        prev_source = row["source"]

    return body + LATEX_TAB_FOOTER


def compile_standalone_pdf(tex_source, pdf_path):
    """Compile `tex_source` in a temporary directory, so that only the final
    PDF ends up on disk (no .aux/.log/.tex leftovers)."""

    pdf_path = Path(pdf_path)

    with tempfile.TemporaryDirectory() as tmp:
        tex_file = Path(tmp) / "table.tex"
        tex_file.write_text(tex_source)

        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd = tmp, capture_output = True, text = True)

        if proc.returncode != 0:
            raise RuntimeError("pdflatex failed:\n" + proc.stdout[-2000:])

        shutil.copy(Path(tmp) / "table.pdf", pdf_path)


def export_latex_table(rows,
                       out_dir = None,
                       name = "tab_results",
                       pdf = True,
                       keep_standalone_tex = False,
                       show_model = True):
    """Write <name>.tex (bare table, for \\input{}) and, optionally, <name>.pdf
    (standalone, built in a temp dir). The standalone .tex is only kept on request."""

    out_dir = Path(out_dir) if out_dir is not None else Path(get_results_dir("RESULTS"))
    out_dir.mkdir(parents = True, exist_ok = True)

    table = build_latex_table(rows, show_model = show_model)
    (out_dir / f"{name}.tex").write_text(table)

    standalone = LATEX_STANDALONE_HEADER + table + LATEX_STANDALONE_FOOTER
    if keep_standalone_tex:
        (out_dir / f"{name}_standalone.tex").write_text(standalone)
    if pdf:
        compile_standalone_pdf(standalone, out_dir / f"{name}.pdf")


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------

def get_spectral_results_table(
        target: str = None,
        ebl: str = "dominguez",
    ):

    """
    Aggregate spectral analysis results of one or all sources, across all blocks, for
    a given EBL model. Returns a list of dictionaries (one per source-block pair).
    """

    # Get all sources or use the one specified
    if target is None or target.lower() == "all":
        sources = get_source_list()
    else:
        sources = [target]

    # Load the block configuration only once
    with open(CONFIGS_DIR / "hess_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    rows = []
    for source in sources:
        for block in get_source_blocks(source):

            # Open spectral analysis fit results
            with open(get_results_dir(source, block, ebl, "gamma-out") / "joint_models.yaml", "r") as f:
                # NOTE: Assuming the target model is the last one!
                data_spec = yaml.safe_load(f)["components"][-1]
                if data_spec["name"] != source:
                    raise Exception("Target source is not last in the list of components!")

            # Time range of the block (NOTE: in MJD)
            data_time = config[source]["blocks"][block]
            time = [data_time["tmin"], data_time["tmax"]]

            pars = data_spec["spectral"]["parameters"]
            model_type = data_spec["spectral"]["intrinsic_model"]["spectral"]["type"]

            # Display info
            print(f"{source} {block} [{model_type}] {time[0]:.3f} - {time[1]:.3f} MJD")
            for par in pars:
                print(f"  {par['name']:<12} {par['value']:g} +- {par.get('error', float('nan')):g} {par['unit']}")

            # Construct the row dictionary
            row = {"source": source, "block": block, "time": time, "model": model_type}
            for key in set(PARAM_MAP.values()):
                row[key] = None
                row[f"{key}_err"] = None

            for par in pars:
                # SBPL smoothness parameter (fixed to 1): not tabulated
                if par["name"] == "beta" and model_type != "LogParabolaSpectralModel":
                    continue
                key = PARAM_MAP.get(par["name"])
                if key is None:
                    continue
                row[key] = par["value"]
                row[f"{key}_err"] = par.get("error")

            rows.append(row)

    print(f"Constructed spectral results table for {len(sources)} sources, "
          f"with a total of {len(rows)} blocks!\nEBL model: {ebl}")

    return rows


if __name__ == "__main__":

    rows = get_spectral_results_table(
        # target = "PKS0903-57",
        ebl = "dominguez", )

    # LaTeX table (.tex to \input, plus standalone PDF)
    export_latex_table(rows, show_model = False)

    # Same data as a pandas table / CSV
    df = pd.DataFrame(rows)
    df.to_csv(Path(get_results_dir("RESULTS")) / "tab_results.csv", index = False)