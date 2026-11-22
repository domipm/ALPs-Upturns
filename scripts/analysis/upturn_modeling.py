# Script used for modeling upturns onto our data
# Generates C-stat grid by fitting upturn model for each pair of E break and Delta Gamma
# using as baseline the best-fit model of the given dataset (including any EBL or bias)
# Requires main fit to be performed for desired dataset
#
# Usage:
#   Full grid (+ automatic polish pass at the end):
#       python upturn_search.py --source SRC --bblock BLOCK --ebl dominguez
#   Faster: 8 processes, one fit per point (the polish pass verifies and fixes the grid afterwards):
#       python upturn_search.py --source SRC --bblock BLOCK --ebl dominguez --jobs 8 --fast
#   Only re-fit bad / missing grid points, without re-running the grid:
#       python upturn_search.py --source SRC --bblock BLOCK --ebl dominguez --polish
#   Force specific grid points to be re-fit (nearest grid point to "DeltaGamma E_break[TeV]"):
#       python upturn_search.py --source SRC --bblock BLOCK --ebl dominguez --polish --refit="-4.21 0.238"
#
# The grid is checkpointed after every row, together with the best-fit parameters of every point
# (upturns_params_<dataset>.json). An interrupted run can be completed with --polish (NaN points are re-fit).

import  os
import  gc
import  json
import  logging
import  argparse
import  multiprocessing
from    concurrent.futures          import  ProcessPoolExecutor, as_completed
from    types                       import  SimpleNamespace

# With several fits running in parallel (--jobs), avoid every process also spawning many BLAS / OpenMP threads
# (export these variables yourself before launching to override)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import  matplotlib.pyplot           as      plt
import  matplotlib.colors           as      clr

import  numpy                       as      np
import  astropy.units               as      u

from    gammapy.modeling            import  Fit
from    gammapy.modeling.models     import  Models, SkyModel
from    gammapy.datasets            import  Datasets

from    alpsup.models       import  UpturnSpectralModel, CompositeSpectralModel
from    alpsup.utils        import  get_source_info, get_etau, parse_kwargs
from    alpsup.plots        import  plot_sed_gammapy
from    alpsup.paths        import  get_results_dir
from    alpsup.logs         import  init_log
from    alpsup.parameters   import  set_param_bounds


# Get logger function (re-bound in main() to the logger returned by init_log)
log = logging.getLogger(__name__)


# Multi-start settings for every grid point
# Random restarts around the baseline (in units of the parameter errors); only used if the other starts disagree
JITTER_SCALES = (0.5, 1.0, 2.0)
# Two independent starts reaching the same C-stat within this tolerance -> minimum accepted
STAT_TOL = 0.05
# Polish: a point is re-fit if a neighbouring (or the baseline) solution beats it by more than this
POLISH_TOL = 0.10


# ================================================ #
# FIT HELPERS: STARTING VALUES, MULTI-START, BOUNDS #
# ================================================ #


def _clip_into_bounds(par):
    """Make sure a parameter starts strictly inside its bounds"""

    lo, hi = par.min, par.max
    value = par.value
    if par.name == "amplitude":
        # Log-scaled parameter spanning many decades (e.g. 1e-17 ... 1e-4): a margin proportional to (hi - lo)
        # would silently push every value below ~1e-7 up to 1e-7, so use a relative margin instead
        if np.isfinite(lo):
            value = max(value, lo * (1 + 1e-6))
        if np.isfinite(hi):
            value = min(value, hi * (1 - 1e-6))
    else:
        eps = 1e-3 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else 0.0
        if np.isfinite(lo):
            value = max(value, lo + eps)
        if np.isfinite(hi):
            value = min(value, hi - eps)
    par.value = value


def _apply(pars, values):
    """Set the (free) parameters present in `values`, clipping them into their bounds"""

    for par in pars:
        if par.name in values and not par.frozen:
            par.value = values[par.name]
            _clip_into_bounds(par)


def _jittered(start, pars, rng, scale):
    """Random (seeded) perturbation of a start vector: log-normal for the amplitude, Gaussian (in units of the error) otherwise"""

    out = {}
    for par in pars:
        if par.name not in start:
            continue
        if par.name == "amplitude":
            out[par.name] = start[par.name] * 10 ** rng.normal(0.0, 0.1 * scale)
        else:
            err = par.error if np.isfinite(par.error) and par.error > 0 else 0.1
            out[par.name] = start[par.name] + rng.normal(0.0, scale * err)
    return out


def at_bounds(pars, rel = 1e-3):
    """Names of free parameters sitting on one of their bounds (informational only, a boundary solution can be legitimate)"""

    out = []
    for par in pars:
        # Amplitude is log-scaled, bounds are meaningless in linear terms
        if par.frozen or par.name == "amplitude":
            continue
        lo, hi = par.min, par.max
        span = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else 1.0
        if np.isfinite(lo) and par.value - lo < rel * span:
            out.append(f"{par.name}@min")
        if np.isfinite(hi) and hi - par.value < rel * span:
            out.append(f"{par.name}@max")
    return out


def fit_grid_point(dataset, target, fit, baseline, warm, rng, tol = STAT_TOL, max_starts = None, extra = ()):
    """
    Minimise the C-stat of one grid point from several starting points and keep the best.

    Candidate starts: `extra` (e.g. the upturn-compensated start), warm starts (solutions of neighbouring grid points),
    the baseline fit, and seeded jitters of the baseline. All candidates are RANKED by their own C-stat before any fit, so a
    bad (e.g. runaway) neighbour solution is never tried first, and quick mode (max_starts) uses the best candidate.
    A fit that ends ABOVE the C-stat of its own start is not a minimum and is discarded.
    Stops as soon as the two best results agree within `tol` (= minimum reproduced by two independent starts).
    Only the optimizer is run (no covariance / Hesse): the grid needs the minimum, not the errors.

    Returns (stat, best_free_parameters) or (nan, None).
    """

    pars = dataset.models[target].parameters

    starts = [e for e in extra if e is not None] + [w for w in warm if w is not None] + [baseline]
    starts += [_jittered(baseline, pars, rng, scale) for scale in JITTER_SCALES]

    ranked = []
    for start in starts:
        _apply(pars, start)
        try:
            stat0 = float(dataset.stat_sum())
        except Exception:
            continue
        if np.isfinite(stat0):
            ranked.append((stat0, start))
    ranked.sort(key = lambda x: x[0])
    # Quick mode: a single start (the polish pass verifies the result afterwards)
    if max_starts:
        ranked = ranked[:max_starts]

    found = []
    for stat0, start in ranked:
        _apply(pars, start)
        try:
            res = fit.optimize(dataset)
        except Exception as e:
            log.warning(f"\tOptimizer exception: {e}")
            continue
        stat = dataset.stat_sum()
        if not (res.success and np.isfinite(stat)):
            continue
        if stat > stat0 + 1e-3:
            log.debug(f"\tDiscarded fit: ended above its own start ({stat:.2f} > {stat0:.2f})")
            continue
        found.append((float(stat), {p.name: float(p.value) for p in pars if not p.frozen}))
        found.sort(key = lambda x: x[0])
        if len(found) >= 2 and found[1][0] - found[0][0] < tol:
            break

    if not found:
        return np.nan, None

    if len(found) == 1 and len(ranked) > 1:
        log.warning("\tOnly one valid start for this point, minimum not cross-checked")

    stat, best = found[0]
    # Leave the dataset in the best-fit state
    _apply(pars, best)
    return stat, best


def compensated_start(ctx, baseline):
    """
    Start vector for the intrinsic parameters such that intrinsic x upturn reproduces the baseline (no-upturn) spectrum
    over the fit range (log-space least squares). Without this, switching the upturn on at a grid point starts the fit far
    from the valley (flux off by orders of magnitude above the upturn break) and the optimizer can run away.
    Call right after configure_point. Returns a dict like `baseline` or None (the other starts are then used).
    """

    pars = ctx.dataset.models[ctx.target].parameters
    free = []
    try:
        from scipy.optimize import least_squares
        spec = ctx.dataset.models[ctx.target].spectral_model
        intr, upt = spec.intrinsic_model, spec.upturn_model
        names = [n for n in baseline if n in intr.parameters.names]
        if not names:
            return None
        free = [pars[n] for n in names]
        energy = np.geomspace(ctx.energy_min, ctx.energy_max, 30) * u.TeV
        val = lambda x: np.asarray(getattr(x, "value", x), dtype = float)
        goal = np.log(np.clip(val(intr(energy)), 1e-300, None)) - np.log(np.clip(val(upt(energy)), 1e-300, None))

        islog = [p.name == "amplitude" for p in free]
        tx = lambda p, v: np.log10(v) if p.name == "amplitude" else v
        lo = np.array([tx(p, p.min) if np.isfinite(p.min) else -np.inf for p in free], dtype = float)
        hi = np.array([tx(p, p.max) if np.isfinite(p.max) else np.inf for p in free], dtype = float)
        x0 = np.clip([tx(p, baseline[p.name]) for p in free], lo, hi)

        def setx(x):
            for p, xi, lg in zip(free, x, islog):
                p.value = 10 ** xi if lg else xi

        def resid(x):
            setx(x)
            r = np.log(np.clip(val(intr(energy)), 1e-300, None)) - goal
            return np.where(np.isfinite(r), r, 1e3)

        sol = least_squares(resid, x0, bounds = (lo, hi))
        setx(sol.x)
        # Non-intrinsic free parameters (e.g. bias) keep their baseline values, so the vector is complete
        return {**baseline, **{p.name: float(p.value) for p in free}}
    except Exception as e:
        log.debug(f"\tCompensated start failed: {e}")
        return None
    finally:
        for p in free:
            p.value = baseline[p.name]


def configure_point(ctx, i, j):
    """
    Restore the baseline parameters, set the grid point (upturn index delta and break energy),
    freeze flags and bounds. Returns the start vector (baseline values of the free parameters).
    """

    pars = ctx.dataset.models[ctx.target].parameters

    # Restore dataset parameters to original values
    for par in pars:
        par.value = ctx.params_org[par.name].value
        par.error = ctx.params_org[par.name].error

    # Fix upturn beta smoothing parameter
    pars["beta_upt"].frozen = True
    # Set break parameters from the grid, and freeze them
    pars["e_upt"].value = ctx.ebreak_vals[j]
    pars["indexdelta_upt"].value = ctx.ibreak_vals[i]
    pars["e_upt"].frozen = True
    pars["indexdelta_upt"].frozen = True
    # Make sure reference is frozen just in case
    pars["reference"].frozen = True

    # Rest of the parameters (index, amplitude, etc.) depend on intrinsic model, should be free

    # Smooth broken power law parameters
    if "ebreak" in pars.names:
        pars["ebreak"].frozen = ctx.kwargs.get("ebreak_frozen", False)
        pars["ebreak"].min = ctx.ebreak_min
        # NOTE: Set SBPL Ebreak parameter below E(tau = tau_limit) so there's no overlap with upturn!
        pars["ebreak"].max = ctx.ebreak_max
        if "indexdelta" in pars.names:
            pars["indexdelta"].max = ctx.kwargs.get("indexdelta_max", 7.5)

    return {p.name: p.value for p in pars if not p.frozen}


def solve_point(ctx, i, j, warm = ()):
    """Fit one grid point (multi-start). Returns (stat, best_free_parameters) or (nan, None)."""

    baseline = configure_point(ctx, i, j)
    comp = compensated_start(ctx, baseline) if getattr(ctx.args, "comp_start", True) else None
    stat, best = fit_grid_point(ctx.dataset, ctx.target, ctx.fit, baseline, warm, ctx.rng,
                                max_starts = 1 if getattr(ctx, "quick", False) else None, extra = (comp,))

    ibreak, ebreak = ctx.ibreak_vals[i], ctx.ebreak_vals[j]
    if best is None:
        log.warning(f"[{i},{j}] No valid fit for Gamma2 = {ibreak:.2f}, Ebreak2 = {ebreak:.2e}")
    else:
        pars = ctx.dataset.models[ctx.target].parameters
        log.info(f"[{i},{j}] Fit success! Gamma2 = {ibreak:.2f}, Ebreak2 = {ebreak:.2e}, C-stat = {stat:.2f}")
        log.info("\t" + ", ".join(f"{k} = {v:.4e}" for k, v in best.items()))
        flagged = at_bounds(pars)
        if flagged:
            log.info(f"\tParameters at bounds: {', '.join(flagged)}")
    return stat, best


def evaluate_point(ctx, i, j, values = None):
    """C-stat of a grid point for a given parameter vector (no fit). values = None -> baseline start vector."""

    baseline = configure_point(ctx, i, j)
    pars = ctx.dataset.models[ctx.target].parameters
    _apply(pars, baseline if values is None else values)
    try:
        stat = float(ctx.dataset.stat_sum())
    except Exception:
        return np.nan
    return stat if np.isfinite(stat) else np.nan


# ============================================ #
# GRID: RUN, SAVE / LOAD, POLISH               #
# ============================================ #


def grid_files(ctx):
    """Output files of the grid: C-stat, Delta C-stat (for plots) and best-fit parameters of every point"""

    d = ctx.args.dataset
    return (ctx.dir_gout / f"upturns_cstat_{d}.npz",
            ctx.dir_gout / f"upturns_dcstat_{d}.npz",
            ctx.dir_gout / f"upturns_params_{d}.json")


def save_grid(ctx, cstat_grid, params):
    """Save the grid and the per-point parameters (written via temporary files, so a checkpoint can't be left half-written)"""

    f_cstat, f_dcstat, f_params = grid_files(ctx)
    zmin = np.nanmin(cstat_grid) if np.isfinite(cstat_grid).any() else 0.0

    for fname, z in ((f_cstat, cstat_grid), (f_dcstat, cstat_grid - zmin)):
        tmp = fname.with_name(fname.stem + ".tmp.npz")
        np.savez(file = tmp, X = ctx.ibreak_grid, Y = ctx.ebreak_grid, Z = z)
        os.replace(tmp, fname)

    tmp = f_params.with_name(f_params.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"ibreak_vals": [float(x) for x in ctx.ibreak_vals],
                   "ebreak_vals": [float(x) for x in ctx.ebreak_vals],
                   "points": params}, f)
    os.replace(tmp, f_params)


def load_grid(ctx):
    """Load a previously saved grid (and, if available, the per-point parameters used for warm starts)"""

    f_cstat, _, f_params = grid_files(ctx)
    if not f_cstat.exists():
        raise FileNotFoundError(f"No saved grid at {f_cstat}: run the full grid first.")

    cstat_grid = np.array(np.load(f_cstat)["Z"], dtype = float)
    if cstat_grid.shape != ctx.ibreak_grid.shape:
        raise ValueError(f"Saved grid has shape {cstat_grid.shape}, but --npoints / --tau-limit give {ctx.ibreak_grid.shape}. Use the same arguments as the original run.")

    ni, nj = cstat_grid.shape
    if f_params.exists():
        with open(f_params, "r") as f:
            saved = json.load(f)
        if not (np.allclose(saved["ibreak_vals"], ctx.ibreak_vals) and np.allclose(saved["ebreak_vals"], ctx.ebreak_vals)):
            raise ValueError("Saved grid axes differ from the requested ones. Use the same arguments as the original run.")
        params = saved["points"]
    else:
        log.warning("No parameter file found (grid from an older version of the script): no warm starts available. "
                    "Use --refit to fix specific points; they will be re-built by continuation from the weak-upturn edge.")
        params = [[None] * nj for _ in range(ni)]

    return cstat_grid, params


def run_grid(ctx):
    """Fit the full grid, using the solutions of the neighbouring points (weaker upturn / lower E_break) as warm starts"""

    ni, nj = len(ctx.ibreak_vals), len(ctx.ebreak_vals)
    cstat_grid = np.full((ni, nj), np.nan)
    params = [[None] * nj for _ in range(ni)]

    for i in range(ni):
        log.info(f"*** deltaindex_brk = {ctx.ibreak_vals[i]:.6f} ***")
        for j in range(nj):
            log.info(f"··· e_brk = {ctx.ebreak_vals[j]:.6f} ···")
            warm = [params[i - 1][j] if i > 0 else None,
                    params[i][j - 1] if j > 0 else None]
            cstat_grid[i, j], params[i][j] = solve_point(ctx, i, j, warm)

        # Checkpoint after every row
        save_grid(ctx, cstat_grid, params)
        # Force python to collect garbage (memory usage)
        gc.collect()

    return cstat_grid, params


# Context of each worker process (built once per process by _init_worker)
_CTX = None


def _init_worker(args, kwargs):
    """Worker start-up: own log file, own copy of the datasets and models"""

    global _CTX, log
    log = init_log(args.source, fname = f"upturn_worker_{os.getpid()}.log", bblock = args.bblock, ebl = args.ebl)
    _CTX = prepare(args, kwargs)
    _CTX.quick = args.fast


def _solve_column(j):
    """Fit one column of the grid (fixed E_break, upturn getting stronger), warm-starting each point from the previous one"""

    ctx = _CTX
    # Seed depends on the column only, so results don't depend on how the columns are scheduled
    ctx.rng = np.random.default_rng([ctx.args.seed, j])
    out, prev = [], None
    for i in range(len(ctx.ibreak_vals)):
        stat, par = solve_point(ctx, i, j, [prev])
        out.append((stat, par))
        if par is not None:
            prev = par
    return j, out


def _make_executor(jobs, args, kwargs):
    # "spawn": every worker builds its own datasets (safe on Linux, macOS and Windows)
    return ProcessPoolExecutor(max_workers = jobs, mp_context = multiprocessing.get_context("spawn"),
                               initializer = _init_worker, initargs = (args, kwargs))


def run_grid_parallel(ctx, jobs):
    """
    Fit the grid with several processes, one E_break column per task (columns are independent chains along the upturn strength).
    The grid is checkpointed whenever a column finishes. Failed columns / points stay NaN and are picked up by the polish pass.
    """

    ni, nj = len(ctx.ibreak_vals), len(ctx.ebreak_vals)
    cstat_grid = np.full((ni, nj), np.nan)
    params = [[None] * nj for _ in range(ni)]

    jobs = min(jobs, nj)
    log.info(f"Running grid with {jobs} parallel processes (one log file per worker: upturn_worker_<pid>.log)")

    # Slowest columns first (low E_break) for better load balance
    with _make_executor(jobs, ctx.args, ctx.kwargs) as ex:
        futures = {ex.submit(_solve_column, j): j for j in range(nj)}
        for fut in as_completed(futures):
            try:
                j, col = fut.result()
            except Exception as e:
                log.error(f"Column {futures[fut]} failed: {e!r} (left as NaN, the polish pass will re-fit it)")
                continue
            for i, (stat, par) in enumerate(col):
                cstat_grid[i, j], params[i][j] = stat, par
            n_ok = int(np.isfinite(cstat_grid[:, j]).sum())
            log.info(f"Column {j} done (E_break = {ctx.ebreak_vals[j]:.3e} TeV): {n_ok}/{ni} points fitted")
            save_grid(ctx, cstat_grid, params)

    return cstat_grid, params


def _neighbours(i, j, shape):
    return [(i + a, j + b) for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= i + a < shape[0] and 0 <= j + b < shape[1]]


def polish_grid(cstat, params, evaluate, solve, force = (), max_rounds = 3, tol = POLISH_TOL, on_update = None):
    """
    Re-fit only the grid points that need it.

    A point needs a re-fit if (a) its C-stat is NaN, (b) it is in `force`, or (c) the stored solution of a neighbour,
    or the baseline start vector, evaluated at this point gives a LOWER C-stat than the stored one by more than `tol`
    (a true minimum can't be beaten by another parameter vector at the same grid point, so the stored fit failed).
    Re-fits start from the neighbours' solutions and are only accepted if they lower the C-stat.

    cstat: 2D array (modified in place), params: nested list of dicts / None (modified in place)
    evaluate(i, j, values) -> stat (no fit), solve(i, j, warm) -> (stat, params)
    Returns the number of improved points.
    """

    shape = cstat.shape
    force = set(force)
    tried = set()
    n_total = 0

    def store(i, j, stat, par):
        """Keep a new solution if it beats the stored one. Returns True if the grid changed."""
        if par is None or not np.isfinite(stat):
            return False
        if np.isfinite(cstat[i, j]) and stat >= cstat[i, j] - 1e-3:
            if params[i][j] is None:
                params[i][j] = par          # only kept as a warm start, C-stat unchanged
            return False
        log.info(f"[{i},{j}] C-stat improved: {cstat[i, j]:.2f} -> {stat:.2f}")
        cstat[i, j], params[i][j] = stat, par
        if on_update is not None:
            on_update()
        return True

    def ensure_chain(i, j):
        """If points below (same E_break, weaker upturn) have no stored solution, build them by continuation from the weak-upturn edge"""
        for k in range(i):
            if params[k][j] is None and (k, j) not in tried:
                tried.add((k, j))
                prev = params[k - 1][j] if k > 0 else None
                stat, par = solve(k, j, [prev] if prev is not None else [])
                store(k, j, stat, par)

    for rnd in range(max_rounds):
        changed = 0
        for i, j in np.ndindex(*shape):
            cur = cstat[i, j]
            cands = [params[a][b] for a, b in _neighbours(i, j, shape) if params[a][b] is not None]

            if (i, j) not in force and np.isfinite(cur):
                tests = [evaluate(i, j, c) for c in cands] + [evaluate(i, j, None)]
                if not any(np.isfinite(t) and t < cur - tol for t in tests):
                    continue
                log.info(f"[{i},{j}] C-stat = {cur:.2f} is beaten by a neighbouring / baseline solution "
                         f"({np.nanmin(tests):.2f}): re-fitting")

            ensure_chain(i, j)
            cands = [params[a][b] for a, b in _neighbours(i, j, shape) if params[a][b] is not None]
            stat, par = solve(i, j, cands)
            changed += store(i, j, stat, par)

        n_total += changed
        log.info(f"Polish round {rnd + 1}: {changed} point(s) improved")
        if not changed:
            break
        # Forced points are only re-fit in the first round
        force = set()

    return n_total


def nearest_point(ctx, ibreak, ebreak):
    """Grid indices closest to a (DeltaGamma, E_break [TeV]) pair"""

    i = int(np.argmin(np.abs(ctx.ibreak_vals - ibreak)))
    j = int(np.argmin(np.abs(np.log10(ctx.ebreak_vals) - np.log10(ebreak))))
    return i, j


# ================================================ #
# SETUP: DATASETS, MODELS, GRID AXES               #
# ================================================ #


def prepare(args, kwargs):
    """Load datasets and models, attach the upturn, define grid axes. Returns a context object shared by all grid functions."""

    target = args.source

    # Get info on source from data class
    source = get_source_info(target)
    target_redshift = source.redshift

    # Define output directory
    dir_gout = get_results_dir(target, args.bblock, args.ebl, output = "gamma-out")

    # Load dataset and best-fit intrinsic model
    log.info("Loading datasets and models...")
    datasets = Datasets.read(
        filename = dir_gout.joinpath(f"{args.dataset}_datasets.yaml"),
        filename_models = dir_gout.joinpath(f"{args.dataset}_models.yaml"), )

    # Get energy range
    dataset_hess = datasets["HESS"]
    try:
        dataset_flat = datasets["Fermi-LAT"]
    except:
        dataset_flat = None
    dataset_energy_range = [None, dataset_hess.counts.geom.axes["energy"].edges[-1].to(u.TeV)]
    if dataset_flat is not None:
        dataset_energy_range[0] = dataset_flat.counts.geom.axes["energy"].edges[0].to(u.TeV)
    else:
        dataset_energy_range[0] = dataset_hess.counts.geom.axes["energy"].edges[0].to(u.TeV)

    # Add spectral upturn to the model (HESS model, including bias term)
    datasets.models[target].spectral_model.upturn_model = UpturnSpectralModel()

    # Define spectral model for Fermi-LAT (unbiased, with the same upturn as HESS, linked parameters)
    # NOTE: Spatial model is read from the Fermi-LAT model (the HESS 1D model has none)
    if dataset_flat is not None:
        spatial_model = datasets["Fermi-LAT"].models[f"{target} Unbiased"].spatial_model
        models = Models()
        for model in datasets.models:
            if model.name != f"{target} Unbiased":
                models.append(model)
        models.append(
            SkyModel(
                name = f"{target} Unbiased", datasets_names = "Fermi-LAT",
                spatial_model = spatial_model,
                spectral_model = CompositeSpectralModel(
                    intrinsic_model = datasets.models[target].spectral_model.intrinsic_model,
                    ebl_model = datasets.models[target].spectral_model.ebl_model,
                    bias = 0,
                    upturn_model = datasets.models[target].spectral_model.upturn_model, ) ) )
        models[f"{target} Unbiased"].spectral_model.parameters["bias"].frozen = True
        datasets.models = models

    # Display dataset and model to fit
    log.info(datasets)
    log.info(datasets.models[target])
    if dataset_flat is not None:
        log.info(datasets.models[f"{target} Unbiased"])

    # Compute energy at tau optical depth limit
    e_tau = get_etau(target_redshift, args.ebl, tau_lim = args.tau_limit)
    e_tau_tev = float(e_tau.to_value(u.TeV))

    # Set break energy to E(tau = tau_lim)
    datasets.models[target].parameters["e_upt"].quantity = e_tau
    # Set index delta value to -0.1 by default
    datasets.models[target].parameters["indexdelta_upt"].value = - 0.1
    log.info("Datasets and models loaded!")

    # Define grid for Index2_brk = gamma2 and E_brk = ebreak2
    ibreak_vals = np.linspace(kwargs.get("ibreak_min", -0.01), -5.00, args.npoints) # from 0 (no break) to -5 (sharp break) linearly spaced
    ebreak_vals = np.logspace(np.log10(e_tau_tev), np.log10(31.6), args.npoints)    # from E(tau) to 31.6 TeV

    # Generate meshgrid
    ibreak_grid, ebreak_grid = np.meshgrid(ibreak_vals, ebreak_vals, indexing = "ij")

    # Define new (copy) dataset object
    dataset = datasets.copy()
    # Store default parameters
    params_org = datasets.models[target].parameters

    # Set parameter bounds
    set_param_bounds(dataset.models[target], bounds = "upt")

    ctx = SimpleNamespace(
        args = args, kwargs = kwargs, target = target, dir_gout = dir_gout,
        dataset = dataset, params_org = params_org,
        ibreak_vals = ibreak_vals, ebreak_vals = ebreak_vals,
        ibreak_grid = ibreak_grid, ebreak_grid = ebreak_grid,
        # SBPL break energy bounds: from the data range up to E(tau = tau_limit), so there's no overlap with the upturn
        ebreak_min = kwargs.get("ebreak_min", float(dataset_energy_range[0].value)),
        ebreak_max = kwargs.get("ebreak_max", e_tau_tev),
        energy_min = float(dataset_energy_range[0].value), energy_max = float(dataset_energy_range[1].value),
        fit = Fit(), rng = np.random.default_rng(args.seed), quick = False, )

    # Make sure the null (start of every grid point) is the best minimum we can find
    refine_null(ctx)
    return ctx


def refine_null(ctx, n = 6):
    """
    Re-fit the NULL (intrinsic model with the upturn effectively off) from several break energies and keep the best minimum.

    The null is the grid point with the weakest upturn (smallest |delta|, highest E_break). Every grid point starts from the
    null parameters, so if the null is a local minimum (e.g. SBPL break stuck at the data threshold) every point inherits that.
    Here the intrinsic break is frozen on `n` energies between its bounds, the rest is fitted, then everything is freed and
    re-fitted. The best result replaces the stored baseline (ctx.params_org). Disable with --kwargs null_scan=False.
    """

    pars = ctx.dataset.models[ctx.target].parameters
    if "ebreak" not in pars.names or ctx.kwargs.get("ebreak_frozen", False) or not ctx.kwargs.get("null_scan", True):
        return

    j = len(ctx.ebreak_vals) - 1
    n = int(ctx.kwargs.get("null_starts", n))
    results = []

    configure_point(ctx, 0, j)
    start_stat = float(ctx.dataset.stat_sum())
    cur = {p.name: float(p.value) for p in pars if not p.frozen}
    try:
        res = ctx.fit.optimize(ctx.dataset)
        if res.success and np.isfinite(ctx.dataset.stat_sum()):
            results.append((float(ctx.dataset.stat_sum()), {p.name: float(p.value) for p in pars if not p.frozen}, "stored baseline"))
    except Exception as e:
        log.warning(f"\tNull refit from stored baseline failed: {e}")

    for eb in np.geomspace(ctx.ebreak_min * 1.02, ctx.ebreak_max * 0.98, n):
        configure_point(ctx, 0, j)
        pars["ebreak"].value = float(eb)
        pars["ebreak"].frozen = True
        try:
            ctx.fit.optimize(ctx.dataset)
            pars["ebreak"].frozen = False
            res = ctx.fit.optimize(ctx.dataset)
        except Exception as e:
            pars["ebreak"].frozen = False
            log.warning(f"\tNull scan failed at ebreak = {eb:.3e}: {e}")
            continue
        stat = float(ctx.dataset.stat_sum())
        if res.success and np.isfinite(stat):
            results.append((stat, {p.name: float(p.value) for p in pars if not p.frozen}, f"ebreak start {eb:.3e}"))
    if not results:
        log.warning("Null scan found no valid fit, keeping the stored baseline")
        return

    results.sort(key = lambda x: x[0])
    log.info("Null scan (C-stat, ebreak): " + "; ".join(f"{r[0]:.2f}, {r[1].get('ebreak', np.nan):.3g}" for r in results))
    stat, best, label = results[0]
    log.info(f"Null: best C-stat = {stat:.2f} (from {label}); stored baseline C-stat at the corner point = {start_stat:.2f}")
    for name, value in best.items():
        ctx.params_org[name].value = value



# ================================================ #
# PLOTS                                            #
# ================================================ #


def plot_cstatgrid(target: str, bblock: str, ebl: str,
                   dataset: str, which: str, rcolors: bool = True, cmap: str = "viridis", plot_best: bool = False):

    # NOTE: Other nice looking colormaps: OrRd + White, Viridis_r + Black

    # Define fname and labels
    if which == "cstat":
        fname = f"upturns_cstat_{dataset}.npz"
        ptitle = r'{} C-stat Surface'.format(target)
        pcblab = r'C-statistic'
    if which == "dcstat":
        fname = f"upturns_dcstat_{dataset}.npz"
        # ptitle = r'{} $\Delta$C-stat Surface'.format(target)
        ptitle = r'{} Upturn Grid'.format(target)
        pcblab = r'$\Delta$C-statistic'

    # Load data file
    data = np.load(file = get_results_dir(target, bblock, ebl, output = "gamma-out").joinpath(fname))
    
    # Select corresponding values
    ibreak_grid = data['X']
    ebreak_grid = data['Y']
    dcstat_grid = data['Z']

    # Define colors
    colors = [cmap, "white"]
    if rcolors == True:
        colors = [cmap + "_r", "black"]
    
    # Plot grid surface
    colormesh = plt.pcolormesh(
        ibreak_grid, ebreak_grid, dcstat_grid, 
        shading = 'auto',
        cmap = colors[0],
        # Use sqrt normalization for color
        norm = clr.PowerNorm(gamma = 0.5),
        edgecolors = (colors[1], 0.15),
        linewidth = 0.001,)

    if which == "dcstat":

        # Plot contour map with confidence levels (68% - 1sigma, 90% - 2sigma, 95% - 3 sigma)
        contours = plt.contour(
            ibreak_grid, ebreak_grid, dcstat_grid,
            levels = [2.30, 4.61, 5.99], colors = colors[1], linewidths = 1.5,)

        # Get best-fit point (where DeltaC-stat = 0)
        idx_bf = np.where(dcstat_grid == 0.0)
        ibreak_bf = ibreak_grid[idx_bf]
        ebreak_bf = ebreak_grid[idx_bf]

        # Plot best fit point
        if plot_best:
            plt.scatter(ibreak_bf, ebreak_bf, label = "Best-fit parameters",
                        s = 100, marker = "*", color = colors[1], zorder = 5)

        # Plot contour labels
        plt.clabel(contours, inline = True, fontsize=10, 
                fmt={2.30: r'1$\sigma$', 4.61: r'2$\sigma$', 5.99: r'3$\sigma$'},)
    
    # Plot colorbar
    plt.colorbar(colormesh, label = pcblab)

    # Add label for dataset
    label_dataset = {
        "joint": r"$\it{Fermi}$-LAT and H.E.S.S.",
        "joint_bias": r"$\it{Fermi}$-LAT and H.E.S.S.",
        "hess": r"H.E.S.S.",
        "hess_bias": r"H.E.S.S.", }

    plt.title(label_dataset[dataset] + f"\nBlock {bblock.split("-")[0].split("block")[-1]}", 
              loc = "left", x = 0.035, y = 0.875, fontsize = 10,
              bbox = dict(facecolor = 'white', edgecolor = 'gray', boxstyle = 'round', alpha = 0.75))

    # Axis labels and formatting
    plt.xlabel(r'$\Delta \Gamma$')
    plt.ylabel(r'$E_{\mathrm{break}}$ [TeV]')

    # if "block1" not in bblock:
    #     ptitle = ptitle.split(" ")[0] + f" (Block {bblock.split("-")[0].split("block")[-1]}) Upturn Grid"
    plt.title(ptitle)
    
    plt.xscale('linear')
    plt.yscale('log')
    # Save plot
    if which == "dcstat" and plot_best:
        plt.legend()
    plt.savefig(get_results_dir(target, bblock, ebl, output = "plots").joinpath(fname.split(".")[0] + ".pdf"), 
                bbox_inches = "tight", )
    plt.close()

    return


def plot_bestfit(target: str, bblock: str, ebl: str, dataset: str):

    # Output gamma-out directory
    dir_gout = get_results_dir(target, bblock, ebl, output = "gamma-out")

    # Load data file
    data = np.load(dir_gout / f"upturns_cstat_{dataset}.npz")

    # Select corresponding values
    ibreak_grid = data['X']
    ebreak_grid = data['Y']
    dcstat_grid = data['Z']

    # Get ebreak, ibreak for lowest cstat

    # Find index of minimum
    idx = np.unravel_index(np.nanargmin(dcstat_grid), dcstat_grid.shape)

    # Extract corresponding values
    ibreak_best = ibreak_grid[idx]
    ebreak_best = ebreak_grid[idx] * u.TeV
    dcstat_min = dcstat_grid[idx]

    # Load best-fit intrinsic model for this block
    model_org = Models.read(dir_gout / f"{dataset}_models.yaml")[target]

    # Add upturn model to the total model with the best-fit parameters
    model_upturn = model_org.copy(copy_data = True)
    model_upturn.spectral_model.upturn_model = UpturnSpectralModel(indexdelta_upt = ibreak_best, e_upt = ebreak_best)

    # Save this compound model
    model_upturn_spectral = model_upturn.spectral_model
    model_upturn = SkyModel(name = f"{target} Upturn", spectral_model = model_upturn_spectral)
    Models([model_upturn]).write(path = dir_gout / "upturn_bestfit.yaml", overwrite = True, )
    
    # Define energy range
    if dataset == "joint" or dataset == "joint_bias":
        energy_range = [1e-3, 31.6] * u.TeV
    else:
        energy_range = [0.1, 31.6] * u.TeV

    # Plot best-fit upturn model
    ax = model_upturn_spectral.plot(energy_range, sed_type = "e2dnde",
                                    color = "tab:orange", label = "Upturn Model")
    model_upturn_spectral.plot_error(energy_range, ax = ax, color = "tab:orange", sed_type = "e2dnde")

    # Plot non-upturned model to compare
    model_org.spectral_model.plot(energy_range, ax = ax, sed_type = "e2dnde", 
                              color = "tab:blue", label = "Best-fit Model", linestyle = "--")
    model_org.spectral_model.plot_error(energy_range, ax = ax, sed_type = "e2dnde", color = "tab:blue")

    # Plot non-upturned flux points
    plot_sed_gammapy(target, bblock, ebl, inst = dataset, ax = ax,
                     plot_model = False, plot_fluxp = True, 
                     color = "tab:blue", label = None,
                     save_plot = False)

    ax.set_ylim(1e-16, 1e-8)

    plt.legend()
    # Save plot
    plt.tight_layout()
    plt.savefig(dir_gout / "upturn_bestfit.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    return


# ================================================ #
# MAIN                                             #
# ================================================ #


def main():

    global log

    # Arguments for script
    parser = argparse.ArgumentParser(description = "Run upturn search (C-stat grid over upturn break energy and index change)")

    parser.add_argument("--source", required = True, help = "Source name (e.g. 1ES0347-121, or all)")

    parser.add_argument("--bblock", required = True, 
                    help = "Which Bayesian block to consider (name of subfolder, for analyzing time selection blocks or different configs)")
    parser.add_argument("--ebl", required = True, 
                        help = "EBL absorption model to use (loaded from EBLTable). Default: dominguez")

    # TODO: REMOVE THIS! JUST SPECTRAL, BUT KEEP OPTION FROM OLD SCRIPTS FOR HESS, FLAT, ETC
    parser.add_argument("--dataset", choices = ["hess", "hess_bias", "joint", "joint_bias"], default = None, help = "Which dataset and model to use for upturn search. Options: {'hess', 'joint'}. Default: None (select based on available instruments)")

    parser.add_argument("--npoints", type = int, default = 20, help = "Number of point to take for each axis in grid search (E_break and DeltaIndex_break)")

    parser.add_argument("--plots-only", action = "store_true", help = "Run only generation of plots from files")
    parser.add_argument("--tau-limit", type = float, default = 1.0, help = "Minimum energy limit for grid search corresponding to optical depth value. Default: 1")

    parser.add_argument("--polish", action = "store_true",
                        help = "Do not run the grid: load the saved one and only re-fit bad or missing points (then save and re-plot)")
    
    parser.add_argument("--no-polish", action = "store_true", help = "Skip the automatic polish pass after a full grid run")
    parser.add_argument("--polish-rounds", type = int, default = 3, help = "Maximum number of polish passes over the grid")
    parser.add_argument("--refit", action = "append", default = [], metavar = '"DELTA EBREAK"',
                        help = 'Force a re-fit of the grid point nearest to this (DeltaGamma, E_break [TeV]), e.g. --refit="-4.21 0.238". Can be repeated. Implies --polish')
   
    parser.add_argument("--jobs", type = int, default = 1,
                        help = "Number of parallel processes for the grid (one E_break column per task, each process loads its own copy of the datasets). Default: 1")
    
    parser.add_argument("--fast", action = "store_true",
                        help = "Single fit per grid point (no cross-check between starts); the polish pass then verifies / fixes the grid")
    
    parser.add_argument("--comp-start", action = argparse.BooleanOptionalAction, default = False,
                        help = "Use the upturn-compensated start (intrinsic x upturn matched to the baseline spectrum) as an extra starting point for each grid point. Default: on. Disable with --no-comp-start")
    
    parser.add_argument("--seed", type = int, default = 12345, help = "Seed for the random restarts (reproducible results)")

    parser.add_argument("--kwargs", nargs = '*', help = "Additional keyword arguments ('key=value')")
    args = parser.parse_args()

    # Get the name of target source
    target = args.source

    # Parse keyword arguments (if given)
    kwargs = {}
    if args.kwargs:
        kwargs = parse_kwargs(args.kwargs)

    # TODO: FIX THIS!
    args.dataset = "joint"

    # Run only plots
    if args.plots_only:
        plot_cstatgrid(target, args.bblock, args.ebl, dataset = args.dataset, which = "cstat")
        plot_cstatgrid(target, args.bblock, args.ebl, dataset = args.dataset, which = "dcstat")
        plot_bestfit(target, args.bblock, args.ebl, dataset = args.dataset)
        exit()

    # --refit implies --polish
    args.polish = args.polish or bool(args.refit)

    # Initialize logging functions (module-level `log` is re-bound so that the helper functions log to the same place)
    # Separate log file for polish runs, so the log of the original grid is not overwritten
    log = init_log(target, fname = "upturn_polish.log" if args.polish else "upturn_modeling.log",
                   bblock = args.bblock, ebl = args.ebl)

    ctx = prepare(args, kwargs)

    # ------------------ #
    # RUN OR LOAD GRID   #
    # ------------------ #

    if args.polish:
        cstat_grid, params = load_grid(ctx)
    else:
        ctx.quick = args.fast
        if args.jobs > 1:
            cstat_grid, params = run_grid_parallel(ctx, args.jobs)
        else:
            cstat_grid, params = run_grid(ctx)
        # Polish pass always uses the full multi-start
        ctx.quick = False

    # ------------------ #
    # POLISH PASS        #
    # ------------------ #

    if args.polish or not args.no_polish:
        force = []
        for s in args.refit:
            d, e = (float(x) for x in s.replace(",", " ").split())
            point = nearest_point(ctx, d, e)
            log.info(f"Forcing re-fit of grid point {point}: Gamma2 = {ctx.ibreak_vals[point[0]]:.2f}, Ebreak2 = {ctx.ebreak_vals[point[1]]:.2e}")
            force.append(point)

        log.info("Polishing grid...")
        polish_grid(cstat_grid, params,
                    evaluate = lambda i, j, values: evaluate_point(ctx, i, j, values),
                    solve = lambda i, j, warm: solve_point(ctx, i, j, warm),
                    force = force, max_rounds = args.polish_rounds,
                    on_update = lambda: save_grid(ctx, cstat_grid, params))

    n_nan = int(np.isnan(cstat_grid).sum())
    if n_nan > 0:
        log.warning(f"{n_nan} grid point(s) have no valid fit (NaN)")

    log.info(f"Saving files to {ctx.dir_gout}")
    save_grid(ctx, cstat_grid, params)

    # Generate plots
    log.info(f"Generating and saving plots...")
    plot_cstatgrid(target, args.bblock, args.ebl, dataset = args.dataset, which = "cstat")
    plot_cstatgrid(target, args.bblock, args.ebl, dataset = args.dataset, which = "dcstat")
    plot_bestfit(target, args.bblock, args.ebl, dataset = args.dataset)

    log.info(f"Upturn modeling complete! :)")

    return


if __name__ == "__main__":

    main()