import  os
import  sys
import  logging

from    pathlib         import  Path

from    alpsup.paths    import  get_results_dir


def init_log(target: str | None = None,  
             bblock: str | None = None,
             ebl: str | None = None ,     
             fname: str = "log.log", 
             path: str | Path | None = None) -> logging.Logger:
    """
    Initialize logging functionality (global for consistency across scripts)

    Parameters
    ----------
    target (str)    : Name of target (for folder saving).
    bblock (str)    : Time/Bayesian block. Default: "baseline".
    ebl (str)       : Name of EBL model (for folder saving). Default: None.
    fname (str)     : Filename of log, including extension. Default: "log.log".
    path (str, Path): If path given, use that one instead of constructing it.

    Returns
    -------
    log (`logging.Logger`)  : Logger instance.
    """

    # Log file path saved to bblock subfolder of target with given file name
    if path is None:
        dir_log = get_results_dir(target, bblock, ebl, output = "logs")
    # If path given, use that one intead!
    else:
        dir_log = Path(path)

    # Create directory if it doesn't exist
    os.makedirs(name = dir_log, exist_ok = True)

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
                logging.FileHandler(filename = str(dir_log.resolve()) + f"/{fname}", mode = "a"), 
                logging.StreamHandler(sys.stdout)], 
        force = True, )
    log = logging.getLogger(__name__)

    # Return logger object
    return log

