import  os
import  sys
import  logging

from    alpsup.paths    import  get_results_dir


def init_log(target: str,  
             bblock: str = "baseline",
             ebl: str | None = None ,     
             fname: str = "log.log", ) -> logging.Logger:
    """
    Initialize logging functionality (global for consistency across scripts)

    Parameters
    ----------
    target (str)    : Name of target (for folder saving).
    fname (str)     : Filename of log, including extension. Default: log.log.
    bblock (str)    : Time/Bayesian block. Default: baseline.

    Returns
    -------
    log (`logging.Logger`)  : Logger instance.
    """

    # Log file path saved to bblock subfolder of target with given file name
    dir_log = get_results_dir(target, bblock, ebl, output = "logs")
    # Create directory if it doesn't exist
    os.makedirs(name = dir_log, exist_ok = True)

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
                logging.FileHandler(filename = str(dir_log.resolve()) + f"/{fname}", mode = "a"), 
                logging.StreamHandler(sys.stdout)], )
    log = logging.getLogger(__name__)

    # Return logger object
    return log

