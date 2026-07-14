Folder containing all input data for the thesis analysis. 

Below is a diagram of the folder structure required for all the scripts to execute properly. The most relevant folders and files are:

- `fermipy-data/` - Contains the isotropic spectral template, and Galactic interstellar emission models obtained directly from [LAT Background Models website](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html) as well as the 4FGL catalog obtained from [LAT Catalog Data Products](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/).

- `gammapy-data/2.0/` - Data from the GammaPy library downloaded for the tutorials. Can also be obtained from the GitHub Repository [GammaPy Data](https://github.com/gammapy/gammapy-data)

- `flat-data` - Fermi-LAT data obtained directly from the [LAT Data Server](https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi). Organized in sub-folders for each source, with data downloaded and formatted using the bash script `get_files.sh`. Each source must have an `events_list.txt` pointing to a list of photon files `*_PH*.fits`, as well as a `spacecraft.fits` file. All files are obtained with the same search parameters.
NOTE: An overall `spacecraft.fits` file in the main directory contains spacecraft data generated from 2008 to 2024 (containing the full observation time period), centered at 1ES0229+200, 15 deg search radius, between 100 MeV and 1 TeV

- `hess-data` - H.E.S.S. data obtained directly from the collaboration, following the `HAP FITS Export` procedure and its file structure. Organized in sub-folders for each source, which in turn are organized depending on the era the data is taken from as `out_hess*`. The most relevant files for each source are the `fits` files for all the runs, as well as the HDU and OBS index files.