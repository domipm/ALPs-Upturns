!/bin/bash

# List of URLs to download (Copied from Fermi-LAT Data Server)
wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/QUERYID_PH00.fits
wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/QUERYID_PH01.fits
# ...
wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/QUERYID_SC00.fits

# Debug info
echo "All files downloaded."

# Generate list of all events (ensure to append ./data for fermipy compatibility)
ls -d "$PWD"/*_PH*.fits > events_list.txt

# Rename spacecraft file
mv L*_SC00.fits spacecraft.fits

# TODO: IMPROVE GLOBAL PATH, MAKE DIRECTORY ABOVE