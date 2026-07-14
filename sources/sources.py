import matplotlib.pyplot as plt
import astropy
import yaml
import numpy as np

from adjustText import adjust_text

from astropy.coordinates import SkyCoord
import astropy.units as u

import matplotlib.colors as mcolors

# Define plot
fig, ax = plt.subplots(subplot_kw = {"projection": "mollweide"})

# Array containing names of sources
texts = []

# Open sources file
with open('./sources.yaml', 'r') as f:

    # Load the yaml file
    data = yaml.full_load(f)

    # Array containing the info for each source
    points = []

    for k, source in enumerate( data["sources"] ):

        if source in ["GRB180720B", "PKS0625-354"]:
            continue

        # Define sky coordinate object
        gal = SkyCoord(
            data["sources"][source]["ra"],
            data["sources"][source]["dec"],
            frame = 'galactic',
            unit = u.deg
        )

        # Append data to array (now including redshift)
        points.append( [source, gal.l.wrap_at('180d').radian, gal.b.radian, data["sources"][source]["z"] ] )

        # Append text annotations to list
        texts.append( ax.text( 
            x = gal.l.wrap_at('180d').radian,
            y = gal.b.radian,
            s = data["sources"][source]["target"],
            ha = "center",
            va = "center",
            fontsize = 6,
            bbox = dict(boxstyle = 'round', alpha = 0.75, color = "white"),
            zorder = 4,
        ) )

# Convert to arrays for convenience
s_vals  = [p[0] for p in points]
x_vals  = np.array([p[1] for p in points])
y_vals  = np.array([p[2] for p in points])
z_vals  = np.array([p[3] for p in points])

# Choose a colormap and normalisation
cmap = mcolors.LinearSegmentedColormap.from_list("blue_red", ["#4575b4", "#d73027"])
# cmap = plt.cm.coolwarm
norm = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())

# Single scatter call, coloured by redshift
sc = ax.scatter(x_vals, y_vals,
                c      = z_vals,
                cmap   = cmap,
                norm   = norm,
                marker = ".",
                zorder = 5)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                    pad = 0.05, fraction = 0.025, shrink = 1)
cbar.set_label("Redshift $z$", fontsize = 8)
cbar.ax.tick_params(labelsize = 6)

# Create the galactic plane in galactic coordinates
l_plane = np.linspace(0, 360, 500) * u.deg
b_plane = np.zeros(500) * u.deg

# Convert to equatorial coordinates (RA/Dec)
galactic_plane = SkyCoord(l=l_plane, b=b_plane, frame='galactic')
ra_plane = galactic_plane.icrs.ra.wrap_at('180d').radian
dec_plane = galactic_plane.icrs.dec.radian
# Annotate galactic plane
ax.text(0.46, 0.85, "Galactic Plane", transform = ax.transAxes, fontsize = 6,
        verticalalignment = 'top', bbox = dict(boxstyle = 'round', alpha = 0.75, color = "white"), 
        color = "#C08283", alpha = 1)

# Plot galactic plane (separate into two to not join lines)
ax.plot(ra_plane[:411], dec_plane[:411],
        color = "red", alpha = 0.25, label = "Galactic Plane", zorder = 3)
ax.plot(ra_plane[412:], dec_plane[412:],
        color = "red", alpha = 0.25, zorder = 3)

# Set visible grid below the points
ax.grid(visible = True, linestyle = ":", zorder = 2)
ax.set_axisbelow(True)
ax.set_xlabel(r"RA [deg]")
ax.set_ylabel(r"DEC [deg]")

# Run automatic text adjustment
adjust_text(texts, expand=(0.5, 1.85))

plt.xticks(fontsize=6, alpha=0.5)
plt.yticks(fontsize=6, alpha=0.5)

# plt.title("Sources Map")
plt.savefig("./sources.pdf", bbox_inches="tight")