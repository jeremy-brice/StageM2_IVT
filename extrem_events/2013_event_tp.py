#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import dask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import calendar


# # Timing of event

# In[32]:


idate = '2013-10-12'
fdate = '2013-10-15'
dom='CH'


# # load tp data

# In[33]:


tp = xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/chirps/daily/chirps-v2.0.2013.days_p05_HMA.nc')
tp = tp.sel(time=slice(idate,fdate))
tp = field_dom(tp,dom)


# In[34]:


tp


# # elev data

# In[35]:


elev = xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/gmted/gmted_on_era5_grid.nc')['elevation']
elev = field_dom(elev,dom)


# In[36]:


latS,latN,lonW,lonE = coord_domain(dom)


# # Spatial Precipitation over event

# In[38]:


#==================== Color grading ============================

step = 2
vmax = 30
levels = np.arange(0, vmax + step, step)

base_cmap = plt.get_cmap("Blues", len(levels)-1)
colors = base_cmap(np.arange(base_cmap.N))
colors[0] = [1, 1, 1, 1]

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(levels, ncolors=len(colors))

#===================== Plot each day ==================

fig, axes = plt.subplots(
    1, 4,
    figsize=(16, 3),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axes = axes.flatten()

for i, ax in enumerate(axes):
    
    data_day = tp.precip.isel(time=i)
    date = str(tp.time.values[i])[:10]

    im = ax.pcolormesh(
        tp.longitude,
        tp.latitude,
        data_day,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.8
    )

    ax.set_extent([lonW, lonE, latS, latN], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=1, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.8)

    # === contours topo (si dispo) ===
    SH = ax.contour(
        elev.longitude,
        elev.latitude,
        elev,
        levels=[500,1000,2000,3000,4000,5000,6000,7000],
        colors="k",
        linewidths=0.6,
        alpha=0.2,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(SH, inline=True, fontsize=9, fmt="%d")

    ax.set_title(date, fontsize=14, weight="bold")

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.8,
        color="gray",
        alpha=0.4,
        linestyle="--"
    )

    if i==0:
        gl.left_labels = True
        gl.bottom_labels = True
        gl.top_labels = False
        gl.right_labels = False
    elif i in [1,2,3]:
        gl.left_labels = False
        gl.bottom_labels = True
        gl.top_labels = False
        gl.right_labels = False

#==================== Colorbar ============================

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend='max')
cbar.set_label("Precipitation (mm/day)", fontsize=16)

#==================== Final ============================

plt.suptitle("2013 event precipitation", fontsize=18, weight="bold")
plt.show()


# In[ ]:




