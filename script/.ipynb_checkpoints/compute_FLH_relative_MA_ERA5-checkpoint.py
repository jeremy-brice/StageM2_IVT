import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# Arguments
iyear = int(sys.argv[1])
fyear = int(sys.argv[2])
dom   = sys.argv[3]

#============================ Loading data ================================

# Temperature at each levels from ERA5
ds = xr.open_dataset("/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/FLH_monthly_era5_1940-2025_MA_025x025.nc")

# Geopotential Height from ERA5
elev = xr.open_dataset("/bettik/PROJECTS/pr-regional-climate/bricej/gmted/gmted_on_era5_MA.nc")
elev = elev.rename({'latitude':'lat','longitude':'lon'})

#Period & Domain Def (maximum is MA)
ds = ds.sel(time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))

ds = field_dom(ds, dom)
elev = field_dom(elev, dom)

# Trier les coordonnées
# xarray interp exige des coordonnées monotoniques croissantes
ds = ds.sortby("lat")
ds = ds.sortby("lon")

elev = elev.sortby("lat")
elev = elev.sortby("lon")

# Interpoler elevation sur la grille FLH
elev_interp = elev["elevation"].interp(
    lat=ds["lat"],
    lon=ds["lon"],
    method="linear"
)

# 5. Calcul du FLH relatif
flh_relative = ds["FLH"] - elev_interp

#========================== Saving Files ==================================

#Saving to dataset
output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly'

ds = flh_relative.to_dataset(name="FLHr")
ds.attrs["units"] = "m"

filename = f"{output_dir}/FLH_relative_monthly_era5_{iyear}-{fyear}_{dom}_025x025.nc"
    
ds.to_netcdf(filename)
print(f'Saved file : {filename}')
