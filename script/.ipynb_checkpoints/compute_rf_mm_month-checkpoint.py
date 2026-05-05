import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import dask

# Arguments
dom   = sys.argv[1]

#####################  ERA5 SF #####################################

ds1= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/'
                         +'sf_MA_1940-2025_ERA5.nc')['sf']
ds1=field_dom(ds1,dom)

#####################  ERA5 TP #####################################

ds2= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/'
                         +'tp_MA_1940-2025_ERA5.nc')['tp']
ds2=field_dom(ds2,dom)

#================================= RF = TP - SF =========================================

rf = ds2 - ds1

#================================= SAVING ===============================================

output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly'
    
ds = rf.to_dataset(name="rf")
ds.attrs["units"] = "mm/month"

# récupérer première et dernière année
first_year = ds.time.dt.year.min().item()
last_year = ds.time.dt.year.max().item()
    
filename = f"{output_dir}/rf_{dom}_{first_year}-{last_year}_ERA5.nc"
    
ds.to_netcdf(filename)

print(f'Saved file : {filename}')