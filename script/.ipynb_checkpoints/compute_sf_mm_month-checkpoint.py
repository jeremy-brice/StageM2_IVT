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

latS,latN,lonW,lonE = coord_domain(dom)

#####################  ERA5 SF #####################################

ds= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/'
                         +'sf_monthly_era5_1940-2025_NH_025x025.nc')['sf']
ds=ds.rename({'valid_time': 'time','latitude':'lat','longitude':'lon'})

ds=field_dom(ds,dom)

# Convert m/day to mm/month
days_in_month = ds['time'].dt.days_in_month
ds= ds * days_in_month * 1000

#================================= SAVING ===============================================

output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly'
    
ds = ds.to_dataset(name="sf")
ds.attrs["units"] = "mm w.e./month"

# récupérer première et dernière année
first_year = ds.time.dt.year.min().item()
last_year = ds.time.dt.year.max().item()
    
filename = f"{output_dir}/sf_{dom}_{first_year}-{last_year}_ERA5.nc"
    
ds.to_netcdf(filename)

print(f'Saved file : {filename}')