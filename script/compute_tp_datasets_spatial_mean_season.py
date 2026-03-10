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
iyear = int(sys.argv[1])
fyear = int(sys.argv[2])
dom   = sys.argv[3]
season   = sys.argv[4]

latS,latN,lonW,lonE = coord_domain(dom)
models = ['APHRO','GPCP','CRU','CHIRPS','HARv2','ERA5']

model1='APHRO'
field1 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1951-2015_APHRO.nc')['tp']
field1=field1.sel(time=slice(str(iyear),str(fyear)))
field1=field_dom(field1,dom)

model2='GPCP'
field2 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1891-2019_GPCP.nc')['tp']
field2=field2.sel(time=slice(str(iyear),str(fyear)))
field2=field_dom(field2,dom)


model3='CRU'
field3 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1901-2020_CRU.nc')['tp']
field3=field3.sel(time=slice(str(iyear),str(fyear)))
field3=field_dom(field3,dom)


model4='CHIRPS'
field4 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1981-2025_CHIRPS.nc')['tp']
field4=field4.sel(time=slice(str(iyear),str(fyear)))
field4=field_dom(field4,dom)


model5='HARv2'
field5 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1980-2023_HARv2.nc')['tp']
field5=field5.sel(time=slice(str(iyear),str(fyear)))
def xy_sel_dom(ds,latS,latN,lonW,lonE):
    #latS,latN,lonW,lonE=coord_domain(domain)
    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.coords:
        print(dim)
        if dim in ['lat', 'latitude','LON']:
            lat_str = dim
        elif dim in ['lon', 'longitude','LAT']:
            lon_str = dim
        else:
            other_dims_str.append(dim)
    mask = (
        (latS < ds[lat_str]) & ( ds[lat_str] < latN) 
    & (lonW < ds[lon_str]) & ( ds[lon_str] < lonE)
    )
    field_dom=ds.where(mask,drop=True)
    return field_dom
field5=xy_sel_dom(field5,latS,latN,lonW,lonE)


model6='ERA5'
field6 = xr.open_dataset(f'/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1940-2025_ERA5.nc')['tp']
field6=field6.sel(time=slice(str(iyear),str(fyear)))
field6=field_dom(field6,dom)


tp_season1 = clim(field1, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)
tp_season2 = clim(field2, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)
tp_season3 = clim(field3, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)
tp_season4 = clim(field4, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)
tp_season5 = clim(field5, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)
tp_season6 = clim(field6, season=season, imon=1, iyr=iyear, fmon=12, fyr=fyear)

tp_season_dict = {
    models[0]: tp_season1,
    models[1]: tp_season2,
    models[2]: tp_season3,
    models[3]: tp_season4,
    models[4]: tp_season5,
    models[5]: tp_season6}


output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/results/climbas/'

for model, data in tp_season_dict.items():
    
    ds = data.to_dataset(name="tp")
    ds.attrs["units"] = "mm/month"
    ds.attrs["long_name"] = f"{season} mean precipitation"
    ds.attrs["description"] = f"Climatological {season} mean over {iyear}–{fyear} in {dom}"
    
    filename = f"{output_dir}/spatial_mean_tp_{season}_{dom}_{iyear}-{fyear}_{model}.nc"
    
    ds.to_netcdf(filename)

    print(f'Saved file : {filename}')