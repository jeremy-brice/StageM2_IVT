import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import ast
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
season = ast.literal_eval(sys.argv[4]) # give type season = [1,2,3] for JFM

latS,latN,lonW,lonE = coord_domain(dom)
models = ['APHRO','GPCP','CRU','CHIRPS','HARv2','ERA5']

# Convertir season = [] en str pour le saving

def months_to_season(months):
    month_letters = {
        1:"J",2:"F",3:"M",4:"A",5:"M",6:"J",
        7:"J",8:"A",9:"S",10:"O",11:"N",12:"D"
    }
    return "".join(month_letters[m] for m in months)  # "".join(["J","F","M"]) --> 'JFM'

season_name = months_to_season(season)


#Data loading for iyear-fyear and in dom

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


#======================= TRENDS CALCULATION ====================================

tp_annual1 = seasonal_selection2(field1, season, iyear, fyear)
par1 = trend_vect(tp_annual1.time,tp_annual1,dim='time')

tp_annual2 = seasonal_selection2(field2, season, iyear, fyear)
par2 = trend_vect(tp_annual2.time,tp_annual2,dim='time')

tp_annual3 = seasonal_selection2(field3, season, iyear, fyear)
par3 = trend_vect(tp_annual3.time,tp_annual3,dim='time')

tp_annual4 = seasonal_selection2(field4, season, iyear, fyear)
par4 = trend_vect(tp_annual4.time,tp_annual4,dim='time')

tp_annual5 = seasonal_selection2(field5, season, iyear, fyear)
par5 = trend_vect(tp_annual5.time,tp_annual5,dim='time')

tp_annual6 = seasonal_selection2(field6, season, iyear, fyear)
par6 = trend_vect(tp_annual6.time,tp_annual6,dim='time')


trend_dict = {
    models[0]: par1,
    models[1]: par2,
    models[2]: par3,
    models[3]: par4,
    models[4]: par5,
    models[5]: par6}


#========================== SAVING ==========================================

output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/results/climbas'
var_names = ['trend', 'intercept', 'rvalue', 'pvalue', 'stderr']

for model, data in trend_dict.items():
    
    ds = xr.Dataset({name: da for name, da in zip(var_names, data)})
    ds.attrs["description"] = f"tp {season_name} linregress over {iyear}–{fyear} in {dom}"
        
    filename = f"{output_dir}/spatial_trend_tp_{season_name}_{dom}_{iyear}-{fyear}_{model}.nc"
        
    ds.to_netcdf(filename)
    
    print(f'Saved file : {filename}')