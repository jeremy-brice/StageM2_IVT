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
import calendar

#=================  Period & Domain ========================

iyear=1981
fyear=2015
dom='HMA'
latS,latN,lonW,lonE,latlim,lonlim = coord_domain('HMA')
models = ['APHRO','GPCP','CRU','HARv2','ERA5']

#=================  Saving netCDF files ========================

output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/climbas/'

#=================  tp Data loading ========================

#####################  APHRO  #####################################

model1='aphro'
res1=str('0.25°x0.25°')
ds1 = xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/aphro/monthly/APHRO_mon_MA_025deg_V1101_EXR1.1951-2015.nc'
                      ,decode_times=False)['precip']
start_date = pd.Timestamp('1951-01-01')
# Create new time index
nmonths = ds1.time.size
time_new = pd.date_range(start=start_date, periods=nmonths, freq='MS')  # 'MS' = Month Start
# Replace time in dataset
ds1['time'] = time_new
ds1=reversing_lat(ds1)
ds1=ds1.rename({'latitude':'lat','longitude':'lon'})
field1_mm_day=field_dom(ds1,dom)
field1_mm_day=field1_mm_day.sel(time=slice(str(iyear),str(fyear)))

# Convert mm/day to mm/month
days_in_month = field1_mm_day['time'].dt.days_in_month
field1= field1_mm_day * days_in_month

#####################  GPCP  #####################################

model2='gpcp'
res2=str('1.0°x1.0°')
ds2= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/gpcp/precip_mon.gpcp_v2020_1891_2019_10.nc')['precip']

field2=field_dom(ds2,dom)
field2=field2.sel(time=slice(str(iyear),str(fyear)))

#####################  CRU  #####################################

model3='cru'
res3=str('0.5°x0.5°')

ds3_raw= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/cru/cru_ts4.05.1901.2020.pre.dat.nc')['pre']
ds3=reversing_lat(ds3_raw)
field3=field_dom(ds3,dom)
field3=field3.sel(time=slice(str(iyear),str(fyear)))

#####################  CHIRPS  #####################################

#model4='chirps'
#res4='0.05°x0.05°'
#
#ds4 = xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/chirps/chirps-v2.0.monthly.nc')['precip']
#ds4=ds4.rename({'latitude':'lat','longitude':'lon'})
#ds4=reversing_lat(ds4)
#field4=field_dom(ds4,dom)
#field4=field4.sel(time=slice(str(iyear),str(fyear)))

#####################  HARv2  #####################################

model5='HARv2'
res5='10kmx10km'

ds5_raw0= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/HARv2/HARv2_mon_d10km_d_2d_prcp_1980-2023.nc')['prcp']
ds5_raw=ds5_raw0*24 #mm/h
# Calculate days in each month
days_in_month5= ds5_raw['time'].dt.days_in_month
# Convert to mm/month
ds5= ds5_raw* days_in_month5

def xy_sel_dom(ds,latS,latN,lonW,lonE):
    #latS,latN,lonW,lonE,latlim,lonlim=coord_domain(domain)
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

# Apply mask (non-NaN where the condition is True)
field5=xy_sel_dom(ds5.sel(time=slice(str(iyear),str(fyear))),latS,latN,lonW,lonE)

#####################  ERA5  #####################################

model6='era5'
res6=str('0.25°x0.25°')

ds6_raw= xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/santolam/era5/'
                         +'tp.mon_era5_1940-2025_NH_025x025.nc')['tp'] * 1000
ds6_raw=ds6_raw * ds6_raw['valid_time'].dt.days_in_month  # convert m/day to mm/month
ds6=ds6_raw.rename({'valid_time': 'time','latitude':'lat','longitude':'lon'})
ds6=ds6.sel(time=slice(str(iyear),str(fyear)))

field6=field_dom(ds6,dom)

#=================  Annual Climatology ========================

tp_mean1 = field1.mean(dim='time', skipna=True)
tp_mean2 = field2.mean(dim='time', skipna=True)
tp_mean3 = field3.mean(dim='time', skipna=True)
#tp_mean4 = field4.mean(dim='time', skipna=True)
tp_mean5 = field5.mean(dim='time', skipna=True)
tp_mean6 = field6.mean(dim='time', skipna=True)

tp_mean_dict = {
    models[0]: tp_mean1,
    models[1]: tp_mean2,
    models[2]: tp_mean3,
    models[3]: tp_mean5,
    models[4]: tp_mean6}

#Save

for model, data in tp_mean_dict.items():
    
    ds = data.to_dataset(name="tp")
    ds.attrs["units"] = "mm/month"
    ds.attrs["long_name"] = "Annual mean precipitation"
    ds.attrs["description"] = "Climatological annual mean over 1981–2015 in HMA"
    
    filename = f"{output_dir}/spatial_mean_tp_annual_{dom}_{iyear}-{fyear}_{model}.nc"
    
    ds.to_netcdf(filename)

    print(f'Saved file : {filename}')




