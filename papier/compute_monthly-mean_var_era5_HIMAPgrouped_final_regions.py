import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import ast
import domain
from domain import *
from climbas import *
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import dask
from scipy import stats

# Arguments
iyear = int(sys.argv[1])
fyear = int(sys.argv[2])
var   = sys.argv[3]

# ============================================ regions & file of var ===============================================

domains = ["Tien Shan", "Pamir Alay", "Pamir", "Hindu Kush", "Karakoram", "Kunlun", "Spiti Lahaul", "Central Himalaya", "Bhutan", "Nyainqentangla", "Inner TP"]

path = '/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/'

files = {'t2m': 't2m_monthly_era5_1940-2025_NH_025x025.nc',
        'FLH': 'FLH_monthly_era5_1940-2025_MA_025x025.nc',
        'tcwv': 'tcwv_monthly_era5_1940-2025_NH_025x025.nc',
        'ivt': 'ivt_monthly_era5_1940-2025_MA_025x025.nc',
        'vimd': 'vimd_monthly_era5_1940-2025_NH_025x025.nc',
        'tp': 'tp_MA_1940-2025_ERA5.nc',
        'sf': 'sf_MA_1940-2025_ERA5.nc',
        'rf': 'rf_MA_1940-2025_ERA5.nc'}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    file = files[var]


# ============================================ data ===============================================

data = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    
    data_var = xr.open_dataset(path + file)[var]
    if var in ['t2m', 'tcwv', 'ivt', 'vimd']:
        data_var = data_var.rename({'valid_time': 'time','latitude':'lat','longitude':'lon'})
    data_var = data_var.sel(time=slice(str(iyear),str(fyear)))
    data_var = field_dom(data_var, 'MA')
        
    for dom in domains:
        data[dom] = himap_grouped2(data_var, dom)


elif var == 'R':
    data['sf'] = {}
    data['tp'] = {}
    
    sf = xr.open_dataset(path + files['sf'])['sf']
    tp = xr.open_dataset(path + files['tp'])['tp']
    
    sf = sf.sel(time=slice(str(iyear),str(fyear)))
    tp = tp.sel(time=slice(str(iyear),str(fyear)))

    sf = field_dom(sf, 'MA')
    tp = field_dom(tp, 'MA')

    for dom in domains:
        data['sf'][dom] = himap_grouped2(sf, dom)
        data['tp'][dom] = himap_grouped2(tp, dom)
    
# ============================================ MONTHLY MEAN ===============================================

data_monthly = {}

data_monthly[var] = {}

if var in ['t2m', 'FLH', 'tcwv', 'ivt', 'vimd', 'tp', 'sf', 'rf']:

    for dom in domains:
        monthly_mean = data[dom].groupby('time.month').mean(dim='time', skipna=True)
        monthly_mean = monthly_mean.mean(dim=['lat', 'lon'], skipna=True)
        data_monthly[var][dom] = monthly_mean

elif var == 'R':

    data_monthly[var]['sf'] = {}
    data_monthly[var]['tp'] = {}
    data_monthly[var]['R'] = {}

    for dom in domains:

        # ================= MEAN =================
        sf_mean = data['sf'][dom].groupby('time.month').mean(dim='time', skipna=True)
        tp_mean = data['tp'][dom].groupby('time.month').mean(dim='time', skipna=True)

        sf_mean = sf_mean.mean(dim=['lat', 'lon'], skipna=True)
        tp_mean = tp_mean.mean(dim=['lat', 'lon'], skipna=True)

        data_monthly[var]['sf'][dom] = sf_mean
        data_monthly[var]['tp'][dom] = tp_mean

        tp_safe = tp_mean.where(tp_mean != 0)
        data_monthly[var]['R'][dom] = sf_mean / tp_safe * 100


# ============================================ SAVING FILE ===============================================

name_months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

rows = []

for dom in domains:

    if var in ['t2m', 'FLH', 'tcwv', 'ivt', 'vimd', 'tp', 'sf', 'rf']:
        da = data_monthly[var][dom]
    elif var == 'R':
        da = data_monthly[var]['R'][dom]
    values = da.values if hasattr(da, "values") else np.array(da)
    row = {"region": dom}
    for i, m in enumerate(name_months):
        row[m] = values[i]
    rows.append(row)

df = pd.DataFrame(rows)

path = '/bettik/PROJECTS/pr-regional-climate/bricej/paper/'
filename = f"{var}_era5_monthly_mean_HIMAPgrouped_regions_{iyear}-{fyear}.csv"

df.to_csv(path + filename, index=False)

print(f"Saved File: {path}{filename}")