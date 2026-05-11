import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import dask

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
        data[dom] = himap_grouped(data_var, dom)


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
        data['sf'][dom] = himap_grouped(sf, dom)
        data['tp'][dom] = himap_grouped(tp, dom)
    

# ============================================ annual selection ===============================================

data_annual = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:

    for dom in domains:
        data_annual[dom] = data[dom].groupby('time.year').mean(dim='time', skipna=True)
        data_annual[dom] = data_annual[dom].mean(dim=['lat','lon'], skipna=True)


elif var == 'R':
    data_annual['sf'] = {}
    data_annual['tp'] = {}
    data_annual['R'] = {}
    for dom in domains:
        data_annual['sf'][dom] = data['sf'][dom].groupby('time.year').mean(dim='time', skipna=True)
        data_annual['sf'][dom] = data_annual['sf'][dom].mean(dim=['lat','lon'], skipna=True)

        data_annual['tp'][dom] = data['tp'][dom].groupby('time.year').mean(dim='time', skipna=True)
        data_annual['tp'][dom] = data_annual['tp'][dom].mean(dim=['lat','lon'], skipna=True)

        data_annual['R'][dom] = data_annual['sf'][dom] / data_annual['tp'][dom] * 100

# ============================================ trends ===============================================

trend = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    for dom in domains:
        trend[dom] = stats.linregress(
                    data_annual[dom].year.values,
                    data_annual[dom].values
                )

elif var == 'R':
    for dom in domains:
        trend[dom] = stats.linregress(
                    data_annual['R'][dom].year.values,
                    data_annual['R'][dom].values
                )
        
# ============================================ saving file ===============================================

rows = []

for region, res in trend.items():
    if isinstance(res, float) and np.isnan(res):
        # cas où NaN (pas assez de données pour avoir une trend)
        rows.append({
            "region": region,
            "slope": np.nan,
            "intercept": np.nan,
            "rvalue": np.nan,
            "pvalue": np.nan,
            "stderr": np.nan
        })
    else:
        rows.append({
            "region": region,
            "slope": res.slope,
            "intercept": res.intercept,
            "rvalue": res.rvalue,
            "pvalue": res.pvalue,
            "stderr": res.stderr
        })

df = pd.DataFrame(rows)

path = '/bettik/PROJECTS/pr-regional-climate/bricej/paper/'
filename = f"{var}_era5_trends_HIMAPgrouped_regions_{iyear}-{fyear}.csv"

df.to_csv(path + filename, index=False)

print(f'Saved File : {path}{filename}')