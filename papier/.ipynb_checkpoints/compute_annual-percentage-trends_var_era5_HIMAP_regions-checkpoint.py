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

domains = ['Eastern Hindu Kush', 
            'Western Himalaya', 
            'Eastern Himalaya',
            'Central Himalaya', 
            'Karakoram', 
            'Western Pamir', 
            'Pamir Alay', 
            'Northern/Western Tien Shan', 
            'Dzhungarsky Alatau', 
            'Western Kunlun Shan', 
            'Nyainqentanglha', 
            'Gangdise Mountains', 
            'Hengduan Shan', 
            'Tibetan Interior Mountains', 
            'Tanggula Shan', 
            'Eastern Tibetan Mountains', 
            'Qilian Shan', 
            'Eastern Kunlun Shan', 
            'Altun Shan', 
            'Eastern Tien Shan', 
            'Central Tien Shan', 
            'Eastern Pamir']

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
    data_var = field_dom(data_var, 'HMA')
        
    for dom in domains:
        data[dom] = himap(data_var, dom)


elif var == 'R':
    data['sf'] = {}
    data['tp'] = {}
    
    sf = xr.open_dataset(path + files['sf'])['sf']
    tp = xr.open_dataset(path + files['tp'])['tp']
    
    sf = sf.sel(time=slice(str(iyear),str(fyear)))
    tp = tp.sel(time=slice(str(iyear),str(fyear)))

    sf = field_dom(sf, 'HMA')
    tp = field_dom(tp, 'HMA')

    for dom in domains:
        data['sf'][dom] = himap(sf, dom)
        data['tp'][dom] = himap(tp, dom)
    

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

# ============================================ percentage trends ===============================================

trend_percentage = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    for dom in domains:
        mean_val = data_annual[dom].mean(skipna=True).item()

        if mean_val != 0: # vérification si la mean value est diffférent de zéro
            slope_pct = (trend[dom].slope / mean_val) * 100
            stderr_pct = (trend[dom].stderr / mean_val) * 100

        else: # mettre NaN si mean value est égal à zéro
            slope_pct = np.nan
            stderr_pct = np.nan

        trend_percentage[dom] = {
            'slope': slope_pct,
            'intercept': trend[dom].intercept,
            'rvalue': trend[dom].rvalue,
            'pvalue': trend[dom].pvalue,
            'stderr': stderr_pct
        }

elif var == 'R':
    for dom in domains:
        mean_val = data_annual['R'][dom].mean(skipna=True).item()

        if mean_val != 0:
            slope_pct = (trend[dom].slope / mean_val) * 100
            stderr_pct = (trend[dom].stderr / mean_val) * 100

        else:
            slope_pct = np.nan
            stderr_pct = np.nan

        trend_percentage[dom] = {
            'slope': slope_pct,
            'intercept': trend[dom].intercept,
            'rvalue': trend[dom].rvalue,
            'pvalue': trend[dom].pvalue,
            'stderr': stderr_pct
        }

# ============================================ saving file ===============================================

rows = []

for region, res in trend_percentage.items():

    rows.append({
    "region": region,
    "slope": res['slope'],
    "intercept": res['intercept'],
    "rvalue": res['rvalue'],
    "pvalue": res['pvalue'],
    "stderr": res['stderr']})

df = pd.DataFrame(rows)

path = '/bettik/PROJECTS/pr-regional-climate/bricej/paper/'
filename = f"{var}_era5_percentage_trends_HIMAP_regions_{iyear}-{fyear}.csv"

df.to_csv(path + filename, index=False)

print(f'Saved File : {path}{filename}')