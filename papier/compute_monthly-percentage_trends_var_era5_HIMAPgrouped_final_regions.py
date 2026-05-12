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
months = ast.literal_eval(sys.argv[4]) # give type season = [1,2,3] for JFM or for one month January give [1]

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
    

# ============================================ monthly selection ===============================================

data_monthly = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:

    for dom in domains:
        monthly_selection_var = seasonal_selection2(data[dom], season_months=months, iyr=iyear, fyr=fyear)
        data_monthly[dom] = monthly_selection_var.mean(dim=['lat','lon'], skipna=True)


elif var == 'R':
    data_monthly['sf'] = {}
    data_monthly['tp'] = {}
    data_monthly['R'] = {}
    for dom in domains:
        monthly_selection_var1 = seasonal_selection2(data['sf'][dom], season_months=months, iyr=iyear, fyr=fyear)
        data_monthly['sf'][dom] = monthly_selection_var1.mean(dim=['lat','lon'], skipna=True)

        monthly_selection_var2 = seasonal_selection2(data['tp'][dom], season_months=months, iyr=iyear, fyr=fyear)
        data_monthly['tp'][dom] = monthly_selection_var2.mean(dim=['lat','lon'], skipna=True)

        tp_safe = data_monthly['tp'][dom].where(data_monthly['tp'][dom] != 0) # verification que tp different de zéro (NaN si c'est le cas)
        data_monthly['R'][dom] = data_monthly['sf'][dom] / tp_safe * 100

# ============================================ trends ===============================================

trend = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    for dom in domains:
        trend[dom] = stats.linregress(
                    data_monthly[dom].time.values,
                    data_monthly[dom].values
                )

elif var == 'R':
    for dom in domains:
        x = data_monthly['R'][dom].time.values
        y = data_monthly['R'][dom].values

        # enlever les NaN
        mask = np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]

        # minimum 5 valeurs valides pour faire une trend si non que des NaN
        if len(y_valid) >= 5:
            trend[dom] = stats.linregress(x_valid,y_valid)
        else:
            trend[dom] = np.nan

# ============================================ percentage trends ===============================================

trend_percentage = {}

if var in ['t2m','FLH','tcwv','ivt','vimd','tp','sf','rf']:
    for dom in domains:
        mean_val = data_monthly[dom].mean(skipna=True).item()

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
        mean_val = data_monthly['R'][dom].mean(skipna=True).item()

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
        "slope": res['slope'],
        "intercept": res['intercept'],
        "rvalue": res['rvalue'],
        "pvalue": res['pvalue'],
        "stderr": res['stderr']})

df = pd.DataFrame(rows)

name_months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_label = '_'.join([name_months[m-1] for m in months]) # ça donne : [1] -> 'Jan'; ou [1,2,3] -> 'Jan_Feb_Mar'

path = '/bettik/PROJECTS/pr-regional-climate/bricej/paper/'
filename = f"{var}_era5_{month_label}_percentage_trends_HIMAPgrouped_regions_{iyear}-{fyear}.csv"

df.to_csv(path + filename, index=False)

print(f'Saved File : {path}{filename}')