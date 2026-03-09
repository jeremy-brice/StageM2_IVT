import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from scipy import stats
import scipy.stats as stats

# Arguments
iyear = int(sys.argv[1])
fyear = int(sys.argv[2])
dom   = sys.argv[3]

#===================  ANNUAL ==============================

ds=xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/tcwv_monthly_era5_1940-2025_NH_025x025.nc')
ds = ds.sel(valid_time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))

tcwv=field_dom(ds,dom)

# pour alléger le dossier 
tcwv = tcwv.drop_vars(['number','expver'], errors='ignore')  # les variables inutiles
tcwv.attrs = {}  # les atributs pas utiles pour les calculs
tcwv['tcwv'].attrs = {}  # les atributs pas utiles pour les calculs
tcwv = tcwv.load()  # charger en mémoire pour pas avoir à relire le fichier pour les calculs

tcwv_year=tcwv.groupby('valid_time.year').mean('valid_time')['tcwv']
tcwv_mean=tcwv_year.mean(dim=['latitude','longitude'])

Mean = np.mean(tcwv_mean.values)
Std = np.std(tcwv_mean.values)
Min = np.min(tcwv_mean.values)
Max = np.max(tcwv_mean.values)

# Linear trend of Annual IVT mean
slope, intercept, r_value, p_value, std_err = stats.linregress(tcwv_mean['year'].values, tcwv_mean.values)
trend = intercept + slope * tcwv_mean['year'].values

stats_series = pd.Series({
    "mean": Mean,
    "std": Std,
    "min": Min,
    "max": Max,
    "slope": slope,
    "intercept": intercept,
    "r_value": r_value,
    "p_value": p_value,
    "std_err": std_err
})
stats_series = stats_series.to_frame(name="value")
stats_series.index.name = "stat"

time_series = pd.DataFrame({
    "year": tcwv_mean['year'].values,
    "tcwv": tcwv_mean.values,
    "trend": trend
})

time_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tcwv_annual_{dom}_{iyear}-{fyear}_ERA5.csv", index=False)
stats_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tcwv_annual_{dom}_{iyear}-{fyear}_ERA5.csv")

print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tcwv_annual_{dom}_{iyear}-{fyear}_ERA5.csv')
print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tcwv_annual_{dom}_{iyear}-{fyear}_ERA5.csv')

#=================== SEASONAL =================================

seasons = {"DJFM": [12,1,2,3], 
           "AM": [4,5], 
           "JJAS": [6,7,8,9], 
           "ON": [10,11]}
start_year = iyear
end_year = fyear

#Rename time dimension for 'clim' function
tcwv = tcwv.rename({'valid_time': 'time'})

#Climatology for each season
for season in seasons:
    
    tcwv_season = seasonal_selection2(tcwv['tcwv'], seasons[season], start_year, end_year)
    tcwv_mean=tcwv_season.mean(dim=['latitude','longitude'])

    Mean = np.mean(tcwv_mean.values)
    Std = np.std(tcwv_mean.values)
    Min = np.min(tcwv_mean.values)
    Max = np.max(tcwv_mean.values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(tcwv_mean['time'].values, tcwv_mean.values)
    trend = intercept + slope * tcwv_mean['time'].values
    
    stats_series = pd.Series({
    "mean": Mean,
    "std": Std,
    "min": Min,
    "max": Max,
    "slope": slope,
    "intercept": intercept,
    "r_value": r_value,
    "p_value": p_value,
    "std_err": std_err
    })
    stats_series = stats_series.to_frame(name="value")
    stats_series.index.name = "stat"

    time_series = pd.DataFrame({
    "year": tcwv_mean['time'].values,
    "tcwv": tcwv_mean.values,
    "trend": trend
    })
    
    time_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tcwv_{season}_{dom}_{iyear}-{fyear}_ERA5.csv', index=False)
    stats_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tcwv_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')

    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tcwv_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')
    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tcwv_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')

    
