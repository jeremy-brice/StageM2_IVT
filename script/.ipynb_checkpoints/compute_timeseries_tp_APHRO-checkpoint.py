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

ds=xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/tp_datasets/tp_HMA_1951-2015_APHRO.nc')
ds = ds.sel(time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))

tp=field_dom(ds,dom)

tp_year=tp.groupby('time.year').mean(dim='time', skipna=True)['tp']
tp_mean=tp_year.mean(dim=['lat','lon'], skipna=True)

Mean = np.mean(tp_mean.values)
Std = np.std(tp_mean.values)
Min = np.min(tp_mean.values)
Max = np.max(tp_mean.values)

# Linear trend of Annual tp mean
slope, intercept, r_value, p_value, std_err = stats.linregress(tp_mean['year'].values, tp_mean.values)
trend = intercept + slope * tp_mean['year'].values

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
    "year": tp_mean['year'].values,
    "tp": tp_mean.values,
    "trend": trend
})

time_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tp_annual_{dom}_{iyear}-{fyear}_APHRO.csv", index=False)
stats_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tp_annual_{dom}_{iyear}-{fyear}_APHRO.csv")

print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tp_annual_{dom}_{iyear}-{fyear}_APHRO.csv')
print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tp_annual_{dom}_{iyear}-{fyear}_APHRO.csv')

#=================== SEASONAL =================================

seasons = {"DJFM": [12,1,2,3], 
           "AM": [4,5], 
           "JJAS": [6,7,8,9], 
           "ON": [10,11]}
start_year = iyear
end_year = fyear

#Climatology for each season
for season in seasons:
    
    tp_season = seasonal_selection2(tp['tp'], seasons[season], start_year, end_year)
    tp_mean=tp_season.mean(dim=['lat','lon'], skipna=True)

    Mean = np.mean(tp_mean.values)
    Std = np.std(tp_mean.values)
    Min = np.min(tp_mean.values)
    Max = np.max(tp_mean.values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(tp_mean['time'].values, tp_mean.values)
    trend = intercept + slope * tp_mean['time'].values
    
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
    "year": tp_mean['time'].values,
    "tp": tp_mean.values,
    "trend": trend
    })
    
    time_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tp_{season}_{dom}_{iyear}-{fyear}_APHRO.csv', index=False)
    stats_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tp_{season}_{dom}_{iyear}-{fyear}_APHRO.csv')

    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_tp_{season}_{dom}_{iyear}-{fyear}_APHRO.csv')
    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_tp_{season}_{dom}_{iyear}-{fyear}_APHRO.csv')

    
