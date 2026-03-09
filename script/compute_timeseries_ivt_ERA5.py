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

ds=xr.open_dataset('/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly/ivt_monthly_era5_1940-2025_MA_025x025.nc')
ds = ds.sel(valid_time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))

ivt=field_dom(ds,dom)

ivt_year=ivt.groupby('valid_time.year').mean('valid_time')['ivt']
ivt_mean=ivt_year.mean(dim=['latitude','longitude'])

Mean = np.mean(ivt_mean.values)
Std = np.std(ivt_mean.values)
Min = np.min(ivt_mean.values)
Max = np.max(ivt_mean.values)

# Linear trend of Annual IVT mean
slope, intercept, r_value, p_value, std_err = stats.linregress(ivt_mean['year'].values, ivt_mean.values)
trend = intercept + slope * ivt_mean['year'].values

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
    "year": ivt_mean['year'].values,
    "ivt": ivt_mean.values,
    "trend": trend
})

time_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_ivt_annual_{dom}_{iyear}-{fyear}_ERA5.csv", index=False)
stats_series.to_csv(f"/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_ivt_annual_{dom}_{iyear}-{fyear}_ERA5.csv")

print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_ivt_annual_{dom}_{iyear}-{fyear}_ERA5.csv')
print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_ivt_annual_{dom}_{iyear}-{fyear}_ERA5.csv')

#=================== SEASONAL =================================

seasons = {"DJFM": [12,1,2,3], 
           "AM": [4,5], 
           "JJAS": [6,7,8,9], 
           "ON": [10,11]}
start_year = iyear
end_year = fyear

#Rename time dimension for 'clim' function
ivt = ivt.rename({'valid_time': 'time'})

#Climatology for each season
for season in seasons:
    
    ivt_season = seasonal_selection2(ivt['ivt'], seasons[season], start_year, end_year)
    ivt_mean=ivt_season.mean(dim=['latitude','longitude'])

    Mean = np.mean(ivt_mean.values)
    Std = np.std(ivt_mean.values)
    Min = np.min(ivt_mean.values)
    Max = np.max(ivt_mean.values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(ivt_mean['time'].values, ivt_mean.values)
    trend = intercept + slope * ivt_mean['time'].values
    
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
    "year": ivt_mean['time'].values,
    "ivt": ivt_mean.values,
    "trend": trend
    })
    
    time_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_ivt_{season}_{dom}_{iyear}-{fyear}_ERA5.csv', index=False)
    stats_series.to_csv(f'/bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_ivt_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')

    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/timeseries_ivt_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')
    print(f'Saved : /bettik/PROJECTS/pr-regional-climate/bricej/results/timeseries/stats_ivt_{season}_{dom}_{iyear}-{fyear}_ERA5.csv')

    
