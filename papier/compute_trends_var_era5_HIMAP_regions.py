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

files = {'t2m': 't2m_monthly_era5_1940-2025_NH_025x025.nc',
        'FLH': 'FLH_monthly_era5_1940-2025_MA_025x025.nc',
        'tcwv': 'tcwv_monthly_era5_1940-2025_NH_025x025.nc',
        'ivt': 'ivt_monthly_era5_1940-2025_MA_025x025.nc',
        'vimd': 'vimd_monthly_era5_1940-2025_NH_025x025.nc',
        'tp': 'tp_MA_1940-2025_ERA5.nc',
        'sf': 'sf_MA_1940-2025_ERA5.nc',
        'rf': 'rf_MA_1940-2025_ERA5.nc'}

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

# ============================================ annual selection ===============================================

data_annual = {}

for dom in domains:
    data_annual[dom] = data[dom].groupby('time.year').mean(dim='time', skipna=True)
    data_annual[dom] = data_annual[dom].mean(dim=['lat','lon'], skipna=True)

# ============================================ trends ===============================================

trend = {}

for dom in domains:
    trend[dom] = stats.linregress(
                data_annual[dom].year.values,
                data_annual[dom].values
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
filename = f"{var}_era5_trends_HIMAP_regions_{iyear}-{fyear}.csv"

df.to_csv(path + filename, index=False)

print(f'Saved File : {path}{filename}')