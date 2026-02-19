import xarray as xr
import numpy as np
from scipy import stats
import scipy.stats as stats
import pandas as pd

def latlon(ds):
    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.dims:
        if dim in ['lat', 'latitude']:
            lat_str = dim
        elif dim in ['lon', 'longitude']:
            lon_str = dim
        else:
            other_dims_str.append(dim)
    return lat_str,lon_str

def clim(ds,season='annual',imon=1,iyr=1979,fmon=12,fyr=2005):
    """
        Compute the climatology from monthly data. For seasonal climatology,
        it is possible to shift the start and the end of you
        period in order to select the full winter season (e.g. Dec 1979, Jan 1980, Feb 1980) 
        or select individual winter months
        without removing any data (e.g. for 'DJF' it will take into
        account the first D and last JF).
        The time dimension needs to be named
        'time'.
        Parameters
        ----------
        ds : xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
            Monthly data to process.
        
        season : int, str, optional
            Season on wchich to compute the climatology. Default is 'annual'.
            Options are:
            - 'annual'
            - single month: int (ex: 1 for January, 2 for February, etc.)
            - any character string (ex: 'DJF', 'JJAS', etc.)
        imon_obs,fmon_obs,iyr_obs,fyr_obs : int,optional
        fyr_obs is set to 2005 in order to avoid erros when data extension is short
        
          Returns
        -------
        clim : xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
            Weighted climatology.
        -------------------------------------
         Example
        -------
         import xarray as xr
         import sys
         sys.path.insert(1, '/home/maria/Documents/MyPythonLibrary/')
         import climbasis as climb
         da = xr.open_dataarray(...)
         clim = climb.clim(da, season='annual', imon=1,iyr=1979,fmon=12,fyr=2005)
         vals=field0.where(field0['time.month'].isin([12,1,2]).resample(time='AS-Dec')).mean('time') 
    """
    field0=ds.sel(time=slice(str(iyr)+"-"+str(imon), str(fyr)+"-"+str(fmon)))
    month = field0['time.month']
    
    if isinstance(season, int):
        season_sel = (month == season)
        print('Use monthly_selection(ds,mon,iyr,fyr)')
        #print(season_sel)
    elif isinstance(season, str) and len(season) > 1:
        season_str = 'JFMAMJJASONDJFMAMJJASOND'
        #print(season_str.index(season)) ## Index gives the position of the season letter in the season str; then we advance
        #print(len(season))
        month_start = season_str.index(season) + 1
        month_end = month_start + len(season) - 1
        #print(month_start)
        #print(month_end)
        if month_end > 12:
            month_end -= 12  #x-=12 equivalent x=x*12 (multiple)
            season_sel = (month >= month_start) | (month <= month_end)
            #print('Check monthly/seasonal selection')
            #print(season_sel)
        else:
            season_sel = (month >= month_start) & (month <= month_end)
            #print('Check monthly/seasonal selection')
            #print(season_sel)

    else:
        raise ValueError(
                    f"""Invalid season argument: '{season}'. Valid values are:
                - 'annual'"
                - single month: int (ex: 1 for January, 2 for February, etc.)
                - any character string (ex: 'DJF', 'JJAS', etc.)")
                """
                )
    clim=field0.sel(time=season_sel).mean('time')
    #clim.attrs['period'] = str(field0[0]['time.year'].values) + '-' + str(ds[-1]['time.year'].values)
    return clim

def trend_vect(x,y,dim):
    '''
    Compute the spatial trend vectorized, instead of grid by grid.
    ex:
    par=trend_vect(vals.time,vals,'time')

    Source:https://stackoverflow.com/questions/52094320/with-xarray-how-to-parallelize-1d-operations-on-a-multidimensional-dataset
    https://github.com/mickaellalande/MC-Toolkit/blob/master/conda_environment_xarray_xesmf_proplot/xarray/advanced-analysis.ipynb
    
    You can also use a loop on lon/lat but way longer!-> spatial_regression()
    '''
    #print('trend-0','intercept-1','rvalue-2','pvalue-3','stderr-4')
    return xr.apply_ufunc(
        stats.linregress, x, y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True
        )

def seasonal_selection(ds,season='MA',iyr=1979,fyr=2005):
    """
        Output of field of seasonal/monthly values 3-D Xarray (nyr,lat,lon). For seasonal ,it is possible to shift the start and the end of you
        period in order to select the full winter season (e.g. Dec 1979, Jan 1980, Feb 1980) or select individual winter months
        without removing any data (e.g. for 'DJF' it will take into
      account the first D and last JF).
        The time dimension needs to be named
        'time'.
        Parameters
        ----------
        ds : xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
            Monthly data to process.
        
        season : int, str, optional
            Season on wchich to compute the climatology. Default is 'annual'.
            Options are:
            - 'annual'
            - single month: int (ex: 1 for January, 2 for February, etc.)
            - any character string (ex: 'DJF', 'JJAS', etc.)
        imon_obs,fmon_obs,iyr_obs,fyr_obs : int,optional
        fyr_obs is set to 2005 in order to avoid erros when data extension is short
        
          Returns
        -------
     seasonal_values: xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
            Weighted climatology.
    """
    if season[0]=='D':
        imon=6;fmon=6
    else:
        imon=1;fmon=12
    field0=ds.sel(time=slice(str(iyr)+"-"+str(imon), str(fyr+1)+"-"+str(fmon)))
    #print(field0.time)
    month = field0['time.month']
    if isinstance(season, int):
        season_sel = (month == season)
        #print(season_sel)
        nyr=fyr-iyr+1
        len_season=int(1)
    elif isinstance(season, str) and len(season) > 1:
        season_str = 'JFMAMJJASONDJFMAMJJASOND'
        #print(season_str.index(season)) ## Index gives the position of the season letter in the season str; then we advance
        len_season=len(season) ##I need this for monthly seasonal selection
        month_start = season_str.index(season) + 1
        month_end = month_start + len(season) - 1
            #print(month_start)
            #print(month_end)
        if month_end > 12:
            month_end -= 12  #x-=12 equivalent x=x*12 (multiple)
            season_sel = (month >= month_start) | (month <= month_end)
            #print('Check monthly/seasonal selection: winter')
            #print(season_sel)
            nyr=fyr-iyr+1
        else:
            season_sel = (month >= month_start) & (month <= month_end)
            #print('Check monthly/seasonal selection')
             #print(season_sel)
            nyr=fyr-iyr+1
    else:
        raise ValueError(
                    f"""Invalid season argument: '{season}'. Valid values are:
                - 'annual'"
                - single month: int (ex: 1 for January, 2 for February, etc.)
                - any character string (ex: 'DJF', 'JJAS', etc.)")
                """
                )
    clim=field0.sel(time=season_sel).mean('time')
    #clim.attrs['period'] = str(field0[0]['time.year'].values) + '-' + str(field0[-1]['time.year'].values)
    coords={'time': np.arange(int(iyr),int(fyr)+1,1), field0.dims[1]: field0.coords[field0.dims[1]], field0.dims[2]: field0.coords[field0.dims[2]]}
    zero = xr.DataArray(np.zeros((nyr,field0.shape[1],field0.shape[2])),coords=coords,dims=[field0.dims[0],field0.dims[1], field0.dims[2]])
    seasonal_values=xr.zeros_like(zero)
    seasonal_anomalies=xr.zeros_like(zero)
    if season[0]=='D':
        print('(D-',iyr,' JF-',int(iyr)+1,' to D-',int(fyr),'JF-',int(fyr)+1)
    for i in range(nyr):
        tmp=field0.sel(time=season_sel).values[i*len_season:(i*len_season+len_season),:,:]
        #tmp=field[12*i+(imon-1):12*i+(fmon),:,:]
        seasonal_values[i,:,:]=np.mean(tmp,axis=0)
        seasonal_anomalies[i,:,:]=seasonal_values[i,:,:]-clim[:,:]   
    #seasonal_values.attrs['period'] = str(field0[0]['time.year'].values) + '-' + str(field0[-1]['time.year'].values)
    #seasonal_anomalies.attrs['period'] = str(field0[0]['time.year'].values) + '-' + str(field0[-1]['time.year'].values)
    return seasonal_values,seasonal_anomalies

def monthly_selection(ds,season,iyr,fyr):
    #https://stackoverflow.com/questions/60791186/select-xarray-dataset-based-on-month
    # Use .groupby('time.month') to organize the data into months
    # then use .groups to extract the indices for each month
    if isinstance(season, int):
        rmon=season
        #print(season_sel)
    elif isinstance(season, str) and season[0]=='m':
        rmon=int(season.split('mon')[1])
    lat_str,lon_str=latlon(ds)
    field_period=ds.sel(time=slice(str(iyr)+"-"+str(1), str(fyr)+"-"+str(12)))
    time_ar=pd.date_range(start=str(iyr)+'-01',end=str(fyr+1)+'-01',freq='YS')
    month_idxs=field_period.groupby('time.month').groups
    # Extract the time indices corresponding to all the Januarys
    selmon_idxs=month_idxs[rmon]
    # Extract the january months by selecting
    valsmon=field_period.isel(time=selmon_idxs)
    anomsmon=valsmon-valsmon.mean('time')
    if len(ds.dims)==3:
        valsmon= xr.DataArray(data=valsmon, dims=["time",ds.dims[1],ds.dims[2]],
            coords=[valsmon.time,valsmon.coords[ds.dims[1]],valsmon.coords[ds.dims[2]]])
        anomsmon= xr.DataArray(data=anomsmon, dims=["time",ds.dims[1],ds.dims[2]],
            coords=[anomsmon.time,anomsmon.coords[ds.dims[1]],anomsmon.coords[ds.dims[2]]])
    elif len(ds.dims)==4:
        valsmon= xr.DataArray(data=valsmon, dims=["time",ds.dims[1],ds.dims[2],ds.dims[3]],
            coords=[valsmon.time,valsmon.coords[ds.dims[1]],valsmon.coords[ds.dims[2]],valsmon.coords[ds.dims[3]]])
        anomsmon= xr.DataArray(data=anomsmon, dims=["time",ds.dims[1],ds.dims[2],ds.dims[3]],
            coords=[anomsmon.time,anomsmon.coords[ds.dims[1]],anomsmon.coords[ds.dims[2]],anomsmon.coords[ds.dims[3]]])
    #time_ar=pd.date_range(start=str(iyr)+'-01',end=str(fyr+1)+'-01',freq='M')
    return valsmon,anomsmon    #time_ar=pd.date_range(start=str(iyr)+'-01',end=str(fyr+1)+'-01',freq='M')

