import xarray as xr

def reversing_lat(ds):
    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.dims:
        if dim in ['lat', 'lon']:
            dsO= ds.reindex(lat=list(reversed(ds.lat)))
        elif dim in ['latitude', 'longitude']:
            dsO= ds.reindex(latitude=list(reversed(ds.latitude)))
        else:
            other_dims_str.append(dim)
    return dsO

def coord_domain(domain):
    """
        Get the latitude and longitude limits for the corresponding zone.
        Parameters
        ----------
        zone : str
            Zone name. Options are:
            - 'GLOB', 'global', 'GLOBAL', 'GLOB-land'
            - 'NH' : North Hemisphere
            - 'HMA' : High Mountain of Asia
            - 'NA' : North America
        Returns
        -------
        latlim, lonlim : slice
            Latitude and longitude limits of the zone.
        -------
    """
    if domain in ['global','GLOB']:
        latS=-90;latN=90; lonW=-180;lonE=180

    # North Hemisphere
    elif domain in ['NH']:
        latS = 0;latN = 90;lonW=-180;lonE=180

    elif domain in ['NHe']:
        latS = 0;latN = 90;lonW=0;lonE=180    
    
    elif domain in ['SH']:
        latS=-90;latN=-20;lonW=-180;lonE=180
        
    elif domain in ['NINO34']:
        latS=-5;latN=5;lonW=-170;lonE=-120
        
    elif domain in ['MA']:
        latS = 10;latN = 60;lonW=40;lonE=130
        
    # High Mountain of Asia (HMA)
    elif domain in ['HMA','GRh']:
        latS = 20;latN = 45;lonW=60;lonE=110

    elif domain in ['HMA+']:
        latS = 10;latN = 45;lonW=60;lonE=105
        
    # HK: Hindu-Kush / Karakoram / Western Himalay
    # CH: Central and Est Himalaya
    # TB: Tibetan Plateau
    elif domain in ['HK']:
        latS = 34;latN = 41;lonW=64;lonE=76

    elif domain in ['HK+']:
        latS = 34;latN = 45;lonW=60;lonE=76
        
    elif domain in ['CH']:
        latS = 23;latN = 30.5;lonW=80;lonE=91.5
        
    elif domain in ['TP']:
        latS = 30.5;latN = 38;lonW=80;lonE=102
        
    elif domain in ['SEH']:
        latS = 20;latN = 30;lonW=91.5;lonE=110
        
    elif domain in ['SL']:
        latS = 29;latN = 36;lonW=76;lonE=80
        
    elif domain in ['MARMer','GRa']:
        latS = 25;latN = 31;lonW=80;lonE=90
        
    elif domain in ['MARFed','GRf']:
        latS = 36;latN = 41;lonW=67;lonE=78
        
    else:
        raise ValueError(
            f"""Invalid zone argument: '{zone}'. Valid zones are:
                - 'GLOB', 'global', 'GLOBAL'
                - 'NH' : North Hemisphere
                - 'HMA' : High Mountain of Asia
                - 'NA' : North America
             """
        )
    return latS,latN,lonW,lonE

def field_dom(ds,domain):
    latS,latN,lonW,lonE=coord_domain(domain)
    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.dims:
        if dim in ['lat', 'latitude','LON']:
            lat_str = dim
        elif dim in ['lon', 'longitude','LAT']:
            lon_str = dim
        else:
            other_dims_str.append(dim)

    if ds[lat_str][0] > ds[lat_str][-1]:
        lat_slice = slice(latN, latS)  # décroissant
    else:
        lat_slice = slice(latS, latN)  # croissant
    
    field = ds.sel(
        {lat_str: lat_slice,
         lon_str: slice(lonW, lonE)}
    )
    return field