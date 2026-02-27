import xarray as xr

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
        latlim=slice(None);lonlim=slice(None)
    # North Hemisphere
    elif domain in ['NH']:
        latS = 0;latN = 90;lonW=-180;lonE=180
        latlim=slice(90,0);lonlim=slice(None)
    elif domain in ['SH']:
        latS=-90;latN=-20;lonW=-180;lonE=180
        latlim=slice(-90,-20);lonlim=slice(None)
    elif domain in ['NINO34']:
        latS=-5;latN=5;lonW=-170;lonE=-120
        latlim=slice(latS,latN);lonlim=slice(None)
    elif domain in ['MA']:
        latS = 10;latN = 60;lonW=40;lonE=130
        latlim=slice(latS,latN);lonlim=slice(lonW,lonE)
    # High Mountain of Asia (HMA)
    elif domain in ['HMA','GRh']:
        latS = 20;latN = 45;lonW=60;lonE=110
        latlim=slice(latS,latN);lonlim=slice(lonW,lonE)
    # HK: Hindu-Kush / Karakoram / Western Himalay
    # CH: Central and Est Himalaya
    # TB: Tibetan Plateau
    elif domain in ['HK']:
        latS = 34;latN = 41;lonW=64;lonE=76
        lonlim = slice(70, 81);latlim = slice(31, 40)
    elif domain in ['CH']:
        latS = 23;latN = 30.5;lonW=79;lonE=91.5
        lonlim= slice(79, 98);latlim= slice(26, 31)
    elif domain in ['TP']:
        latS = 30.5;latN = 38;lonW=80;lonE=102
        lonlim= slice(81, 104);latlim= slice(31, 39)
    elif domain in ['SEH']:
        latS = 20;latN = 30;lonW=91.5;lonE=110
        lonlim= slice(81, 104);latlim= slice(31, 39)
    elif domain in ['MARMer','GRa']:
        latS = 25;latN = 31;lonW=80;lonE=90
        lonlim= slice(lonW, lonE);latlim= slice(latS, latN)
    elif domain in ['MARFed','GRf']:
        latS = 36;latN = 41;lonW=67;lonE=78
        lonlim= slice(lonW, lonE);latlim= slice(latS, latN)
    else:
        raise ValueError(
            f"""Invalid zone argument: '{zone}'. Valid zones are:
                - 'GLOB', 'global', 'GLOBAL'
                - 'NH' : North Hemisphere
                - 'HMA' : High Mountain of Asia
                - 'NA' : North America
             """
        )
    return latS,latN,lonW,lonE,latlim,lonlim

def field_dom(ds,domain):
    latS,latN,lonW,lonE,latlim,lonlim=coord_domain(domain)
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
    
    #field = ds.where((latS < ds.coords[lat_str]) & ( ds.coords[lat_str] < latN) & (lonW < ds.coords[lon_str]) & ( ds.coords[lon_str] < lonE),drop=True)
    field = ds.sel(
        {lat_str: lat_slice,
         lon_str: slice(lonW, lonE)}
    )
    #print('Domain;latS,latN,lonW,lonE:',domain,latS,latN,lonW,lonE)
    return field