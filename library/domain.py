import xarray as xr
import geopandas as gpd
import numpy as np
import regionmask

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
        
    elif domain in ['GRp']:
        latS = 36.19;latN = 41.38;lonW=67.53;lonE=77.16

    elif domain in ['GRq']:
        latS = 25.92;latN = 31.10;lonW=81.76;lonE=90.20

    elif domain in ['Tien Shan']:
        latS = 39.33;latN = 46.02;lonW=69.31;lonE=88.70

    elif domain in ['Hundu-Kush']:
        latS = 34.86;latN = 37.49;lonW=69.27;lonE=74.65

    elif domain in ['Pamir']:
        latS = 37.19;latN = 39.67;lonW=70.47;lonE=75.06

    elif domain in ['Spiti-Lahul']:
        latS = 31.04;latN = 35.84;lonW=72.86;lonE=80.95

    elif domain in ['Karakoram']:
        latS = 33.70;latN = 38.20;lonW=73.74;lonE=79.26

    elif domain in ['West-Nepal']:
        latS = 27.97;latN = 31.85;lonW=77.63;lonE=84.11

    elif domain in ['Bhutan']:
        latS = 27.15;latN = 29.89;lonW=88.89;lonE=95.44

    elif domain in ['Everest']:
        latS = 27.11;latN = 29.58;lonW=83.57;lonE=89.39

    elif domain in ['Nayainqentangla']:
        latS = 27.77;latN = 31.53;lonW=91.73;lonE=98.32

    elif domain in ['Pamir-Altai']:
        latS = 37.47;latN = 40.41;lonW=66.02;lonE=75.10

    elif domain in ['Kunlun']:
        latS = 35.01;latN = 39.67;lonW=73.10;lonE=85.64

    elif domain in ['TP2']:
        latS = 27.64;latN = 39.81;lonW=77.52;lonE=103.32
        
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
        if dim in ['lat', 'latitude','LAT']:
            lat_str = dim
        elif dim in ['lon', 'longitude','LON']:
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


def himap(ds, domain):
    """
        Get the data for the corresponding zone.
        Parameters
        ----------
        domain : str
            Zone name. Options are:
            - Eastern Hindu Kush, Western Himalaya, Eastern Himalaya, 
            Central Himalaya, Karakoram, Western Pamir, Pamir Alay, 
            Northern/Western Tien Shan, Dzhungarsky Alatau, 
            Western Kunlun Shan, Nyainqentanglha, Gangdise Mountains, 
            Hengduan Shan, Tibetan Interior Mountains, Tanggula Shan, 
            Eastern Tibetan Mountains, Qilian Shan, Eastern Kunlun Shan, 
            Altun Shan, Eastern Tien Shan, Central Tien Shan, Eastern Pamir.
        ds : xarray.Dataset (or xarray.DataArray ?) 
        
        Returns
        -------
        ds_region : slice
            Data only in the wanted zone, elsewhere is NaN values.
            A grid cannot be in two different regions. 
            But some of the grids at the limits can be a little bit in the other region just next to the one chosen.
        -------
    """

    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.dims:
        if dim in ['lat', 'latitude','LAT']:
            lat_str = dim
        elif dim in ['lon', 'longitude','LON']:
            lon_str = dim
        else:
            other_dims_str.append(dim)
    
    # Shapefile for the different regions
    shp_path = '/bettik/PROJECTS/pr-regional-climate/bricej/HMA_regions_Bolch/boundary_mountain_regions_hma_v3.shp'
    shp = gpd.read_file(shp_path)

    if shp.crs != "EPSG:4326":
        shp = shp.to_crs("EPSG:4326")

    region = shp[shp["Primary_ID"] == domain]

    # masque précis
    regions = regionmask.Regions(region.geometry.values) # polygone de la région
    mask = regions.mask(ds) # masque NaN ou zéro si dans le masque
    
    ds = ds.where(mask == 0) # mask == 0 créer un masque booléen True/False pour extraire seulement la région voulue

    # découpe la zone pour alléger le dataset
    lonW, latS, lonE, latN = region.total_bounds
    if ds[lat_str][0] > ds[lat_str][-1]:
        lat_slice = slice(latN, latS)  # décroissant
    else:
        lat_slice = slice(latS, latN)  # croissant
    ds_region = ds.sel(
        {lat_str: lat_slice,
         lon_str: slice(lonW, lonE)}
    )
    
    return ds_region


def himap_grouped(ds, domain_group):
    """
    Sélectionne un groupe de régions HMA.

    Parameters
    ----------
    domain_group : str
        Nom du groupe : "Tien Shan", "Pamir Alay", "Pamir", "Hindu Kush", "Karakoram", "Kunlun", "Spiti Lahaul", "Central Himalaya", "Bhutan", "Nyainqentangla", "Inner TP")
    ds : xarray.Dataset or DataArray

    Returns
    -------
    ds_region : Dataset/DataArray
    """

    # Détection des dimensions
    lat_str = ''
    lon_str = ''
    other_dims_str = []
    for dim in ds.dims:
        if dim in ['lat', 'latitude','LAT']:
            lat_str = dim
        elif dim in ['lon', 'longitude','LON']:
            lon_str = dim
        else:
            other_dims_str.append(dim)

    # Grouped regions
    region_groups = {
        "Tien Shan": [
            "Eastern Tien Shan", "Dzhungarsky Alatau",
            "Northern/Western Tien Shan", "Central Tien Shan"
        ],
        "Pamir Alay": ["Pamir Alay"],
        "Pamir": ["Western Pamir"],
        "Hindu Kush": ["Eastern Hindu Kush"],
        "Karakoram": ["Karakoram"],
        "Kunlun": ["Eastern Pamir", "Western Kunlun Shan"],
        "Spiti Lahaul": ["Western Himalaya"],
        "Central Himalaya": ["Central Himalaya"],
        "Bhutan": ["Eastern Himalaya"],
        "Nyainqentangla": ["Nyainqentanglha"],
        "Inner TP": [
            "Gangdise Mountains", "Tibetan Interior Mountains",
            "Eastern Kunlun Shan", "Altun Shan", "Hengduan Shan",
            "Tanggula Shan", "Eastern Tibetan Mountains", "Qilian Shan"
        ]
    }

    # Charger shapefile
    shp_path = '/bettik/PROJECTS/pr-regional-climate/bricej/HMA_regions_Bolch/boundary_mountain_regions_hma_v3.shp'
    shp = gpd.read_file(shp_path)

    if shp.crs != "EPSG:4326":
        shp = shp.to_crs("EPSG:4326")

    # Sélection multiple des régions
    region = shp[shp["Primary_ID"].isin(region_groups[domain_group])]

    # Masque
    regions = regionmask.Regions(region.geometry.values)
    mask = regions.mask(ds)
    ds = ds.where(~np.isnan(mask)) # permet de garder toutes les régions dans la région voulue

    # Limites box globale du groupe
    lonW, latS, lonE, latN = region.total_bounds

    if ds[lat_str][0] > ds[lat_str][-1]:
        lat_slice = slice(latN, latS) # décroissant 
    else:
        lat_slice = slice(latS, latN) # croissant

    ds_region = ds.sel({
        lat_str: lat_slice,
        lon_str: slice(lonW, lonE)
    })

    return ds_region