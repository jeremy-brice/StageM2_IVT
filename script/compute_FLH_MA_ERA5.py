import sys
sys.path.insert(1, '/home/bricej/MyPythonLibrary/StageM2_IVT/library/')
import domain
from domain import *
from climbas import *
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# Arguments
iyear = int(sys.argv[1])
fyear = int(sys.argv[2])
dom   = sys.argv[3]

#============================ Loading data ================================

# Temperature at each levels from ERA5
t = xr.open_mfdataset("/bettik/PROJECTS/pr-regional-climate/santolam/era5/monthly/TLevs_mon.era5_*_025x025.nc", combine="by_coords")

# Geopotential Height from ERA5
z = xr.open_mfdataset("/bettik/PROJECTS/pr-regional-climate/santolam/era5/monthly/HGTLevs_mon.era5_*_025x025.nc", combine="by_coords")

#Period & Domain Def (maximum is MA)
t = t.sel(valid_time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))
z = z.sel(valid_time=slice(f"{iyear}-01-01", f"{fyear}-12-31"))

t = field_dom(t, dom)
z = field_dom(z, dom)

# Making the data lighter 
t = t.drop_vars(['number','expver'], errors='ignore')  # les variables inutiles
t.attrs = {}  # les atributs pas utiles pour les calculs
t['t'].attrs = {}  # les atributs pas utiles pour les calculs
t = t.load()  # charger en mémoire pour pas avoir à relire le fichier pour les calculs

z = z.drop_vars(['number','expver'], errors='ignore')  # les variables inutiles
z.attrs = {}  # les atributs pas utiles pour les calculs
z['z'].attrs = {}  # les atributs pas utiles pour les calculs
z = z.load()  # charger en mémoire pour pas avoir à relire le fichier pour les calculs

# K -> °C
t = t.t - 273.15

# m2/s2 -> m
z = z.z / 9.81

#============================== Calcul of Freezing Level Height (FLH) ==============================

# Décalage vertical pour pouvoir comparer les températures des pressures levels entre elles
t_shift = t.shift(pressure_level=1)
z_shift = z.shift(pressure_level=1)

# Multiplication entre normal et décalage (shift) : Détection crossing 0°C
crossing = (t * t_shift) <= 0 # True si inférieur ou égale à 0

# Premier crossing (Premier True) : si signe est négatif alors on a changé de signe et donc FLH entre ces deux niveaux 
idx = crossing.argmax(dim="pressure_level")

Ti = t.isel(pressure_level=idx)
Ti1 = t_shift.isel(pressure_level=idx)

Hi = z.isel(pressure_level=idx)
Hi1 = z_shift.isel(pressure_level=idx)

# FLH interpolation entre les deux niveaux de pression trouvé juste avant
FLH = Hi + Ti * (Hi1 - Hi) / (Ti - Ti1)

eps = 0.1
FLH = xr.where(abs(Ti - Ti1) > eps, FLH, np.nan) # évite la division par zéro au cas où (numériquement ici très proche de zéro = instable)

# Vérification si il y a bien le crossing qui existe
FLH = FLH.where(crossing.any(dim="pressure_level")) # si pas de crossing par température 0 alors argmax=0 et donc il faut mettre NaN
FLH = FLH.where(t.max(dim="pressure_level") > 0) # Si toute la colonne > 0 °C
FLH = FLH.where(t.min(dim="pressure_level") < 0) # Si toute la colonne < 0 °C

#Enlève les valeurs abberantes à cause de la façon dont on calcule
FLH = xr.where(FLH > 10000, np.nan, FLH)

#========================== Saving Files ==================================

#Rename dimensions
FLH = FLH.reset_coords("pressure_level", drop=True)
FLH=FLH.rename({'valid_time': 'time','latitude':'lat','longitude':'lon'})

#Saving to dataset
output_dir = '/bettik/PROJECTS/pr-regional-climate/bricej/era5/monthly'

ds = FLH.to_dataset(name="FLH")
ds.attrs["units"] = "m"

filename = f"{output_dir}/FLH_monthly_era5_{iyear}-{fyear}_{dom}_025x025.nc"
    
ds.to_netcdf(filename)
print(f'Saved file : {filename}')
