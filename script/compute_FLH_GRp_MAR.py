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
t = xr.open_mfdataset(f'/bettik/PROJECTS/pr-regional-climate/santolam/MARout_post/{dom}/spin2/work/monthly/TTZ_monmean_MARv3.14_ER5_spin2_{dom}_*.nc', combine="by_coords")['TTZ']

# Geopotential Height from MAR (pas encore)
z = 
