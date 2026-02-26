import numpy as np
import xarray as xr

def compute_ivt(uivt, vivt):
    ivt = (uivt['viwve']**2 + vivt['viwvn']**2)**0.5
    return xr.Dataset({'ivt': ivt})