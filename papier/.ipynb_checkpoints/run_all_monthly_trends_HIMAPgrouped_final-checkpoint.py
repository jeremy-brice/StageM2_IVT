import os

for m in range(1, 13):
    cmd = f"python compute_monthly-percentage_trends_var_era5_HIMAPgrouped_final_regions.py 1960 2019 rf [{m}]"
    os.system(cmd)