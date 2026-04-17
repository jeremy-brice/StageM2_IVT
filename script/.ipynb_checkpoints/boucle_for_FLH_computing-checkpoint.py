import os

for yr in range(1940, 2025, 2):
    next_year = yr + 1
    print(f"Running {yr}-{next_year} MA")

    os.system(f"python compute_FLH_MA_ERA5.py {yr} {next_year} MA")