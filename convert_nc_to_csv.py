from os import listdir
from os.path import isfile, join
import os
import xarray as xr
import numpy as np
import pandas as pd
from pprint import pprint
"""
https://climate.esa.int/en/odp/#/dashboard

SSS: Up to 2019 (included)
SST: Up to 2016 (included)

Get data from 2016
Ocean colour/chlorophyll-a: Up to 2022

Get data from:
- Run this command for SSS: podaac-data-downloader -c ECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4 -d sss-outdir -sd 2012-01-01T00:00:00Z -ed 2018-01-01T00:00:00Z
- Link for SST data: wget https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/201201/oisst-avhrr-v02r01.20120101.nc	


https://urs.earthdata.nasa.gov/oauth/authorize?client_id=HrBaq4rVeFgp2aOo6PoUxA&response_type=code&redirect_uri=https://archive.podaac.earthdata.nasa.gov/login&state=%2Fpodaac-ops-cumulus-protected%2FECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4%2FOCEAN_TEMPERATURE_SALINITY_day_mean_2012-01-01_ECCO_V4r4_latlon_0p50deg.nc

"""



def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])


sss_indir = 'sss-outdir/'
sst_indir = 'sst-outdir/'
sst_outdir = 'sst-csv-sst_indir/'
sss_outdir = 'sss-csv-sst_indir/'
var = 'sst' # 'SALT' for SSS, sst for SST

if var == 'sst':
    print('open')
    lat_list = [x-0.25 for x in [y/2 for y in range(-179, 181, 1)]]
    lon_list = [x+0.25 for x in [y/2 for y in range(-360, 360, 1)]]
    in_dict = {}
    for a in lat_list:
        in_dict[a] = {}
        for b in lon_list:
            in_dict[a][b] = 0
    #pprint(in_dict)
    print('dictionary initialized')
    #quit()
    file_count = 1
    sst_onlyfiles = [f for f in listdir(sst_indir) if isfile(sst_indir + f) and f[-2:]=="nc"]
    print(isfile(sss_indir + listdir(sss_indir)[0]) and listdir(sss_indir)[0][-2:]=="nc")
    sss_onlyfiles = [f for f in listdir(sss_indir) if isfile(sss_indir + f) and f[-2:]=="nc"]
    print(sst_onlyfiles, sss_onlyfiles)
    for (f, g) in zip(sst_onlyfiles, sss_onlyfiles):
        #print(f'starting {sst_indir + f}')
        sst_data = xr.open_dataset(sst_indir + f)
        sss_data = xr.open_dataset(sss_indir + g)

        #pprint(sss_data.values.to_dataframe())
        sst_dfs = sst_data['sst'].to_dataframe()
        sss_dfs = sss_data['SALT'].to_dataframe()
        print('got dataframes')
        sst_dfs.dropna()
        sss_dfs.dropna()
        print('dropped null values')

        sss_index = sss_dfs.index.tolist()
        sst_index = sst_dfs.index.tolist()

        print('got indexes')

        sss_values = sss_dfs.values.tolist()
        sst_values = sst_dfs.values.tolist()

        sst_dfs.iloc[0:0]
        sss_dfs.iloc[0:0]
        print('got values')

        sss_d = [] 
        sst_d = []

        sst_df = [[list(x)[2],list(x)[3]-180,list(y)[0]] for (x, y) in zip(sst_index, sst_values)]
        #print(sst_df)
        #quit()
        #print('converted sst data to list')
        for (x, y) in zip(sss_index, sss_values):
            #print(x, y, list(x)+list(y))
            if x[1] < -10:
                break
            sss_d.append([list(x)[2],list(x)[3],list(y)[0]])
        sss_df = sss_d
        #pprint(sss_df)
        #quit()
        #print('l')
#        print(sss_df.values.tolist())
        #sss_df = [list(x)+list(y) for (x, y) in zip(sss_df.index.tolist(), sss_df.values.tolist())]
        out_list = []
        print('preparing to merge data')
        ctr = 0

        for z in sss_df:
            in_dict[z[0]][z[1]] = z[2]
        print('in_dict filled. Going to merge data.')

        for x in sst_df: #SST: 0.875, 0.625, 0.375, 0.125 SSS: 0.75, 0.25
            print(x)
            """ctr += 1
            if ctr == 100:
                print(x)
                ctr = 0"""
            if np.isnan(x[2]):
                continue
            if abs(x[0]-int(x[0])) != 0.875 and abs(x[0]-int(x[0])) != 0.375:
                continue
            if abs(x[1]-int(x[1])) != 0.875 and abs(x[1]-int(x[1])) != 0.375:
                continue
            sst_value = x[2]
            prep_lat = 0
            prep_lon = 0
            if x[0] > 0:
                prep_lat = x[0]-0.125
            else:
                prep_lat = x[0]+0.125
            if x[1] > 0:
                prep_lon = x[1]-0.125
            else:
                prep_lon = x[1]+0.125
            sss_value = in_dict[prep_lat][prep_lon]
            if not np.isnan(sss_value) and not np.isnan(sst_value):
                out_list.append([sst_value, sss_value])
                print(x, ' done')
        with open('data_to_use/' + 'day1.csv', 'w') as f:
            f.write('sst, sss\n')
            for x in out_list:
                f.write(f'{x[0]},{x[1]}\n')

        #df2= df.sample(n=3)
        #df = df.sample(frac=)
        #df2 = out_list
        #df2.to_csv(sst_outdir + f[:-3] + '.csv')
        print(f"File {file_count} of {len(sst_onlyfiles)} completed.")
        file_count+=1
        #quit()

