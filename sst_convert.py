from os import listdir
from os.path import isfile, join
import os
import datetime
import sys
import xarray as xr
import numpy as np
import pandas as pd
from pprint import pprint
import ast
import sqlite3

conn = sqlite3.connect("data.db")
cur = conn.cursor()




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


sst_indir = 'sst-outdir/'

sst_outdir = 'sst-csv-dir'


var = 'sst' # 'SALT' for SSS, sst for SST, chlor_a for Chl-a

if var == 'sst':
    print('open')
    # use glob
    sst_onlyfiles = [f for f in listdir(sst_outdir) if isfile(sst_outdir + f) and f[-3:]=="csv"]
    sst_in = sst_onlyfiles = [f for f in listdir(sst_indir) if isfile(sst_indir + f) and f[-2:]=="nc"]

    print(f'Count of files in dir: {len(sst_onlyfiles)}')

    for (h, i) in zip(sst_in, sst_onlyfiles):
        curr_datafile = os.path.join(sst_outdir, i)

        #sst_dfs = pd.read_csv(curr_datafile)

        #data_list = df.values.tolist()
        """chla_data = xr.open_dataset( curr_datafile)

        #pprint(sss_data.values.to_dataframe())

        chla_dfs = chla_data['chlor_a'].to_dataframe()"""

        print('got dataframes')
        sst_data = xr.open_dataset(sst_indir + h)['sst']
        sst_data.to_dataframe().to_csv(curr_datafile)
        print('converted to csv')
        sst_dfs = pd.read_csv(curr_datafile)
        sst_dfs.dropna()
        print('dropped null values')


        #preparing('chla conversion')
        """
        chla_index = []
        with open('chl-index.txt', 'r') as f:
            for l in f:
                print(type(l))
                chla_index = output = ast.literal_eval(l)""" 

        #chla_index = chla_dfs.index.tolist()
        #print(chla_index)
  
        #print('got indexes')

        #print(chla_values)
        print('got values')
        #print(sst_df)
        #quit()
        #print('converted sst data to list')
        #pprint(sss_df)
        #print('l')
#        print(sss_df.values.tolist())
        #sss_df = [list(x)+list(y) for (x, y) in zip(sss_df.index.tolist(), sss_df.values.tolist())]
        out_list = []
        print('preparing to merge data')
        ctr = 0

        for x in sst_dfs.values: #SST: 0.875, 0.625, 0.375, 0.125 SSS: 0.75, 0.25
            print(x) # Data structure of x: [nan, Timestamp('2012-01-16 00:00:00'), -90.0, -180.0, 0.0, nan]

            ctr+=1
            if ctr == 1:
                continue
            if np.isnan(x[4]):
                continue
            #pprint(x)
            time = x[0]
            lat = float(x[2])
            lon = float(x[3])


            #date_ = x[0]
            sst = x[4]
            print(x[4])
            cur.execute("INSERT INTO temperature (t_date, t_lat, t_lon, t_value) VALUES (?, ?, ?, ?)", (time, lat, lon, sst))
            """ctr += 1
            if ctr == 100:
                print(x)
                ctr = 0"""
        
        conn.commit()

            

        #df2= df.sample(n=3)
        #df = df.sample(frac=)
        #df2 = out_list
        #df2.to_csv(sst_outdir + f[:-3] + '.csv')
        #print(f"File {file_count} of {len(sst_onlyfiles)} completed.")
        #quit()


