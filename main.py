#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns

"""
Get data from:
- Run this command for SSS: podaac-data-downloader -c ECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4 -d sss-outdir -sd 2012-01-01T00:00:00Z -ed 2018-01-01T00:00:00Z
- Run this command for SST: podaac-data-downloader -c L3S_LEO_DY-STAR-v2.81 -d sss-outdir -sd 2012-01-01T00:00:00Z -ed 2012-01-01T00:00:00Z


https://urs.earthdata.nasa.gov/oauth/authorize?client_id=HrBaq4rVeFgp2aOo6PoUxA&response_type=code&redirect_uri=https://archive.podaac.earthdata.nasa.gov/login&state=%2Fpodaac-ops-cumulus-protected%2FECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4%2FOCEAN_TEMPERATURE_SALINITY_day_mean_2012-01-01_ECCO_V4r4_latlon_0p50deg.nc

"""

print('hello')
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# For sss

sss_csv = 'noaa_aoml_19fa_f217_8e36_c7ba_c5e0_5983.csv'
sst_csv = 'erdMH1sstd8dayR20190SQ_2f2d_cfb8_a7d1.csv'

df = pd.read_csv(sss_csv)
data_lst = df.values.tolist()
df = pd.read_csv(sst_csv)
data_list = df.values.tolist()
out_x_list = []
out_y_list = []
"""
data_lst = df.values.tolist()

out_time_list_sss = [] # time
out_y_list_sss = [] # sss


for y in data_lst: #sss (y)
    if y[1] == "0.25" and y[2] == "0.25":
        #print(y)
        out_time_list_sss.append(y[0]) # 
        out_y_list_sss.append(float(y[-1]))

#print(out_x_list_sss, out_y_list_sss, len(out_x_list_sss), len(out_y_list_sss))

# For sst


df = pd.read_csv(sst_csv)

sst_data_list = df.values.tolist()

for a in sst_data_list: # a is sst, l is sss 
    if a[1] == "-0.3958333" and a[2] == "-0.02082316" and a[0] in out_time_list_sss:
        #print(a)
        k = []
        for l in data_lst: # for each sss
            if l[0] == a[0]: # if sss time = sst time
                k = l
        if not math.isnan(float(a[3])): # If the sst does not have a null value
            #print(a)       
            out_x_list.append(float(a[3])) # sst
            out_y_list.append(float(k[3])) # sss
"""

for a, b in zip(data_lst, data_list): #a: sss, b: sst
    print(a, b)
    out_x_list.append(float(a[3])) # sst
    out_y_list.append(float(k[3]))

quit()
#print(f'out_x_list is {out_x_list}, out_y_list is {out_y_list}, lengths are {len(out_x_list)}, {len(out_y_list)}')

trainRatio = 0.8
validationSplit = 0.8
validation_split_mark = int(validationSplit*trainRatio*len(out_x_list)) + 1
test_split_mark = int(trainRatio*len(out_x_list)) + 1

# x is labels, y is features

(training_x, training_y, validation_x, validation_y, testing_x, testing_y) = ([ [x] for x in out_x_list[:validation_split_mark]], [ [x] for x in out_y_list[:validation_split_mark]], [ [x] for x in out_x_list[validation_split_mark:test_split_mark]], [[x] for x in out_y_list[validation_split_mark:test_split_mark]], [[x] for x in out_x_list[test_split_mark:]], [[x] for x in out_y_list[test_split_mark:]])
print(training_x)

(training_x, training_y, validation_x, validation_y, testing_x, testing_y) = (np.array(training_x), np.array(training_y), np.array(validation_x), np.array(validation_y), np.array(testing_x), np.array(testing_y))


#print(np.array(training_x))
input_shape = training_x.shape[1:] # Fix this <-- Why does it think training_x nparray is a list?
normalizer = tf.keras.layers.Normalization(axis=-1, input_shape=input_shape)
print(f'input_shape is {input_shape}')

normalizer.adapt(training_x)

print(training_y)
sst_normalizer = layers.Normalization(input_shape=[1,], axis=None)
sst_normalizer.adapt(training_y)

sst_model = tf.keras.Sequential([
    sst_normalizer,
    layers.Dense(units=1)
])

sst_model.summary()

sst_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = sst_model.fit(
    training_x,
    training_y,
    epochs=200,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_data=(validation_x, validation_y))

test_results = {}

test_results['sst_model'] = sst_model.evaluate(
    testing_x,
    testing_y, verbose=0)

print(testing_y)
inlist = [x[0] for x in testing_y]
print(min(inlist), max(inlist))
rmse = test_results['sst_model']
print('Normalized RMSE: ', rmse/(max(inlist) - min(inlist)))

print(test_results)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [sss]')
plt.legend()
plt.grid(True)
plt.show()

y = sst_model.predict([[36]])
print('Prediction for ', 26, ' is ', y)