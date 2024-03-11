#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import tensorflow_probability as tfp
from sklearn.metrics import r2_score
"""
Get data from:
- Run this command for SSS: podaac-data-downloader -c ECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4 -d sss-outdir -sd 2012-01-01T00:00:00Z -ed 2018-01-01T00:00:00Z
- Run this command for SST: podaac-data-downloader -c L3S_LEO_DY-STAR-v2.81 -d sst-outdir -sd 2012-01-01T00:00:00Z -ed 2012-01-01T00:00:00Z


https://urs.earthdata.nasa.gov/oauth/authorize?client_id=HrBaq4rVeFgp2aOo6PoUxA&response_type=code&redirect_uri=https://archive.podaac.earthdata.nasa.gov/login&state=%2Fpodaac-ops-cumulus-protected%2FECCO_L4_TEMP_SALINITY_05DEG_DAILY_V4R4%2FOCEAN_TEMPERATURE_SALINITY_day_mean_2012-01-01_ECCO_V4r4_latlon_0p50deg.nc

"""

# Historical

print('hello')
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import sqlite3

conn = sqlite3.connect("data.db")
cur = conn.cursor()

conn2 = sqlite3.connect("summer-data.db")
cur2 = conn2.cursor()
print(tf.__version__)

cur.execute('SELECT * FROM triples;')
data = cur.fetchall()
print(data[int(len(data)/2)])
print()
print()
out_x_list = [[x[0], x[2]] for x in data]
out_y_list = [x[1] for x in data]

cur2.execute('SELECT * FROM triples;')
data2 = cur.fetchall()

out_x_list2 = [[x[0], x[2]] for x in data]
out_y_list2 = [x[1] for x in data]

out_x_list = out_x_list + out_x_list2
out_y_list = out_y_list + out_y_list2

trainRatio = 0.8
validationSplit = 0.8
validation_split_mark = int(validationSplit*trainRatio*len(out_x_list)) + 1
test_split_mark = int(trainRatio*len(out_x_list)) + 1

# x is labels, y is features

(training_x, training_y, validation_x, validation_y, testing_x, testing_y) = ([ [x] for x in out_x_list[:validation_split_mark]], [ [x] for x in out_y_list[:validation_split_mark]], [ [x] for x in out_x_list[validation_split_mark:test_split_mark]], [[x] for x in out_y_list[validation_split_mark:test_split_mark]], [[x] for x in out_x_list[test_split_mark:]], [[x] for x in out_y_list[test_split_mark:]])
#print(training_x)



testing_x = [[x] for x in out_x_list2]
testing_y = [[x] for x in out_y_list2]

print('Correlation between SST and Chl-a:', tfp.stats.correlation(
    [[k[0][0]] for k in testing_x], testing_y, sample_axis=0, event_axis=-1, keepdims=False, name=None))

print('Correlation between SSS and Chl-a:', tfp.stats.correlation(
    [[k[0][1]] for k in testing_x], testing_y, sample_axis=0, event_axis=-1, keepdims=False, name=None))

only_sst_data = [x[0][0] for x in training_x]
only_sss_data = [x[0][1] for x in training_x]

(training_x, training_y, validation_x, validation_y, testing_x, testing_y) = (np.array(training_x), np.array(training_y), np.array(validation_x), np.array(validation_y), np.array(testing_x), np.array(testing_y))


#print(np.array(training_x))
input_shape = training_x.shape[1:] # Fix this <-- Why does it think training_x nparray is a list?
normalizer = tf.keras.layers.Normalization(axis=-1, input_shape=input_shape)
print(f'input_shape is {input_shape}')

normalizer.adapt(training_x)

print(training_y)
chl_normalizer = layers.Normalization(input_shape=[1, 2,], axis=None)
chl_normalizer.adapt(training_y)

chl_model = tf.keras.Sequential([
    chl_normalizer,
    layers.Dense(units=1)
])

#print(training_y.shape)
chl_model.summary()

chl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mean_absolute_error')

history = chl_model.fit(
    training_x,
    training_y,
    epochs=10,
    # Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_data=(validation_x, validation_y))

test_results = {}

print('Testing data length: ', len(testing_x), len(testing_y))
test_results['sst_model'] = chl_model.evaluate(
    testing_x,
    testing_y, verbose=0)

#print(testing_y)
inlist = [x[0] for x in testing_y]
print(min(inlist), max(inlist))
rmse = test_results['sst_model']
print(rmse, 'Normalized RMSE: ', rmse/(max(inlist) - min(inlist)))

#print(test_results)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#"""
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
#plt.ylim([0, 7])
plt.xlabel('Epoch')
plt.ylabel('Error [mg/m3 Chl-a]')
#plt.ylim([0, 2])
plt.legend()
plt.grid(True)
#plt.show()
#"""

# Current SSS: 35.1
# Current SST: 4.825
# Current Chl-a: 0.173

# Limit SSS: 37.4
# Limit SST: 29

# Testing SSS: 36
# Testing SST: 6
# Testing Chl-a: 0.148

# Ratio: 0.85549132948
# Whale Original Population: 20000
# New Population of Whale: 17109
# North Atlantic right whale: 364
# New Population of Right Whale: 311

#print(np.array(testing_x[0]).shape)
"""
test = [4.825, 35.1] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)
test = [6, 36] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)
"""

"""
25.5, 35: 0.094
25.5, 38: 0.071
27, 32: 0.03
27, 37: 0.133
27, 39: 0.221
"""

test = [11.28, 31.4] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)

test = [11.28, 34.4] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)

test = [11.71, 31.4] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)

test = [11.71, 34.4] #SST, SSS
y = chl_model.predict(test)
print('Prediction for ', test, ' is ', y)

x = input('Save the model? ')
if x == "Yes":
    chl_model.save('chl_model.keras')

