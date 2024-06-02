#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import tensorflow_probability as tfp
from sklearn.metrics import r2_score

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


chl_model = keras.models.load_model("chl_model.keras")

test = [21.90928951, 36.64385559] #SST, SSS
y = chl_model.predict(test)[0][0]
print('Prediction for ', test, ' is ', y)
