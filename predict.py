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

test = [12.406, 32.224] #SST, SSS
y = chl_model.predict(test)[0][0]
print('Prediction for ', test, ' is ', y)
