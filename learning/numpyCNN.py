import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.datasets import mnist

# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# selecting a subset of data (200 images)
x_train = x_train[:200]
y = y_train[:200]

X = x_train.T
X = X/255

y.resize((200,1))
y = y.T

#checking value
pd.Series(y[0]).value_counts()

# converting into binary classification
for i in range(y.shape[1]):
    if y[0][i] >4:
        y[0][i] = 1
    else:
        y[0][i] = 0

#checking value counts
pd.Series(y[0]).value_counts()

print(X.shape, y.shape, f.shape)