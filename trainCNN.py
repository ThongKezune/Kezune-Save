import glob
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import ast
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from keras.callbacks import ModelCheckpoint


# Load MNIST dataset
label = ['moving1', 'leftclick', 'rolldown', 'zoombig', 'zoomsmall', 'rollup', 'rightclick']
x = []
y = []
for i in range(7):
    for data in glob.glob(label[i]+'.csv'):
        print(data)
        with open(data) as f:
            reader = csv.reader(f)
            for row in reader:
                t = []
                for xyz in row:
                    s = xyz.strip('[]')
                    arr = [float(x) for x in s.split()]
                    t.append(arr)
                x.append(t)
                y.append(i)
print(type(x))
print(type(x[0]))
print(x[0])
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(X_train[0]))
print(X_train[0].shape)
print(X_train[0])

x_train = X_train.reshape(-1, 3, 21, 1)
x_test = X_test.reshape(-1, 3, 21, 1)
print(y_train)
# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(21, 3)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(7, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('1')
# Train the model
model.fit(X_train, y_train, epochs=100)
print('2')
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
# Lưu model
model.save('my_model.h5')