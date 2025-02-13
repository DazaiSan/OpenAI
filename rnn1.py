import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()[:20]
x_train, x_test = x_train/255.0, x_test/255.0
# print(x_train.shape)

model = Sequential()

model.add(LSTM(2, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(1, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=1, validation_data=(x_test,y_test))