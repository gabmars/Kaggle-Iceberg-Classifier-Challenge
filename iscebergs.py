import pandas as pd
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

train=pd.read_json('train.json')

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
Y_train = np.array(train["is_iceberg"])

# Задаем seed для повторяемости результатов
np.random.seed(42)

# Размер мини-выборки
batch_size = 75
# Количество классов изображений
nb_classes = 2
# Количество эпох для обучения
nb_epoch = 10
# Размер изображений
img_rows, img_cols = 75, 75
# Количество каналов в изображении: RGB
img_channels = 3

# Нормализуем данные
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255

# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(75, (5, 5), padding='same',
                        input_shape=(75, 75, 2), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(75, (5, 5), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(150, (5, 5), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(150, (5, 5), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(900, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(1, activation='softmax'))

# Задаем параметры оптимизации
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# Обучаем модель
model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)

# Оцениваем качество обучения модели на тестовых данных
#scores = model.evaluate(X_test, Y_test, verbose=0)
#print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))