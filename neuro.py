import numpy
from tensorflow.python import tf2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
seed = 21
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # loading in the data
# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs || приводим к унитарному коду, чтобы нейросеть могла распознать
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

model = Sequential() #создание модели
#функция активации
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) #защита от переобучения
#акетная нормализация нормализует входные данные, поступающие в следующий слой, гарантируя,
# что сеть всегда создает функции активации с тем же распределением, которое нам нужно:
model.add(BatchNormalization())
#свёрточный слой
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
#объединяющий слой
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#повторяем слои
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#сжатие данных
model.add(Flatten())
model.add(Dropout(0.2))
#первый плотный слой
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#В этом последнем слое мы уравниваем количество классов с числом нейронов.
# функция активации softmax выбирает нейрон с наибольшей вероятностью в качестве своего выходного значения
model.add(Dense(class_num))
model.add(Activation('softmax'))
#Оптимизатор - это то, что настроит веса в вашей сети так, чтобы приблизиться к точке с наименьшими потерями.
epochs = 25 #количество эпох
optimizer = 'adam' #наиболее эффективный алгоритм оптимизации
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) #компиляция модели
print(model.summary())