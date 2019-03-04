#LIBRERIAS
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras import metrics as mt
from keras.optimizers import RMSprop
from keras.callbacks import Callback,EarlyStopping
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, AveragePooling2D
import matplotlib.pyplot as plt
#VARIABLES GLOBALES
ROWS = 64
COLS = 64
CHANNELS = 3
optimizer = "rmsprop"
objective = 'categorical_crossentropy'
nb_epoch = 60
batch_size = 32
#DIRECTORIOS
train_dir="./train/"
test_dir="./test/"
#LEER LISTA DE IMAGENES
train_images = [train_dir+i for i in os.listdir(train_dir)]
test_images =  [test_dir+i for i in os.listdir(test_dir)]
#FUNCION LEER IMAGEN
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
#FUNCION REDIMENSIONAR IMAGEN
def prep_data(images):
    count = len(images)
    data = np.ndarray((count,ROWS, COLS,CHANNELS),dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%100 == 0: print('Procesadas {} de {}'.format(i, count))
        if i==count: print('Procesadas {} de {}'.format(i, count))
    return data
#LEER IMAGENES
train=prep_data(train_images)
test=prep_data(test_images)
#GENERAR OBJETIVOS
labels = []
for i in train_images:
    if 'lim' in i:
        labels.append(0)
    elif 'paradero' in i:
        labels.append(1)
    elif 'pare' in i:
        labels.append(2)
    elif 'pro' in i:
        labels.append(3)
categorias = to_categorical(labels,num_classes=4)
#MODELO DE RED NEURONAL CONVOLUCIONADA
def senales():
    modelo = Sequential()
    modelo.add(Conv2D(32,(3, 3), padding='same', input_shape=(ROWS, COLS,3), activation='relu'))
    modelo.add(Conv2D(32,(3, 3), padding='same',activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(4))
    modelo.add(Activation('softmax'))
    modelo.compile(loss=objective, optimizer=optimizer, metrics=["accuracy",mt.mean_squared_error])
    return modelo
#GENERAR MODELO
modelo=senales()
print(modelo.summary())
#DETECCION DE PERDIDAS
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
#ENTRENAR E HISTORIAL DE PERDIDAS
def entrenar():
    history=LossHistory()
    modelo.fit(train, categorias, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.30, shuffle=True,callbacks=[history])
    predicciones=modelo.predict(test)
    return predicciones,history
modelo.save('mpredictionalt.h5')
predicciones,historia=entrenar()
plt.plot(range(len(historia.losses)),historia.losses)
plt.savefig('perdida.png')
plt.plot(range(len(historia.val_losses)),historia.val_losses)
plt.savefig('perdida interacion.png')
#PROBAR MODELO
for i in range(0,12):
    index=np.where(predicciones[i]==predicciones[i].max())
    print index
    plt.imshow(test[i])
    plt.show()
modelo.save('modelo2D.h5')
