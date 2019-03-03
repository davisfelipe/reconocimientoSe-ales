import os,cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

alto=64
ancho=64
canales=3

dir_aprendizaje="./train/"
dir_prueba="./test/"


imagenes_entrenamiento = [dir_aprendizaje+i for i in os.listdir(dir_aprendizaje)]
imagenes_prueba = [dir_prueba+i for i in os.listdir(dir_prueba)]

def leer_imagen(ruta):
    imagen=cv2.imread(ruta,cv2.IMREAD_COLOR)
    return cv2.resize(imagen, (alto,ancho), interpolation=cv2.INTER_CUBIC)

def redefinir_imagen(imagenes):
    contar=len(imagenes)
    datos=np.ndarray((contar,alto,ancho,canales), dtype=np.uint8)
    for i, archivo in enumerate(imagenes):
        imagen=leer_imagen(archivo)
        datos[i]=imagen.Tcanales
        if i%5 ==0:
            print('Redefinida {} imagenes de {}'.format(i,contar))
    return datos
datosa=redefinir_imagen(imagenes_entrenamiento)
datosb=redefinir_imagen(imagenes_prueba)
clasificacion = []
for i in imagenes_entrenamiento:
    if 'cat' in i:
        clasificacion.append(1)
    else:
        clasificacion.append(0)

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def senal():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, alto, ancho), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

modelo=senal()
nb_epoch = 10
batch_size = 16

def run_catdog():
    
    history = LossHistory()
    modelo.fit(datosa, clasificacion, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])
    

    predictions = modelo.predict(imagenes_prueba, verbose=0)
    return predictions, history

predictions, history = run_catdog()
    
    


