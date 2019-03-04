import numpy as np
import cv2
from keras.models import load_model
video = cv2.VideoCapture(1)
modelo=load_model('mpd.h5')

def leer_imagen():
    _,imagen=video.read()
    cv2.imshow('imagen',imagen)
    return cv2.resize(imagen,(128,128),interpolation=cv2.INTER_CUBIC)
def prep_imagen():
    data=np.ndarray((1,3,128,128),dtype=np.uint8)
    data[0]=leer_imagen().T    
    return np.expand_dims(np.array(data[0]),axis=0)
def tipo_senal(array):
    if array.max()>0.90:
        index = np.where(array==array.max())
        print array
        if index[1]==0:
            return "LIMITE"
        elif index[1]==1:
            return "PARADERO"
        elif index[1]==2:
            return "PARE"
        elif index[1]==3:
            return "PROHIBIDO"
    else:
        return "NADA"
while(True):
    frame=leer_imagen()
    frame=prep_imagen()
    predicciones=modelo.predict(frame)
    print(tipo_senal(predicciones))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

'''
def leer_imagen():
    _,imagen=video.read()
    cv.imshow('imagen',imagen)
    return imagen,cv.resize(imagen,(128,128),interpolation=cv.INTER_CUBIC)
def prep_imagen():
    data=np.ndarray((1,3,128,128),dtype=np.uint8)
    mostrar,imagen=leer_imagen()
    data[0]=imagen.T
    return data[0],mostrar
def tipo_senal(array):
    if array.max()>0.90:
        index = np.where(array==array.max())
        print array
        if index[1]==0:
            return "LIMITE"
        elif index[1]==1:
            return "PARADERO"
        elif index[1]==2:
            return "PARE"
        elif index[1]==3:
            return "PROHIBIDO"
    else:
        return "NADA"

while True:
    frame, imagen=prep_imagen()
    prediccion=model.predict(np.expand_dims(np.array(frame),axis=0))
    print(tipo_senal(prediccion))
def leer_imagen():
    _,img=video.read()
    return img,cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
def prep_data():
    data=np.ndarray((1,3,128,128),dtype=np.uint8)
    mostrar,imagen=leer_imagen()
    data[0]=imagen.T
    return data[0],mostrar
while True:
    frame,imagen=prep_data()
    img_array=np.array(frame)
    img_array=np.expand_dims(img_array,axis=0)
    predicciones=model.predict(img_array)
    imgagen = cv2.circle(imagen,(56,56),32,(0,0,255),1)
    cv2.imshow('ventana',imagen)
    index=np.where(predicciones==predicciones.max())
    print index
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
_,frame=video.read()
imagen=cv2.resize(frame,(256,256),interpolation=cv2.INTER_CUBIC)
im = Image.fromarray(imagen, 'RGB')
img_array = np.array(im)
img_array = np.expand_dims(img_array, axis=0)
print(img_array.shape)
prediction = model.predict(img_array)
'''
