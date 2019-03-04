#LIBRERIAS
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
#VARIABLES GLOBALES
video = cv2.VideoCapture(0)
modelo1=load_model('modelo2Dm1.h5')
modelo2=load_model('modelo2Dm2.h5')
modelo3=load_model('modelo2Dm3.h5')
texto="NADA"
tamano=64
#LECTURA Y VISUALIZACION DE LA IMAGEN
def leer_imagen():
    _,imagen=video.read()
    cp_imagen=imagen
    cp_imagen=cv2.putText(cp_imagen,texto,(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 4)
    cv2.imshow('imagen',cp_imagen)
    return cv2.resize(imagen,(64,64),interpolation=cv2.INTER_CUBIC),cv2.resize(imagen,(128,128),interpolation=cv2.INTER_CUBIC)
#REDIMENSIONAR IMAGEN
def prep_imagen():
    data1=np.ndarray((1,64,64,3),dtype=np.uint8)
    data2=np.ndarray((1,128,128,3),dtype=np.uint8)
    data1[0],data2[0]=leer_imagen()    
    return np.expand_dims(np.array(data1[0]),axis=0),np.expand_dims(np.array(data2[0]),axis=0)
#IDENTIFICAR SENAL
def tipo_senal(array):
    if array.max()>0.70:
        index = np.where(array==array.max())
        index=int(index[0])
        if index==0:
            return 'LIMITE {:.3%}'.format(array[index])
        elif index==1:
            return 'PARADERO {:.3%}'.format(array[index])
        elif index==2:
            return 'PARE {:.3%}'.format(array[index])
        elif index==3:
            return 'PROHIBIDO {:.3%}'.format(array[index])
    else:
        return "NADA"
#EJECUCION

while(True):
    frame1,frame2=prep_imagen()
    a1=modelo1.predict(frame1)
    a2=modelo2.predict(frame2)
    a3=modelo3.predict(frame1)
    promedio=pd.DataFrame([a1[0],a2[0],a3[0]],columns=['a','b','c','d'])
    print(promedio.mean())
    texto=tipo_senal(promedio.mean())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()