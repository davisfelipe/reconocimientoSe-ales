#LIBRERIAS
import numpy as np
import cv2
from keras.models import load_model
#VARIABLES GLOBALES
video = cv2.VideoCapture(0)
modelo=load_model('modelo1D.h5')
texto="Nada"
#LECTURA Y VISUALIZACION DE LA IMAGEN
def leer_imagen():
    _,imagen=video.read()
    cp_imagen=imagen
    cp_imagen=cv2.putText(cp_imagen,texto,(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 4)
    cv2.imshow('imagen',cp_imagen)
    return cv2.resize(imagen,(128,128),interpolation=cv2.INTER_CUBIC)
#REDIMENSIONAR IMAGEN
def prep_imagen():
    data=np.ndarray((1,3,128,128),dtype=np.uint8)
    data[0]=leer_imagen().T    
    return np.expand_dims(np.array(data[0]),axis=0)
#IDENTIFICAR SENAL
def tipo_senal(array):
    if array.max()>0.50:
        index = np.where(array==array.max())
        if index[1]==0:
            return 'LIMITE'+str(array[index]*100)
        elif index[1]==1:
            return 'PARADERO'+str(array[index]*100)
        elif index[1]==2:
            return 'PARE'+str(array[index]*100)
        elif index[1]==3:
            return 'PROHIBIDO'+str(array[index]*100)
    else:
        return "NADA"
#EJECUCION
while(True):
    frame=leer_imagen()
    frame=prep_imagen()
    predicciones=modelo.predict(frame)
    texto=tipo_senal(predicciones)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()