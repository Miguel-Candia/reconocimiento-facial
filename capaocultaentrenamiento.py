import cv2 as cv
import os
import numpy as np
# tiempo de entrenamiento
from time import time

dataRuta='C:/Repository/Python OpenCV/reconocimientofacial1/Data'

listData = os.listdir(dataRuta)
print('data',listData)

ids=[]
rostrosData=[]
id=0
timeStart= time()
for fila in listData:
    rutacompleta= dataRuta+'/'+fila
    print('Iniciando lectura...')
    for archivo in os.listdir(rutacompleta):
        print('Imagenes: ',fila+ '/'+archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0))

    id=id+1

    timeEndRead=time()
    timeWholeRead =timeEndRead - timeStart
    print('Tiempo Total de lectura: ',timeWholeRead)

trainingEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
print('============ Inicio Entrenamiento===========')
timeEndTraining=time()

trainingEigenFaceRecognizer.train(rostrosData, np.array(ids))

timeWholeTraining= timeEndTraining - timeWholeRead

print('tiempo entrenamiento total: ',timeWholeTraining)
# Guardamos nuestro entrenamiento
trainingEigenFaceRecognizer.write('trainingEigenFaceRecognizer.xml')

print('============ Fin Entrenamiento===========')






