import cv2 as cv
# libreria para crear archivos
import os
# para manipular la resolucion de la camara
import imutils

modelo='FotosMartina'
ruta1 = 'C:/Repository/Python OpenCV/reconocimientofacial1'

rutacompleta = ruta1+ '/' + modelo

# en caso de no existir la ruta la crea
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

# abre la camara y cargamos los ruidos (para reconocimiento facial)
camara=cv.VideoCapture(0)
ruidos = cv.CascadeClassifier('C:\Repository\Python OpenCV\Entrenamientos opencv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')

id=0

while True:
    respuesta,captura = camara.read()
    if(respuesta==False):break

    grises = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()
    captura=imutils.resize(captura,width=640)
    cara=ruidos.detectMultiScale(grises,1.3,5)

    #Creamos nuestro rectangulo
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (160,160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/img_{}.jpg'.format(id),rostrocapturado)
        id=id+1


    cv.imshow("resultado Rostro", captura)

    if id==351:
        break

    # if cv.waitKey(1)==ord('s'):
    #     break
camara.release()
cv.destroyAllWindows()