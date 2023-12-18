import cv2 as cv
import os


dataRuta='C:/Repository/Python OpenCV/reconocimientofacial1/Data'
listData = os.listdir(dataRuta)

trainingEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()

trainingEigenFaceRecognizer.read('trainingEigenFaceRecognizer.xml')


ruidos = cv.CascadeClassifier('C:\Repository\Python OpenCV\Entrenamientos opencv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)

while True:
    _,captura=camara.read()
    grises = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (160,160), interpolation=cv.INTER_CUBIC)
        rostrocapturado = cv.cvtColor(rostrocapturado, cv.COLOR_BGR2GRAY)
        result = trainingEigenFaceRecognizer.predict(rostrocapturado)
        cv.putText(captura,'{}'.format(result), (x,y-5), 1,1.3 , (0,255,0),1,cv.LINE_AA)
        if result[1] < 9000:
            cv.putText(captura,'{}'.format(listData[result[0]]), (x,y-20), 2,1.1 , (255,0,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)
        else:
            cv.putText(captura,"Desconocido", (x,y-20), 2,0.7 , (0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)



        
    cv.imshow('Resultados', captura)

    if cv.waitKey(1)==ord('s'):
        break
camara.release()
cv.destroyAllWindows()



