# Date: Apr 20, 2023
# Developer: Bekzod Abduxalilov
# Language: Python

import numpy
import cv2

prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel' 
min_aniqlik = 0.2

classes = ["Orqa fon","Samalyot","Velosiped","Qush","Kema","Baklashka","Avtobus",
           "Mashina","Mushuk","Stol","Sigir","Ovqatlanish Stoli","Kuchuk","Ot","Mototsikl","Odam",
           "Tuvakdagi Gul","Qo'y","Divan","Poyezd","Televizor monitori"]

numpy.random.seed(543210)
ranglar = numpy.random.uniform(0, 255, size=(len(classes), 3))
# obyektdagi to'rtburchak ramkadagi ranglarni turli xil bo'lishini ta'minlaydi

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
camera = cv2.VideoCapture(0)

while True:
    _, rasm = camera.read()
    
    boyi, eni = rasm.shape[0], rasm.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(rasm, (700,700)), 0.007, (300,300), 130)

    net.setInput(blob)
    aniqlangan_obyektlar = net.forward()


    for i in range(aniqlangan_obyektlar.shape[2]):
        aniqlik = aniqlangan_obyektlar[0][0][i][2]
    
        if aniqlik > min_aniqlik:
            class_index = int(aniqlangan_obyektlar[0][0][i][1])
        
            yuqori_chap_x = int(aniqlangan_obyektlar[0][0][i][3] * eni)
            yuqori_chap_y = int(aniqlangan_obyektlar[0][0][i][4] * boyi)
            pastki_ong_x = int(aniqlangan_obyektlar[0][0][i][5] * eni)
            pastki_ong_y = int(aniqlangan_obyektlar[0][0][i][6] * boyi)
        
            prediction_text = f"{classes[class_index]}: {aniqlik:.2f}%"
            cv2.rectangle(rasm, (yuqori_chap_x, yuqori_chap_y), (pastki_ong_x, pastki_ong_y), ranglar[class_index], 3)
            cv2.putText(rasm, prediction_text, (yuqori_chap_x, 
                    yuqori_chap_y - 15 if yuqori_chap_y > 30 else yuqori_chap_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ranglar[class_index], 2)

    cv2.namedWindow("Aniqlangan obyektlar", cv2.WINDOW_NORMAL)
    cv2.imshow("Aniqlangan obyektlar", rasm)
    cv2.waitKey(5)   

        

     
cv2.destroyAllWindows()
camera.release()

    
    
