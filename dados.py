#Ejemplo de captura de webcam
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
   
    # Mostrar el frame en una ventana
    cv2.imshow("Webcam", frame)

    # Salir del bucle si se presiona la tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

imagen = frame
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
if gray is None:
    print("IMAGEN NO LEIDA")
    exit(1)

imagenCanny = cv2.Canny(gray, 200, 350)

cv2.imshow("Imagen Canny", imagenCanny)
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Original", gray)


imagen_copia = imagen.copy()
imagen_final = imagen.copy()


(contornos,_) = cv2.findContours(imagenCanny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_filtrados = [] #Almacenamos los filtrados

#Recorremos todos los contornos, con un factor epsilon relativamente laxo para detectar los poligonos. A partir de ahí, en función del area
#(para descartar objetos muy pequeños), y de los lados, nos quedamos con los cuadrados/rectangulos correspondientes.
for c in contornos:
    epsilon = 0.05*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    print(len(approx))
    x,y,w,h = cv2.boundingRect(approx)
    #cv2.putText(imagen,'Area: ' + str(cv2.contourArea(c)), (x,y-5),1,1,(0,255,0),1)
    if cv2.contourArea(c) >= 280 and len(approx) ==4:
        cv2.putText(imagen, 'Lados: ' +str(len(approx)) + ' Area: ' + str(cv2.contourArea(c)) , (x,y-5),1,1,(0,255,0),1)
        contornos_filtrados.append(c)



cv2.drawContours(imagen,contornos_filtrados,-1,(0,0,255), 2)
#cv2.imshow("contornos", img_contours)
cv2.imshow("contornos2", imagen)

#Guardo los contornos en una imagen negra del mismo tamaño que la original, para tener solo lo que nos interesa en la siguiente imagen a procesar.
stencil = np.zeros(imagen_copia.shape).astype(imagen.dtype)
color = [255, 255, 255]
cv2.fillPoly(stencil, contornos_filtrados, color)
result = cv2.bitwise_and(imagen_copia, stencil)
cv2.imwrite("result.jpg", result)



gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
imagenCanny2 = cv2.Canny(gray2, 200, 350)

cv2.imshow("Imagen Canny", imagenCanny2)

contornos, jerarquia = cv2.findContours(imagenCanny2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Filtramos los contornos internos basandonos en su area
contornos_internos = []
for i in range(len(contornos)):
    area = cv2.contourArea(contornos[i])
    if area < 200:
        contornos_internos.append(contornos[i])

#Los dibujamos en la imagen original.
cv2.drawContours(imagen, contornos_internos, -1, (0, 0, 255), 2)
cv2.drawContours(result,contornos_internos,-1,(0,0,255), 2)
#cv2.imshow("contornos", img_contours)
cv2.imshow("canny2", imagenCanny2)

cv2.imshow("contornos3", result)


#Calculamos cuantos contornos internos tiene cada contorno externo dentro.
for contorno_externo in contornos_filtrados:
    contornos_dentro = 0
    for contorno_interno in contornos_internos:
        todos_dentro = True
        puntos_interno = contorno_interno.reshape(-1, 2).astype(float)
        for punto in puntos_interno:
            x, y = punto
            if cv2.pointPolygonTest(contorno_externo, (x, y), False) < 0:
                todos_dentro = False
                break
        if todos_dentro:
            contornos_dentro += 1
    print("Contornos internos dentro del contorno externo: {}".format(contornos_dentro))
    puntotexto = random.choice(contorno_externo.reshape(-1, 2).astype(float))
    x, y = puntotexto 
    cv2.putText(imagen_final, 'Ha salido un ' + str(int(contornos_dentro/2)), (int(x),int(y)-5), 1, 1, (0,255,0), 1) #Por algún motivo encuentra siempre el doble de contornos de los que deberia, así que le aplico un factor de conversion

cv2.imshow("final", imagen_final)

cv2.waitKey(0)
cv2.destroyAllWindows()