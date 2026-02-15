import cv2
import os
import numpy as np

labels = []
facesData = []
label = 0
datapath = r'C:\Users\user\Desktop\Reconocimiento Facial_python\Data'
peopleList = os.listdir(datapath)
print('Lista de personas : ', peopleList)

for nameDir in peopleList:
    personPath = os.path.join(datapath, nameDir) # Mejor usar join para rutas
    print('Leyendo las imágenes de la carpeta: ' + nameDir)
    
    for fileName in os.listdir(personPath):
        print('Rostros: ' + nameDir + '/' + fileName)
        
        # --- CORRECCIÓN AQUÍ ---
        imagePath = os.path.join(personPath, fileName)
        image = cv2.imread(imagePath, 0) # El '0' la lee directamente en escala de grises
        
        if image is not None:
            # Es vital que todas las fotos tengan el mismo tamaño (ej. 150x150)
            # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
            facesData.append(image)
            labels.append(label)
        # -----------------------
        
    label = label + 1

# Crear el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenar el modelo
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")