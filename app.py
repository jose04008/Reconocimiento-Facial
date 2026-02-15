import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import os
import numpy as np

# 1. Configuración de rutas y carga de datos
# Adaptado de tu archivo 'Reconociemiento Facial.py'
DATAPATH = 'Data' 
MODELO_PATH = 'modeloLBPHFace.xml'

# Intentamos cargar los nombres de las carpetas (personas)
if os.path.exists(DATAPATH):
    imagePaths = os.listdir(DATAPATH)
else:
    imagePaths = []

# Inicializar el reconocedor y el clasificador
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(MODELO_PATH):
    face_recognizer.read(MODELO_PATH)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Clase para procesar el video en tiempo real
class FaceProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convertimos el frame de la web a formato OpenCV
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        
        # Detección de rostros
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            
            # Predicción con tu modelo entrenado
            result = face_recognizer.predict(rostro)

            # Lógica de etiquetas (basada en tu código original)
            if result[1] < 70:
                nombre = imagePaths[result[0]]
                color = (0, 255, 0) # Verde si lo conoce
            else:
                nombre = 'Desconocido'
                color = (0, 0, 255) # Rojo si no lo conoce

            # Dibujar en el frame
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{nombre}", (x, y-25), 2, 1, color, 1, cv2.LINE_AA)

        return img

# 3. Interfaz de Usuario de Streamlit
def main():
    st.set_page_config(page_title="IA Facial - Ing. Industrial", layout="wide")
    
    st.title("🛡️ Sistema de Reconocimiento Facial")
    st.sidebar.header("Opciones del Proyecto")
    
    menu = ["Visualización", "Configuración"]
    choice = st.sidebar.selectbox("Menú", menu)

    if choice == "Visualización":
        st.subheader("Cámara en vivo")
        st.write("Presiona 'Start' para iniciar la detección basada en el modelo LBPH.")
        
        # Este componente reemplaza al while True y cap.read()
        webrtc_streamer(
    key="facerecog",
    video_transformer_factory=FaceProcessor,
    rtc_configuration={  # Esto ayuda a establecer la conexión a través de firewalls
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

    elif choice == "Configuración":
        st.subheader("Estado del Sistema")
        st.write(f"**Personas en base de datos:** {len(imagePaths)}")
        st.write(imagePaths)
        if st.button("Verificar Modelo"):
            if os.path.exists(MODELO_PATH):
                st.success("Modelo 'modeloLBPHFace.xml' cargado correctamente.")
            else:
                st.error("No se encontró el archivo del modelo.")

if __name__ == "__main__":
    main()