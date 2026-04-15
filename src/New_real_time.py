import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Modelos
# YOLO: detecta si hay una persona en el frame
yolo = YOLO("yolov8n.pt")

# MTCNN: detecta y alinea la cara dentro del recorte de la persona
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=False,
    device=device,
    thresholds=[0.5, 0.6, 0.6]  # Mismo umbral que en generate_embeddings
)

# FaceNet: genera el embedding de 512 dimensiones
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Modelos cargados en:", device)

# Cargar base de datos de embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array([item["embedding"] for item in data])
known_names      = [item["name"] for item in data]
known_roles      = [item["role"] for item in data]

print(f"Base de datos cargada: {len(data)} embeddings de {len(set(known_names))} personas")

# Umbral de distancia euclidiana
# InceptionResnetV1 con vggface2 produce vectores normalizados
# Distancia < 0.9 → misma persona (ajustable según pruebas)
UMBRAL = 0.9

def reconocer_rostro(face_crop_bgr):
    """
    Recibe un recorte BGR de OpenCV (zona de la persona detectada por YOLO).
    Retorna (nombre, rol, distancia) si reconoce, o (None, None, None) si no.
    """

    # Convertir BGR → RGB (PIL)
    img_rgb = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

    # MTCNN detecta y alinea la cara dentro del recorte
    face_tensor = mtcnn(img_rgb)

    if face_tensor is None:
        return None, None, None  # No se detectó cara clara

    # Normalizar: [0,255] → [-1,1]
    face_tensor = (face_tensor / 127.5) - 1.0

    # Agregar dimensión batch y mover a GPU
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Generar embedding
    with torch.no_grad():
        embedding = resnet(face_tensor)[0].cpu().numpy()

    # Calcular distancia euclidiana contra todos los embeddings conocidos
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_dist  = np.min(distances)
    index     = np.argmin(distances)

    if min_dist < UMBRAL:
        return known_names[index], known_roles[index], min_dist
    else:
        return None, None, min_dist
    
# Cámara 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Control de FPS: procesar reconocimiento cada N frames
# YOLO corre en todos los frames, pero el embedding solo cada SKIP_FRAMES
SKIP_FRAMES = 3
frame_count  = 0

# Cache del último resultado para mostrarlo entre frames procesados
last_results = {}  # {box_id: (label, color)}

import time

puerta_abierta    = False
tiempo_apertura   = None
DURACION_PUERTA   = 5  # segundos

print("Sistema de acceso iniciado... (ESC para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    frame_count += 1
    procesar = (frame_count % SKIP_FRAMES == 0)

    # YOLO detecta personas
    results = yolo(frame, verbose=False)[0]
    current_results = {}

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])

        if cls != 0:  # Solo personas
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if procesar:
            # Recorte de la persona detectada por YOLO
            margin      = 30
            x1c = max(0, x1 - margin)
            y1c = max(0, y1 - margin)
            x2c = min(frame.shape[1], x2 + margin)
            y2c = min(frame.shape[0], y2 + margin)
            face_crop   = frame[y1c:y2c, x1c:x2c]

            if face_crop.size == 0:
                continue

            # Reconocimiento facial
            nombre, rol, distancia = reconocer_rostro(face_crop)

            if nombre:
                color = (0, 255, 0)   # Verde: acceso concedido
                label = f"{nombre} ({rol}) {distancia:.2f}"
                if not puerta_abierta:
                    puerta_abierta  = True
                    tiempo_apertura = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] ACCESO CONCEDIDO - {nombre} ({rol})")
                 
            elif distancia is not None:
                color = (0, 0, 255)   # Rojo: no autorizado
                label = f"NO AUTORIZADO {distancia:.2f}"
            else:
                color = (0, 165, 255) # Naranja: buscando rostro
                label = "BUSCANDO ROSTRO..."

            

            current_results[i] = (label, color, x1, y1, x2, y2)

        elif i in last_results:
            # Reusar resultado del frame anterior
            current_results[i] = last_results[i]

    last_results = current_results

    # Verificar si la puerta debe cerrarse
    if puerta_abierta and (time.time() - tiempo_apertura >= DURACION_PUERTA):
        puerta_abierta = False
        print(f"[{time.strftime('%H:%M:%S')}] PUERTA CERRADA")

    # Log de no autorizado (solo cada SKIP_FRAMES para no saturar consola)
    if procesar:
        for i, (label, color, x1, y1, x2, y2) in current_results.items():
            if label.startswith("NO AUTORIZADO"):
                print(f"[{time.strftime('%H:%M:%S')}] ACCESO DENEGADO - Persona no autorizada")

    # Dibujar resultados
    for i, (label, color, x1, y1, x2, y2) in current_results.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS en pantalla 
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Mostrar estado de la puerta en pantalla
    if puerta_abierta:
        segundos_restantes = DURACION_PUERTA - int(time.time() - tiempo_apertura)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 255, 0), -1)
        cv2.putText(frame, "PUERTA ABIERTA", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.putText(frame, f"Cerrando en {segundos_restantes}s", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 180), -1)
        cv2.putText(frame, "PUERTA CERRADA", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    cv2.imshow("Control de Acceso - UPB", frame)

    if cv2.waitKey(1) == 27:  # ESC para salir
        break

# Cierre
cap.release()
cv2.destroyAllWindows()
print("Sistema de acceso cerrado correctamente.")