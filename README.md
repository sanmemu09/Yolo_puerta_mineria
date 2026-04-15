# Sistema de Control de Acceso - UPB
Sistema de reconocimiento facial para control de acceso a puertas, desarrollado con YOLO, MTCNN y FaceNet.

Realizado por:
   -Santiago Mendoza Muñoz
   -Camilo Armenta
   -Valentina Rendon
   
---

## Requisitos del sistema

- Python 3.11
- Camara web
- GPU Nvidia con CUDA (opcional, pero recomendado para mejor rendimiento)

---

## Estructura del proyecto

```
smart-door-access-yolo/
│
├── dataset/                    # Fotos organizadas por rol y nombre
│   ├── Estudiantes/
│   │   ├── Nombre Apellido/
│   │   │   ├── foto1.jpg
│   │   │   └── foto2.jpeg
│   └── Profesor/
│       └── Nombre Apellido/
│           └── foto1.jpg
│
├── src/
│   ├── generate_embeddings.py  # Genera el archivo embeddings.pkl
│   └── real_time_access.py     # Sistema de acceso en tiempo real
│
├── embeddings.pkl              # Base de datos de embeddings (ya generado)
├── requirements.txt
└── README.md
```

---

## Instalacion

### Paso 1: Instalar PyTorch

**Si tienes GPU Nvidia:**

Verifica tu version de CUDA con:
```bash
nvidia-smi
```

Luego instala PyTorch con soporte CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Si NO tienes GPU Nvidia (solo CPU):**
```bash
pip install torch torchvision torchaudio
```

### Paso 2: Instalar facenet-pytorch sin romper PyTorch

```bash
pip install facenet-pytorch --no-deps
```

> IMPORTANTE: No omitas el --no-deps. Sin el, pip puede reemplazar tu instalacion de PyTorch por una version incompatible.

### Paso 3: Instalar el resto de dependencias

```bash
pip install opencv-python ultralytics pillow numpy
```

---

## Verificacion de instalacion

Corre este script para confirmar que todo esta bien:

```python
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

print("PyTorch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No detectada (se usara CPU)")
print("facenet-pytorch: OK")
```

---

## Uso

### Generar embeddings (solo si agregas fotos nuevas)

Si el archivo `embeddings.pkl` ya existe, puedes saltarte este paso.

```bash
python src/generate_embeddings.py
```

El script recorre la carpeta `dataset/`, detecta las caras y guarda los embeddings en `embeddings.pkl`.

### Iniciar el sistema de acceso

```bash
python src/real_time_access.py
```

Presiona **ESC** para cerrar el sistema.

---

## Como agregar una persona nueva

1. Crea una carpeta con su nombre dentro del rol correspondiente:
   ```
   dataset/Estudiantes/Nombre Apellido/
   ```
2. Agrega entre 3 y 5 fotos en formato `.jpg` o `.jpeg`.
   - Fotos con buena iluminacion
   - Cara centrada y visible
   - Evitar perfiles o fotos borrosas
3. Vuelve a correr `generate_embeddings.py` para regenerar el `.pkl`.

---

## Comportamiento del sistema

| Color del recuadro | Significado |
|---|---|
| Verde | Persona reconocida, acceso concedido |
| Rojo | Persona no autorizada |
| Naranja | No se detecto un rostro claro |

Cuando se concede acceso:
- La pantalla muestra **PUERTA ABIERTA** durante 5 segundos
- Se registra en consola con hora, nombre y rol

---

## Rendimiento esperado

| Configuracion | Velocidad aproximada |
|---|---|
| GPU Nvidia (CUDA) | Tiempo real fluido |
| CPU | 1-3 segundos por frame procesado |

---

## Notas tecnicas

- Modelo de deteccion de personas: `yolov8n.pt` (COCO, clase 0 = persona)
- Modelo de deteccion facial: `MTCNN` de `facenet-pytorch`
- Modelo de reconocimiento: `InceptionResnetV1` preentrenado en `vggface2`
- Distancia usada: Euclidiana
- Umbral de reconocimiento: `0.8` (ajustable en `real_time_access.py`)
