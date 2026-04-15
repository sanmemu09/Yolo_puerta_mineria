import os
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# MTCNN: detecta y alinea la cara
# keep_all=False → solo la cara más prominente
# post_process=False → devuelve tensor crudo listo para el modelo
mtcnn = MTCNN(
    image_size=160,       # FaceNet espera 160x160
    margin=20,            # Margen alrededor de la cara detectada
    keep_all=False,       # Una sola cara por foto
    post_process=False,   # Sin normalización extra, la haremos manual
    device=device,
    thresholds=[0.5, 0.6, 0.6]  # Default es [0.6, 0.7, 0.7]
)

# InceptionResnetV1: genera el embedding de 512 dimensiones
# pretrained='vggface2' → entrenado en VGGFace2, muy robusto para reconocimiento facial
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Modelos cargados correctamente en:", device)


dataset_path = "dataset"
embeddings = []
formatos_validos = ('.jpg', '.jpeg', ".png")

for role in sorted(os.listdir(dataset_path)):        # Estudiantes / Profesor
    role_path = os.path.join(dataset_path, role)

    if not os.path.isdir(role_path):
        continue

    for person in sorted(os.listdir(role_path)):     # Valentina, Juan...
        person_path = os.path.join(role_path, person)

        if not os.path.isdir(person_path):
            continue

        fotos_procesadas = 0
        fotos_omitidas = 0

        for img_name in sorted(os.listdir(person_path)):

            # Filtrar solo jpg y jpeg
            if not img_name.lower().endswith(formatos_validos):
                continue

            img_path = os.path.join(person_path, img_name)

            try:
                # Cargar imagen
                img = Image.open(img_path).convert('RGB')

                # Detectar y alinear cara → tensor [3, 160, 160]
                face_tensor = mtcnn(img)

                if face_tensor is None:
                    print(f"  Omitida {img_name} de {person}: No se detectó rostro.")
                    fotos_omitidas += 1
                    continue

                # Normalizar manualmente: de [0,255] a [-1,1]
                face_tensor = (face_tensor / 127.5) - 1.0

                # Agregar dimensión batch → [1, 3, 160, 160]
                face_tensor = face_tensor.unsqueeze(0).to(device)

                # Generar embedding en GPU, sin calcular gradientes
                with torch.no_grad():
                    embedding = resnet(face_tensor)

                # Convertir a lista para guardar en pkl
                embedding_list = embedding[0].cpu().numpy().tolist()

                embeddings.append({
                    "name": person,
                    "role": role,
                    "embedding": embedding_list
                })

                fotos_procesadas += 1
                print(f"{person} - {img_name}")

            except Exception as e:
                print(f" Error en {img_name} de {person}: {e}")
                fotos_omitidas += 1

        print(f"  → {person}: {fotos_procesadas} procesadas, {fotos_omitidas} omitidas\n")

# Guardar embeddings en archivo .pkl
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Resumen final
print(f"\n{'='*50}")
print(f"  Proceso finalizado.")
print(f"  Total de embeddings guardados: {len(embeddings)}")

# Desglose por persona
from collections import Counter
conteo = Counter(e["name"] for e in embeddings)
print(f"\n  Desglose por persona:")
for nombre, cantidad in sorted(conteo.items()):
    print(f"    - {nombre}: {cantidad} embedding(s)")

print(f"\n  Archivo guardado: embeddings.pkl")
print(f"{'='*50}")
