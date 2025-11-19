import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ------------------------
# 1. CARREGAR MODELO
# ------------------------
MODEL_PATH = "final_CNN_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Modelo carregado com sucesso!")
except Exception as e:
    st.error("Erro ao carregar o modelo: " + str(e))
    st.stop()

# ------------------------
# FUNÇÃO DE PRÉ-PROCESSAMENTO SEM OPENCV
# ------------------------
def preprocess(pil_img):
    # converter PIL → numpy em escala de cinza
    img = np.array(pil_img).astype("uint8")

    # binarização simples
    img = np.where(img > 80, 255, 0).astype("uint8")

    # inverter (MNIST = fundo preto, número branco)
    img = 255 - img

    # encontrar pixels brancos (o número)
    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        return np.zeros((28, 28, 1), dtype="float32")

    # encontrar bounding box (equivalente ao cv2.boundingRect)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # cortar ao redor do dígito
    img_crop = img[y_min:y_max+1, x_min:x_max+1]

    # redimensionar mantendo proporção para 20x20
    pil_crop = Image.fromarray(img_crop)
    pil_crop = pil_crop.resize((20, 20), Image.Resampling.LANCZOS)

    # criar 28x28 com padding 4px
    padded = Image.new("L", (28, 28))
    padded.paste(pil_crop, (4, 4))

    # converter para array normalizado
    arr = np.array(padded).astype("float32") / 255.0
    return arr.reshape(28, 28, 1)

# ------------------------
# 2. CANVAS PARA DESENHAR
# ------------------------
st.subheader("Desenhe seu número:")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_color="#FFFFFF",
    background_color="#000000",
    stroke_width=8,
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ------------------------
# 3. BOTÃO DE PREDIÇÃO
# ------------------------
if st.button("Reconhecer Dígito"):
    if canvas.image_data is None:
        st.warning("Desenhe algo primeiro!")
    else:
        # imagem do canvas → PIL
        img = canvas.image_data
        pil_img = Image.fromarray(img.astype("uint8")).convert("L")

        st.subheader("Imagem usada:")
        st.image(pil_img.resize((140, 140)))

        # aplicar pré-processamento NOVO (sem cv2)
        arr = preprocess(pil_img)
        arr = arr.reshape(1, 28, 28, 1)

        # previsão
        pred = model.predict(arr, verbose=0)
        numero = np.argmax(pred)

        st.subheader(f"Resultado: **{numero}**")

        # probabilidades
        st.write("Probabilidades por classe:")
        for i, p in enumerate(pred[0]):
            st.write(f"{i}: {p:.4f}")

def preprocess(pil_img):
    # converter PIL → NumPy (grayscale)
    img = np.array(pil_img).astype(np.uint8)

    # remover ruído do fundo
    img = np.where(img < 50, 0, img)
    img = np.where(img > 200, 255, img)

    # inverter (MNIST = traço claro)
    img = 255 - img

    # detectar pixels do dígito
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # recortar a região ativa
    img_crop = img[y_min:y_max+1, x_min:x_max+1]

    # calcular nova proporção (igual MNIST)
    h, w = img_crop.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    # redimensionar mantendo proporção
    img_resized = Image.fromarray(img_crop).resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_resized = np.array(img_resized)

    # criar canvas 28×28
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # calcular offsets para centralizar
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    # colar dígito centralizado
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    # normalizar
    canvas = canvas.astype("float32") / 255.0

    return canvas.reshape(28, 28, 1)
