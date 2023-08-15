import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import ImageGrab

# Cargamos el modelo guardado
model = tf.keras.models.load_model('C:/Users/diego/Documents/Programacion/Python/Deep Learning/modelo_guardado')


# Definimos una función para capturar una imagen de la pantalla y hacer una predicción con el modelo
def detect_object():
    # Capturamos una imagen de la pantalla
    img = ImageGrab.grab()
    # La convertimos en un arreglo de Numpy
    img_array = img_to_array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # Normalizamos la imagen

    # Hacemos una predicción en la imagen capturada
    predictions = model.predict(img_array)
    if predictions[0] < 0.5:
        print('El objeto NO se encuentra en pantalla')
    else:
        print('El objeto SÍ se encuentra en pantalla')

# Ejecutamos la función
detect_object()
