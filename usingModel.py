from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo entrenado
model = load_model('ruta/al/modelo')

# Cargar una imagen de prueba
img_path = 'ruta/a/la/imagen/de/prueba.jpg'
img = image.load_img(img_path, target_size=(150, 150))

# Preprocesar la imagen para que sea compatible con el modelo
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Realizar la predicción
pred = model.predict(x)

# Imprimir el resultado
if pred[0] > 0.5:
    print("La imagen contiene el objeto de interés.")
else:
    print("La imagen NO contiene el objeto de interés.")
