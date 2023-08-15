import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ruta de directorio raíz de datos
data_dir = r"C:\Users\diego\Documents\Programacion\Python\Deep Learning\dataset"

# Rutas de entrenamiento y validación
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Tamaño de imagen de entrada
img_size = (224, 224)

# Generador de imágenes de entrenamiento
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary')

# Generador de imágenes de validación
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))

export_path = 'C:/Users/diego/Documents/Programacion/Python/Deep Learning/modelo_guardado'

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True
)

