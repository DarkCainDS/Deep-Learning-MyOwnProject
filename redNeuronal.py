import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

# cargar los datos de entrenamiento y prueba
train_data = ...
test_data = ...

# definir la arquitectura de la red neuronal
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(2, activation='softmax')

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# crear una instancia del modelo
model = MyModel()

# definir la función de pérdida y el optimizador
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# definir las métricas a seguir durante el entrenamiento
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# definir la función de entrenamiento
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # realizar una predicción con el modelo y calcular la pérdida
    predictions = model(images)
    loss = loss_object(labels, predictions)
  # calcular los gradientes y actualizar los pesos del modelo
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # actualizar las métricas
  train_loss(loss)
  train_accuracy(labels, predictions)

# definir la función de prueba
@tf.function
def test_step(images, labels):
  # realizar una predicción con el modelo y calcular la pérdida
  predictions = model(images)
  t_loss = loss_object(labels, predictions)
  # actualizar las métricas
  test_loss(t_loss)
  test_accuracy(labels, predictions)

# entrenar el modelo
EPOCHS = 10

for epoch in range(EPOCHS):
  # reiniciar las métricas
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  # entrenar el modelo
  for images, labels in train_data:
    train_step(images, labels)

  # evaluar el modelo en los datos de prueba
  for test_images, test_labels in test_data:
    test_step(test_images, test_labels)

  # imprimir las métricas de entrenamiento y prueba
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
model.save('modelo.h5')

