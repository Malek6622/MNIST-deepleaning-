import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
data, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
train, test = data['train'], data['test']
num = 0.1 * info.splits['train'].num_examples
num = tf.cast(num, tf.int64)
numt = info.splits['test'].num_examples
numt= tf.cast(numt, tf.int64)
def scale(img, a):
    img = tf.cast(img, tf.float32)
    img /= 255.
    return img, a
s_t_v = train.map(scale)
test_d = test.map(scale)
bu = 10000
s_t_val = s_t_v.shuffle(bu)
val = s_t_val.take(num)
train_d = s_t_val.skip(num)
ba = 100
train_d = train_d.batch(ba)
val = val.batch(num)
test_d = test_d.batch(numt)
v_in, v_t = next(iter(val))
layers = 50
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(layers, activation='relu'), 
    tf.keras.layers.Dense(layers_, activation='relu'), 
    tf.keras.layers.Dense(output_size, activation='softmax') 
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 5
model.fit(train_data, epochs=epochs, validation_data=(v_in, v_t), verbose =2)
test_loss, test_accuracy = model.evaluate(test_d)
print(test_loss, test_accuracy*100)
