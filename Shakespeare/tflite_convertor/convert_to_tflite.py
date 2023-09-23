import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

"""Define the base model LSTM.

To be compatible with TFLite Model Personalization, we need to define a
base model and a head model. 

Here we are using an identity layer for base model, which just passes the 
input as it is to the head model.
"""
chars: str = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
seq_len = 80
# set sub-model size
p=0.95

hidden_size = int(128 * p)
base = tf.keras.Sequential(
    [tf.keras.Input(shape=(seq_len,seq_len,)), tf.keras.layers.Lambda(lambda x: x)]
)
base.summary()
base.compile(loss="categorical_crossentropy", optimizer="adam")
run_model = tf.function(lambda x: base(x))
# This is important, let's fix the input size.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([None, len(chars), seq_len], base.inputs[0].dtype))
base.save("identity_model", save_format="tf", signatures=concrete_func)

"""Define the head model.

This is the model architecture that we will train using Flower. 
"""

# Set Model
head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(seq_len,seq_len,)),
        tf.keras.layers.LSTM(hidden_size,return_sequences=True),
        tf.keras.layers.LSTM(hidden_size,return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(chars), activation="softmax", ),
    ]
)

head.compile(loss="categorical_crossentropy", optimizer="adam")
head.summary()

run_model = tf.function(lambda x: head(x))
# This is important, let's fix the input size.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([None, len(chars), len(chars)], head.inputs[0].dtype))
# model directory.
MODEL_DIR = "keras_lstm"
head.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

model = tf.keras.models.load_model(MODEL_DIR)
model.summary()
#model = head

"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.bottleneck.tflite
inference.tflite
initialize.tflite
optimizer.tflite
train_head.tflite
Copy them in your Android code under the assets/model directory.
"""


base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    80, base_path, heads.KerasModelHead(model), optimizers.Adam(0.001), train_batch_size=128
)
converter.convert_and_save("tflite_model")
