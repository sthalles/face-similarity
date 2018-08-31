import tensorflow as tf

tf.enable_eager_execution()
import numpy as np
tfe = tf.contrib.eager
from lib.densenet import DenseNet
from sklearn.metrics import accuracy_score
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def normalize(image, label):
    return tf.to_float(tf.expand_dims(image, 2)), label

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(normalize)
train_dataset = train_dataset.shuffle(256)
train_dataset = train_dataset.batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(normalize)
test_dataset = test_dataset.shuffle(1024)
test_dataset = test_dataset.batch(64)

def loss(predictions, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,10), logits=predictions))

learning_rates = [0.01, 0.001, 0.0001]
regularizations = [1e-3, 1e-4, 1e-5]

def apply_gradients(optimizer, grads, vars_, global_step=None):
  """Functional style apply_grads for `tfe.defun`."""
  optimizer.apply_gradients(zip(grads, vars_), global_step=global_step)

counter = 0
for learning_rate in learning_rates:
    for weight_decay in regularizations:

        print("Experiment #: {0}, Learning rate {1}, weight decay: {2}".format(counter, learning_rate, weight_decay))

        model = DenseNet(k=8, weight_decay=weight_decay, num_outputs=10,
                         units_per_block=[6, 6], momentum=0.997, epsilon=0.001,
                         initial_pool=False)

        #model.call = tfe.defun(model.call)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


        # Training loop
        for (i, (x, y)) in enumerate(train_dataset):
            # Calculate derivatives of the input function with respect to its parameters.

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                train_loss_np = loss(logits, y)

            grads = tape.gradient(train_loss_np, model.variables)

            # Apply the gradient to the model
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=tf.train.get_or_create_global_step())


        avg_loss =[]
        avg_acc = []
        for (i, (x_, y_)) in enumerate(test_dataset):
            logits = model(x_, training=False)
            val_loss_np = loss(logits, y_)
            avg_loss.append(val_loss_np.numpy())
            pred = tf.argmax(logits, 1)
            acc = accuracy_score(y_, pred)
            avg_acc.append(acc)

        print("Testing (avg) loss: {0}, Val (avg) accuracy: {1}".format(np.mean(avg_loss), np.mean(avg_acc)))
        counter += 1
        print("---------------------------")
