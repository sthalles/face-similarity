import tensorflow as tf

print(tf.__version__)

tf.enable_eager_execution()

import os
import numpy as np
from lib.densenet import DenseNet
from read_data import *
import matplotlib.pyplot as plt
from contrastive import contrastive_loss
from models.convnet import ConvNet
from lib.cyclical_lr import cyclical_learning_rate

tfe = tf.contrib.eager

tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('eval_model_every_n_steps', 500,
                            'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_id', None,
                            'Model folder name to be loaded.')
tf.app.flags.DEFINE_float('max_learning_rate', 0.001,
                            'Maximum learning rate value.')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Number of training pairs per iteration.')

FLAGS = tf.app.flags.FLAGS

train_filenames = ['/home/thalles/PycharmProjects/eager-dense-nets/dataset_tfrecords/train.tfrecords']
train_dataset = tf.data.TFRecordDataset(train_filenames)
train_dataset = train_dataset.map(tf_record_parser, num_parallel_calls=2)
train_dataset = train_dataset.map(random_flip_left_right, num_parallel_calls=2)
# train_dataset = train_dataset.map(random_image_rotation, num_parallel_calls=2)
train_dataset = train_dataset.map(random_resize_and_crop, num_parallel_calls=2)
#train_dataset = train_dataset.map(random_distortions, num_parallel_calls=2)
# train_dataset = train_dataset.map(normalizer)
train_dataset = train_dataset.repeat(100)
train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.batch(FLAGS.batch_size)

test_filenames = ['/home/thalles/PycharmProjects/eager-dense-nets/dataset_tfrecords/val.tfrecords']
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)
# test_dataset = test_dataset.map(random_image_rotation)
test_dataset = test_dataset.map(random_resize_and_crop)
# test_dataset = test_dataset.map(normalizer)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(256)

args = {"k": 12,
        "weight_decay": 2e-4,
        "num_outputs": 16,
        "units_per_block": [6,12,24,16],
        "momentum": 0.99,
        "epsilon": 0.001,
        "initial_pool": True}

base_lr = FLAGS.max_learning_rate / 3
max_beta1 = 0.95
base_beta1 = 0.85

number_of_examples = sum(1 for _ in tf.python_io.tf_record_iterator(train_filenames[0]))
epoch_length = number_of_examples // FLAGS.batch_size
stepsize = 4 * epoch_length

#stepsize = 3124
print("number_of_examples {0}, epoch_length: {1}, stepsize: {2}".format(number_of_examples,epoch_length,stepsize))

get_lr_and_beta1 = cyclical_learning_rate(base_lr=base_lr, max_lr=FLAGS.max_learning_rate,
                                        max_mom=max_beta1, base_mom=base_beta1,
                                        stepsize=stepsize,
                                        decrease_base_by=0.15)

model = DenseNet(**args)

checkpoint_dir = FLAGS.work_dir

if FLAGS.model_id is None:
    process_id = os.getpid()
    print("Running instance #:", process_id)
else:
    process_id = FLAGS.model_id
    print("Prepare to load model id {0}".format(FLAGS.model_id))

checkpoint_dir = os.path.join(checkpoint_dir, str(process_id))

train_writer = tf.contrib.summary.create_file_writer(os.path.join(checkpoint_dir, "train"))
val_writer = tf.contrib.summary.create_file_writer(os.path.join(checkpoint_dir, "val"))

learning_rate_tf = tfe.Variable(base_lr)
beta1_tf = tfe.Variable(max_beta1)

global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf, beta1=beta1_tf, beta2=0.99)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

if FLAGS.model_id is not None:
    try:
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("Model {0} restored with success.".format(FLAGS.model_id))
    except:
        print("Error loading model id {0}".format(FLAGS.model_id))

current_best_val_avg_loss = np.inf
running_val_avg_loss = 0.0
for (batch, (Xi, Xj, label)) in enumerate(train_dataset):

    with train_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):

        with tf.GradientTape() as tape:
            # siamese net inference
            GX1 = model(Xi, training=True)
            GX2 = model(Xj, training=True)

            # compute contrastive loss
            train_loss_np, _ = contrastive_loss(GX1, GX2, label)
            tf.contrib.summary.scalar('loss', train_loss_np)
            tf.contrib.summary.scalar('learning_rate', learning_rate_tf)
            tf.contrib.summary.scalar('beta1', beta1_tf)

        # compute grads w.r.t model parameters and update weights
        grads = tape.gradient(train_loss_np, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

        new_lr, new_beta1 = next(get_lr_and_beta1)
        learning_rate_tf.assign(new_lr)
        beta1_tf.assign(new_beta1)

    if global_step.numpy() % FLAGS.eval_model_every_n_steps == 0:
        mean_similarity = []
        mean_dissimilarity = []

        counter = 0
        for (batch, (Xi, Xj, labels)) in enumerate(test_dataset):

            GX1 = model(Xi, training=False)
            GX2 = model(Xj, training=False)
            val_loss_np, Dw = contrastive_loss(GX1, GX2, labels)
            running_val_avg_loss += val_loss_np.numpy()
            counter += 1

            for i in range(labels.shape[0]):
                if labels[i].numpy() == 0:
                    mean_similarity.append(Dw[i])
                else:
                    mean_dissimilarity.append(Dw[i])

        running_val_avg_loss /= counter # get mean validation loss

        if running_val_avg_loss < current_best_val_avg_loss:
            current_best_val_avg_loss = running_val_avg_loss
            # save the model
            root.save(file_prefix=checkpoint_prefix)
            print(
                "Model saved. Best avg loss: {0}\t Global step: {1}".format(current_best_val_avg_loss, global_step.numpy()))

        mean_similarity_np = np.mean(mean_similarity)
        mean_dissimilarity_np = np.mean(mean_dissimilarity)
        print("Mean similarity of similar images:", mean_similarity_np)
        print("Mean similarity of dissimilar images:", mean_dissimilarity_np)
        with val_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', val_loss_np)
            tf.contrib.summary.scalar('mean_similarity', mean_similarity_np)
            tf.contrib.summary.scalar('mean_dissimilarity', mean_dissimilarity_np)
            tf.contrib.summary.scalar('embedding_mean_distance', np.abs(mean_similarity_np - mean_dissimilarity_np))

