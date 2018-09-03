import tensorflow as tf

print(tf.__version__)

tf.enable_eager_execution()

import json
import os
import numpy as np
from lib.densenet import DenseNet
from read_data import *
import matplotlib.pyplot as plt
from contrastive import contrastive_loss
from lib.cyclical_lr import cyclical_learning_rate

tfe = tf.contrib.eager

tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('eval_model_every_n_steps', 1000,
                            'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_id', None,
                            'Model folder name to be loaded.')
tf.app.flags.DEFINE_float('max_learning_rate', 3e-3,
                            'Maximum learning rate value.')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Number of training pairs per iteration.')
tf.app.flags.DEFINE_integer('growth_rate', 12,
                            'Densenet growth_rate factor.')
tf.app.flags.DEFINE_float('l2_regularization', 0.009,
                            'Weight decay regularization penalty.')
tf.app.flags.DEFINE_integer('num_outputs', 16,
                            'Number of output units for DenseNet.')
tf.app.flags.DEFINE_list('units_per_block', [6,12,24,16],
                            'DenseNet units and blocks architecture.')
tf.app.flags.DEFINE_float('momentum', 0.997,
                            'Momentum for batch normalization.')
tf.app.flags.DEFINE_float('epsilon', 0.001,
                            'Epsilon for batch normalization.')
tf.app.flags.DEFINE_bool('initial_pool', True,
                            'Should the DenseNet include the first pooling layer.')

FLAGS = tf.app.flags.FLAGS

train_filenames = ['./dataset_tfrecords/train_1.tfrecords', './dataset_tfrecords/train_2.tfrecords']
train_dataset = tf.data.TFRecordDataset(train_filenames)
train_dataset = train_dataset.map(tf_record_parser, num_parallel_calls=2)
train_dataset = train_dataset.map(random_flip_left_right, num_parallel_calls=2)
# train_dataset = train_dataset.map(random_image_rotation, num_parallel_calls=2)
train_dataset = train_dataset.map(random_resize_and_crop, num_parallel_calls=2)
#train_dataset = train_dataset.map(random_distortions, num_parallel_calls=2)
# train_dataset = train_dataset.map(normalizer)
train_dataset = train_dataset.repeat(50)
train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.batch(FLAGS.batch_size)

test_filenames = ['./dataset_tfrecords/val_200k.tfrecords']
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)
# test_dataset = test_dataset.map(random_image_rotation)
test_dataset = test_dataset.map(random_resize_and_crop)
# test_dataset = test_dataset.map(normalizer)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(256)

args = {"k": FLAGS.growth_rate,
        "weight_decay": FLAGS.l2_regularization,
        "num_outputs": FLAGS.num_outputs,
        "units_per_block": FLAGS.units_per_block,
        "momentum": FLAGS.momentum,
        "epsilon": FLAGS.epsilon,
        "initial_pool": FLAGS.initial_pool}

base_lr = FLAGS.max_learning_rate / 3
max_beta1 = 0.95
base_beta1 = 0.85

# number_of_examples = sum(1 for _ in tf.python_io.tf_record_iterator(train_filenames[0]))
# epoch_length = number_of_examples // FLAGS.batch_size
# stepsize = 4 * epoch_length

stepsize = 8196
# print("number_of_examples {0}, epoch_length: {1}, stepsize: {2}".format(number_of_examples,epoch_length,stepsize))

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
# optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate_tf, weight_decay=FLAGS.weight_decay,
#                                           beta1=beta1_tf, beta2=0.99)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf,
                                   beta1=beta1_tf, beta2=0.99)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

# print(dict(FLAGS.flag_values_dict()))
with open(checkpoint_dir + "/train/" + 'meta.json', 'w') as fp:
    json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=4)
    print("Training meta-file saved.")

if FLAGS.model_id is not None:
    try:
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("Model {0} restored with success.".format(FLAGS.model_id))
    except:
        print("Error loading model id {0}".format(FLAGS.model_id))

current_best_val_avg_loss = np.inf

for (batch, (Xi, Xj, train_labels)) in enumerate(train_dataset):

    with train_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):

        with tf.GradientTape() as tape:
            # siamese net inference
            GX1 = model(Xi, training=True)
            GX2 = model(Xj, training=True)

            # compute contrastive loss
            train_loss_np, _ = contrastive_loss(GX1, GX2, train_labels)
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
        mean_val_loss = []
        for (batch, (Xi, Xj, val_labels)) in enumerate(test_dataset):

            GX1 = model(Xi, training=False)
            GX2 = model(Xj, training=False)
            val_loss_np, Dw = contrastive_loss(GX1, GX2, val_labels)
            mean_val_loss.append(val_loss_np.numpy())

            for i in range(val_labels.shape[0]):
                if val_labels[i].numpy() == 0:
                    mean_similarity.append(Dw[i])
                else:
                    mean_dissimilarity.append(Dw[i])

        mean_val_loss = np.mean(mean_val_loss)
        if mean_val_loss < current_best_val_avg_loss:
            current_best_val_avg_loss = mean_val_loss
            # save the model
            root.save(file_prefix=checkpoint_prefix)
            print(
                "Model saved. Best avg validation loss: {0}\t Global step: {1}".format(current_best_val_avg_loss, global_step.numpy()))

        mean_similarity_np = np.mean(mean_similarity)
        mean_dissimilarity_np = np.mean(mean_dissimilarity)
        print("Mean similarity of similar images:", mean_similarity_np)
        print("Mean similarity of dissimilar images:", mean_dissimilarity_np)
        with val_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', mean_val_loss)
            tf.contrib.summary.scalar('mean_similarity', mean_similarity_np)
            tf.contrib.summary.scalar('mean_dissimilarity', mean_dissimilarity_np)
            tf.contrib.summary.scalar('embedding_mean_distance', np.abs(mean_similarity_np - mean_dissimilarity_np))

