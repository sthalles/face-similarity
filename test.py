import tensorflow as tf

print(tf.__version__)

tf.enable_eager_execution()

import os
import numpy as np
from lib.densenet import DenseNet
from read_data import *
import matplotlib.pyplot as plt
from contrastive import contrastive_loss
import json

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

tfe = tf.contrib.eager

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 12235,
                            'Model folder name to be loaded.')

test_filenames = ['./dataset_tfrecords/val_32.tfrecords']
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.map(random_resize_and_crop)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(1)

checkpoint_dir = FLAGS.work_dir
checkpoint_dir = os.path.join(checkpoint_dir, str(FLAGS.model_id))

# load training metadata
with open(checkpoint_dir + '/train/meta.json', 'r') as fp:
    training_args = Dotdict(json.load(fp))

args = {"k": training_args.growth_rate,
        "weight_decay": training_args.l2_regularization,
        "num_outputs": training_args.num_outputs,
        "units_per_block": training_args.units_per_block,
        "momentum": training_args.momentum,
        "epsilon": training_args.epsilon,
        "initial_pool": training_args.initial_pool}

model = DenseNet(**args)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

try:
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Model {} successfully loaded.".format(FLAGS.model_id))
except:
    print("Error loading model: {}".format(FLAGS.model_id))

current_best_avg_loss = np.inf
running_loss_avg = 0
mean_similarity = []
mean_dissimilarity = []

for (batch, (Xi, Xj, label)) in enumerate(test_dataset):

    with tf.contrib.summary.record_summaries_every_n_global_steps(100):

        GX1 = model(Xi, training=False)
        GX2 = model(Xj, training=False)
        _, Dw = contrastive_loss(GX1, GX2, label)

        for i in range(label.shape[0]):
            if label[i].numpy() == 0:
                mean_similarity.append(Dw[i])
            else:
                mean_dissimilarity.append(Dw[i])

        f, axarr = plt.subplots(1, 2)
        plt.title('Similariry: ' + str(Dw) + "\tLabel: " + str(label.numpy()))
        axarr[0].imshow(tf.squeeze(Xi))
        axarr[1].imshow(tf.squeeze(Xj))
        plt.show()

        #print("Label:",label)
        #print("Similarity:", Dw)

mean_similarity_np = np.mean(mean_similarity)
mean_dissimilarity_np = np.mean(mean_dissimilarity)

print("Mean similarity of similar images:", mean_similarity_np)
print("Mean similarity of dissimilar images:", mean_dissimilarity_np)


