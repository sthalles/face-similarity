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

tfe = tf.contrib.eager

tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 2482,
                            'Model folder name to be loaded.')

FLAGS = tf.app.flags.FLAGS

filenames = ['/home/thalles/Documents/datasets/celeb_faces/dataset_tfrecords/test.tfrecords']
test_dataset = tf.data.TFRecordDataset(filenames)
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.map(central_image_crop)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(1)

# model = DenseNet(k=8)
model = ConvNet()

checkpoint_dir = FLAGS.work_dir
checkpoint_dir = os.path.join(checkpoint_dir, str(FLAGS.model_id))

writer = tf.contrib.summary.create_file_writer(checkpoint_dir)
global_step = tf.train.get_or_create_global_step()
writer.set_as_default()

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

try:
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))
except:
    print("Error")

current_best_avg_loss = np.inf
running_loss_avg = 0
for (batch, (Xi, Xj, label)) in enumerate(test_dataset):

    with tf.contrib.summary.record_summaries_every_n_global_steps(100):

        for (batch, (Xi, Xj, label)) in enumerate(test_dataset):

            GX1 = model(Xi, training=False)
            GX2 = model(Xj, training=False)
            Dw = tf.norm((GX1 - GX2), ord="euclidean").numpy()

            f, axarr = plt.subplots(1, 2)
            plt.title('Similariry: ' + str(Dw) + "\tLabel: " + str(label.numpy()))
            axarr[0].imshow(tf.squeeze(Xi))
            axarr[1].imshow(tf.squeeze(Xj))
            plt.show()
            #print("Label:",label)
            #print("Similarity:", Dw)
