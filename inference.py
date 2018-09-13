import tensorflow as tf

print(tf.__version__)

tf.enable_eager_execution()

import os
from lib.densenet import DenseNet
from pre_processing import *
import matplotlib.pyplot as plt
from contrastive import contrastive_loss
import json
from utils import Dotdict

tfe = tf.contrib.eager

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 5708,
                            'Model folder name to be loaded.')

test_filenames = ['./dataset_tfrecords/test_200k.tfrecords']
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.map(random_resize_and_crop)
test_dataset = test_dataset.map(normalizer)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(8)

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

mean_similarity = []
mean_dissimilarity = []

for (batch, (Xi, Xj, label)) in enumerate(test_dataset):

    with tf.contrib.summary.record_summaries_every_n_global_steps(100):

        GX1 = model(Xi, training=False)
        GX2 = model(Xj, training=False)
        _, Dw = contrastive_loss(GX1, GX2, label, margin=2.)

        f, axarr = plt.subplots(2, 8, figsize=(16,4))
        f.subplots_adjust(hspace=0.3)

        for i in range(label.shape[0]):

            Si = denormalize(Xi[i]).numpy()
            Sj = denormalize(Xj[i]).numpy()

            if label[i].numpy() == 0:
                mean_similarity.append(Dw[i])
            else:
                mean_dissimilarity.append(Dw[i])

            axarr[0, i].set_title('Sim: ' + str(Dw[i].numpy()))
            print('Similariry: ' + str(Dw[i].numpy()) + "\tLabel: " + str(label[i].numpy()))
            axarr[0,i].imshow(np.squeeze(Si))
            axarr[0,i].set_axis_off()

            axarr[1,i].set_title("Label: " + str(label[i].numpy()))
            axarr[1,i].imshow(np.squeeze(Sj))
            axarr[1,i].set_axis_off()

        plt.show()

mean_std_similarity_np = np.std(mean_similarity)
mean_std_dissimilarity_np = np.std(mean_dissimilarity)
mean_similarity_np = np.mean(mean_similarity)
mean_dissimilarity_np = np.mean(mean_dissimilarity)

print("Mean similarity {0} Mean Std: {1}.".format(mean_similarity_np, mean_std_similarity_np))
print("Mean dissimilarity {0} Mean Std: {1}.".format(mean_dissimilarity_np, mean_std_dissimilarity_np))


