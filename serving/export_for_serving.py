import tensorflow as tf

print(tf.__version__)

# tf.enable_eager_execution()

import os
from lib.densenet import DenseNet
from pre_processing import *
import matplotlib.pyplot as plt
from contrastive import contrastive_loss
import json
from utils import Dotdict

tfe = tf.contrib.eager

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('work_dir', '../tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 31911,
                            'Model folder name to be loaded.')

checkpoint_dir = FLAGS.work_dir
checkpoint_dir = os.path.join(checkpoint_dir, str(FLAGS.model_id))

# load training metadata
with open(checkpoint_dir + '/meta.json', 'r') as fp:
    training_args = Dotdict(json.load(fp))

args = {"k": training_args.growth_rate,
        "weight_decay": training_args.l2_regularization,
        "num_outputs": training_args.num_outputs,
        "units_per_block": training_args.units_per_block,
        "momentum": training_args.momentum,
        "epsilon": training_args.epsilon,
        "initial_pool": training_args.initial_pool}

model = DenseNet(**args)

checkpoint_prefix = os.path.join(checkpoint_dir, "face_sim_model.ckpt")
root = tfe.Checkpoint(model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

try:
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Model {} successfully loaded.".format(FLAGS.model_id))
except:
    print("Error loading model: {}".format(FLAGS.model_id))

model(tf.ones((1,128,128,3), dtype=tf.float32), training=False)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save(file_prefix=checkpoint_prefix)
