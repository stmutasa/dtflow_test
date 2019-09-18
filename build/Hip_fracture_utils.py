"""
For the CUMC Test, we will use a garden fracture classifier network from a previous paper
It's a simple custom 10-20 layer network that takes a hip radiograph and tries to predict whether there is:
0: No fracture
1: A nondisplaced femoral neck fracture (Garden I/II)
2: A displaced femoral neck fracture (Garden III/IV)

Data will be split between 3 computers:
exx - 4 GPU CPU in Sachin's office
radphysics - the DGX system on campus
skynet - 2 GPU workstation in Simi's house

This file will contain the utility files
"""

import tensorflow as tf
import SODLoader as SDL
from pathlib import Path

sdl = SDL.SODLoader(str(Path.home()))

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS


# Load the protobuf
def load_protobuf(filenames, training=True, batch_size=32):
    """
    Loads the protocol buffer into a form to send to shuffle
    :param filenames: Name of the tfrecords files
    :param training: Training or testing
    :param batch_size:
    :return:
    """

    # Create a dataset from the protobuf
    dataset = tf.data.TFRecordDataset(filenames)

    # Shuffle
    if training: dataset = dataset.shuffle(buffer_size=1000)

    _records_call = lambda dataset: \
        sdl.load_tfrecords(dataset, [850, 850, 1], tf.float32)

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Warp the data set
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch, cache, prefetch, then repeat
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat()

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

    def __init__(self, distords):
        self._distords = distords

    def __call__(self, record):

        if self._distords:  # Training

            # Data Augmentation ------------------  Random flip, contrast, noise

            # Random flip
            record['data'] = tf.image.random_flip_left_right(tf.image.random_flip_up_down(record['data']))

            # # Random contrast
            record['data'] = tf.image.random_contrast(record['data'], lower=0.97, upper=1.03)

            # Reshape image to 256x256
            record['data'] = tf.image.resize_images(record['data'], [256, 256])

            # Random gaussian noise
            T_noise = tf.random_uniform([1], 0, 0.1)
            noise = tf.random_uniform(shape=[256, 256, 1], minval=-T_noise, maxval=T_noise)
            record['data'] = tf.add(record['data'], tf.cast(noise, tf.float32))

        else:  # Validation

            # Reshape image to 256x256
            record['data'] = tf.image.resize_images(record['data'], [256, 256])

        return record


import SODNetwork as SDN

# Instantialize class
sdn = SDN.SODMatrix(summary=True, phase_train=True)


def forward_pass(images, phase_train=True):
    """
    Perform the forward pass. 256x256 version
    :param images: Input images
    :param phase_train: bool, whether this is the testing or training phase
    :return:
    """

    K = 16

    # First layer is conv
    print('Input Images: ', images)

    # Residual blocks
    conv = sdn.convolution('Conv1', images, 3, K, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual1', conv, 3, K * 2, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual2', conv, 3, K * 4, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual3', conv, 3, K * 8, 2, phase_train=phase_train)  # 16x16
    conv = sdn.residual_layer('Residual4', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual5', conv, 3, K * 16, 2, phase_train=phase_train)
    conv = sdn.inception_layer('Inception6', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual7', conv, 3, K * 32, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual8', conv, 3, K * 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual9', conv, 3, K * 32, 1, phase_train=phase_train)

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True, override=3)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)

    return Logits, L2_loss


"""
Now the total loss calculation
"""


def total_loss(logits, labels):
    """
    For the loss function
    :param logits: calculated raw outputs
    :param labels: real labels
    :return: the loss value
    """

    logits, labels = tf.squeeze(logits), tf.squeeze(labels)

    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=FLAGS.num_classes, dtype=tf.uint8)

    # Calculate  loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labels), logits=logits)

    # Reduce to scalar
    loss = tf.reduce_mean(loss)

    return loss
