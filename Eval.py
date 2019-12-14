"""
Distributed tensorflow example.

We aim to create a program that starts an arbitrary amount of containers
and networks them together with the goal of making a distributed computing tensorflow system
In this case, it will be a synchronized training session.

Note this uses the old tensorflow distributed computing libraries introduced in 0.10:
 https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md. The new version
at https://www.tensorflow.org/guide/distribute_strategy does not currently support custom training loops.

LOG Device placement results after sanity checks.
- Weights and biases being placed on parameter servers in round robin
- Convolutions being placed on current worker
- Nothing being placed on any other worker from this worker
- Global step var being placed on PS0
- train/gradients/convolutions placed on current worker
- train/GradientDescent/update_b1 placed on parameter servers (apply gradient descent)
- Input data being placed on the specific worker

TODO:
- Custom dataset
- Deploy on K8's cluster: This code is working on a single minikube node in the "kubernetes" branch of this project

"""

# This is the baseline process that every container will have a copy of
import tensorflow as tf
import time, os
import Utils as utils
import SODTester as SDT
import glob

# Define command line arguments to define how this process will run. TF.app.flags is similar to sys.argv
FLAGS = tf.app.flags.FLAGS

"""
Network flags
"""

tf.app.flags.DEFINE_integer('batch_size', 105, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('learning_rate', 3e-3, """Initial learning rate""")
tf.app.flags.DEFINE_integer('epoch_size', 105, """How many images were loaded""")

# This represents the per worker batch size
batch_size = FLAGS.batch_size

# We Want to start with a low learning rate n to stabilize distributed training, then increase to k*n after
# 5 epochs with k representing number of replicas per the paper by He et. al
learning_rate = FLAGS.learning_rate

root = '/home/stmutasa/PycharmProjects/dtflow_test/'


# Define a custom training class
def test():
    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Run the input custom function and bring back the data iterator object
        dataset_iterator = generate_inputs(batch_size)

        # Get the inputs.
        examples = dataset_iterator.get_next()
        images, labels, accno = examples['data'], examples['label2'], examples['hip_id']

        # Define input shape
        images = tf.reshape(images, [FLAGS.batch_size, 256, 256, 1])

        # Display the images
        tf.summary.image('Test IMG', tf.reshape(images[0], shape=[1, 256, 256, 1]), 4)

        # Calculate the logits
        logits, _ = utils.forward_pass(images=images, phase_train=False)

        # -------------------  Housekeeping functions  ----------------------

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(0.999)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_MAE, best_epoch = 0.25, 0

        # Tester instance
        sdt = SDT.SODTester(True, False)

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=(root + 'data/checkpoints/'))

                # Initialize iterator
                mon_sess.run([var_init, dataset_iterator.initializer])

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the model
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)

                    # model_checkpoint_path: "data/checkpoints/model.ckpt-1604"
                    Epoch = int(ckpt.model_checkpoint_path.split('-')[-1])

                else:
                    print('No checkpoint file found')
                    break

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = FLAGS.epoch_size // FLAGS.batch_size

                # Tester instance
                sdt = SDT.SODTester(True, False)

                try:
                    while step < max_steps:
                        # Load some metrics for testing
                        lbl1, logtz, pt = mon_sess.run([labels, logits, accno])

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    datad, _, _ = sdt.combine_predictions(lbl1, logtz, pt, FLAGS.batch_size)
                    sdt.calculate_metrics(logtz, lbl1, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print('------ Current Best AUC: %.4f (Epoch: %s) --------' % (best_MAE, best_epoch))

                    # Lets save runs that perform well
                    if sdt.AUC >= best_MAE:
                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filenames
                        checkpoint_file = os.path.join((root + 'data/'), ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC)))
                        csv_file = os.path.join((root + 'data/'), ('E_%s_AUC_%0.2f.csv' % (Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)
                        utils.sdl.save_Dict_CSV(datad, csv_file)

                        # Save a new best MAE
                        best_MAE = sdt.AUC
                        best_epoch = Epoch

                    # Shut down the session
                    mon_sess.close()

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob((root + 'data/checkpoints/') + '*.index')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:
                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob((root + 'data/checkpoints/') + '*.index')


def generate_inputs(batch_size):
    """
    Function to generate inputs and return a tf.data iterator
    :param local_batch_size: total batch size for this worker
    :return:
    """

    # Retreive local filenames, exclude the testing files
    all_files = utils.sdl.retreive_filelist('tfrecords', False, path=(root + 'data/'))
    filenames = [x for x in all_files if 'Test' in x]

    # Return data as a dictionary
    return utils.load_protobuf(filenames, training=False, batch_size=batch_size)


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    test()


if __name__ == '__main__':
    tf.app.run()
