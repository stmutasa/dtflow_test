"""
Distributed tensorflow example.

We aim to create a program that starts an arbitrary amount of containers
and networks them together with the goal of making a distributed computing server

LOG Device placement results:
- Weights and biases being placed on parameter servers in round robin - CORRECT
- Convolutions being placed on current worker - CORRECT
- Nothing being placed on any other worker from this worker - CORRECT
- Global step var being placed on PS0 - CORRECT
- train/gradients/convolutions placed on current worker
- train/GradientDescent/update_b1 placed on parameter servers (apply gradient descent)
- Input data being placed on the specific worker - CORRECT

TODO:
- Sanity checks
- Get a real example running

"""

# This is the baseline file that each worker will have reference to.

import tensorflow as tf
import os, time, datetime
import numpy as np

# Define command line arguments to define how it will run as ps or worker
FLAGS = tf.app.flags.FLAGS

# Define some of the command line arguments
tf.app.flags.DEFINE_integer('task_index', 0,
                           'Index of task within the job')

tf.app.flags.DEFINE_string('ps_hosts', 'localhost:3222,localhost:3223',
                           'Comma separated list of hostname:port pairs')

tf.app.flags.DEFINE_string('job_name', 'worker',
                           'Either ps or worker')

tf.app.flags.DEFINE_string('worker_hosts', 'localhost:3224,localhost:3225',
                           'Comma separated list of hostname:port pairs')

# Set tensorflow verbosity: 0 = all, 1 = no info, 2 = no info/warning, 3 = nothing
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# config
batch_size = 10

# We Want to start with a low learning rate n to stabilize distributed training, then increase to k*n after
# 5 epochs with k representing number of replicas per the paper by He et. al
learning_rate = 0.0005

def main(_):

    # Parse the command line arguments to get a lists of parameter servers and hosts
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.distribute.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    # Calculate the global batch size
    global_batch_size = batch_size * len(worker_hosts)

    if FLAGS.job_name == "ps":

        # You need your parameter servers to constantly listen to possible commits from the workers.
        # This is done using the server.join() method.
        # This method tells TensorFlow to block and listen for requests until the server shuts down.
        server.join()

    elif FLAGS.job_name == "worker":

        # Place any following variables on parameter servers in a round robin manner
        # Everything else is placed on the first device of the worker specified.
        with tf.device(tf.compat.v1.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            # Retreive the data iterator object
            dataset_iterator = generate_inputs(batch_size)

            # Run the training step. We return the optimizer too to create the sync hook
            # This is lazy programming, should just make more classes
            train_op, opt = train_step(dataset_iterator.get_next(), global_batch_size)

            # Global step defined on a random ps
            global_step = tf.train.get_or_create_global_step()

        """
        Now set up the session for this process
        """

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(train_op)

            def after_run(self, run_context, run_values):
                if self._step % 200 == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = 200 * global_batch_size / duration
                    sec_per_batch = float(duration / 200)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        # Define the configuration proto for the session.
        config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Make a sync replicas hook that handles init and queues. True if this is chief worker
        sync_replicas_hook = opt.make_session_run_hook((FLAGS.task_index == 0), num_tokens=0)

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=1000000), sync_replicas_hook, tf.train.NanTensorHook(train_op), _LoggerHook()]

        # The MonitoredTrainingSession takes care of session initialization, saving, restoring
        # and closing when done or an error occurs.
        print ('\n******Worker %s: Starting monitored session...\n' %FLAGS.task_index)
        with tf.compat.v1.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=(FLAGS.task_index == 0),
                                                hooks=hooks,
                                                config=config) as mon_sess:

            print('\n******Worker %s: Session started!! Starting loops\n' % FLAGS.task_index)
            mon_sess.run(dataset_iterator.initializer)

            while not mon_sess.should_stop():

                # Run a training step
                ce, st = mon_sess.run([train_op, global_step])
                #print('Cross Entropy %s, step %s' % (ce, st))

            if mon_sess.should_stop():
                print ('Training Done, Shutting down...')
                pass

                # Terminate everything here

def train_step(inputs, global_batch_size):

    """
    Defines one step of the training.
    :param inputs: Inputs from the data iterator
    :param global_batch_size: worker batch size * num of workers
    :return: The reduced loss
    """

    # Get the inputs.
    images, labels = inputs

    # Calculate the logits
    logits = define_model(images)

    # Calculate loss
    loss = calculate_loss(logits, labels)

    # Scale the total loss by the GLOBAL and not local (replica) batch size.
    loss *= (1.0/global_batch_size)

    # Return the training operation optimizer
    train_op, opt = calculate_optimizer(loss)

    return train_op, opt


def generate_inputs(local_batch_size):

    """
    Function to generate inputs and return a tf.data iterator
    :param local_batch_size: total batch size for this worker
    :return:
    """

    # Make a dummy mnist batch
    dummy_inputs = np.random.uniform(-1.0, 1.0, [200, 784]).astype((np.float32))
    dummy_labels = np.random.randint(0, 10, [200]).astype(np.float32)

    # Assume the size is the same
    assert dummy_inputs.shape[0] == dummy_labels.shape[0]

    # Create the dataset object. Shuffle, then batch it, then make it repeat indefinitely
    dataset= tf.data.Dataset.from_tensor_slices((dummy_inputs, dummy_labels)).shuffle(200).repeat().batch(local_batch_size)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Add to collection
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    # Return data as a dictionary
    return iterator


def define_model(inputs):

    """
    Defines a rudimentary model
    :param inputs: tf.data inputs
    :return: The calculated logits
    """

    # Biases to be placed on the ps in round robin
    with tf.name_scope('biases'):

        b1 = tf.compat.v1.get_variable('b1', [100], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=True)
        b2 = tf.compat.v1.get_variable('b2', [10], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=True)

    # Weights, for the ps's in round robin
    with tf.name_scope('weights'):

        # Dummy weights
        W1 = tf.compat.v1.get_variable('W1', [784, 100], dtype=tf.float32, initializer=tf.initializers.he_normal(), trainable=True)
        W2 = tf.compat.v1.get_variable('W2', [100, 10], dtype=tf.float32, initializer=tf.initializers.he_normal(), trainable=True)

    # Actual calculations, copied on to all the workers
    with tf.name_scope('convolutions'):

        # y is our prediction
        conv1 = tf.add(tf.matmul(inputs, W1), b1)
        conv1_a = tf.nn.sigmoid(conv1)
        conv2 = tf.add(tf.matmul(conv1_a, W2), b2)
        logits = tf.nn.softmax(conv2)

    return logits


def calculate_loss(logits, labels):
    """
    Calculates the loss
    :param logits: Logits, output from the model
    :param labels: The labels from tf.data
    :return: The total loss
    """

    # Cost function name scope
    with tf.name_scope('cross_entropy'):

        loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(labels, tf.float32) * tf.log(logits), reduction_indices=[1]))

    return loss


def calculate_optimizer(loss):

    """
    Calculates the gradients and returns the tensorflow optimizer object
    This is also the point of synchronization
    :param loss: The calculated total loss
    :return: train_op
    """

    # Retreive global step variable (saved on ps:0 typically)
    global_step = tf.train.get_or_create_global_step()

    # Specify the optimizer name scope
    with tf.name_scope('train'):

        # Construct the optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        """
        This is the point of synchronization by modifying the regular optimizer with the sync_replicas_optimizer.
        We need to tell it how many replicas we want (x) so that at each step the optimizer collects x gradients
        before applying them to the variables.
        Prevents stale gradients but drawbacks are that this will slow us on a heterogenous cluster
        Uses a token system to prevent dead workers from stopping training
        """
        replicas = len(FLAGS.worker_hosts.split(","))
        print('\n*******Number of Replicas: %s\n*********'%replicas)

        # Also where you would maintain moving averages with the variable_averages argument
        optimizer = tf.compat.v1.train.SyncReplicasOptimizer(optimizer,
                                                             replicas_to_aggregate=replicas,
                                                             total_num_replicas=replicas)

        # Construct the training operation
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Control graph execution. i.e. before you return, make sure train op is evaluated
        with tf.control_dependencies([train_op]):
            return train_op, optimizer



if __name__ == "__main__":
    if tf.io.gfile.exists("/tmp/train_logs"):
        tf.io.gfile.rmtree("/tmp/train_logs")
    tf.io.gfile.makedirs("/tmp/train_logs")
    tf.compat.v1.app.run(main=main)