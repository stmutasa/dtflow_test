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
import time, datetime
import Hip_fracture_utils as utils

# Define command line arguments to define how this process will run. TF.app.flags is similar to sys.argv
FLAGS = tf.app.flags.FLAGS

# Define some of the command line arguments
tf.app.flags.DEFINE_integer('task_index', 0,
                            'Index of task within the job')

tf.app.flags.DEFINE_string('ps_hosts', 'localhost:3222',
                           'Comma separated list of hostname:port pairs')

tf.app.flags.DEFINE_string('job_name', 'worker',
                           'Either ps or worker')

tf.app.flags.DEFINE_string('worker_hosts', 'localhost:3224',
                           'Comma separated list of hostname:port pairs')

"""
Network flags
"""

tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('learning_rate', 3e-3, """Initial learning rate""")
tf.app.flags.DEFINE_integer('epoch_size', 2200, """How many images were loaded""")
tf.app.flags.DEFINE_integer('num_epochs', 450, """Number of epochs to run""")

# This represents the per worker batch size
batch_size = FLAGS.batch_size

# We Want to start with a low learning rate n to stabilize distributed training, then increase to k*n after
# 5 epochs with k representing number of replicas per the paper by He et. al
learning_rate = FLAGS.learning_rate


def main(_):

    # Parse the command line arguments to get a lists of parameter servers and hosts
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster specification from the parameter server and worker hosts. Every process has a copy of this
    # To know which other systems it has to wait on
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Define the configuration proto for the session.
    config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Create and start this specific server
    server = tf.distribute.Server(cluster,
                                  job_name=FLAGS.job_name,
                                  task_index=FLAGS.task_index,
                                  config=config)

    # Calculate the global batch size
    global_batch_size = batch_size * len(worker_hosts)

    if FLAGS.job_name == "ps":

        # You need your parameter servers to constantly listen to possible commits from the workers.
        # This is done using the server.join() method.
        # This method tells TensorFlow to block and listen for requests until the server shuts down.
        server.join()

    elif FLAGS.job_name == "worker":

        # Everything run within the context of replica_device_setter will
        # Place any following variables on parameter servers in a round robin manner
        # Everything else is placed on the first device of the worker specified.
        # You can check this with log_device_placement below
        with tf.device(tf.compat.v1.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            # Run the input custom function and bring back the data iterator object
            dataset_iterator = generate_inputs(batch_size)

            # Run the training step custom function. We return the optimizer too to create the sync hook
            train_op, opt, loss = train_step(dataset_iterator.get_next(), global_batch_size)

            # -------------------  Housekeeping functions  ----------------------

            # Merge the summaries
            all_summaries = tf.summary.merge_all()

        """
        Now set up the session for this process
        tf.train.MonitoredTrainingSession, takes care of session initialization, saving, restoring
        and closing when done or an error occurs. It also appears to have special handling of distributed sessions
        """

        # Set the intervals
        max_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs)
        checkpoint_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs) // 100

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):

                # Called before the session is created
                print('\n******Worker %s: Starting monitored session...\n' % FLAGS.task_index)

                # Retreive local filenames, exclude the testing files
                afs = utils.sdl.retreive_filelist('tfrecords', False, path='data/')
                fns = [x for x in afs if 'Test' not in x]
                print('*' * 10, 'Files for worker %s: %s' % (FLAGS.task_index, fns))

                self._step = -1
                self._start_time = time.time()

            def after_create_session(self, session, coord):

                # Called after the session is made and graph is finalized
                print('\n******Worker %s: Session started!! Starting loops\n' % FLAGS.task_index)

            def before_run(self, run_context):

                # Called before each step
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):

                # Called after each step, only if successful
                if self._step % checkpoint_steps == 0:

                    # Display the elapsed time
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = 200 * global_batch_size / duration
                    sec_per_batch = float(duration / 200)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.datetime.now(), self._step, (loss_value * 1e6),
                                        examples_per_sec, sec_per_batch))

        # Make a sync replicas hook that handles initialization and queues. This is key to making the process
        # synchronized (not needed if asynchronous). The 1st parameter needs to know if this is the master worker
        # Num tokens=0 seems to be needed to prevent the master worker from hijacking all the tokens initially
        sync_replicas_hook = opt.make_session_run_hook((FLAGS.task_index == 0), num_tokens=0)

        # Make a profiler hook to track memory and gpu usage
        _ProfilerHook = tf.train.ProfilerHook(save_steps=max_steps // 8, output_dir='data/checkpoints/',
                                              show_memory=True, show_dataflow=True)

        # Make a summary hook
        summary_hook = tf.train.SummarySaverHook(save_steps=max_steps // 10, output_dir='data/checkpoints/', summary_op=all_summaries)

        # Group our hooks for the monitored train sessions.
        hooks = [tf.train.StopAtStepHook(last_step=max_steps), sync_replicas_hook, tf.train.NanTensorHook(train_op),
                 _ProfilerHook, _LoggerHook(), summary_hook]

        # Scafford object for finalizing the graph
        scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(), dataset_iterator.initializer))

        # The MonitoredTrainingSession takes care of a lot of boilerplate code. It also needs to know if this is
        # The master worker
        with tf.compat.v1.train.MonitoredTrainingSession(master=server.target,
                                                         is_chief=(FLAGS.task_index == 0),
                                                         hooks=hooks,
                                                         scaffold=scaffold,
                                                         checkpoint_dir='data/checkpoints/',
                                                         save_checkpoint_steps=max_steps // 6) as mon_sess:

            while not mon_sess.should_stop():

                # Run a training step
                mon_sess.run(train_op)

            if mon_sess.should_stop():
                print('Training Done, Shutting down...')
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
    images, labels = inputs['data'], inputs['label2']

    # Define input shape
    images = tf.reshape(images, [FLAGS.batch_size, 256, 256, 1])

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(images[0], shape=[1, 256, 256, 1]), 4)

    # Calculate the logits
    logits, l2_loss = define_model(images)

    # Calculate loss
    loss = calculate_loss(logits, labels)

    # Scale the total loss by the GLOBAL and not local (replica) batch size.
    loss *= (1.0 / global_batch_size)

    # Add in L2 term
    loss += l2_loss

    # Update the moving average batch norm ops
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Retreive the training operation with the applied gradients
    with tf.control_dependencies(extra_update_ops):
        # Return the training operation optimizer
        train_op, opt = calculate_optimizer(loss)

    return train_op, opt, loss


def generate_inputs(local_batch_size):
    """
    Function to generate inputs and return a tf.data iterator
    :param local_batch_size: total batch size for this worker
    :return:
    """

    # Retreive local filenames, exclude the testing files
    all_files = utils.sdl.retreive_filelist('tfrecords', False, path='data/')
    filenames = [x for x in all_files if 'Test' not in x]

    print('*' * 10, 'Files for worker %s: %s' % (FLAGS.task_index, filenames))

    # Return data as a dictionary
    return utils.load_protobuf(filenames, training=True, batch_size=local_batch_size)


def define_model(inputs):
    """
    Defines the model
    :param inputs: tf.data inputs
    :return: The calculated logits
    """

    # Actually returns logits and L2 loss
    return utils.forward_pass(images=inputs, phase_train=True)


def calculate_loss(logits, labels):
    """
    Calculates the loss
    :param logits: Logits, output from the model
    :param labels: The labels from tf.data
    :return: The total loss
    """

    # Calculate the loss
    loss = utils.total_loss(logits, labels)

    # Output the losses
    tf.summary.scalar('Cross Entropy', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

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

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', loss)

    # Specify the optimizer name scope
    with tf.name_scope('train'):
        # Construct the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Maintain average weights to smooth out training
        variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)

        # Applies the average to the variables in the trainable ops collection
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        """
        This is the point of synchronization by modifying the regular optimizer with the sync_replicas_optimizer.
        We need to tell it how many replicas we want (x) so that at each step the optimizer collects x gradients
        before applying them to the variables.
        Prevents stale gradients but drawbacks are that this will slow us on a heterogenous cluster
        Uses a token system to prevent dead workers from stopping training
        """

        # First retreive the number of replicas
        replicas = len(FLAGS.worker_hosts.split(","))
        print('\n*******Number of Replicas: %s\n*********' % replicas)

        # Then introduce the custom sync_replicas optimizer
        # This is also where you would sync the moving averages with the variable_averages argument
        optimizer = tf.compat.v1.train.SyncReplicasOptimizer(optimizer,
                                                             replicas_to_aggregate=replicas,
                                                             total_num_replicas=replicas,
                                                             variable_averages=variable_averages,
                                                             variables_to_average=tf.trainable_variables())

        # Construct the training operation
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
        for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

        # Control graph execution. i.e. before you return, make sure train op is evaluated
        with tf.control_dependencies([train_op, variable_averages_op]):
            return train_op, optimizer


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)
