import argparse, glob, os
import constants as c
import tensorflow as tf

def conv_layers(scope_name, inpt, channel_sizes, kernels, strides):
    """ Makes arbitrary convolutional layers on top of given layer. Returns
        flattened result (ready for FC & output layers).

        :param scope_name: The variable scope name for the conv layers
        :param inpt: The existing layer to build on top of
        :param channel_sizes: The desired number of channels after each conv layer
        :param kernels: The kernel for each conv layer
        :param strides: The stried for each conv layer

        :return: Flattened resulting tensor after new conv layers
    """
    with tf.variable_scope(scope_name, reuse=False):
        conv_tensor = inpt
        for channels, kernel_len, stride_len in zip(channel_sizes, kernels, strides):
            kernel = (kernel_len, kernel_len)
            stride = (1, stride_len, stride_len, 1)
            conv_tensor = tf.layers.conv2d(conv_tensor, filters=channels,
                                           kernel_size=kernel,
                                           strides=stride, padding=c.PADDING,
                                           activation=tf.nn.relu,
                                           name="%dx%d" % (kernel_len, kernel_len))

        conv_shape = conv_tensor.shape
        flattened_conv = tf.reshape(conv_tensor, shape=[conv_shape[0].value, -1],
                                    name="flattened")
    return flattened_conv

def fc_layers(scope_name, inpt, channel_sizes):
    """ Makes arbitrary fully-connected layers on top of given layer. Returns
        resulting tensor.

        :param scope_name: The variable scope name for the FC layers
        :param inpt: The existing layer to build on top of
        :param channel_sizes: The desired number of channels after each conv layer

        :return: The resulting tensor after new FC layers
    """
    with tf.variable_scope(scope_name, reuse=False):
        fc_layer = inpt
        for i, channels in enumerate(channel_sizes):
            fc_layer = tf.layers.dense(fc_layer, channels, name='fc_%d' % i,
                                       activation=tf.nn.relu)
    return fc_layer

def gen_batches(tup):
    """ Returns a generator of batches, given data.

        :param tup: All the data to return in batches; tuple of
                    (images, labels, feedback)

        :return: A generator of batches, each represented as a tuple of
                 (images, labels, feedback)
    """
    imgs, labels, feedback = tup
    for s in xrange(0, len(imgs), c.BATCH_SIZE):
        e = s + c.BATCH_SIZE
        yield imgs[s:e], labels[s:e], feedback[s:e]

def delete_model_files():
	""" Deletes existing files in model and summaries directories. """
	model_files = glob.glob(os.path.join(c.MODEL_DIR, "*"))
	summary_files = glob.glob(os.path.join(c.SUMMARY_DIR, "*"))
	for f in model_files+summary_files:
		os.remove(f)

def get_args():
    """ Parses args and returns a parser. """
    parser = argparse.ArgumentParser()

    parser.add_argument('--new',
                        help="Delete old model.",
                        dest='new',
                        action='store_true',
                        default=False)

    parser.add_argument('--val',
                        help="Validate model.",
                        dest='val',
                        action='store_true',
                        default=False)

    parser.add_argument('--train',
                        help="Train model.",
                        dest='train',
                        action='store_true',
                        default=False)

    return parser.parse_args()
