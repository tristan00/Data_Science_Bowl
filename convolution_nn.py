import tensorfow as tf



def model(features, labels, mode):
    x = tf.reshape(features, shape=[-1, features, 1, 1])

    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)