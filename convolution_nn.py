import tensorfow as tf



def model(x, y):
    x = tf.reshape(x, shape=[-1, x, y, 4])