import tensorfow as tf
import numpy as np
import logging
import time
import traceback

nodes_per_layer = 2500
hm_epochs = 10
batch_size = 100

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DNN():

    def __init_(self):
        pass

    def fit(self, x_train, x_test, y_train, y_test):
        start_time = time.time()
        x = tf.placeholder('float', [None, self.input_width])
        y = tf.placeholder('float', [None, 1])
        prediction, prob = self.neural_network_model(nodes_per_layer, x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver(tf.global_variables())

        learning_log = []
        output_log = []
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:x_test, y:y_test, prob:1})
        learning_log.append((0, accuracy_float, None))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(x_train):
                try:
                    start = i
                    end = i + batch_size
                    batch_x = np.array(x_train[start:end])
                    batch_y = np.array(y_train[start:end])
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, prob: .5})
                    epoch_loss += c
                    i += batch_size
                    logger.info(
                        "Batch {0} of epoch {1} completed, loss: {2}, time:{3}".format(i / batch_size, epoch + 1, c,
                                                                                       time.time() - start_time))
                except:
                    traceback.print_exc()
                    list_batch = batch_x.tolist()
                    print(list_batch)

            logger.info("Epoch {0} completed out of {1}, loss: {2}".format(epoch + 1, hm_epochs, epoch_loss))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob: 1.0})
            accuracy_float = accuracy.eval(session=sess, feed_dict={x: x_test, y: y_test, prob: 1})
            learning_log.append((epoch + 1, accuracy_float, epoch_loss))
            for i in learning_log:
                logger.info(i)




def neural_network_model(x, input_width):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_width, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, 1])),
                        'biases': tf.Variable(tf.random_normal([1]))}

    prob = tf.placeholder_with_default(1.0, shape=())
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    l2_dropout = tf.nn.dropout(l2, prob)
    l3 = tf.add(tf.matmul(l2_dropout, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.leaky_relu(l3)
    l3_dropout = tf.nn.dropout(l3, prob)
    output = tf.add(tf.matmul(l3_dropout , output_layer['weights']), output_layer['biases'])
    return output, prob