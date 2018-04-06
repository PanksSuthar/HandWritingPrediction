
# Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys



# TBD
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#TBD
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#TBD
def weightVariables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#TBD
def biasVariables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Print checkoint
def printCheckPoint(msg):
    print("*****************************")
    print("***************")
    print("********")
    print(msg)
    print("********")
    print("***************")
    print("*****************************")

def createModel():

    # Import Data " :: Importing data from MNIST (Data storage) or local"
    ''' if Data is not availabe at  data folder , it will fetch from online repository
        Used offline way in this problem
    '''
    _mnist  = input_data.read_data_sets("data/", one_hot=True)
    printCheckPoint("Data Loaded")

    _session = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Reshaping image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    printCheckPoint("Image Reshaped")

    #-------------------------------- Convolutional Layer 1 ---------------------------------
    # Setting up weights and bias
    convolutionalLayer1_weights = weightVariables([5, 5, 1, 32])
    convolutionalLayer1_bias = biasVariables([32])

    # tf.layers.conv2d and tf.nn.conv2d is same ; The first one internally called second wone only
    # There is difference of syntax and parameters
    hiddenLayer_conv1 = tf.nn.relu(conv2d(x_image, convolutionalLayer1_weights) + convolutionalLayer1_bias)
    hiddenLayer_pool1 = max_pool_2x2(hiddenLayer_conv1)

    printCheckPoint("Convolutional Layer 1 created")

    # -------------------------------- Convolutional Layer 2 ---------------------------------
    # Setting up weights and bias
    convolutionalLayer2_weights = weightVariables([5, 5, 32, 64])
    convolutionalLayer2_bias = biasVariables([64])

    # tf.layers.conv2d and tf.nn.conv2d is same ; The first one internally called second one only
    # There is difference of syntax and parameters
    hiddenLayer_conv2 = tf.nn.relu(conv2d(hiddenLayer_pool1, convolutionalLayer2_weights) + convolutionalLayer2_bias)
    hiddenLayer_pool2 = max_pool_2x2(hiddenLayer_conv2)

    printCheckPoint("Convolutional Layer 2 created")

    # -------------------------------- Fully Connected Layer 1 ---------------------------------
    # Weights and Bias
    fullyConnectedLayer_weights1 = weightVariables([7 * 7 * 64, 1024])
    fullyConnectedLayer_bias1 = biasVariables([1024])

    # Flattern to 1D
    hiddenLayer_pool2_flat = tf.reshape(hiddenLayer_pool2, [-1, 7 * 7 * 64])
    hiddenLayer_fullyConnected1 = tf.nn.relu(tf.matmul(hiddenLayer_pool2_flat, fullyConnectedLayer_weights1) + fullyConnectedLayer_bias1)

    printCheckPoint("Fully Connected Layer 1 created")

    # -------------------------------- Fully Connected Layer 2 with dropout ---------------------------------
    keep_prob = tf.placeholder(tf.float32)
    hiddenLayer_fullyConnected_drop = tf.nn.dropout(hiddenLayer_fullyConnected1, keep_prob)

    fullyConnectedLayer_weights2 = weightVariables([1024, 10])
    fullyConnectedLayer_bias2 = biasVariables([10])

    printCheckPoint("Fully Connected Layer 2 created")

    #------------------------------ Output layer-------------------------------------
    y_outputLayer = tf.nn.softmax(tf.matmul(hiddenLayer_fullyConnected_drop, fullyConnectedLayer_weights2) + fullyConnectedLayer_bias2)

    printCheckPoint("Output  Layer 1 created")
    #----------------------------- Defining Loss and Optimizer-------------------------------
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_outputLayer))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_outputLayer, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    printCheckPoint("Loss function defined")

    # ----------------------------------- Train Model ---------------------------------
    # description to be added
    printCheckPoint("Training Model")
    saver = tf.train.Saver()
    _session.run(tf.initialize_all_variables())
    # with tf.Session() as sess:
    # sess.run(init_op)
    for i in range(10000):
        batch = _mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    save_path = saver.save(_session, str(sys.path[0])+"\model\model.ckpt")
    print("Model saved in file: ", save_path)

    print("Test accuracy %g" % accuracy.eval(feed_dict={
        x: _mnist.test.images, y_: _mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    createModel()