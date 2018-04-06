# Imports
import sys,os
import tensorflow as tf
from PIL import Image, ImageFilter
import glob

def predictValue(imgValue):

    # ------------------ Other Functions
    # TBD
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # TBD
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # TBD
    def weightVariables(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # TBD
    def biasVariables(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    # Defining weights and other variables
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Reshaping image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # -------------------------------- Convolutional Layer 1 ---------------------------------
    # Setting up weights and bias
    convolutionalLayer1_weights = weightVariables([5, 5, 1, 32])
    convolutionalLayer1_bias = biasVariables([32])

    # tf.layers.conv2d and tf.nn.conv2d is same ; The first one internally called second wone only
    # There is difference of syntax and parameters
    hiddenLayer_conv1 = tf.nn.relu(conv2d(x_image, convolutionalLayer1_weights) + convolutionalLayer1_bias)
    hiddenLayer_pool1 = max_pool_2x2(hiddenLayer_conv1)

    # -------------------------------- Convolutional Layer 2 ---------------------------------
    # Setting up weights and bias
    convolutionalLayer2_weights = weightVariables([5, 5, 32, 64])
    convolutionalLayer2_bias = biasVariables([64])

    # tf.layers.conv2d and tf.nn.conv2d is same ; The first one internally called second one only
    # There is difference of syntax and parameters
    hiddenLayer_conv2 = tf.nn.relu(conv2d(hiddenLayer_pool1, convolutionalLayer2_weights) + convolutionalLayer2_bias)
    hiddenLayer_pool2 = max_pool_2x2(hiddenLayer_conv2)

    # -------------------------------- Fully Connected Layer 1 ---------------------------------
    # Weights and Bias
    fullyConnectedLayer_weights1 = weightVariables([7 * 7 * 64, 1024])
    fullyConnectedLayer_bias1 = biasVariables([1024])

    # Flattern to 1D
    hiddenLayer_pool2_flat = tf.reshape(hiddenLayer_pool2, [-1, 7 * 7 * 64])
    hiddenLayer_fullyConnected1 = tf.nn.relu(
        tf.matmul(hiddenLayer_pool2_flat, fullyConnectedLayer_weights1) + fullyConnectedLayer_bias1)

    # -------------------------------- Fully Connected Layer 2 with dropout ---------------------------------
    keep_prob = tf.placeholder(tf.float32)
    hiddenLayer_fullyConnected_drop = tf.nn.dropout(hiddenLayer_fullyConnected1, keep_prob)

    fullyConnectedLayer_weights2 = weightVariables([1024, 10])
    fullyConnectedLayer_bias2 = biasVariables([10])

    # ------------------------------ Output layer-------------------------------------
    y_outputLayer = tf.nn.softmax(
        tf.matmul(hiddenLayer_fullyConnected_drop, fullyConnectedLayer_weights2) + fullyConnectedLayer_bias2)

    # ----------------------------------- PredictValue ---------------------------------
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model\model.ckpt")
        print("Model restored.")
        prediction = tf.argmax(y_outputLayer, 1)
        return prediction.eval(feed_dict={x: [imgValue], keep_prob: 1.0}, session=sess)




def imagePrepare(imgPath):

    # Open Image
    img = Image.open(imgPath).convert('L')
    width = float(img.size[0])
    height = float(img.size[1])

    # Creates white canvas of 28x28 pixels
    canvasedImage = Image.new('L', ( 28, 28), (255))

    # check which dimension is bigger
    if width > height:

        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width

        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheigth = 1

        # resize and sharpen
        resizedAndSharpenImage = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # caculate horizontal pozition
        wtop = int(round(((28 - nheight) / 2), 0))

        # paste resized image on white canvas
        canvasedImage.paste(img, (4, wtop))
    else:

        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1

        # resize and sharpen
        resizedAndSharpenImage = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # caculate vertical pozition
        wleft = int(round(((28 - nwidth) / 2), 0))

        # paste resized image on white canvas
        canvasedImage.paste(img, (wleft, 4))

    # Enable to save image
    # canvasedImage.save("sample.png")

    tv = list(canvasedImage.getdata())  # get pixel values
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    #print(tva)
    return tva

def main(imagePath):
    imgvalue = imagePrepare(imagePath)
    predictInt = predictValue(imgvalue)
    print(predictInt)


if __name__ == '__main__':
    x = input("Enter complete Path to file : ")
    main(x)
