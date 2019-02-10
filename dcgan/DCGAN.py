import tensorflow as tf
import numpy as np

# batch normalization node
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                            )

class DCGAN:
    def __init__(self,
                 img_shape = [64,64,3],
                 batch_size = 128,
                 total_epoch = 5,
                 learning_rate = 0.0002,
                 noise_n = 100):

        #hyper parameter
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.learning_rate = learning_rate
        self.noise_n = noise_n


        #generate Variable

        # 4x4 filter
        self.G_W1 = tf.Variable(tf.truncated_normal([4, 4, 1024, 100], stddev=0.02), name="G_W1")
        # batch-norm
        self.G_bn1 = batch_norm(name="G_bn1")

        # 4x4 filter
        self.G_W2 = tf.Variable(tf.truncated_normal([4, 4, 512, 1024], stddev=0.02), name='G_W2')
        # batch-norm
        self.G_bn2 = batch_norm(name="G_bn2")

        # 4x4 filter
        self.G_W3 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='G_W3')
        # batch-norm
        self.G_bn3 = batch_norm(name="G_bn3")

        # 4x4 filter
        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='G_W4')
        # batch-norm
        self.G_bn4 = batch_norm(name="G_bn4")

        # 4x4 filter
        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, 3, 128], stddev=0.02), name='G_W5')


        # discriminate Variable

        # 4x4 filter
        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, 3, 128], stddev=0.02), name='D_W1')

        # 4x4 filter
        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='D_W2')
        # batch-norm
        self.D_bn2 = batch_norm(name="D_bn2")

        # 4x4 filter
        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='D_W3')
        # batch_norm
        self.D_bn3 = batch_norm(name="D_bn3")

        # 4x4 filter
        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, 512, 1024], stddev=0.02), name='D_W4')
        # batch_norm
        self.D_bn4 = batch_norm(name="D_bn4")

        # 4x4 filter
        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, 1024, 1], stddev=0.02), name='D_W5')

        self.D_var_list = [self.D_W1, self.D_W2, self.D_W3, self.D_W4, self.D_W5]
        self.G_var_list = [self.G_W1, self.G_W2, self.G_W3, self.G_W4, self.G_W5]

    # generate
    def generate(self,z):

        # 1x1x100
        input_ = tf.reshape(z, [self.batch_size, 1, 1, 100])

        # 1x1x100 -> 4x4x1024
        layer_1 = tf.nn.conv2d_transpose(input_
                                         , self.G_W1
                                         , output_shape=[self.batch_size, 4, 4, 1024]
                                         , strides=[1, 4, 4, 1])

        # activation function
        layer_1 = tf.nn.relu(self.G_bn1(layer_1))

        # 4x4x1024 -> 8x8x512
        layer_2 = tf.nn.conv2d_transpose(layer_1
                                         , self.G_W2
                                         , output_shape=[self.batch_size, 8, 8, 512]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_2 = tf.nn.relu(self.G_bn2(layer_2))

        # 8x8x512 -> 16x16x256
        layer_3 = tf.nn.conv2d_transpose(layer_2
                                         , self.G_W3
                                         , output_shape=[self.batch_size, 16, 16, 256]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_3 = tf.nn.relu(self.G_bn3(layer_3))

        # 16x16x256 -> 32x32x128
        layer_4 = tf.nn.conv2d_transpose(layer_3
                                         , self.G_W4
                                         , output_shape=[self.batch_size, 32, 32, 128]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_4 = tf.nn.relu(self.G_bn4(layer_4))

        # 32x32x128 -> 64x64x3
        layer_5 = tf.nn.conv2d_transpose(layer_4
                                         , self.G_W5
                                         , output_shape=[self.batch_size, 64, 64, 3]
                                         , strides=[1, 2, 2, 1])

        output_ = tf.nn.tanh(layer_5)

        return output_

    def discriminate(self, img):
        # 64x64x3 -> 32x32x128
        layer_1 = tf.nn.conv2d(img
                               , self.D_W1
                               , strides=[1, 2, 2, 1]
                               , padding='SAME')

        # activation function
        layer_1 = tf.nn.leaky_relu(layer_1, alpha=0.2)

        # 32x32x128 -> 16x16x256
        layer_2 = tf.nn.conv2d(layer_1
                               , self.D_W2
                               , strides=[1, 2, 2, 1]
                               , padding='SAME')

        # activation function
        layer_2 = tf.nn.leaky_relu(self.D_bn2(layer_2), alpha=0.2)

        # 16x16x256 -> 8x8x512
        layer_3 = tf.nn.conv2d(layer_2
                               , self.D_W3
                               , strides=[1, 2, 2, 1]
                               , padding='SAME')

        # activation function
        layer_3 = tf.nn.leaky_relu(self.D_bn3(layer_3), alpha=0.2)

        # 8x8x512 -> 4x4x1024
        layer_4 = tf.nn.conv2d(layer_3
                               , self.D_W4
                               , strides=[1, 2, 2, 1]
                               , padding='SAME')

        # activation function
        layer_4 = tf.nn.leaky_relu(self.D_bn4(layer_4), alpha=0.2)

        # 4x4x1024 -> 1x1x1
        layer_5 = tf.nn.conv2d(layer_4, self.D_W5, strides=[1, 4, 4, 1], padding='SAME')
        layer_5 = tf.reshape(layer_5, [self.batch_size, 1])

        # percentage
        output_ = tf.nn.sigmoid(layer_5)

        return output_

    def test_generator(self, noise_z, batch_size, sess):
        noise_z = np.array(noise_z).reshape([batch_size, 100])

        Z = tf.placeholder(tf.float32, [batch_size, 100])
        # 1x1x100
        input_ = tf.reshape(Z, [batch_size, 1, 1, 100])

        # 1x1x100 -> 4x4x1024
        layer_1 = tf.nn.conv2d_transpose(input_
                                         , self.G_W1
                                         , output_shape=[batch_size, 4, 4, 1024]
                                         , strides=[1, 4, 4, 1])

        # activation function
        layer_1 = tf.nn.relu(self.G_bn1(layer_1))

        # 4x4x1024 -> 8x8x512
        layer_2 = tf.nn.conv2d_transpose(layer_1
                                         , self.G_W2
                                         , output_shape=[batch_size, 8, 8, 512]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_2 = tf.nn.relu(self.G_bn2(layer_2))

        # 8x8x512 -> 16x16x256
        layer_3 = tf.nn.conv2d_transpose(layer_2
                                         , self.G_W3
                                         , output_shape=[batch_size, 16, 16, 256]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_3 = tf.nn.relu(self.G_bn3(layer_3))

        # 16x16x256 -> 32x32x128
        layer_4 = tf.nn.conv2d_transpose(layer_3
                                         , self.G_W4
                                         , output_shape=[batch_size, 32, 32, 128]
                                         , strides=[1, 2, 2, 1])

        # activation function
        layer_4 = tf.nn.relu(self.G_bn4(layer_4))

        # 32x32x128 -> 64x64x3
        layer_5 = tf.nn.conv2d_transpose(layer_4
                                         , self.G_W5
                                         , output_shape=[batch_size, 64, 64, 3]
                                         , strides=[1, 2, 2, 1])

        output_ = tf.nn.tanh(layer_5)

        generated_samples = sess.run(output_, feed_dict={Z: noise_z})
        generated_samples = (generated_samples + 1.) / 2.
        return generated_samples
