import tensorflow as tf
import reader

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
                                            reuse=tf.AUTO_REUSE     # if tensorflow vesrion < 1.4, delete this line
                                            )

class pix2pix:
    def __init__(self,
                 img_shape = [256,256,3],
                 batch_size = 128,
                 total_epoch = 5,
                 learning_rate = 0.0002):
        # hyper parameter
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.learning_rate = learning_rate

        # generate Variable(encoder-decoder)
        # ENCODING
        self.G_W1 = tf.Variable(tf.truncated_normal([4,4,3,64],stddev=0.2), name="G_W1")

        self.G_W2 = tf.Variable(tf.truncated_normal([4,4,64,128],stddev=0.2),name="G_W2")
        self.G_bn2 = batch_norm(name="G_bn2")

        self.G_W3 = tf.Variable(tf.truncated_normal([4,4,128,256],stddev=0.2),name="G_W3")
        self.G_bn3 = batch_norm(name="G_bn3")

        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.2), name="G_W4")
        self.G_bn4 = batch_norm(name="G_bn4")

        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W5")
        self.G_bn5 = batch_norm(name="G_bn5")

        self.G_W6 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W6")
        self.G_bn6 = batch_norm(name="G_bn6")

        self.G_W7 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W7")
        self.G_bn7 = batch_norm(name="G_bn7")

        self.G_W8 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W8")
        self.G_bn8 = batch_norm(name="G_bn8")

        # DECODING
        self.G_W9 = tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.2), name="G_W9")
        self.G_bn9 = batch_norm(name="G_bn9")

        self.G_W10 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W10")
        self.G_bn10 = batch_norm(name="G_bn10")

        self.G_W11 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W11")
        self.G_bn11 = batch_norm(name="G_bn11")

        self.G_W12 = tf.Variable(tf.truncated_normal([4, 4, 512, 512+512], stddev=0.2), name="G_W12")
        self.G_bn12 = batch_norm(name="G_bn12")

        self.G_W13 = tf.Variable(tf.truncated_normal([4, 4, 256, 512+512], stddev=0.2), name="G_W13")
        self.G_bn13 = batch_norm(name="G_bn13")

        self.G_W14 = tf.Variable(tf.truncated_normal([4, 4, 128, 256+256], stddev=0.2), name="G_W14")
        self.G_bn14 = batch_norm(name="G_bn14")

        self.G_W15 = tf.Variable(tf.truncated_normal([4, 4, 64, 128+128], stddev=0.2), name="G_W15")
        self.G_bn15 = batch_norm(name="G_bn15")

        self.G_W16 = tf.Variable(tf.truncated_normal([4, 4, 3, 64+64], stddev=0.2), name="G_W16")
        self.G_bn16 = batch_norm(name="G_bn16")

        # discriminate
        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, 3, 64],stddev=0.02), name='D_W1')
        self.D_bn1 = batch_norm(name="D_bn1")

        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev=0.02), name='D_W2')
        self.D_bn2 = batch_norm(name="D_bn2")

        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name='D_W3')
        self.D_bn3 = batch_norm(name="D_bn3")

        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name='D_W4')
        self.D_bn4 = batch_norm(name="D_bn4")

        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, 512, 1], stddev=0.02), name='D_W5')


        def generate(self,img):
            # 256x256x3 -> 128x128x64
            layer_1 = tf.nn.conv2d(img,self.G_bn1,strides=[1,2,2,1],padding='SAME')
            layer_1 = tf.nn.leaky_relu(layer_1,alpha=0.2)

























