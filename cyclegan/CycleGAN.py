import tensorflow as tf
from ops import *
from utils import *

class cyclegan:
    def __init__(self,
                 img_shape,
                 batch_size,
                 learning_rate = 0.0002,
                 keep_prob = 0.5):
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        self.G_W1 = tf.Variable(tf.truncated_normal([7,7,3,64],stddev=0.2), name='G_W1')

        self.G_W2 = tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.2), name='G_W2')
        self.G_bn2 = batch_norm(name="G_bn2")

        self.G_W3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.2), name='G_W3')
        self.G_bn3 = batch_norm(name="G_bn3")

        self.G_res1 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res1')
        self.G_bn_res1 = batch_norm(name="G_bn_res1")

        self.G_res2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res2')
        self.G_bn_res2 = batch_norm(name="G_bn_res2")

        self.G_res3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res3')
        self.G_bn_res3 = batch_norm(name="G_bn_res3")

        self.G_res4 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res4')
        self.G_bn_res4 = batch_norm(name="G_bn_res4")

        self.G_res4 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res5')
        self.G_bn_res5 = batch_norm(name="G_bn_res5")

        self.G_res5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.2), name='G_res6')
        self.G_bn_res6 = batch_norm(name="G_bn_res6")

        self.G_W4 = tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.2), name='G_W4')

        self.G_W5 = tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.2), name='G_W5')

        self.G_W6 = tf.Variable(tf.truncated_normal([7,7,64,3], stddev=0.2), name='G_W6')


    def generate(self,img,scope='generator'):
        with tf.variable_scope(scope,reuse=False):
            layer_1 = conv(img,64,kernel=7,stride=1,pad=3,pad_type='reflect',scope='layer_1')
            layer_1 = instance_norm(layer_1)
            layer_1 = relu(layer_1)

            layer_2 = conv(layer_1,128,kernel=3,stride=2,pad=1,pad_type='zero',scope='layer_2')
            layer_2 = instance_norm(layer_2)
            layer_2 = relu(layer_2)

            layer_3 = conv(layer_2, 256, kernel=3, stride=2, pad=1, pad_type='zero', scope='layer_3')
            layer_3 = instance_norm(layer_3)
            layer_3 = relu(layer_3)

            layer_4 = resblock(layer_3,256,scope='resblock_1')
            layer_5 = resblock(layer_4, 256, scope='resblock_2')
            layer_6 = resblock(layer_5, 256, scope='resblock_3')
            layer_7 = resblock(layer_6, 256, scope='resblock_4')
            layer_8 = resblock(layer_7, 256, scope='resblock_5')
            layer_9 = resblock(layer_8, 256, scope='resblock_6')

            layer_10 = deconv(layer_9,)

