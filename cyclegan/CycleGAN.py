import tensorflow as tf

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


    def generate(self,img):
        def residule_block(x,dim):
            y = tf.nn.conv2d(x,)


        # 256x256x3 -> 128x128x64
        layer_1 = tf.nn.conv2d(img,self.G_W1,strides=[1,2,2,1],padding='SAME')
        layer_1 = tf.nn.relu(layer_1,alpha=0.2)
        # 128x128x64 -> 64x64x128
        layer_2 = tf.nn.conv2d(layer_1,self.G_W2,strides=[1,2,2,1],padding='SAME')
        layer_2 = self.G_bn2(layer_2)
        layer_2 = tf.nn.relu(layer_2,alpha=0.2)
        #64x64x128 -> 32x32x256
        layer_3 = layer_res_1 = tf.nn.conv2d(layer_2,self.G_W3,strides=[1,2,2,1],padding='SAME')
        layer_3 = self.G_bn3(layer_3)
        layer_3 = tf.nn.relu(layer_3, alpha=0.2)
        #32x32x256 -> 32x32x256
        layer_4 = layer_res_2 = tf.nn.conv2d(layer_3,self.G_res1,strides=[1,2,2,1],padding='SAME')
        layer_4 = self.G_bn_res1(layer_4)
        layer_4 = tf.nn.relu(layer_4, alpha=0.2)

        layer_4 = tf.nn.conv2d(layer_3,self.G_res1,strides=[1,2,2,1],padding='SAME')
        

