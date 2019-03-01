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

    def generate(self,img,scope='generator'):
        #resnet
        with tf.variable_scope(scope,reuse=False):
            #Down Sampling
            layer_1 = conv(img,64,kernel=7,stride=1,pad=3,pad_type='reflect',scope='G_layer_1')
            layer_1 = instance_norm(layer_1)
            layer_1 = relu(layer_1)

            layer_2 = conv(layer_1,128,kernel=3,stride=2,pad=1,pad_type='zero',scope='G_layer_2')
            layer_2 = instance_norm(layer_2)
            layer_2 = relu(layer_2)

            layer_3 = conv(layer_2, 256, kernel=3, stride=2, pad=1, pad_type='zero', scope='G_layer_3')
            layer_3 = instance_norm(layer_3)
            layer_3 = relu(layer_3)

            layer_4 = resblock(layer_3,256,scope='G_resblock_1')
            layer_5 = resblock(layer_4, 256, scope='G_resblock_2')
            layer_6 = resblock(layer_5, 256, scope='G_resblock_3')
            layer_7 = resblock(layer_6, 256, scope='G_resblock_4')
            layer_8 = resblock(layer_7, 256, scope='G_resblock_5')
            layer_9 = resblock(layer_8, 256, scope='G_resblock_6')

            #UP Sampling
            layer_10 = deconv(layer_9,128,kernel=3,stride=2,scope='G_layer_10')
            layer_10 = instance_norm(layer_10)
            layer_10 = relu(layer_10)

            layer_11 = deconv(layer_10,64,kernel=3,stride=2,scope='G_layer_11')
            layer_11 = instance_norm(layer_11)
            layer_11 = relu(layer_11)

            output_ = conv(layer_11,3,kernel=7,stride=1,pad=3,pad_type='reflect',scope='G_layer_12')
            output_ = tanh(output_)

            return output_

        def discriminate(self,img,scope='discriminate'):
            with tf.variable_scope(scope, reuse=False):
                #128x128x64
                layer_1 = conv(img,64,kernel=4,stride=2,pad=1,pad_type='zero',scope='D_layer_1')
                layer_1 = lrelu(layer_1,0.2)
                #64x64x128
                layer_2 = conv(layer_1,128,kernel=4,stride=2,pad=1,pad_type='zero',scope='D_layer_2')
                layer_2 = instance_norm(layer_2)
                layer_2 = lrelu(layer_2,0.2)
                #32x32x256
                layer_3 = conv(layer_2,256,kernel=4,stride=2,pad=1,pad_type='zero',scope='D_layer_3')
                layer_3 = instance_norm(layer_3)
                layer_3 = lrelu(layer_3,0.2)
                #16x16x512
                layer_4 = conv(layer_3,512,kernel=4,stride=2,pad=1,pad_type='zero',scope='D_layer_4')
                layer_4 = instance_norm(layer_4)
                layer_4 = lrelu(layer_4,0.2)
                #16x16x1
                output_ = conv(layer_4,1,kernel=4,stride=1,pad=1,pad_type='zero',scope='D_layer_5')
                #not used sigmoid because LSGAN

                return output_





