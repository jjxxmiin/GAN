from ops import *
from utils import *
from CycleGAN import cyclegan
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder

# init
img_ch = 3
img_size = 256
learning_rate = 0.0001
# feature weight
gan_w = 1.0 # X -> Y'            1 *loss(Y:Y')
cycle_w = 10.0 # X -> Y' -> X''  10*loss(X:X'')
identity_w = 5.0 # Y -> X'       5 *loss(X:X')
epoch = 3
iteration = 100000
batch_size = 1

# data dir init
dir_name = "G://dataset//vangogh2photo//"

train_A_dir = dir_name + "trainA"
train_B_dir = dir_name + "trainB"
test_A_dir = dir_name + "testA"
test_B_dir = dir_name + "testB"
log_dir = "logs"
checkpoint_dir = "checkpoint"
sample_dir = "sample"

# dataset processing
train_A_dataset = glob(train_A_dir + "/*.*")
train_B_dataset = glob(train_B_dir + "/*.*")
dataset_num = max(len(train_A_dataset), len(train_B_dataset))

Image_Data_Class = ImageData(img_size, img_ch)

trainA = tf.data.Dataset.from_tensor_slices(train_A_dataset)
trainB = tf.data.Dataset.from_tensor_slices(train_B_dataset)

trainA = trainA.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(batch_size)).repeat()
trainB = trainB.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(batch_size)).repeat()

trainA_iterator = trainA.make_one_shot_iterator()
trainB_iterator = trainB.make_one_shot_iterator()

train_A = trainA_iterator.get_next()
train_B = trainB_iterator.get_next()

# build model
G_ab = cyclegan.generate(train_A,scope="generate_B")
G_ba = cyclegan.generate(train_B,scope="generate_A")

G_aba = cyclegan.generate(G_ab,reuse=True,scope="generate_A")
G_bab = cyclegan.generate(G_ba,reuse=True,scope="generate_B")

G_aa = cyclegan.generate(train_A,reuse=True,scope="generate_A")
G_bb = cyclegan.generate(train_B,reuse=True,scope="generate_B")

D_real_A = cyclegan.discriminate(train_A,scope="discriminate_A")
D_real_B = cyclegan.discriminate(train_B,scope="discriminate_B")

D_fake_A = cyclegan.discriminate(G_ba,reuse=True,scope="discriminate_A")
D_fake_B = cyclegan.discriminate(G_ab,reuse=True,scope="discriminate_B")

# loss
# X' -> X
# Y' -> Y
identity_loss_A = L1_loss(G_aa,train_A)
identity_loss_B = L1_loss(G_bb,train_B)

# X -> Y'
# Y -> X'
G_cost_A = g_cost(D_fake_A)
G_cost_B = g_cost(D_fake_B)

# X -> Y' vs real
D_cost_A = d_cost(D_real_A,D_fake_A)
D_cost_B = d_cost(D_real_B,D_fake_B)

# X -> Y' -> X'' vs X
recon_cost_A = L1_loss(G_aba,train_A)
recon_cost_B = L1_loss(G_bab,train_B)

G_loss_A = gan_w * G_cost_A + \
            recon_cost_A + \
            identity_loss_A

G_loss_B =  G_cost_B + \
            recon_cost_B + \
            identity_loss_B

D_loss_A = gan_w * D_cost_A
D_loss_B = gan_w * D_cost_B

G_loss = G_loss_A + G_loss_B
D_loss = D_loss_A + D_loss_B

# optimizer
t_vars = tf.trainable_variables()
G_var_list = [var for var in t_vars if 'generate' in var.name]
D_var_list = [var for var in t_vars if 'discriminate' in var.name]

G_train = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(G_loss,var_list=G_var_list)
D_train = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(D_loss,var_list=D_var_list)

# summary
all_G_loss = tf.summary.scalar("Generator_loss", G_loss)
all_D_loss = tf.summary.scalar("Discriminator_loss", D_loss)
G_A_loss = tf.summary.scalar("G_A_loss", G_loss_A)
G_B_loss = tf.summary.scalar("G_B_loss", G_loss_B)
D_A_loss = tf.summary.scalar("D_A_loss", D_loss_A)
D_B_loss = tf.summary.scalar("D_B_loss", D_loss_B)

Gen_loss = tf.summary.merge([G_A_loss, G_B_loss, all_G_loss])
Dis_loss = tf.summary.merge([D_A_loss, D_B_loss, all_D_loss])

# image
fake_A = G_ba
fake_B = G_ab

real_A = train_A
real_B = train_B

# test
test_image = tf.placeholder(tf.float32,[1,img_size,img_size,img_ch], name="test_image")
test_fake_A = cyclegan.generate(test_image,reuse=True,scope="generate_A")
test_fake_B = cyclegan.generate(test_image,reuse=True,scope="generate_B")

# train
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # tensorboard
    writer = tf.summary.FileWriter(log_dir,sess.graph)

    # model loading
    could_load, checkpoint_counter = load(checkpoint_dir,sess,saver)

    if could_load:
        start_epoch = (int)(checkpoint_counter / iteration)
        start_batch_id = checkpoint_counter - start_epoch * iteration
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print(" [!] Load failed...")

    start_time = time.time()

    for epoch in range(start_epoch,epoch):
        for i in range(start_batch_id,iteration):
            # UPDATE D_LOSS
            _, d_loss ,summary_str = sess.run([D_train,D_loss,Dis_loss])
            writer.add_summary(summary_str,counter)
            # UPDATE G_LOSS
            batch_A_images,batch_B_images,fake_A_,fake_B_, _, g_loss,summary_str = sess.run([real_A,real_B,fake_A,fake_B,G_train,G_loss,Gen_loss])
            writer.add_summary(summary_str, counter)

            counter += 1
            if i % 100 == 0:
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, i, iteration, time.time() - start_time, d_loss, g_loss))

            if i == start_batch_id or i % 300 == 0:
                save_images(fake_A_, [batch_size, 1],'G://{}//fake_A_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, i + 1))
                #save_images(fake_B_, [batch_size, 1],'G://{}//fake_B_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, i + 1))
                #save_images(batch_A_images, [batch_size, 1],'G://{}//batch_A_images_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, i + 1))
                save_images(batch_B_images, [batch_size, 1],'G://{}//batch_B_images_{:02d}_{:06d}.jpg'.format(sample_dir, epoch, i + 1))

            # checkpoint save
            if i % 1000 == 0:
                save(checkpoint_dir,counter,sess,saver)


        start_batch_id = 0

        save(checkpoint_dir,counter,sess,saver)