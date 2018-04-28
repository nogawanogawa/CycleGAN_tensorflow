import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *
from img_gen import *


output_path = "./output"
check_dir = "./output/checkpoints/"

img_layer = 3
temp_check = 0
max_epoch = 1
max_images = 100
h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class CycleGAN():

    # ネットワークのセットアップ
    def __init__(self):

        # パラメータの定義
        self.img_height = 256
        self.img_width = 256
        self.img_size = self.img_height * self.img_width
        self.pool_size = 50

        # placeholder(変換前用)
        self.x = tf.placeholder(tf.float32, [batch_size, self.img_width, self.img_height, img_layer], name="x")
        self.y = tf.placeholder(tf.float32, [batch_size, self.img_width, self.img_height, img_layer], name="y")

        # placeholder(discriminator向け教師データ(x側のDiscriminatorにはx以外、y側のDiscriminatorにはy以外を使用))
        self.fake_pool_x = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, img_layer], name="fake_pool_x")
        self.fake_pool_y = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, img_layer], name="fake_pool_y")

        # loop制御用
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        # 学習率
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:

            # 1 教師データから画像を変換
            self.fake_y = generator(self.x, name="g_x") # X -> Y
            self.fake_x = generator(self.y, name="g_y") # Y -> X

            # 2 教師データを正しいと見分ける(discriminator)
            self.rec_x = discriminator(self.x, "d_x")
            self.rec_y = discriminator(self.y, "d_y")

            scope.reuse_variables()

            # 3 1で生成された画像を偽物と見分ける(discriminator)
            self.fake_rec_x = discriminator(self.fake_x, "d_x")
            self.fake_rec_y = discriminator(self.fake_y, "d_y")

            # 4 1で生成された画像をもとに戻す
            self.cyc_x = generator(self.fake_y, "g_y") # X -> Y -> X
            self.cyc_y = generator(self.fake_x, "g_x") # Y -> X -> Y

            scope.reuse_variables()

            # 5 間違いの教師データを間違いと見分ける(discriminator)
            self.fake_pool_rec_x = discriminator(self.fake_pool_x, "d_x")
            self.fake_pool_rec_y = discriminator(self.fake_pool_y, "d_y")

        #Lossの計算
        self.loss_calc()

        #画像取得用ネットワークの初期化
        self.input_setup()


    # 入力画像ファイルに関する設定
    def input_setup(self):

        # 集合Xの教師データの設定(horse)
        filenames_x = tf.train.match_filenames_once("./input/horse2zebra/trainA/*.jpg")
        self.queue_length_x = tf.size(filenames_x)

        # 集合Xの教師データの設定(zebra)
        filenames_y = tf.train.match_filenames_once("./input/horse2zebra/trainB/*.jpg")
        self.queue_length_y = tf.size(filenames_y)

        filename_queue_x = tf.train.string_input_producer(filenames_x)
        filename_queue_y = tf.train.string_input_producer(filenames_y)

        # 画像の読み込み
        image_reader = tf.WholeFileReader()
        _, image_file_x = image_reader.read(filename_queue_x)
        _, image_file_y = image_reader.read(filename_queue_y)

        # jpegファイルを整形
        self.image_x = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_x, \
                                                                channels=3),[self.img_height,self.img_width]),127.5),1)
        self.image_y = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_y, \
                                                                channels=3),[self.img_height,self.img_width]),127.5),1)

    # 実行時の画像取得
    def input_read(self, sess):

        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files_x = sess.run(self.queue_length_x)
        num_files_y = sess.run(self.queue_length_y)

        self.fake_images_x = np.zeros((self.pool_size,1,self.img_height, self.img_width, img_layer))
        self.fake_images_y = np.zeros((self.pool_size,1,self.img_height, self.img_width, img_layer))

        self.A_input = np.zeros((max_images, batch_size, self.img_height, self.img_width, img_layer))
        self.B_input = np.zeros((max_images, batch_size, self.img_height, self.img_width, img_layer))

        for i in range(max_images):
            # 画像 -> Tensor
            image_tensor = sess.run(self.image_x)
            if(image_tensor.size == self.img_size*batch_size*img_layer):
                self.A_input[i] = image_tensor.reshape((batch_size, self.img_height, self.img_width, img_layer))

        for i in range(max_images):
            # 画像 -> Tensor
            image_tensor = sess.run(self.image_y)
            if(image_tensor.size == self.img_size*batch_size*img_layer):
                self.B_input[i] = image_tensor.reshape((batch_size, self.img_height, self.img_width, img_layer))

        coord.request_stop()
        coord.join(threads)

    def loss_calc(self):

        # Cycle Consistency Loss
        cyc_loss = tf.reduce_mean(tf.abs(self.x-self.cyc_x)) + tf.reduce_mean(tf.abs(self.y-self.cyc_y))

        # Adversarial Loss
        #  Discriminatorのloss(偽物を偽物と見分ける)
        disc_loss_x = tf.reduce_mean(tf.squared_difference(self.fake_rec_x,1))
        disc_loss_y = tf.reduce_mean(tf.squared_difference(self.fake_rec_y,1))

        # Generatorのloss
        g_loss_x = cyc_loss*10 + disc_loss_y
        g_loss_y = cyc_loss*10 + disc_loss_x

        # Discriminatorのloss(本物を本物と見分ける)
        d_loss_x = (tf.reduce_mean(tf.square(self.fake_pool_rec_x)) + tf.reduce_mean(tf.squared_difference(self.rec_x,1)))/2.0
        d_loss_y = (tf.reduce_mean(tf.square(self.fake_pool_rec_y)) + tf.reduce_mean(tf.squared_difference(self.rec_y,1)))/2.0

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_x_vars = [var for var in self.model_vars if 'd_x' in var.name]
        g_x_vars = [var for var in self.model_vars if 'g_x' in var.name]
        d_y_vars = [var for var in self.model_vars if 'd_y' in var.name]
        g_y_vars = [var for var in self.model_vars if 'g_y' in var.name]

        self.d_x_trainer = optimizer.minimize(d_loss_x, var_list=d_x_vars)
        self.d_y_trainer = optimizer.minimize(d_loss_y, var_list=d_y_vars)
        self.g_x_trainer = optimizer.minimize(g_loss_x, var_list=g_x_vars)
        self.g_y_trainer = optimizer.minimize(g_loss_y, var_list=g_y_vars)

        self.g_x_loss_summ = tf.summary.scalar("g_x_loss", g_loss_x)
        self.g_y_loss_summ = tf.summary.scalar("g_y_loss", g_loss_y)
        self.d_x_loss_summ = tf.summary.scalar("d_x_loss", d_loss_x)
        self.d_y_loss_summ = tf.summary.scalar("d_y_loss", d_loss_y)

    def fake_image_pool(self, num_fakes, fake, fake_pool):

        if(num_fakes < self.pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,self.pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake

    def train(self):

        # 初期化
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:

            sess.run(init)

            #教師データの読み込み
            self.input_read(sess)

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step),100):
                print ("epoch : ", epoch)

                # 学習率を設定
                # 始め100epochは0.0002、その後は線形に減少
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                for ptr in range(0,max_images):
                    print("In the iteration ",ptr)

                    # Optimizing the G_x network
                    _, fake_y_temp, summary_str = sess.run([self.g_x_trainer, self.fake_y, self.g_x_loss_summ], \
                                                            feed_dict={self.x:self.A_input[ptr], \
                                                                        self.y:self.B_input[ptr],\
                                                                        self.lr:curr_lr})
                    fake_y_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_y_temp, self.fake_images_y)

                    # Optimizing the D_y network
                    _, summary_str = sess.run([self.d_y_trainer, self.d_y_loss_summ],\
                                                feed_dict={self.x:self.A_input[ptr], \
                                                            self.y:self.B_input[ptr], \
                                                            self.lr:curr_lr, \
                                                            self.fake_pool_y:fake_y_temp1})

                    # Optimizing the G_y network
                    _, fake_x_temp, summary_str = sess.run([self.g_y_trainer, self.fake_x, self.g_y_loss_summ],\
                                                            feed_dict={self.x:self.A_input[ptr],\
                                                                        self.y:self.B_input[ptr], \
                                                                        self.lr:curr_lr})
                    fake_x_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_x_temp, self.fake_images_x)

                    # Optimizing the D_x network
                    _, summary_str = sess.run([self.d_x_trainer, self.d_x_loss_summ],\
                                                feed_dict={self.x:self.A_input[ptr], \
                                                            self.y:self.B_input[ptr],\
                                                            self.lr:curr_lr, \
                                                            self.fake_pool_x:fake_x_temp1})

                    self.num_fake_inputs+=1

                    # 画像を保存(x->y, y->x)
                    if(ptr == max_images-1) :
                        fig = plot(self.A_input[ptr], fake_y_temp, self.B_input[ptr], fake_x_temp1)
                        plt.savefig('output/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
                        plt.close(fig)

                sess.run(tf.assign(self.global_step, epoch + 1))



# Resnet Block
def resnet_block(inputres, dim, name="resnet"):

    with tf.variable_scope(name):

        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)

        return tf.nn.relu(out_res + inputres)

# Generator
def generator(inputgen, name="generator", reuse=None):
    if reuse:
        scope.reuse_variables()

    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        x = conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
        x = conv2d(x, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        x = conv2d(x, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        x = resnet_block(x, ngf*4, "r1")
        x = resnet_block(x, ngf*4, "r2")
        x = resnet_block(x, ngf*4, "r3")
        x = resnet_block(x, ngf*4, "r4")
        x = resnet_block(x, ngf*4, "r5")
        x = resnet_block(x, ngf*4, "r6")
        x = resnet_block(x, ngf*4, "r7")
        x = resnet_block(x, ngf*4, "r8")
        x = resnet_block(x, ngf*4, "r9")

        x = general_deconv2d(x, [batch_size,128,128,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        x = general_deconv2d(x, [batch_size,256,256,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
        x = conv2d(x, img_layer, f, f, 1, 1, 0.02,"SAME","c6",do_relu=False)

        # Adding the tanh layer
        x = tf.nn.tanh(x,"t1")

        return x

# Discriminator
def discriminator(inputdisc, name="discriminator", reuse=None):
    if reuse:
        scope.reuse_variables()

    with tf.variable_scope(name):
        f = 4

        x = conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        x = conv2d(x, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        x = conv2d(x, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        x = conv2d(x, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        x = conv2d(x, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return x
