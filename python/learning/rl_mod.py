#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Cyrille Morin.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import warnings
import numpy as np
from gnuradio import gr
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import pmt
import threading
import time
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["TF_XLA_FLAGS"] = "tf_xla_auto_jit=2"

codebook_qpsk   = [[[-1,-1],[-1,1],[1,-1],[1,1]]]
codebook_4pam   = [[[-3,0], [-1,0], [3,0], [1,0]]]
codebook_8psk   = [[[-1,-1], [-np.sqrt(2.0),0], [0,np.sqrt(2.0)], [-1,1], [0,-np.sqrt(2.0)], [1,-1], [1,1], [np.sqrt(2.0),0]]]
codebook_16qam  = [[[-3,-3], [-3,-1], [-3,3], [-3,1], [-1,-3], [-1,-1], [-1,3], [-1,1], [3,-3], [3,-1], [3,3], [3,1], [1,-3], [1,-1], [1,3], [1,1]]]
codebook_64qam  = [[[3,3], [3,1], [3,5], [3,7], [3,-3], [3,-1], [3,-5], [3,-7], [1,3], [1,1], [1,5], [1,7], [1,-3], [1,-1], [1,-5], [1,-7],
                  [5,3], [5,1], [5,5], [5,7], [5,-3], [5,-1], [5,-5], [5,-7], [7,3], [7,1], [7,5], [7,7], [7,-3], [7,-1], [7,-5], [7,-7],
                  [-3,3], [-3,1], [-3,5], [-3,7], [-3,-3], [-3,-1], [-3,-5], [-3,-7], [-1,3], [-1,1], [-1,5], [-1,7], [-1,-3], [-1,-1], [-1,-5], [-1,-7],
                  [-5,3], [-5,1], [-5,5], [-5,7], [-5,-3], [-5,-1], [-5,-5], [-5,-7], [-7,3], [-7,1], [-7,5], [-7,7], [-7,-3], [-7,-1], [-7,-5], [-7,-7]]]
codebook_256qam = [[[-15,15],[-15,13],[-15,9],[-15,11],[-15,1],[-15,3],[-15,7],[-15,5],[-15,-15],[-15,-13],[-15,-9],[-15,-11],[-15,-1],[-15,-3],[-15,-7],[-15,-5],
                   [-13,15],[-13,13],[-13,9],[-13,11],[-13,1],[-13,3],[-13,7],[-13,5],[-13,-15],[-13,-13],[-13,-9],[-13,-11],[-13,-1],[-13,-3],[-13,-7],[-13,-5],
                   [-9,15],[-9,13],[-9,9],[-9,11],[-9,1],[-9,3],[-9,7],[-9,5],[-9,-15],[-9,-13],[-9,-9],[-9,-11],[-9,-1],[-9,-3],[-9,-7],[-9,-5],
                   [-11,15],[-11,13],[-11,9],[-11,11],[-11,1],[-11,3],[-11,7],[-11,5],[-11,-15],[-11,-13],[-11,-9],[-11,-11],[-11,-1],[-11,-3],[-11,-7],[-11,-5],
                   [-1,15],[-1,13],[-1,9],[-1,11],[-1,1],[-1,3],[-1,7],[-1,5],[-1,-15],[-1,-13],[-1,-9],[-1,-11],[-1,-1],[-1,-3],[-1,-7],[-1,-5],
                   [-3,15],[-3,13],[-3,9],[-3,11],[-3,1],[-3,3],[-3,7],[-3,5],[-3,-15],[-3,-13],[-3,-9],[-3,-11],[-3,-1],[-3,-3],[-3,-7],[-3,-5],
                   [-7,15],[-7,13],[-7,9],[-7,11],[-7,1],[-7,3],[-7,7],[-7,5],[-7,-15],[-7,-13],[-7,-9],[-7,-11],[-7,-1],[-7,-3],[-7,-7],[-7,-5],
                   [-5,15],[-5,13],[-5,9],[-5,11],[-5,1],[-5,3],[-5,7],[-5,5],[-5,-15],[-5,-13],[-5,-9],[-5,-11],[-5,-1],[-5,-3],[-5,-7],[-5,-5],
                   [15,15],[15,13],[15,9],[15,11],[15,1],[15,3],[15,7],[15,5],[15,-15],[15,-13],[15,-9],[15,-11],[15,-1],[15,-3],[15,-7],[15,-5],
                   [13,15],[13,13],[13,9],[13,11],[13,1],[13,3],[13,7],[13,5],[13,-15],[13,-13],[13,-9],[13,-11],[13,-1],[13,-3],[13,-7],[13,-5],
                   [9,15],[9,13],[9,9],[9,11],[9,1],[9,3],[9,7],[9,5],[9,-15],[9,-13],[9,-9],[9,-11],[9,-1],[9,-3],[9,-7],[9,-5],
                   [11,15],[11,13],[11,9],[11,11],[11,1],[11,3],[11,7],[11,5],[11,-15],[11,-13],[11,-9],[11,-11],[11,-1],[11,-3],[11,-7],[11,-5],
                   [1,15],[1,13],[1,9],[1,11],[1,1],[1,3],[1,7],[1,5],[1,-15],[1,-13],[1,-9],[1,-11],[1,-1],[1,-3],[1,-7],[1,-5],
                   [3,15],[3,13],[3,9],[3,11],[3,1],[3,3],[3,7],[3,5],[3,-15],[3,-13],[3,-9],[3,-11],[3,-1],[3,-3],[3,-7],[3,-5],
                   [7,15],[7,13],[7,9],[7,11],[7,1],[7,3],[7,7],[7,5],[7,-15],[7,-13],[7,-9],[7,-11],[7,-1],[7,-3],[7,-7],[7,-5],
                   [5,15],[5,13],[5,9],[5,11],[5,1],[5,3],[5,7],[5,5],[5,-15],[5,-13],[5,-9],[5,-11],[5,-1],[5,-3],[5,-7],[5,-5]]]

class rl_mod(gr.basic_block):
    """
    Learnable RL-like modulator
    """
    def __init__(self, tag_name="packet_num", bpmsg = 2, packet_len=256, batch_size=1, mod = 1000, lr = 0.05, exp_noise = 0.01, on = True , save_folder=""):
        gr.basic_block.__init__(self,
            name="rl_mod",
            in_sig=[np.int8, ],
            out_sig=[np.complex64, np.complex64, (np.complex64,2**bpmsg)])

        self.tag_name = tag_name
        self.packet_len = packet_len
        self.batch_size = mod#batch_size
        self.train_modulo = mod
        self.exploration_noise = exp_noise
        self.learning_rate = lr
        self.training = True if on==1 else False

        self.grad_buffer = []
        self.msg_list = []
        self.resetting = False
        self.loading = False
        print("HERE!!!!!!!!!!!!!!!")
        self.save_folder = save_folder + "/"

        self.set_output_multiple(self.packet_len*self.batch_size)
        self.message_port_register_in(pmt.intern('losses'))
        self.set_msg_handler(pmt.intern('losses'), self.message_callback)

        self.bits_per_msg = bpmsg
        self.real_chan_uses = 2

        # self.model = self.TX_Model(self.bits_per_msg, self.real_chan_uses)
        self.embedding = tf.Variable(tf.random.uniform([2**self.bits_per_msg,self.real_chan_uses], minval=-2, maxval=2), trainable=True)
        if self.bits_per_msg == 2:
            self.embedding = tf.Variable(np.array(codebook_qpsk[0],dtype=np.float32), trainable=True)
        if self.bits_per_msg == 3:
            self.embedding = tf.Variable(np.array(codebook_8psk[0],dtype=np.float32), trainable=True)
        if self.bits_per_msg == 4:
            self.embedding = tf.Variable(np.array(codebook_16qam[0],dtype=np.float32), trainable=True)
        if self.bits_per_msg == 6:
            self.embedding = tf.Variable(np.array(codebook_64qam[0],dtype=np.float32), trainable=True)
        if self.bits_per_msg == 8:
            self.embedding = tf.Variable(np.array(codebook_256qam[0],dtype=np.float32), trainable=True)


        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def TX_Model(self, k, n):
      model = tf.keras.Sequential([
            tf.keras.layers.Embedding(2**k,n, embeddings_regularizer=tf.keras.regularizers.l2(l=1))
            # tf.keras.layers.Embedding(2**n,n, embeddings_constraint=tf.keras.constraints.UnitNorm([0,1]))
      ])
      return model

    def message_callback(self, msg):
        if not self.training:
            return
        else :
            # threading.Thread(target=self.handle_callback_data, args=(msg,)).start()
            self.handle_callback_data(msg)
            # print(len(self.msg_list))
    def save_model(self):
        file_name = self.save_folder + "_" + time.strftime("%y%m%d%H%M", time.localtime()) + "_" + str(self.bits_per_msg) + "_TX.pickle"
        file_obj = open(file_name,'wb')
        pickle.dump(self.embedding.numpy(), file_obj, protocol=4) # Rajouté pour compatibilité avec Python2
        file_obj.close()

    def load_model(self, name):
        self.to_load = self.save_folder + "_" + name + "_" + str(self.bits_per_msg) + "_TX.pickle"
        self.loading = True
        self.resetting = True



    def handle_callback_data(self, msg):
        num_list, losses = pmt.to_python(msg)
        for i in range(len(num_list)):
            while len(self.grad_buffer)>0:
                stored = self.grad_buffer.pop(0)
                rx_num = num_list[i]
                if stored[0] < rx_num:
                    # print('TX: We lost a loss: {} key is {}'.format(stored[0], rx_num))
                    continue
                elif stored[0] > rx_num:
                    self.grad_buffer.insert(0, stored) #Lets put it back in the buffer
                    break
                else:
                    with stored[1][1] as tape:
                        final_loss = stored[1][0] * losses[self.packet_len*i: self.packet_len*(i+1)]  # Stored grad * received crossentropy loss
                        # print(losses[self.packet_len*i: self.packet_len*(i+1)])
                        mean_loss = tf.reduce_mean(final_loss, axis=0)
                    # print(final_loss)
                    grad = tape.gradient(mean_loss, self.embedding)
                    self.optimizer.apply_gradients(zip([grad], [self.embedding,]))
                    del tape
                    break

    def reset(self):
        print("TX: Resetting")
        self.resetting = True

    def set_training_state(self, on):
        self.training = True if on==1 else False
        print("TX: Training is now {}".format('On' if self.training else 'Off'))

    def set_alternate(self, mod):
        self.train_modulo = mod
        self.batch_size=mod
        print('TX: Changing training modulo to {}'.format(self.train_modulo))

    def set_lr(self, lr):
        self.learning_rate = lr
        self.optimizer.learning_rate = lr
        print("TX: Changing learning rate to {}".format(self.learning_rate))

    def set_explo_noise(self, noise):
        self.exploration_noise = noise
        print("TX: Changing exploration noise to {}".format(self.exploration_noise))

    def forecast(self, noutput_items, ninputs):
        #setup size of input_items[i] for work call
        ninput_items_required = []
        for i in range(ninputs):
            ninput_items_required.append(self.packet_len*self.batch_size)
        return ninput_items_required

    # @tf.function
    def inference(self, input):
        norm = self.embedding/tf.reduce_mean(tf.norm(self.embedding,axis=1, ord=2))
        clean = tf.nn.embedding_lookup(norm, input)
        return clean

    # @tf.function
    def inference_and_explo(self, input):
        norm = self.embedding/tf.reduce_mean(tf.norm(self.embedding,axis=1, ord=2))
        clean = tf.nn.embedding_lookup(norm, input)

        noisy = tf.stop_gradient(np.sqrt(1-self.exploration_noise**2)*clean + tf.random.normal(clean.shape, stddev=self.exploration_noise))
        loss = -2 * tf.reduce_sum(tf.abs(noisy-clean), axis=1)/(self.exploration_noise**2)
        return loss, noisy, clean

    def general_work(self, input_items, output_items):

        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))

        if self.resetting:
            if self.loading:
                emb = pickle.load(open(self.to_load ,'rb'), encoding='latin1')
                print(emb)
                self.embedding = tf.Variable(emb, trainable = True)
                print("Loading model TX")
                self.loading = False
            else:
                self.embedding = tf.Variable(tf.random.uniform([2**self.bits_per_msg,self.real_chan_uses], minval=-1, maxval=1), trainable=True)
            # self.model = self.TX_Model(self.real_chan_uses)
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
            self.grad_buffer = []
            self.resetting = False

        #Verify presence of tags
        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))
        if len(tags0) == 0:
            print("TX: We are missing tags at input")
            print("TX: Tag expected at index: {}".format(self.nitems_read(0)))
            tags1 = self.get_tags_in_window(0,  0 , 1, pmt.intern("packet_len"))
            if len(tags1)!=0:
                print("TX: packet length tag was there though")
            self.consume(0, self.packet_len)
            return 0

        # Who is going to learn from this packet.
        learner = 'TX'
        if (self.train_modulo > 0):
            if ((pmt.to_python(tags0[0].value) // self.train_modulo) %2) ==0:
                learner = 'RX'

        output_samples = 0
        # bits = np.float32(np.unpackbits(np.reshape(np.uint8(input_items[0][:self.packet_len*self.batch_size]), (-1,1)), axis=1, count=2, bitorder='little'))
        if learner == 'TX' and self.training:
            messages = np.int32(np.uint8(input_items[0][:self.packet_len*self.batch_size]))
            for i in range(self.batch_size):
                with tf.GradientTape(persistent=True) as tape:
                    loss, noisy, clean = self.inference_and_explo(messages[self.packet_len*i: self.packet_len*(i+1)])
                    self.grad_buffer.append((pmt.to_python(tags0[0].value), (loss, tape)))
                    output_items[0][self.packet_len*i: self.packet_len*(i+1)] = noisy.numpy().view(dtype=np.complex64)[:,0]
                    output_items[1][self.packet_len*i: self.packet_len*(i+1)] = clean.numpy().view(dtype=np.complex64)[:,0]
            output_samples = self.packet_len * self.batch_size
        else:
            messages = np.int32(np.uint8(input_items[0][:self.packet_len*self.batch_size]))
            out = self.inference(messages)
            output_items[0][:self.packet_len*self.batch_size] = out.numpy().view(dtype=np.complex64)[:,0]
            output_items[1][:self.packet_len*self.batch_size] = out.numpy().view(dtype=np.complex64)[:,0]
            # while len(self.msg_list) > 0:
            #     self.handle_callback_data(self.msg_list.pop(0))
            output_samples = self.packet_len*self.batch_size


        #Output constellation points
        msg = np.arange(2**self.bits_per_msg, dtype = np.int32)
        output = self.inference(msg).numpy().view(dtype=np.complex64)[:,0]
        output_items[2][0] = output

        self.consume(0, output_samples)        #self.consume_each(len(input_items[0]))
        self.produce(0, output_samples)
        self.produce(1, output_samples)
        self.produce(2, 1)
        return 0
