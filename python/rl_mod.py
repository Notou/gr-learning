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

class rl_mod(gr.basic_block):
    """
    Learnable RL-like modulator
    """
    def __init__(self, tag_name="packet_num", bpmsg = 2, packet_len=256, batch_size=1, mod = 1000, lr = 0.05, exp_noise = 0.01, on = True ):
        gr.basic_block.__init__(self,
            name="rl_mod",
            in_sig=[np.int8, ],
            out_sig=[np.complex64, ])

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

        self.set_output_multiple(self.packet_len*self.batch_size)
        self.message_port_register_in(pmt.intern('losses'))
        self.set_msg_handler(pmt.intern('losses'), self.message_callback)

        self.bits_per_msg = bpmsg
        self.real_chan_uses = 2

        # self.model = self.TX_Model(self.bits_per_msg, self.real_chan_uses)
        self.embedding = tf.Variable(tf.random.uniform([2**self.bits_per_msg,self.real_chan_uses], minval=-2, maxval=2), trainable=True)

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
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        print("TX: Changing learning rate to {}".format(self.learning_rate))

    def set_explo_noise(self, noise):
        self.exploration_noise = noise
        print("TX: Changing exploration noise to {}".format(self.exploration_noise))

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = self.packet_len*self.batch_size

    @tf.function
    def inference(self, input):
        norm = self.embedding/tf.reduce_mean(tf.norm(self.embedding,axis=1, ord=2))
        clean = tf.nn.embedding_lookup(norm, input)
        return clean

    @tf.function
    def inference_and_explo(self, input):
        norm = self.embedding/tf.reduce_mean(tf.norm(self.embedding,axis=1, ord=2))
        clean = tf.nn.embedding_lookup(norm, input)

        noisy = tf.stop_gradient(clean + tf.random.normal(clean.shape, stddev=self.exploration_noise))
        loss = -1 * tf.reduce_sum(tf.square(noisy-clean), axis=1)/(self.exploration_noise**2)
        return loss, noisy

    def general_work(self, input_items, output_items):

        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))

        if self.resetting:
            self.embedding = tf.Variable(tf.random.uniform([2**self.bits_per_msg,self.real_chan_uses], minval=-1, maxval=1), trainable=True)
            # self.model = self.TX_Model(self.real_chan_uses)
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
            self.grad_buffer = []
            self.resetting = False

        #Verify presence of tags
        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))
        if len(tags0) == 0:
            print("TX: We are missing tags at input")

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
                    loss, noisy = self.inference_and_explo(messages[self.packet_len*i: self.packet_len*(i+1)])
                    self.grad_buffer.append((pmt.to_python(tags0[0].value), (loss, tape)))
                    output_items[0][self.packet_len*i: self.packet_len*(i+1)] = noisy.numpy().view(dtype=np.complex64)[:,0]
            output_samples = self.packet_len * self.batch_size
        else:
            messages = np.int32(np.uint8(input_items[0][:self.packet_len*self.batch_size]))
            out = self.inference(messages)
            output_items[0][:self.packet_len*self.batch_size] = out.numpy().view(dtype=np.complex64)[:,0]
            # while len(self.msg_list) > 0:
            #     self.handle_callback_data(self.msg_list.pop(0))
            output_samples = self.packet_len*self.batch_size



        self.consume(0, output_samples)        #self.consume_each(len(input_items[0]))
        return output_samples
