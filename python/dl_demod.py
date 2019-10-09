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
import time

class dl_demod(gr.basic_block):
    """
    Implements a learnable deep learning based demodulator
    """
    def __init__(self, tag_name="packet_num", packet_len=256, batch_size=1, mod = 1000, lr = 0.05, on = True ):
        gr.basic_block.__init__(self,
            name="DL Demod",
            in_sig=[np.complex64, np.int8],
            out_sig=[np.int8, np.int8])
        self.tag_name = tag_name
        self.packet_len = packet_len
        self.batch_size = batch_size
        self.train_modulo = mod
        self.printoften = 1000
        self.printcount = 0
        self.resetting = False
        self.set_output_multiple(self.packet_len*self.batch_size)
        self.message_port_register_out(pmt.intern("losses"))

        #model parameters
        self.learning_rate = lr
        self.training = True if on==1 else False
        self.epochs = 1

        #Create model
        self.model = self.RX_Model()

        self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.train_loss = tf.metrics.Mean(name='train_loss')
        # train_loss_2 = tf.metrics.Mean(name='train_loss_2')
        self.accuracy = tf.metrics.Precision()

        # self.avg_grad = 0
        # self.avg_appl = 0

    def RX_Model(self, output_size = 2):
      model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(2**output_size, activation='elu', input_shape=(2,)),
          tf.keras.layers.Dense(2**output_size, activation='elu'),
          tf.keras.layers.Dense(output_size, activation='sigmoid')
      ])
      return model

    def reset(self):
        print("RX : Resetting")
        self.resetting = True

    def set_training_state(self, on):
        self.training = True if on==1 else False
        print("RX : Training is now {}".format('On' if self.training else 'Off'))

    def set_alternate(self, mod):
        self.train_modulo = mod
        print('RX : Changing training modulo to {}'.format(self.train_modulo))

    def set_lr(self, lr):
        self.learning_rate = lr
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        print("RX : Changing learning rate to {}".format(self.learning_rate))


    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = self.packet_len*self.batch_size

    def general_work(self, input_items, output_items):
        input = input_items[0]
        labels = input_items[1]

        if self.resetting:
            self.model = self.RX_Model()
            self.resetting = False

        #Verify presence of tags? Not mandatory
        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))
        # tags1 = self.get_tags_in_window(1,  0 , 1, pmt.intern(self.tag_name))

        if len(tags0) == 0 :#or len(tags1) == 0:
            print("RX : We are missing tags at input")
        # if pmt.to_python(tags0[0].value) != pmt.to_python(tags1[0].value):
        #     print("Packet number mismatch")

        learner = 'TX'
        if (self.train_modulo > 0):
            if ((pmt.to_python(tags0[0].value) // self.train_modulo) %2) ==0:
                learner = 'RX'


        #Preprocess data and labels
        input_real = np.stack((input[:self.packet_len*self.batch_size].real, input[:self.packet_len*self.batch_size].imag), axis=-1)
        # print(input_real.shape)
        label = np.float32(np.unpackbits(np.reshape(np.uint8(labels[:self.packet_len*self.batch_size]), (-1,1)), axis=1, count=2, bitorder='little'))

        with tf.GradientTape(persistent=True) as tape:

            estimation = self.model(input_real)
            # print(estimation[:10])
            loss = self.loss_function(label, estimation)
            overall_loss = tf.reduce_sum(loss, axis=1)/2
            self.accuracy.update_state(label, estimation)
            self.train_loss.update_state(overall_loss)
            if self.printcount == self.printoften:
                print("RX : Accuracy reached: {} Loss is {}".format(self.accuracy.result(), self.train_loss.result()))
                self.accuracy.reset_states()
                self.train_loss.reset_states()
                self.printcount=0
            else:
                self.printcount +=1
        if learner == 'RX' and self.training:
            # grad_start = time.time()
            rx_grad = tape.gradient(loss, self.model.trainable_variables)
            # grad_time = time.time()-grad_start
            # appl_start = time.time()
            self.optimizer.apply_gradients(zip(rx_grad, self.model.trainable_variables))
            # appl_time = time.time() - appl_start
            # self.avg_grad += grad_time
            # self.avg_appl += appl_time
        else:
            # print("RX loss: {}".format(overall_loss.shape))
            dict = pmt.dict_add(pmt.make_dict(), tags0[0].value, pmt.to_pmt(overall_loss.numpy()))
            self.message_port_pub(pmt.intern("losses"), dict)
        del tape

        # if self.printcount == self.printoften:
        #     print("Average gradient computation time: {} application time: {}".format(self.avg_grad/self.printoften, self.avg_appl/self.printoften))
        #     self.avg_grad = 0
        #     self.avg_appl = 0

        output_bits = np.reshape(np.packbits(np.uint8(tf.math.round(estimation)), axis=-1, bitorder='little'),(-1,))


        output_items[0][:self.packet_len*self.batch_size] = output_bits
        output_items[1][:self.packet_len*self.batch_size] = labels[:self.packet_len*self.batch_size]


        self.consume(0, self.packet_len*self.batch_size)        #self.consume_each(len(input_items[0]))
        self.consume(1, self.packet_len*self.batch_size)
        return self.packet_len*self.batch_size
