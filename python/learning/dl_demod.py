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
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["TF_XLA_FLAGS"] = "tf_xla_auto_jit=2"

class dl_demod(gr.basic_block):
    """
    Implements a learnable deep learning based demodulator
    """
    def __init__(self, tag_name="packet_num", bitwise = True, bpmsg = 2, packet_len=256, batch_size=1, mod = 1000, lr = 0.05, on = True , save_folder=""):
        gr.basic_block.__init__(self,
            name="DL Demod",
            in_sig=[np.complex64, np.int8],
            out_sig=[(np.float32,bpmsg), np.int8, np.int8])
        self.tag_name = tag_name
        self.packet_len = packet_len
        self.batch_size = batch_size
        self.train_modulo = mod
        self.printoften = 1000
        self.printcount = 0
        self.resetting = False
        self.loading = False
        self.set_output_multiple(self.packet_len*self.batch_size)
        self.message_port_register_out(pmt.intern("losses"))

        self.bits_per_msg = bpmsg
        self.real_chan_uses = 2
        self.starting = True

        self.bitwise = bitwise

        self.save_folder = save_folder + "/"

        #model parameters
        self.learning_rate = lr
        self.training = True if on==1 else False
        self.epochs = 1
        print("Newwwww!!!!!!!!!!!!!!!!!!!!!!")
        #Create model
        self.model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)

        if self.bitwise:
            print("Bitwise!!!!!!!!!!!!!!!!!!!")    # Bitwise
            self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits # Bitwise
        else:
            print("Symbol wise")# Symbol wise
            self.loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #Symbol wise
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.train_loss = tf.metrics.Mean(name='train_loss')


    def RX_Model(self, input_size = 2, output_size = 2):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(2**(output_size+1), activation='elu', input_shape=(input_size,)),
          tf.keras.layers.Dense(2**output_size, activation='elu'),
        ])
        if self.bitwise:
            model.add(tf.keras.layers.Dense(output_size, activation=None))  # Bitwise
        else:
            model.add(tf.keras.layers.Dense(2**output_size, activation=None))  # Symbol wise
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
        self.optimizer.learning_rate = lr
        print("RX : Changing learning rate to {}".format(self.learning_rate))

    def save_model(self):
        file_name = self.save_folder + "_" + time.strftime("%y%m%d%H%M", time.localtime()) + "_" + str(self.bits_per_msg) + "_RX.h5"
        self.model.save(file_name)

    def load_model(self, name):
        self.to_load = self.save_folder + "_" + name + "_" + str(self.bits_per_msg) + "_RX.h5"
        self.loading = True
        self.resetting = True



    def forecast(self, noutput_items, ninputs):
        #setup size of input_items[i] for work call
        ninput_items_required = []
        for i in range(ninputs):
            ninput_items_required.append(self.packet_len*self.batch_size)
        return ninput_items_required


    @tf.function
    def train(self, input, label, bitwise):
        with tf.GradientTape(persistent=False) as tape:
            estimation, overall_loss = self.inference(input, label, bitwise)
            mean_loss = tf.reduce_mean(overall_loss, axis=0)
        self.optimizer.minimize(mean_loss, self.model.trainable_variables, tape=tape)
        return estimation

    @tf.function
    def inference(self, input, label, bitwise):
        estimation = self.model(input)
        loss = self.loss_function(label, estimation)
        if bitwise:
            overall_loss = tf.reduce_sum(loss, axis=1)  # Bitwise
        else:
            overall_loss = loss  # Symbol wise
        return estimation, overall_loss

    def general_work(self, input_items, output_items):
        input = input_items[0]
        labels = input_items[1]
        samples = self.packet_len * (len(output_items[0])//self.packet_len)

        if self.resetting:
            self.model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
            if self.loading:
                self.model = tf.keras.models.load_model(self.to_load)
                self.loading = False
                print("Loading model RX")
            self.resetting = False

        #Verify presence of tags? Not mandatory
        tags0 = self.get_tags_in_window(0,  0 , samples, pmt.intern(self.tag_name))
        tags1 = self.get_tags_in_window(0,  0 , samples, pmt.intern(self.tag_name))

        if len(tags0) == 0 :
            print("RX : We are missing tags at input")
        if len(tags1) == 0 :
            print("RX : We are missing tags at input1")
        if pmt.to_python(tags0[0].value) != pmt.to_python(tags1[0].value):
            print("RX : Tag mismatch between inputs")

        lengths = []
        numbers = []
        for tag in tags0:
            num = pmt.to_python(tag.value)
            numbers.append(num)
            learner_tx = True
            if ((num // self.train_modulo) %2) ==0 and self.training==True:
                learner_tx = False
            if len(lengths) > 0 and lengths[-1][0] == learner_tx:
                lengths[-1][1] += 1
            else:
                lengths.append([learner_tx, 1])

        # learner = 'TX'
        # if (self.train_modulo > 0):
        #     if ((pmt.to_python(tags0[0].value) // self.train_modulo) %2) ==0:
        #         learner = 'RX'


        #Preprocess data and labels
        input_real = np.stack((input[:samples].real, input[:samples].imag), axis=-1)
        if self.bitwise:
            label = np.float32(np.unpackbits(np.reshape(np.uint8(labels[:samples]), (-1,1)), axis=1, count=self.bits_per_msg, bitorder='little'))
            # label = np.float32(np.unpackbits(np.reshape(np.uint8(labels[:self.packet_len*self.batch_size]), (-1,1)), axis=1, count=self.bits_per_msg, bitorder='little')) # Bitwise
        else:
            label = np.reshape(np.uint8(labels[:samples]), (-1,1))
            # label = np.reshape(np.uint8(labels[:self.packet_len*self.batch_size]), (-1,1))  # Symbol wise

        # print("RX : Lengths list: {}".format(lengths))
        packets_seen = 0

        for chunk in lengths:
            start = self.packet_len*packets_seen
            stop = self.packet_len*(packets_seen+ chunk[1])
            if chunk[0]: # It is not ours to learn on these packets
                # begin = time.time()
                estimation, overall_loss = self.inference(input_real[start:stop], label[start:stop], self.bitwise)
                # end  = time.time()
                # print("RX inference {} per packet over {} packets".format((end-begin)/chunk[1], chunk[1]))

                if self.training:
                    for i in range(chunk[1]): # We send seperate messages for the losses of each packet  (because we cannot garantee continuity)
                        tuple = pmt.to_pmt((numbers[packets_seen: chunk[1]+packets_seen], overall_loss.numpy()))
                        self.message_port_pub(pmt.intern("losses"), tuple)
            else:       # Let's learn!
                # begin = time.time()
                estimation = self.train(input_real[start:stop], label[start:stop], self.bitwise)
                # end  = time.time()
                # print("RX estimation {} per packet over {} packets".format((end-begin)/chunk[1], chunk[1]))

            output_items[0][start : stop] = estimation[:,:self.bits_per_msg]
            if self.bitwise:
                output_bits = np.reshape(np.packbits(np.uint8((tf.sign(estimation)+1)/2), axis=-1, bitorder='little'),(-1,)) # Bitwise
            else:
                output_bits = tf.argmax(estimation, axis=1)   # Symbol wise
            output_items[1][start : stop] = output_bits
            packets_seen += chunk[1]


        # output_items[0][:self.packet_len*self.batch_size] = output_bits
        output_items[2][:samples] = labels[:samples]


        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))
        tags1 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))

        if len(tags0) == 0 :
            print("RX : We are missing tags at input")
        if len(tags1) == 0 :
            print("RX : We are missing tags at input1")
        if pmt.to_python(tags0[0].value) != pmt.to_python(tags1[0].value):
            print("RX : Tag mismatch between inputs")

        self.consume(0, samples)
        self.consume(1, samples)
        return samples
