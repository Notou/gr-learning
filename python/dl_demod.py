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
    def __init__(self, tag_name="packet_num", bitwise = True, bpmsg = 2, packet_len=256, batch_size=1, mod = 1000, lr = 0.05, on = True ):
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

        self.bits_per_msg = bpmsg
        self.real_chan_uses = 2

        self.bitwise = bitwise

        #model parameters
        self.learning_rate = lr
        self.training = True if on==1 else False
        self.epochs = 1

        #Create model
        self.model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)

        if self.bitwise:
            print("Bitwise")    # Bitwise
            self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits # Bitwise
        else:
            print("Symbol wise")# Symbol wise
            self.loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #Symbol wise
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.train_loss = tf.metrics.Mean(name='train_loss')


    def RX_Model(self, input_size = 2, output_size = 2):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(2**(output_size+1), activation='relu', input_shape=(input_size,)),
          tf.keras.layers.Dense(2**output_size, activation='relu'),
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
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        print("RX : Changing learning rate to {}".format(self.learning_rate))


    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = self.packet_len*self.batch_size

    def general_work(self, input_items, output_items):
        input = input_items[0]
        labels = input_items[1]
        samples = self.packet_len * (len(output_items[0])//self.packet_len)

        if self.resetting:
            self.model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
            self.resetting = False

        #Verify presence of tags? Not mandatory
        tags0 = self.get_tags_in_window(0,  0 , samples, pmt.intern(self.tag_name))

        if len(tags0) == 0 :
            print("RX : We are missing tags at input")

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
        if self.bitwise: # Testé
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
                estimation = self.model(input_real[start:stop])
                loss = self.loss_function(label[start:stop], estimation)
                if self.bitwise:
                    overall_loss = tf.reduce_sum(loss, axis=1)  # Bitwise
                else:
                    overall_loss = loss  # Symbol wise
                if self.training:
                    for i in range(chunk[1]): # We send seperate messages for the losses of each packet  (because we cannot garantee continuity)
                        tuple = pmt.to_pmt((numbers[packets_seen: chunk[1]+packets_seen], overall_loss.numpy()))
                        self.message_port_pub(pmt.intern("losses"), tuple)
            else:       # Let's learn!
                with tf.GradientTape(persistent=False) as tape:
                    estimation = self.model(input_real[start:stop])
                    loss = self.loss_function(label[start:stop], estimation)
                    if self.bitwise:
                        overall_loss = tf.reduce_sum(loss, axis=1)  # Bitwise
                    else:
                        overall_loss = loss  # Symbol wise
                    mean_loss = tf.reduce_mean(overall_loss, axis=0)

                rx_grad = tape.gradient(mean_loss, self.model.trainable_variables) # tape is lost after this line when persitence is off
                self.optimizer.apply_gradients(zip(rx_grad, self.model.trainable_variables))

            if self.bitwise:
                output_bits = np.reshape(np.packbits(np.uint8((tf.sign(estimation)+1)/2), axis=-1, bitorder='little'),(-1,)) # Bitwise
            else:
                output_bits = tf.argmax(estimation, axis=1)   # Symbol wise
            output_items[0][start : stop] = output_bits
            packets_seen += chunk[1]


        # output_items[0][:self.packet_len*self.batch_size] = output_bits
        output_items[1][:samples] = labels[:samples]


        self.consume(0, samples)
        self.consume(1, samples)
        return samples
