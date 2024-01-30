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

class dl_eq_demod(gr.basic_block):
    """
    docstring for block dl_eq_demod
    """
    def __init__(self, occupied_carriers, pilot_carriers, tag_name="packet_num", bitwise = True, bpmsg = 2, packet_len=256, batch_size=1, mod = 1000, lr = 0.05, on = True, vector_size=64):
        gr.basic_block.__init__(self,
            name="dl_eq_demod",
            in_sig=[(np.complex64, vector_size), np.int8],
            out_sig=[np.int8, np.int8, np.complex64, np.complex64])
        self.user_symbols = len(occupied_carriers[0])
        self.occupied_carriers = np.array(occupied_carriers[0])+(vector_size//2)
        self.pilot_carriers = pilot_carriers
        self.vector_size = vector_size

        self.tag_name = tag_name
        self.packet_len = packet_len
        self.batch_size = batch_size
        self.train_modulo = mod
        self.printoften = 1000
        self.printcount = 0
        self.resetting = False
        self.set_output_multiple(self.packet_len)
        self.message_port_register_out(pmt.intern("losses"))

        self.bits_per_msg = bpmsg
        self.real_chan_uses = 2
        self.starting = True

        self.bitwise = bitwise

        #model parameters
        self.learning_rate = lr
        self.training = True if on==1 else False
        self.epochs = 1

        #Create model
        self.demod_model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)
        self.eq_model = self.EQ_Model(vector_size, self.user_symbols)
        print(self.packet_len // self.user_symbols)
        print(self.occupied_carriers)

        if self.bitwise:
            print("Bitwise")    # Bitwise
            self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits # Bitwise
        else:
            print("Symbol wise")# Symbol wise
            self.loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #Symbol wise
        self.eq_optimizer = tf.optimizers.Adam(self.learning_rate *0.1)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.train_loss = tf.metrics.Mean(name='train_loss')

    def RX_Model(self, input_size = 2, output_size = 2):

        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(2**(output_size+1), activation='relu', input_shape=(input_size,)),
          # tf.keras.layers.Dense(2**output_size, activation='relu'),
        ])
        if self.bitwise:
            model.add(tf.keras.layers.Dense(output_size, activation=None))  # Bitwise
        else:
            model.add(tf.keras.layers.Dense(2**output_size, activation=None))  # Symbol wise
        return model

    def EQ_Model(self, input_size, output_size):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_size, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2*input_size, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='relu'),
            # tf.keras.layers.Dense(2*output_size, activation=None),
            tf.keras.layers.Dense(2*output_size, activation='sigmoid'),
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
        self.optimizer.learning_rate = lr
        self.eq_optimizer.learning_rate = lr
        print("RX : Changing learning rate to {}".format(self.learning_rate))

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        ninput_items_required[0] = self.packet_len //self.user_symbols
        ninput_items_required[1] = self.packet_len

    def general_work(self, input_items, output_items):
        input = input_items[0]
        # print(input.shape)
        labels = input_items[1]
        samples = self.packet_len #* (len(output_items[0])//self.packet_len)
        frame_amount = self.packet_len // self.user_symbols
        used_frames = frame_amount

        if self.resetting:
            self.demod_model = self.RX_Model(self.real_chan_uses, self.bits_per_msg)
            self.eq_model = self.EQ_Model(self.vector_size, self.user_symbols)
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
            self.eq_optimizer = tf.optimizers.Adam(self.learning_rate *0.1)
            self.resetting = False

        #Verify presence of tags? Not mandatory
        tags0 = self.get_tags_in_window(0,  0 , samples, pmt.intern(self.tag_name))

        offset_tags = self.get_tags_in_window(0,  0 , samples, pmt.intern("ofdm_sync_carr_offset"))
        for tag in offset_tags:
            if pmt.to_python(tag.value) > 0:
                print("We've seen an offset at RX")

        if len(tags0) == 0 :
            print("RX : We are missing tags at input")

        learner = 'TX'
        if (self.train_modulo > 0):
            if ((pmt.to_python(tags0[0].value) // self.train_modulo) %2) ==0:
                learner = 'RX'


        # Preprocess inputs
        pilots_real = np.reshape(np.stack((input[:used_frames,:].real, input[:used_frames,:].real), axis=-1), [used_frames,-1, 2])
        if self.bitwise:
            label = np.float32(np.unpackbits(np.reshape(np.uint8(labels[:samples]), (-1,1)), axis=1, count=self.bits_per_msg, bitorder='little'))
        else:
            label = np.reshape(np.uint8(labels[:samples]), (-1,1))
        # print(input[:self.packet_len // self.user_symbols, [0,2,4,8, 16, 32]].shape)
        user_input = input[:frame_amount, self.occupied_carriers]
        #Inference process
        with tf.GradientTape(persistent=True) as tape:
            # print(self.occupied_carriers)
            # print(self.pilot_carriers, pilots_real.shape)
            # print(self.eq_model(pilots_real))
            chan_est = tf.reshape(self.eq_model(pilots_real), [used_frames, 2,-1])
            # chan_est = tf.complex(chan_est[:,0,:], chan_est[:,1,:])
            chan_est = tf.complex(10*chan_est[:,0,:]*tf.math.cos(chan_est[:,1,:]*np.pi*2), 10*chan_est[:,0,:]*tf.math.sin(chan_est[:,1,:]*np.pi*2))

            # print(chan_est)
            corrected_input = user_input / ((chan_est*1)+0)
            # print(corrected_input.shape)
            flat_corrected = tf.reshape(corrected_input, [-1])

            # print(np.stack((flat_corrected.real, flat_corrected.imag), axis=-1).shape)
            demod = self.demod_model(tf.stack((tf.math.real(flat_corrected), tf.math.imag(flat_corrected)), axis=-1))
            # print(demod)
            loss = self.loss_function(label, demod)

        rx_grad = tape.gradient(loss, self.demod_model.trainable_variables)
        eq_grad = tape.gradient(loss, self.eq_model.trainable_variables)
        # print(eq_grad)
        self.optimizer.apply_gradients(zip(rx_grad, self.demod_model.trainable_variables))
        self.optimizer.apply_gradients(zip(eq_grad, self.eq_model.trainable_variables))


        if self.bitwise:
            output_bits = np.reshape(np.packbits(np.uint8((tf.sign(demod)+1)/2), axis=-1, bitorder='little'),(-1,)) # Bitwise
        else:
            output_bits = tf.argmax(demod, axis=1)   # Symbol wise


        output_items[0][:self.packet_len] = output_bits
        output_items[1][:samples] = labels[:samples]
        output_items[2][:self.packet_len] = flat_corrected
        output_items[3][:self.user_symbols] = chan_est[0]
        self.add_item_tag(2, self.nitems_written(2) , tags0[0].key, tags0[0].value)

        self.consume(0, (self.packet_len//self.user_symbols))
        self.consume(1, self.packet_len)
        self.produce(0, self.packet_len)
        self.produce(1, self.packet_len)
        self.produce(2, self.packet_len)
        self.produce(3, self.user_symbols)
        return 0
