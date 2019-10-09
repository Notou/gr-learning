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

class rl_mod(gr.basic_block):
    """
    Learnable RL-like modulator
    """
    def __init__(self, tag_name="packet_num", packet_len=256, batch_size=1, mod = 1000, lr = 0.05, exp_noise = 0.01, on = True ):
        gr.basic_block.__init__(self,
            name="rl_mod",
            in_sig=[np.int8, ],
            out_sig=[np.complex64, ])

        self.tag_name = tag_name
        self.packet_len = packet_len
        self.batch_size = 1#batch_size
        self.train_modulo = mod
        self.exploration_noise = exp_noise
        self.learning_rate = lr
        self.training = True if on==1 else False

        self.grad_buffer = []
        self.resetting = False

        self.set_output_multiple(self.packet_len*self.batch_size)
        self.message_port_register_in(pmt.intern('losses'))
        self.set_msg_handler(pmt.intern('losses'), self.message_callback)

        self.bits_per_msg = 2
        self.real_chan_uses = 2

        self.model = self.TX_Model(self.real_chan_uses)

        # self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def TX_Model(self, n):
      model = tf.keras.Sequential([
            tf.keras.layers.Embedding(2**n,n)
            # tf.keras.layers.Dense(n, activation=None, use_bias=False, input_shape=(n,))
      ])
      return model

    def message_callback(self, msg):
        if not self.training:
            return
        losses = pmt.to_python(msg)
        while len(self.grad_buffer)>0:
            stored = self.grad_buffer.pop(0)
            rx_num = list(losses.keys())[0]
            if stored[0] < rx_num:
                print('TX: We lost a loss: {} key is {}'.format(stored[0], rx_num))
                continue
            elif stored[0] > rx_num:
                self.grad_buffer.insert(0, stored) #Lets put it back in the buffer
                break
            else:
                with stored[1][1] as tape:
                    final_loss = stored[1][0] * losses.get(rx_num)  # Stored grad * received crossentropy loss

                grad = tape.gradient(final_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
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

    def general_work(self, input_items, output_items):
        tags0 = self.get_tags_in_window(0,  0 , 1, pmt.intern(self.tag_name))

        if self.resetting:
            self.model = self.TX_Model(self.real_chan_uses)
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

        # bits = np.float32(np.unpackbits(np.reshape(np.uint8(input_items[0][:self.packet_len*self.batch_size]), (-1,1)), axis=1, count=2, bitorder='little'))
        messages = np.uint8(input_items[0][:self.packet_len*self.batch_size])
        with tf.GradientTape(persistent=True) as tape:
            clean = self.model(messages)
            if (pmt.to_python(tags0[0].value) % 100) ==0:
                print(clean[1])
            if learner == 'TX' and self.training:
                clean = clean/tf.sqrt(2*tf.reduce_mean(tf.square(clean)))
                noisy = tf.stop_gradient(clean + tf.random.normal(clean.shape, stddev=self.exploration_noise))
                loss = -1 * tf.reduce_sum(tf.square(noisy-clean), axis=1)/(self.exploration_noise**2)
        if learner == 'TX' and self.training:
            self.grad_buffer.append((pmt.to_python(tags0[0].value), (loss, tape)))
            symbols = noisy
        else:
            symbols = clean

        # Normalise power of output vector
        out = symbols/tf.sqrt(2*tf.reduce_mean(tf.square(symbols)))
        output_items[0][:self.packet_len*self.batch_size] = out.numpy().view(dtype=np.complex64)[:,0]

        self.consume(0, self.packet_len*self.batch_size)        #self.consume_each(len(input_items[0]))
        return self.packet_len*self.batch_size
