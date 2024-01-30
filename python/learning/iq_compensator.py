#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018 Cyrille Morin.
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

import numpy
from gnuradio import gr
import tensorflow as tf
import pmt

class iq_compensator(gr.basic_block):
    """
    docstring for block iq_compensator
    """
    def __init__(self, frame_size = 100, hidden_units = 100, batch_size = 5, iq_magnitude = 0, learning_rate = 1e-4):
        gr.basic_block.__init__(self,
                                name="iq_compensator",
                                in_sig=[numpy.complex64],
                                out_sig=[numpy.complex64])
        self.frame_size = frame_size
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.batch_number = 0
        self.learning_rate = learning_rate
        self.iq_magnitude = iq_magnitude
        self.step = 0
        self.last_error = 0
        self.batch_gradients = []
        self.batch_losses = []
        self.waiting_for_samples = True
        self.init_graph()
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)

        self.message_port_register_in(pmt.intern('error'))
        self.set_msg_handler(pmt.intern('error'), self.message_callback)

    def forecast(self, noutput_items, ninput_items_required):
        # setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
                ninput_items_required[i] = self.frame_size

    def set_mag_value(self, iq_magnitude):
        self.iq_magnitude = iq_magnitude

    def general_work(self, input_items, output_items):
        if (not self.waiting_for_samples):
            #print("Not waiting for samples")
            return 0
            #pass
        #print("Running")
        if  len(input_items[0]) < self.frame_size:
            print("RL input size: ", len(input_items[0]))
        output_items[0][0:self.frame_size], gradient= self.session.run(
            [self.complexOutput, self.policyGradient], feed_dict={self.observations: input_items[0][0:self.frame_size], self.error_approx: self.last_error})
        self.batch_gradients.append(gradient)
        self.consume(0, self.frame_size)
        self.waiting_for_samples = False
        return self.frame_size

    def message_callback(self, msg):
        loss = pmt.to_python(msg)
        #print("Loss= ", loss)
        self.batch_losses.append(loss)
        self.batch_number +=1
        if self.batch_number == self.batch_size:
            # print("learning ", self.batch_losses, len(self.batch_gradients))
            gradient_L1 = 0
            gradient_L1_bias = 0
            gradient_L2 = 0
            gradient_L2_bias = 0
            for index in range(0, self.batch_size):
                gradient_L1 += (self.batch_losses[index] * self.batch_gradients[index][0]) / self.batch_size
                gradient_L1_bias += (self.batch_losses[index] * self.batch_gradients[index][1]) / self.batch_size
                gradient_L2 += (self.batch_losses[index] * self.batch_gradients[index][2]) / self.batch_size
                gradient_L2_bias += (self.batch_losses[index] * self.batch_gradients[index][3]) / self.batch_size
            self.session.run(self.updateGrads, feed_dict={self.batch_grad_L1: gradient_L1, self.batch_grad_L1_bias: gradient_L1_bias, self.batch_grad_L2: gradient_L2, self.batch_grad_L2_bias: gradient_L2_bias})
            self.batch_gradients = []
            self.last_error = numpy.sum(self.batch_losses) / self.batch_size
            print(self.last_error)
            self.batch_losses = []
            self.batch_number = 0

        self.waiting_for_samples = True

    def init_graph(self):
        # hyperparameters
        H = self.hidden_units  # number of hidden layer neurons

        D = self.frame_size  # input dimensionality
        #H = 40 #Hidden size
        O = 1 #Output size

        # This defines the network as it goes from taking an observation of the environment to
        # giving a probability of chosing to the action of moving left or right.
        self.observations = tf.placeholder(tf.complex64, [D], name="input")
        self.true_offset = tf.placeholder(tf.float32, name="offset")
        self.error_approx = tf.placeholder(tf.float32, name="approxerror")
        print(self.observations.shape)
        floatMatrix = tf.stack([tf.real(self.observations), tf.imag(self.observations)], axis = 0)
        print(floatMatrix)
        floatInput = tf.reshape(floatMatrix, [1,-1])
        print(floatInput)

        L1 = tf.layers.dense(floatInput, H, activation=tf.nn.elu) #, kernel_regularizer= tf.contrib.layers.l2_regularizer(0.5))
        L2 = tf.layers.dense(L1, O, activation=None) #, kernel_regularizer= tf.contrib.layers.l2_regularizer(0.5))
        self.exploration = L2 + 0.05 * tf.random_normal(L2.shape)
        print(self.exploration)
        #floatOutput = tf.reshape(exploration, [2, D])
        floatOutput = (floatMatrix - self.exploration[0][0])# * self.exploration[0][1]
        self.complexOutput = tf.complex(floatOutput[0], floatOutput[1])

        tvars = tf.trainable_variables()
        adam = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)  # Our optimizer
        log = tf.log(self.exploration)
        self.policyGradient = tf.gradients(log, tvars)

        # self.loss =  tf.losses.mean_squared_error(self.true_offset, self.exploration)
        # self.updateGrads = adam.minimize(self.loss, var_list=tvars)
        # Once we have collected a series of gradients from multiple episodes, we apply them.
        # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
        self.batch_grad_L1 = tf.placeholder(tf.float32, name="batch_grad_L1")
        self.batch_grad_L1_bias = tf.placeholder(tf.float32, name="batch_grad_L1_bias")
        self.batch_grad_L2 = tf.placeholder(tf.float32, name="batch_grad_L2")
        self.batch_grad_L2_bias = tf.placeholder(tf.float32, name="batch_grad_L2_bias")

        self.updateGradient = [self.batch_grad_L1, self.batch_grad_L1_bias, self.batch_grad_L2, self.batch_grad_L2_bias]
        self.updateGrads = adam.apply_gradients(zip(self.updateGradient,tvars))
