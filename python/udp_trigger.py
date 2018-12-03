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

import numpy as np
import pmt
import struct
from gnuradio import gr
import socket
import threading

class udp_trigger(gr.sync_block):
    """
    Triggers the transmission od a previously received message upon reception of a utp packet containing the right tx_number
    """
    def __init__(self, tx_number=0,ip_addr='127.0.0.1', port=3500):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='UDP trigger',   # will show up in GRC
            in_sig=None,
            out_sig=None
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.tx_number = tx_number
        self.ip_addr = ip_addr
        self.port = port
        self.message_port_register_out(pmt.intern("out"))
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)
        self.message_buffer = None

        self.socket_thread = threading.Thread(target=self.handle_packet)
        self.socket_thread.daemon = True
        self.socket_thread.start()


    def handle_msg(self, msg):
        self.message_buffer = msg



    def __exit__(self, exc_type, exc_value, traceback):
        # print("Exit")
        self.s.close()

    def handle_packet(self):

        # initialize a socket, think of it as a cable
        # SOCK_DGRAM specifies that this is UDP
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.s.bind((self.ip_addr,self.port))
        while 1:
            data = self.s.recv(1024)
            number = struct.unpack('B',data)[0]
            # print(number)
            if number != self.tx_number:
                print("Wrong target")
                continue
            if self.message_buffer != None:
                self.message_port_pub(pmt.intern('out'),self.message_buffer)


    def work(self, input_items, output_items):
        """example: multiply with constant"""

        return 0
