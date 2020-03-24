#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RL Demod training
# Author: Cyrille Morin
# GNU Radio version: 3.8.1.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import eng_notation
import bokehgui
from gnuradio import fec
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
import time
import functools
from bokeh.client import push_session
from bokeh.plotting import curdoc
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import uhd
from gnuradio import zeromq
from gnuradio.digital.utils import tagged_streams
import learning
import numpy as np
import threading

class dl_demod_RX(gr.top_block):
    def __init__(self, doc, ip="127.0.0.1"):
        gr.top_block.__init__(self, "RL Demod training")
        self.doc = doc
        self.plot_lst = []
        self.widget_lst = []

        ##################################################
        # Parameters
        ##################################################
        self.ip = ip

        ##################################################
        # Variables
        ##################################################
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.packet_length_tag_key = packet_length_tag_key = "packet_length"
        self.occupied_carriers = occupied_carriers = (list(range(-22, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 23)),)
        self.length_tag_key_0 = length_tag_key_0 = "frame_len"
        self.length_tag_key = length_tag_key = "packet_len"
        self.header_mod = header_mod = digital.constellation_bpsk()
        self.fft_len = fft_len = 64
        self.errors = errors = 0
        self.bits_per_symbol = bits_per_symbol = 4
        self.ber = ber = 0
        self.variable_qtgui_label_0_0_val = variable_qtgui_label_0_0 = ber
        self.variable_qtgui_label_0_val = variable_qtgui_label_0 = errors
        self.variable_0 = variable_0 = len(occupied_carriers[0])
        self.training_mod = training_mod = 1
        self.timestamp = timestamp = '1'
        self.t_state = t_state = 1
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.snr_stop = snr_stop = 29
        self.snr_step = snr_step = 0.5
        self.snr_start = snr_start = 0
        self.samp_rate = samp_rate = 0.3e6
        self.rx_lr = rx_lr = -2
        self.rolloff = rolloff = 0
        self.puncpat = puncpat = '11'
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bits_per_symbol)
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_static(fft_len,  occupied_carriers, pilot_carriers, pilot_symbols, 1)
        self.packet_len = packet_len = 400
        self.mag = mag = 10
        self.ldpc_enc = ldpc_enc = fec.ldpc_encoder_make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist")
        self.ldpc_dec = ldpc_dec = fec.ldpc_decoder.make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist", 0.5, 50)
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=packet_length_tag_key, frame_len_tag_key=length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, header_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols)
        self.hdr_format = hdr_format = digital.header_format_ofdm(occupied_carriers, 1, length_tag_key,)
        self.gain_rx = gain_rx = 5
        self.freq = freq = 900e6
        self.bits_per_symbol_0 = bits_per_symbol_0 = len(occupied_carriers[0])

        ##################################################
        # Blocks
        ##################################################
        self.training_mod_textbox = bokehgui.textbox(self.widget_lst, str(1), 'Alternate training every' +": ")
        self.training_mod_textbox.add_callback(
          lambda attr, old, new: self.set_training_mod(int(new)))
        self._t_state_options = [    1,     0,   ]
        self._t_state_labels = [      'On',      'Off',  ]

        self.t_state_radiobutton = bokehgui.radiobutton(self.widget_lst, None, self._t_state_labels, inline = True)
        self.t_state_radiobutton.add_callback(
                  lambda new: self.set_t_state(int(self._t_state_options[new])))
        self.rx_lr_slider = bokehgui.slider(self.widget_lst, 'RX Learning rate (log)' +":", -5, 0, 0.1, 1, -2)
        self.rx_lr_slider.add_callback(lambda attr, old, new: self.set_rx_lr(new))
        self.mag_slider = bokehgui.slider(self.widget_lst, 'Eb/N0' +":", 0, 40, 0.1, 1, 10)
        self.mag_slider.add_callback(lambda attr, old, new: self.set_mag(new))
        self.learning_ber_bf_0 = learning.ber_bf(False, 100, -7.0, 1)
        self.gain_rx_slider = bokehgui.slider(self.widget_lst, 'Amplitude Rx' +":", 0, 90, 0.5, 1, 5)
        self.gain_rx_slider.add_callback(lambda attr, old, new: self.set_gain_rx(new))
        self.blocks_probe_signal_x_0 = blocks.probe_signal_f()
        self.zeromq_sub_msg_source_0_0 = zeromq.sub_msg_source("tcp://"+ip+":50001", 1000)
        self.zeromq_pub_msg_sink_0_0_0 = zeromq.pub_msg_sink('tcp://*:50002', 1000)
        self.variable_qtgui_label_0_0 = bokehgui.label(self.widget_lst, str(variable_qtgui_label_0_0), 'BER' +": ")
        self.variable_qtgui_label_0 = bokehgui.label(self.widget_lst, str(variable_qtgui_label_0), 'Error count' +": ")
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )


        self.uhd_usrp_source_0.set_samp_rate(samp_rate)

        self.uhd_usrp_source_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_source_0.set_center_freq(freq, 0)

        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)

        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_gain(gain_rx, 0)
        self.timestamp_textbox = bokehgui.textbox(self.widget_lst, str('1'), 'Load timestamp' +": ")
        self.timestamp_textbox.add_callback(
          lambda attr, old, new: self.set_timestamp(str(new)))
        self.learning_dl_demod_0 = learning.dl_demod('packet_num', False, bits_per_symbol, packet_len, 20, training_mod, 10**rx_lr , t_state, '/home/cyrille/Gnu-Radio/modules/gr-learning/examples/saved_models')
        self.learning_align_0 = learning.align('packet_num', packet_len, 1, 48)
        self.fft_vxx_1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        def _errors_probe():
            while True:

                val = self.learning_ber_bf_0.total_errors()
                try:
                    self.set_errors(val)
                except AttributeError:
                    pass
                time.sleep(1.0 / (0.5))
        _errors_thread = threading.Thread(target=_errors_probe)
        _errors_thread.daemon = True
        _errors_thread.start()

        self.digital_packet_headerparser_b_0 = digital.packet_headerparser_b(header_formatter.base())
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, fft_len//4, False, 0.9)
        self.digital_ofdm_serializer_vcc_payload = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key_0, packet_length_tag_key, 1, '', True)
        self.digital_ofdm_serializer_vcc_header = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key_0, '', 0, '', True)
        self.digital_ofdm_frame_equalizer_vcvc_1 = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), fft_len//4, length_tag_key_0, True, 0)
        self.digital_ofdm_frame_equalizer_vcvc_0 = digital.ofdm_frame_equalizer_vcvc(header_equalizer.base(), fft_len//4, length_tag_key_0, True, 1)
        self.digital_ofdm_chanest_vcvc_0 = digital.ofdm_chanest_vcvc(sync_word1, sync_word2, 1, 0, 3, False)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            3,
            fft_len,
            fft_len//4,
            length_tag_key_0,
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            int(samp_rate),
            (),
            0)
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(header_mod.base())
        self.bokehgui_time_sink_x_0 = bokehgui.time_sink_f_proc(1024, samp_rate, "Decoded minus label",
         1
        )

        self.bokehgui_time_sink_x_0_plot = bokehgui.time_sink_f(self.doc, self.plot_lst, self.bokehgui_time_sink_x_0, is_message =   False)

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        legend_list = []
        for i in  range(    1  ):
          if len(labels[i]) == 0:
            legend_list.append("Data {0}".format(i))
          else:
            legend_list.append(labels[i])
        self.bokehgui_time_sink_x_0_plot.initialize(log_x = False, log_y = False, update_time = 500, legend_list = legend_list)

        self.bokehgui_time_sink_x_0_plot.set_y_axis([-10, 10])
        self.bokehgui_time_sink_x_0_plot.set_y_label('Difference' + '(' +""+')')
        self.bokehgui_time_sink_x_0_plot.set_x_label('Symbol' + '(' +""+')')

        self.bokehgui_time_sink_x_0_plot.enable_tags(-1, True)
        self.bokehgui_time_sink_x_0_plot.set_trigger_mode(bokehgui.TRIG_MODE_FREE, bokehgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.bokehgui_time_sink_x_0_plot.enable_grid(False)
        self.bokehgui_time_sink_x_0_plot.enable_axis_labels(True)
        self.bokehgui_time_sink_x_0_plot.disable_legend(not True)
        self.bokehgui_time_sink_x_0_plot.set_layout(*((2,0,1,3)))

        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "blue", "blue", "blue"]
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        styles = ["solid", "solid", "solid", "solid", "solid",
                  "solid", "solid", "solid", "solid", "solid"]
        markers = [None, None, None, None, None,
                  None, None, None, None, None]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in  range(     1  ):
          self.bokehgui_time_sink_x_0_plot.format_line(i, colors[i], widths[i], styles[i], markers[i], alphas[i])
        self.bokehgui_time_const_x_0 = bokehgui.time_sink_c_proc(1024, samp_rate, "Received noisy constellation",  1)

        self.bokehgui_time_const_x_0_plot = bokehgui.const_sink_c(self.doc, self.plot_lst, self.bokehgui_time_const_x_0, is_message = False)
        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        legend_list = []
        for i in range(  1 ):
          if len(labels[i]) == 0:
              if(i % 2 == 0):
                  legend_list.append("Re{{Data {0}}}".format(i/2))
              else:
                  legend_list.append("Im{{Data {0}}}".format(i/2))
          else:
              legend_list.append(labels[i])

        self.bokehgui_time_const_x_0_plot.initialize(update_time = 500,
                                    legend_list = legend_list)

        self.bokehgui_time_const_x_0_plot.set_y_axis([-2, 2])
        self.bokehgui_time_const_x_0_plot.set_y_label('Q Channel' + '(' +""+')')

        self.bokehgui_time_const_x_0_plot.set_x_label('I Channel' + '(' +""+')')
        self.bokehgui_time_const_x_0_plot.enable_tags(-1, False)
        self.bokehgui_time_const_x_0_plot.set_trigger_mode(bokehgui.TRIG_MODE_FREE, bokehgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.bokehgui_time_const_x_0_plot.enable_grid(True)
        self.bokehgui_time_const_x_0_plot.enable_axis_labels(True)
        self.bokehgui_time_const_x_0_plot.disable_legend(not True)
        self.bokehgui_time_const_x_0_plot.set_layout(*((0,1,2,2)))
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "blue", "blue", "blue"]
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        markers = ['o', 'o', 'o', 'o', 'o',
                   'o', 'o', 'o', 'o', 'o']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(    1  ):
          self.bokehgui_time_const_x_0_plot.format_line(i, colors[i], widths[i], 'None', markers[i], alphas[i])
        self.bokehgui_frequency_sink_x_0 = bokehgui.freq_sink_c_proc(1024,
                             firdes.WIN_BLACKMAN_hARRIS,
                             freq, samp_rate,
                             "Rx signal",                     1                    )
        self.bokehgui_frequency_sink_x_0_plot = bokehgui.freq_sink_c(self.doc, self.plot_lst, self.bokehgui_frequency_sink_x_0, is_message =False)
        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        legend_list = []

        for i in  range(1):
            if len(labels[i]) == 0:
                legend_list.append("Data {0}".format(i))
            else:
                legend_list.append(labels[i])

        self.bokehgui_frequency_sink_x_0_plot.initialize(update_time = 100,                           legend_list = legend_list)

        self.bokehgui_frequency_sink_x_0_plot.set_y_axis([-140, 10])
        self.bokehgui_frequency_sink_x_0_plot.set_y_label('Relative Gain' + '(' +'dB'+')')
        self.bokehgui_frequency_sink_x_0_plot.set_x_label('Frequency' + '(' +"Hz"+')')

        self.bokehgui_frequency_sink_x_0_plot.set_trigger_mode(bokehgui.TRIG_MODE_FREE,0.0, 0, "")

        self.bokehgui_frequency_sink_x_0_plot.enable_grid(False)
        self.bokehgui_frequency_sink_x_0_plot.enable_axis_labels(True)
        self.bokehgui_frequency_sink_x_0_plot.disable_legend(not True)
        self.bokehgui_frequency_sink_x_0_plot.set_layout(*((3,0,1,3)))
        self.bokehgui_frequency_sink_x_0_plot.enable_max_hold()
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "dark red", "dark green", "dark blue"]
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        styles = ["solid", "solid", "solid", "solid", "solid",
                  "solid", "solid", "solid", "solid", "solid"]
        markers = [None, None, None, None, None,
                   None, None, None, None, None]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in  range(1):
            self.bokehgui_frequency_sink_x_0_plot.format_line(i, colors[i], widths[i], styles[i], markers[i], alphas[i])
        self.blocks_sub_xx_0 = blocks.sub_ff(1)
        self.blocks_repack_bits_bb_0_0_1_1 = blocks.repack_bits_bb(bits_per_symbol, 1, '', False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0_0_1 = blocks.repack_bits_bb(bits_per_symbol, 1, '', False, gr.GR_LSB_FIRST)
        self.blocks_pdu_to_tagged_stream_0_0 = blocks.pdu_to_tagged_stream(blocks.byte_t, 'packet_len')
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*bits_per_symbol)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, fft_len+0*(fft_len//4) + 10)
        self.blocks_char_to_float_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0 = blocks.char_to_float(1, 1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        def _ber_probe():
            while True:

                val = self.blocks_probe_signal_x_0.level()
                try:
                    self.set_ber(val)
                except AttributeError:
                    pass
                time.sleep(1.0 / (0.5))
        _ber_thread = threading.Thread(target=_ber_probe)
        _ber_thread.daemon = True
        _ber_thread.start()

        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, np.sqrt(10**(np.log10(1/bits_per_symbol)-(mag/10.0)))*1, -1)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(-2.0/fft_len)

        if self.widget_lst:
            input_t = bokehgui.bokeh_layout.widgetbox(self.widget_lst)
            widgetbox = bokehgui.bokeh_layout.WidgetLayout(input_t)
            widgetbox.set_layout(*((0, 0, 2, 1)))
            list_obj = [widgetbox] + self.plot_lst
        else:
            list_obj = self.plot_lst
        layout_t = bokehgui.bokeh_layout.create_layout(list_obj, "stretch_both")
        self.doc.add_root(layout_t)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_packet_headerparser_b_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.msg_connect((self.learning_dl_demod_0, 'losses'), (self.zeromq_pub_msg_sink_0_0_0, 'in'))
        self.msg_connect((self.zeromq_sub_msg_source_0_0, 'out'), (self.blocks_pdu_to_tagged_stream_0_0, 'pdus'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.bokehgui_time_const_x_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.learning_align_0, 0))
        self.connect((self.blocks_char_to_float_0_0, 0), (self.blocks_sub_xx_0, 0))
        self.connect((self.blocks_char_to_float_0_0_0, 0), (self.blocks_sub_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.blocks_pdu_to_tagged_stream_0_0, 0), (self.learning_align_0, 1))
        self.connect((self.blocks_repack_bits_bb_0_0_1, 0), (self.learning_ber_bf_0, 1))
        self.connect((self.blocks_repack_bits_bb_0_0_1_1, 0), (self.learning_ber_bf_0, 0))
        self.connect((self.blocks_sub_xx_0, 0), (self.bokehgui_time_sink_x_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_packet_headerparser_b_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.fft_vxx_1, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_header, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_1, 0), (self.digital_ofdm_serializer_vcc_payload, 0))
        self.connect((self.digital_ofdm_serializer_vcc_header, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_payload, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.digital_ofdm_frame_equalizer_vcvc_1, 0))
        self.connect((self.learning_align_0, 1), (self.learning_dl_demod_0, 1))
        self.connect((self.learning_align_0, 0), (self.learning_dl_demod_0, 0))
        self.connect((self.learning_ber_bf_0, 0), (self.blocks_probe_signal_x_0, 0))
        self.connect((self.learning_dl_demod_0, 1), (self.blocks_char_to_float_0_0, 0))
        self.connect((self.learning_dl_demod_0, 2), (self.blocks_char_to_float_0_0_0, 0))
        self.connect((self.learning_dl_demod_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.learning_dl_demod_0, 2), (self.blocks_repack_bits_bb_0_0_1, 0))
        self.connect((self.learning_dl_demod_0, 1), (self.blocks_repack_bits_bb_0_0_1_1, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.bokehgui_frequency_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))

    def get_ip(self):
        return self.ip

    def set_ip(self, ip):
        self.ip = ip

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_packet_length_tag_key(self):
        return self.packet_length_tag_key

    def set_packet_length_tag_key(self, packet_length_tag_key):
        self.packet_length_tag_key = packet_length_tag_key
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_bits_per_symbol_0(len(self.occupied_carriers[0]))
        self.set_hdr_format(digital.header_format_ofdm(self.occupied_carriers, 1, self.length_tag_key,))
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))
        self.set_variable_0(len(self.occupied_carriers[0]))

    def get_length_tag_key_0(self):
        return self.length_tag_key_0

    def set_length_tag_key_0(self, length_tag_key_0):
        self.length_tag_key_0 = length_tag_key_0
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key
        self.set_hdr_format(digital.header_format_ofdm(self.occupied_carriers, 1, self.length_tag_key,))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))
        self.analog_frequency_modulator_fc_0.set_sensitivity(-2.0/self.fft_len)
        self.blocks_delay_0.set_dly(self.fft_len+0*(self.fft_len//4) + 10)

    def get_errors(self):
        return self.errors

    def set_errors(self, errors):
        self.errors = errors
        self.set_variable_qtgui_label_0_val(self.errors)
        self.variable_qtgui_label_0.set_value(str(self.errors))

    def get_bits_per_symbol(self):
        return self.bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol
        self.set_payload_mod(digital.qam_constellation(constellation_points=2**self.bits_per_symbol))
        self.analog_noise_source_x_0.set_amplitude(np.sqrt(10**(np.log10(1/self.bits_per_symbol)-(self.mag/10.0)))*1)
        self.blocks_repack_bits_bb_0_0_1.set_k_and_l(self.bits_per_symbol,1)
        self.blocks_repack_bits_bb_0_0_1_1.set_k_and_l(self.bits_per_symbol,1)

    def get_ber(self):
        return self.ber

    def set_ber(self, ber):
        self.ber = ber
        self.set_variable_qtgui_label_0_0_val(self.ber)
        self.variable_qtgui_label_0_0.set_value(str(self.ber))

    def get_variable_qtgui_label_0_0(self):
        return self.variable_qtgui_label_0_0

    def set_variable_qtgui_label_0_0(self, variable_qtgui_label_0_0):
        self.variable_qtgui_label_0_0 = variable_qtgui_label_0_0
        self.variable_qtgui_label_0_0.set_value(str(self.ber))

    def get_variable_qtgui_label_0(self):
        return self.variable_qtgui_label_0

    def set_variable_qtgui_label_0(self, variable_qtgui_label_0):
        self.variable_qtgui_label_0 = variable_qtgui_label_0
        self.variable_qtgui_label_0.set_value(str(self.errors))

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_training_mod(self):
        return self.training_mod

    def set_training_mod(self, training_mod):
        self.training_mod = training_mod
        self.learning_dl_demod_0.set_alternate(self.training_mod)

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        self.learning_dl_demod_0.load_model(self.timestamp)

    def get_t_state(self):
        return self.t_state

    def set_t_state(self, t_state):
        self.t_state = t_state
        self.learning_dl_demod_0.set_training_state(self.t_state)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_snr_stop(self):
        return self.snr_stop

    def set_snr_stop(self, snr_stop):
        self.snr_stop = snr_stop

    def get_snr_step(self):
        return self.snr_step

    def set_snr_step(self, snr_step):
        self.snr_step = snr_step

    def get_snr_start(self):
        return self.snr_start

    def set_snr_start(self, snr_start):
        self.snr_start = snr_start

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate/6)
        self.bokehgui_frequency_sink_x_0.set_frequency_range(self.freq, self.samp_rate)
        self.bokehgui_time_const_x_0.set_samp_rate(self.samp_rate)
        self.bokehgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_lr(self):
        return self.rx_lr

    def set_rx_lr(self, rx_lr):
        self.rx_lr = rx_lr
        self.learning_dl_demod_0.set_lr(10**self.rx_lr )

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff

    def get_puncpat(self):
        return self.puncpat

    def set_puncpat(self, puncpat):
        self.puncpat = puncpat

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_payload_equalizer(self):
        return self.payload_equalizer

    def set_payload_equalizer(self, payload_equalizer):
        self.payload_equalizer = payload_equalizer

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len

    def get_mag(self):
        return self.mag

    def set_mag(self, mag):
        self.mag = mag
        self.analog_noise_source_x_0.set_amplitude(np.sqrt(10**(np.log10(1/self.bits_per_symbol)-(self.mag/10.0)))*1)

    def get_ldpc_enc(self):
        return self.ldpc_enc

    def set_ldpc_enc(self, ldpc_enc):
        self.ldpc_enc = ldpc_enc

    def get_ldpc_dec(self):
        return self.ldpc_dec

    def set_ldpc_dec(self, ldpc_dec):
        self.ldpc_dec = ldpc_dec

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_header_equalizer(self):
        return self.header_equalizer

    def set_header_equalizer(self, header_equalizer):
        self.header_equalizer = header_equalizer

    def get_hdr_format(self):
        return self.hdr_format

    def set_hdr_format(self, hdr_format):
        self.hdr_format = hdr_format

    def get_gain_rx(self):
        return self.gain_rx

    def set_gain_rx(self, gain_rx):
        self.gain_rx = gain_rx
        self.uhd_usrp_source_0.set_gain(self.gain_rx, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.bokehgui_frequency_sink_x_0.set_frequency_range(self.freq, self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_bits_per_symbol_0(self):
        return self.bits_per_symbol_0

    def set_bits_per_symbol_0(self, bits_per_symbol_0):
        self.bits_per_symbol_0 = bits_per_symbol_0


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--ip", dest="ip", type=str, default='localhost',
        help="Set IP [default=%(default)r]")
    return parser


def main(top_block_cls=dl_demod_RX, options=None):
    if options is None:
        options = argument_parser().parse_args()
    serverProc, port = bokehgui.utils.create_server()
    def killProc(signum, frame, tb):
        tb.stop()
        tb.wait()
        serverProc.terminate()
        serverProc.kill()
    time.sleep(1)
    try:
        # Define the document instance
        doc = curdoc()
        doc.title = "RL Demod training - Cyrille Morin"
        session = push_session(doc, session_id="dl_demod_RX",
                               url = "http://localhost:" + port + "/bokehgui")
        # Create Top Block instance
        tb = top_block_cls(doc, ip=options.ip)
        try:
            tb.start()
            signal.signal(signal.SIGTERM, functools.partial(killProc, tb=tb))
            session.loop_until_closed()
        finally:
            print("Exiting the simulation. Stopping Bokeh Server")
            tb.stop()
            tb.wait()
    finally:
        serverProc.terminate()
        serverProc.kill()


if __name__ == '__main__':
    main()
