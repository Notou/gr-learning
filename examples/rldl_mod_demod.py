#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RL Mod/Demod training
# Author: Cyrille Morin
# GNU Radio version: 3.8.2.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import eng_notation
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import fec
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import uhd
import time
from gnuradio import zeromq
from gnuradio.digital.utils import tagged_streams
from gnuradio.qtgui import Range, RangeWidget
import learning
import numpy as np

from gnuradio import qtgui

class rldl_mod_demod(gr.top_block, Qt.QWidget):

    def __init__(self, puncpat='11'):
        gr.top_block.__init__(self, "RL Mod/Demod training")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RL Mod/Demod training")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "rldl_mod_demod")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Parameters
        ##################################################
        self.puncpat = puncpat

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
        self.bits_per_symbol = bits_per_symbol = 4
        self.variable_0 = variable_0 = len(occupied_carriers[0])
        self.tx_lr = tx_lr = -3
        self.tx_explo = tx_explo = 0.15
        self.training_mod = training_mod = 1
        self.timestamp = timestamp = '1'
        self.t_state = t_state = 1
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.snr_stop = snr_stop = 29
        self.snr_step = snr_step = 0.5
        self.snr_start = snr_start = 0
        self.save_0 = save_0 = 0
        self.save_0_args = save_0_args = None
        self.samp_rate = samp_rate = 0.3e6
        self.rx_lr = rx_lr = -2
        self.rolloff = rolloff = 0
        self.reset_tx = reset_tx = 0
        self.reset_tx_args = reset_tx_args = None
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bits_per_symbol,differential=False,mod_code=digital.utils.mod_codes.GRAY_CODE)
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_static(fft_len,  occupied_carriers, pilot_carriers, pilot_symbols, 1)
        self.packet_len = packet_len = 400
        self.mag = mag = 10
        self.ldpc_enc = ldpc_enc = fec.ldpc_encoder_make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist")
        self.ldpc_dec_0 = ldpc_dec_0 = fec.ldpc_decoder.make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist", 0.5, 50)
        self.ldpc_dec = ldpc_dec = fec.ldpc_decoder.make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist", 0.5, 50)
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=packet_length_tag_key, frame_len_tag_key=length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, header_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols)
        self.hdr_format = hdr_format = digital.header_format_ofdm(occupied_carriers, 1, length_tag_key,)
        self.gain_rx = gain_rx = 5
        self.gain = gain = 10
        self.freq = freq = 900e6
        self.bits_per_symbol_0 = bits_per_symbol_0 = len(occupied_carriers[0])

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_tab_widget_0 = Qt.QTabWidget()
        self.qtgui_tab_widget_0_widget_0 = Qt.QWidget()
        self.qtgui_tab_widget_0_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.qtgui_tab_widget_0_widget_0)
        self.qtgui_tab_widget_0_grid_layout_0 = Qt.QGridLayout()
        self.qtgui_tab_widget_0_layout_0.addLayout(self.qtgui_tab_widget_0_grid_layout_0)
        self.qtgui_tab_widget_0.addTab(self.qtgui_tab_widget_0_widget_0, 'Mainoutput')
        self.qtgui_tab_widget_0_widget_1 = Qt.QWidget()
        self.qtgui_tab_widget_0_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.qtgui_tab_widget_0_widget_1)
        self.qtgui_tab_widget_0_grid_layout_1 = Qt.QGridLayout()
        self.qtgui_tab_widget_0_layout_1.addLayout(self.qtgui_tab_widget_0_grid_layout_1)
        self.qtgui_tab_widget_0.addTab(self.qtgui_tab_widget_0_widget_1, 'Debug')
        self.top_grid_layout.addWidget(self.qtgui_tab_widget_0)
        self._tx_lr_range = Range(-5, 0, 0.1, -3, 200)
        self._tx_lr_win = RangeWidget(self._tx_lr_range, self.set_tx_lr, 'TX Learning rate (log)', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._tx_lr_win, 8, 4, 1, 2)
        for r in range(8, 9):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._tx_explo_range = Range(0.001, 0.5, 0.001, 0.15, 200)
        self._tx_explo_win = RangeWidget(self._tx_explo_range, self.set_tx_explo, 'TX Exploration noise', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._tx_explo_win, 9, 4, 1, 2)
        for r in range(9, 10):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._training_mod_tool_bar = Qt.QToolBar(self)
        self._training_mod_tool_bar.addWidget(Qt.QLabel('Alternate training every' + ": "))
        self._training_mod_line_edit = Qt.QLineEdit(str(self.training_mod))
        self._training_mod_tool_bar.addWidget(self._training_mod_line_edit)
        self._training_mod_line_edit.returnPressed.connect(
            lambda: self.set_training_mod(int(str(self._training_mod_line_edit.text()))))
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._training_mod_tool_bar, 4, 1, 1, 1)
        for r in range(4, 5):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        # Create the options list
        self._t_state_options = (1, 0, )
        # Create the labels list
        self._t_state_labels = ('On', 'Off', )
        # Create the combo box
        # Create the radio buttons
        self._t_state_group_box = Qt.QGroupBox('Training' + ": ")
        self._t_state_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._t_state_button_group = variable_chooser_button_group()
        self._t_state_group_box.setLayout(self._t_state_box)
        for i, _label in enumerate(self._t_state_labels):
            radio_button = Qt.QRadioButton(_label)
            self._t_state_box.addWidget(radio_button)
            self._t_state_button_group.addButton(radio_button, i)
        self._t_state_callback = lambda i: Qt.QMetaObject.invokeMethod(self._t_state_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._t_state_options.index(i)))
        self._t_state_callback(self.t_state)
        self._t_state_button_group.buttonClicked[int].connect(
            lambda i: self.set_t_state(self._t_state_options[i]))
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._t_state_group_box, 4, 0, 1, 1)
        for r in range(4, 5):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._rx_lr_range = Range(-5, 0, 0.1, -2, 200)
        self._rx_lr_win = RangeWidget(self._rx_lr_range, self.set_rx_lr, 'RX Learning rate (log)', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._rx_lr_win, 10, 4, 1, 2)
        for r in range(10, 11):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._mag_range = Range(snr_start, 40, 0.1, 10, 200)
        self._mag_win = RangeWidget(self._mag_range, self.set_mag, 'Eb/N0', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._mag_win, 2, 0, 2, 4)
        for r in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.learning_rl_mod_0 = learning.rl_mod('packet_num', bits_per_symbol, packet_len, 1, training_mod, 10**tx_lr , tx_explo, t_state , '/home/cyrille/Gnu-Radio/modules/gr-learning/examples/saved_models')
        self._gain_rx_range = Range(0, 90, .2, 5, 200)
        self._gain_rx_win = RangeWidget(self._gain_rx_range, self.set_gain_rx, 'Amplitude Rx', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._gain_rx_win, 0, 4, 2, 2)
        for r in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._gain_range = Range(0, 90, .2, 10, 200)
        self._gain_win = RangeWidget(self._gain_range, self.set_gain, 'Amplitude', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._gain_win, 0, 0, 2, 4)
        for r in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.zeromq_sub_msg_source_0_1 = zeromq.sub_msg_source('tcp://127.0.0.1:50002', 100, False)
        self.zeromq_sub_msg_source_0_0 = zeromq.sub_msg_source('tcp://127.0.0.1:50001', 100, False)
        self.zeromq_pub_msg_sink_0_0_0 = zeromq.pub_msg_sink('tcp://127.0.0.1:50002', 100, True)
        self.zeromq_pub_msg_sink_0_0 = zeromq.pub_msg_sink('tcp://127.0.0.1:50001', 100, True)
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
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_gain(gain_rx, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            length_tag_key,
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_gain(gain, 0)
        self._timestamp_tool_bar = Qt.QToolBar(self)
        self._timestamp_tool_bar.addWidget(Qt.QLabel('Load timestamp' + ": "))
        self._timestamp_line_edit = Qt.QLineEdit(str(self.timestamp))
        self._timestamp_tool_bar.addWidget(self._timestamp_line_edit)
        self._timestamp_line_edit.returnPressed.connect(
            lambda: self.set_timestamp(str(str(self._timestamp_line_edit.text()))))
        self.top_grid_layout.addWidget(self._timestamp_tool_bar)
        _save_0_push_button = Qt.QPushButton('&Save TX model')
        def save_0_handler():
          if self.save_0_args is not None:
            self.set_save_0(self.learning_rl_mod_0.save_model())
          else:
            self.set_save_0(self.learning_rl_mod_0.save_model())
        _save_0_push_button.clicked.connect( save_0_handler )
        self.top_grid_layout.addWidget(_save_0_push_button)
        _reset_tx_push_button = Qt.QPushButton('&Reset TX model state')
        def reset_tx_handler():
          if self.reset_tx_args is not None:
            self.set_reset_tx(self.learning_rl_mod_0.reset())
          else:
            self.set_reset_tx(self.learning_rl_mod_0.reset())
        _reset_tx_push_button.clicked.connect( reset_tx_handler )
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(_reset_tx_push_button, 5, 0, 1, 1)
        for r in range(5, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_time_sink_x_2 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_2.set_update_time(0.10)
        self.qtgui_time_sink_x_2.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_2.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_2.enable_tags(True)
        self.qtgui_time_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_2.enable_autoscale(False)
        self.qtgui_time_sink_x_2.enable_grid(False)
        self.qtgui_time_sink_x_2.enable_axis_labels(True)
        self.qtgui_time_sink_x_2.enable_control_panel(False)
        self.qtgui_time_sink_x_2.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_2.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_2.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_2.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_2.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_2.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_2.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_2.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_2.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_2_win = sip.wrapinstance(self.qtgui_time_sink_x_2.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_layout_1.addWidget(self._qtgui_time_sink_x_2_win)
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'Scope Plot', #name
            3, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "packet_num")
        self.qtgui_time_sink_x_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0.enable_grid(False)
        self.qtgui_time_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0.enable_stem_plot(False)


        labels = ['Scope Plot', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(3):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_time_sink_x_0_0_win, 6, 2, 2, 2)
        for r in range(6, 8):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            "Transmitted noiseless constellation", #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(True)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['Classical', 'Learned', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_const_sink_x_0_0_win, 6, 4, 2, 2)
        for r in range(6, 8):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "Received noisy constellation", #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['Classical', 'Learned', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_const_sink_x_0_win, 6, 0, 2, 2)
        for r in range(6, 8):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.learning_tag_numerotation_0 = learning.tag_numerotation('packet_num', packet_len, 4096, "bb")
        self.learning_dl_demod_0 = learning.dl_demod('packet_num', True, bits_per_symbol, packet_len, 20, training_mod, 10**rx_lr , t_state, '/home/cyrille/Gnu-Radio/modules/gr-learning/examples/saved_models')
        self.learning_align_0_0 = learning.align('packet_num', packet_len, 400, 1, 48)
        self.learning_align_0 = learning.align('packet_num', packet_len, 400, 1, 48)
        self.fft_vxx_1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fec_extended_encoder_1_0_0 = fec.extended_encoder(encoder_obj_list=ldpc_enc, threading= None, puncpat=puncpat)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(hdr_format, length_tag_key)
        self.digital_packet_headerparser_b_0 = digital.packet_headerparser_b(header_formatter.base())
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, fft_len//4, False, 0.9)
        self.digital_ofdm_serializer_vcc_payload = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key_0, packet_length_tag_key, 1, '', True)
        self.digital_ofdm_serializer_vcc_header = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key_0, '', 0, '', True)
        self.digital_ofdm_frame_equalizer_vcvc_1 = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), fft_len//4, length_tag_key_0, True, 0)
        self.digital_ofdm_frame_equalizer_vcvc_0 = digital.ofdm_frame_equalizer_vcvc(header_equalizer.base(), fft_len//4, length_tag_key_0, True, 1)
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + fft_len//4,
            rolloff,
            length_tag_key)
        self.digital_ofdm_chanest_vcvc_0 = digital.ofdm_chanest_vcvc(sync_word1, sync_word2, 1, 0, 3, False)
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, sync_word2), length_tag_key, True)
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
        self.digital_constellation_decoder_cb_1 = digital.constellation_decoder_cb(payload_mod.base())
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(header_mod.base())
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, samp_rate/6,True)
        self.blocks_tagged_stream_to_pdu_0_0 = blocks.tagged_stream_to_pdu(blocks.byte_t, 'packet_len')
        self.blocks_tagged_stream_mux_0_0_0 = blocks.tagged_stream_mux(gr.sizeof_char*1, length_tag_key, 0)
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, length_tag_key, 0)
        self.blocks_tag_gate_0_0_0 = blocks.tag_gate(gr.sizeof_gr_complex * 1, False)
        self.blocks_tag_gate_0_0_0.set_single_key("ofdm_sync_chan_taps")
        self.blocks_sub_xx_0_0_0 = blocks.sub_ff(1)
        self.blocks_sub_xx_0_0 = blocks.sub_ff(1)
        self.blocks_sub_xx_0 = blocks.sub_ff(1)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0_0_0 = blocks.repack_bits_bb(8, 1, length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(1, bits_per_symbol, '', False, gr.GR_LSB_FIRST)
        self.blocks_pdu_to_tagged_stream_0_0 = blocks.pdu_to_tagged_stream(blocks.byte_t, 'packet_len')
        self.blocks_null_sink_0_0 = blocks.null_sink(gr.sizeof_gr_complex*2**bits_per_symbol)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*bits_per_symbol)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_cc(0.01, 1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, fft_len+0*(fft_len//4) + 10)
        self.blocks_deinterleave_0 = blocks.deinterleave(gr.sizeof_gr_complex*1, packet_len)
        self.blocks_char_to_float_0_0_1_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0_1 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0 = blocks.char_to_float(1, 1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 2, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, np.sqrt(10**(np.log10(1/bits_per_symbol)-(mag/10.0)))*1, -1)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(-2.0/fft_len)



        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_tagged_stream_to_pdu_0_0, 'pdus'), (self.zeromq_pub_msg_sink_0_0, 'in'))
        self.msg_connect((self.digital_packet_headerparser_b_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.msg_connect((self.learning_dl_demod_0, 'losses'), (self.zeromq_pub_msg_sink_0_0_0, 'in'))
        self.msg_connect((self.zeromq_sub_msg_source_0_0, 'out'), (self.blocks_pdu_to_tagged_stream_0_0, 'pdus'))
        self.msg_connect((self.zeromq_sub_msg_source_0_1, 'out'), (self.learning_rl_mod_0, 'losses'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_random_source_x_0_0, 0), (self.fec_extended_encoder_1_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_deinterleave_0, 0))
        self.connect((self.blocks_char_to_float_0_0, 0), (self.blocks_sub_xx_0, 0))
        self.connect((self.blocks_char_to_float_0_0_0, 0), (self.blocks_sub_xx_0, 1))
        self.connect((self.blocks_char_to_float_0_0_0_0, 0), (self.blocks_sub_xx_0_0, 1))
        self.connect((self.blocks_char_to_float_0_0_0_0_0, 0), (self.blocks_sub_xx_0_0_0, 1))
        self.connect((self.blocks_char_to_float_0_0_1, 0), (self.blocks_sub_xx_0_0, 0))
        self.connect((self.blocks_char_to_float_0_0_1_0, 0), (self.blocks_sub_xx_0_0_0, 0))
        self.connect((self.blocks_deinterleave_0, 1), (self.learning_align_0, 0))
        self.connect((self.blocks_deinterleave_0, 0), (self.learning_align_0_0, 0))
        self.connect((self.blocks_deinterleave_0, 1), (self.qtgui_const_sink_x_0, 1))
        self.connect((self.blocks_deinterleave_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.blocks_pdu_to_tagged_stream_0_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.blocks_tagged_stream_mux_0_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.blocks_tagged_stream_mux_0_0_0, 1))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.learning_tag_numerotation_0, 0))
        self.connect((self.blocks_sub_xx_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.blocks_sub_xx_0_0, 0), (self.qtgui_time_sink_x_0_0, 1))
        self.connect((self.blocks_sub_xx_0_0_0, 0), (self.qtgui_time_sink_x_0_0, 2))
        self.connect((self.blocks_tag_gate_0_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0_0_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.learning_align_0, 1))
        self.connect((self.blocks_throttle_0, 0), (self.learning_align_0_0, 1))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_packet_headerparser_b_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1, 0), (self.blocks_char_to_float_0_0_1, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.fft_vxx_1, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_header, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_1, 0), (self.digital_ofdm_serializer_vcc_payload, 0))
        self.connect((self.digital_ofdm_serializer_vcc_header, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_payload, 0), (self.blocks_tag_gate_0_0_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_repack_bits_bb_0_0_0, 0))
        self.connect((self.fec_extended_encoder_1_0_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.digital_ofdm_frame_equalizer_vcvc_1, 0))
        self.connect((self.learning_align_0, 1), (self.blocks_char_to_float_0_0_0_0_0, 0))
        self.connect((self.learning_align_0, 1), (self.learning_dl_demod_0, 1))
        self.connect((self.learning_align_0, 0), (self.learning_dl_demod_0, 0))
        self.connect((self.learning_align_0_0, 1), (self.blocks_char_to_float_0_0_0_0, 0))
        self.connect((self.learning_align_0_0, 1), (self.blocks_char_to_float_0_0_1_0, 0))
        self.connect((self.learning_align_0_0, 0), (self.digital_constellation_decoder_cb_1, 0))
        self.connect((self.learning_dl_demod_0, 1), (self.blocks_char_to_float_0_0, 0))
        self.connect((self.learning_dl_demod_0, 2), (self.blocks_char_to_float_0_0_0, 0))
        self.connect((self.learning_dl_demod_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.learning_rl_mod_0, 2), (self.blocks_null_sink_0_0, 0))
        self.connect((self.learning_rl_mod_0, 0), (self.blocks_tagged_stream_mux_0, 2))
        self.connect((self.learning_rl_mod_0, 1), (self.qtgui_const_sink_x_0_0, 1))
        self.connect((self.learning_tag_numerotation_0, 0), (self.blocks_tagged_stream_to_pdu_0_0, 0))
        self.connect((self.learning_tag_numerotation_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.learning_tag_numerotation_0, 0), (self.learning_rl_mod_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_time_sink_x_2, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "rldl_mod_demod")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_puncpat(self):
        return self.puncpat

    def set_puncpat(self, puncpat):
        self.puncpat = puncpat

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

    def get_bits_per_symbol(self):
        return self.bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol
        self.set_payload_mod(digital.qam_constellation(constellation_points=2**self.bits_per_symbol,differential=False,mod_code=digital.utils.mod_codes.GRAY_CODE))
        self.analog_noise_source_x_0.set_amplitude(np.sqrt(10**(np.log10(1/self.bits_per_symbol)-(self.mag/10.0)))*1)
        self.blocks_repack_bits_bb_0_0.set_k_and_l(1,self.bits_per_symbol)

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_tx_lr(self):
        return self.tx_lr

    def set_tx_lr(self, tx_lr):
        self.tx_lr = tx_lr
        self.learning_rl_mod_0.set_lr(10**self.tx_lr )

    def get_tx_explo(self):
        return self.tx_explo

    def set_tx_explo(self, tx_explo):
        self.tx_explo = tx_explo
        self.learning_rl_mod_0.set_explo_noise(self.tx_explo)

    def get_training_mod(self):
        return self.training_mod

    def set_training_mod(self, training_mod):
        self.training_mod = training_mod
        Qt.QMetaObject.invokeMethod(self._training_mod_line_edit, "setText", Qt.Q_ARG("QString", str(self.training_mod)))
        self.learning_dl_demod_0.set_alternate(self.training_mod)
        self.learning_rl_mod_0.set_alternate(self.training_mod)

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        Qt.QMetaObject.invokeMethod(self._timestamp_line_edit, "setText", Qt.Q_ARG("QString", str(self.timestamp)))
        self.learning_dl_demod_0.load_model(self.timestamp)
        self.learning_rl_mod_0.load_model(self.timestamp)

    def get_t_state(self):
        return self.t_state

    def set_t_state(self, t_state):
        self.t_state = t_state
        self._t_state_callback(self.t_state)
        self.learning_dl_demod_0.set_training_state(self.t_state)
        self.learning_rl_mod_0.set_training_state(self.t_state )

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

    def get_save_0(self):
        return self.save_0

    def set_save_0(self, save_0):
        self.save_0 = save_0

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate/6)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_2.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
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

    def get_reset_tx(self):
        return self.reset_tx

    def set_reset_tx(self, reset_tx):
        self.reset_tx = reset_tx

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
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len_pmt(self.packet_len)

    def get_mag(self):
        return self.mag

    def set_mag(self, mag):
        self.mag = mag
        self.analog_noise_source_x_0.set_amplitude(np.sqrt(10**(np.log10(1/self.bits_per_symbol)-(self.mag/10.0)))*1)

    def get_ldpc_enc(self):
        return self.ldpc_enc

    def set_ldpc_enc(self, ldpc_enc):
        self.ldpc_enc = ldpc_enc

    def get_ldpc_dec_0(self):
        return self.ldpc_dec_0

    def set_ldpc_dec_0(self, ldpc_dec_0):
        self.ldpc_dec_0 = ldpc_dec_0

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

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_bits_per_symbol_0(self):
        return self.bits_per_symbol_0

    def set_bits_per_symbol_0(self, bits_per_symbol_0):
        self.bits_per_symbol_0 = bits_per_symbol_0




def argument_parser():
    parser = ArgumentParser()
    return parser


def main(top_block_cls=rldl_mod_demod, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()

if __name__ == '__main__':
    main()
