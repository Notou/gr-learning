#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RL Mod/Demod training
# Author: Cyrille Morin
# GNU Radio version: 3.8.1.0

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
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio.digital.utils import tagged_streams
from gnuradio.qtgui import Range, RangeWidget
import learning
import numpy as np
import time
import threading
from gnuradio import qtgui

class rldl_mod_demod(gr.top_block, Qt.QWidget):

    def __init__(self):
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
        # Variables
        ##################################################
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, 1, -1,1,),)
        self.pilot_carriers = pilot_carriers = ((-27, -21, -7, 7, 21,27),)
        self.packet_length_tag_key = packet_length_tag_key = "packet_length"
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.length_tag_key_0 = length_tag_key_0 = "frame_len"
        self.length_tag_key = length_tag_key = "packet_len"
        self.header_mod = header_mod = digital.constellation_bpsk()
        self.fft_len = fft_len = 64
        self.errors = errors = 0
        self.bits_per_symbol = bits_per_symbol = 2
        self.variable_qtgui_label_0 = variable_qtgui_label_0 = errors
        self.variable_0 = variable_0 = len(occupied_carriers[0])
        self.tx_lr = tx_lr = -3
        self.tx_explo = tx_explo = 0.15
        self.training_mod = training_mod = 10
        self.t_state = t_state = 1
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.stop_button = stop_button = 0
        self.stop_button_args = stop_button_args = None
        self.start_button = start_button = 0
        self.start_button_args = start_button_args = None
        self.snr_stop = snr_stop = 29
        self.snr_step = snr_step = 0.5
        self.snr_start = snr_start = 0
        self.samp_rate = samp_rate = 0.3e6
        self.rx_lr = rx_lr = -2
        self.rolloff = rolloff = 0
        self.reset_rx = reset_rx = 0
        self.reset_rx_args = reset_rx_args = None
        self.reset_ber = reset_ber = 0
        self.reset_ber_args = reset_ber_args = None
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bits_per_symbol)
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_static(fft_len,  occupied_carriers, pilot_carriers, pilot_symbols, 1)
        self.packet_len = packet_len = 1008
        self.mag = mag = 10
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=packet_length_tag_key, frame_len_tag_key=length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, header_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols)
        self.hdr_format = hdr_format = digital.header_format_ofdm(occupied_carriers, 1, length_tag_key,)
        self.gain_rx = gain_rx = 40
        self.gain = gain = 30
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
        self._t_state_options = (0, 0, )
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
        self._mag_range = Range(snr_start, 40, 0.01, 10, 200)
        self._mag_win = RangeWidget(self._mag_range, self.set_mag, 'Eb/N0', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._mag_win, 2, 0, 2, 4)
        for r in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.learning_sweeper_0 = learning.sweeper(self, np.arange(snr_start,snr_stop,snr_step), 200, 'learning_ber_bf_0', 'blocks_probe_signal_x_0', self.set_mag)
        self.learning_dl_eq_demod_0 = learning.dl_eq_demod(occupied_carriers, pilot_carriers, 'packet_num', True, bits_per_symbol, packet_len, 1, training_mod, 10**rx_lr , t_state, 64)
        self.learning_ber_bf_0 = learning.ber_bf(False, 100, -7.0, bits_per_symbol)
        self._variable_qtgui_label_0_tool_bar = Qt.QToolBar(self)

        if None:
            self._variable_qtgui_label_0_formatter = None
        else:
            self._variable_qtgui_label_0_formatter = lambda x: str(x)

        self._variable_qtgui_label_0_tool_bar.addWidget(Qt.QLabel('Error count' + ": "))
        self._variable_qtgui_label_0_label = Qt.QLabel(str(self._variable_qtgui_label_0_formatter(self.variable_qtgui_label_0)))
        self._variable_qtgui_label_0_tool_bar.addWidget(self._variable_qtgui_label_0_label)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._variable_qtgui_label_0_tool_bar, 4, 2, 1, 2)
        for r in range(4, 5):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
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
        _stop_button_push_button = Qt.QPushButton('&Stop sweep')
        def stop_button_handler():
          if self.stop_button_args is not None:
            self.set_stop_button(self.learning_sweeper_0.stop_sweep())
          else:
            self.set_stop_button(self.learning_sweeper_0.stop_sweep())
        _stop_button_push_button.clicked.connect( stop_button_handler )
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(_stop_button_push_button, 9, 0, 1, 1)
        for r in range(9, 10):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        _start_button_push_button = Qt.QPushButton('&Start sweep')
        def start_button_handler():
          if self.start_button_args is not None:
            self.set_start_button(self.learning_sweeper_0.start_sweep())
          else:
            self.set_start_button(self.learning_sweeper_0.start_sweep())
        _start_button_push_button.clicked.connect( start_button_handler )
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(_start_button_push_button, 8, 0, 1, 1)
        for r in range(8, 9):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        _reset_rx_push_button = Qt.QPushButton('&Reset RX model state')
        def reset_rx_handler():
          if self.reset_rx_args is not None:
            self.set_reset_rx(self.learning_dl_eq_demod_0.reset())
          else:
            self.set_reset_rx(self.learning_dl_eq_demod_0.reset())
        _reset_rx_push_button.clicked.connect( reset_rx_handler )
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(_reset_rx_push_button, 5, 1, 1, 1)
        for r in range(5, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        _reset_ber_push_button = Qt.QPushButton("Reset &BER last: {}".format(reset_ber))
        def reset_ber_handler():
          if self.reset_ber_args is not None:
            self.set_reset_ber(self.learning_ber_bf_0.reset_counters())
          else:
            self.set_reset_ber(self.learning_ber_bf_0.reset_counters())
        _reset_ber_push_button.clicked.connect( reset_ber_handler )
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(_reset_ber_push_button, 5, 2, 1, 2)
        for r in range(5, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_vector_sink_f_0_0 = qtgui.vector_sink_f(
            len(np.arange(snr_start,snr_stop,snr_step)),
            snr_start,
            snr_step,
            "Eb/N0",
            "BER (log)",
            "BER Curve",
            2 # Number of inputs
        )
        self.qtgui_vector_sink_f_0_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0_0.set_y_axis(-6, 0)
        self.qtgui_vector_sink_f_0_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0_0.enable_grid(True)
        self.qtgui_vector_sink_f_0_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0_0.set_ref_level(0)

        labels = ['Measure', 'QPSK', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_vector_sink_f_0_0_win, 8, 2, 3, 2)
        for r in range(8, 11):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(2, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'Scope Plot', #name
            1 #number of inputs
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


        for i in range(1):
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
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            64, #size
            samp_rate, #samp_rate
            "", #name
            1 #number of inputs
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_TAG, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "packet_num")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


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
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_layout_1.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_number_sink_1_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1
        )
        self.qtgui_number_sink_1_0.set_update_time(0.10)
        self.qtgui_number_sink_1_0.set_title("BER")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1_0.set_min(i, 0)
            self.qtgui_number_sink_1_0.set_max(i, 100)
            self.qtgui_number_sink_1_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1_0.set_label(i, labels[i])
            self.qtgui_number_sink_1_0.set_unit(i, units[i])
            self.qtgui_number_sink_1_0.set_factor(i, factor[i])

        self.qtgui_number_sink_1_0.enable_autoscale(False)
        self._qtgui_number_sink_1_0_win = sip.wrapinstance(self.qtgui_number_sink_1_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_number_sink_1_0_win, 4, 4, 2, 2)
        for r in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0_1 = qtgui.const_sink_c(
            1024, #size
            "", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0_1.set_update_time(0.10)
        self.qtgui_const_sink_x_0_1.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_1.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_1.enable_autoscale(False)
        self.qtgui_const_sink_x_0_1.enable_grid(True)
        self.qtgui_const_sink_x_0_1.enable_axis_labels(True)


        labels = ['', '', '', '', '',
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

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_1.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_1.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_1.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_1.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_1.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_1.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_1_win = sip.wrapinstance(self.qtgui_const_sink_x_0_1.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_layout_1.addWidget(self._qtgui_const_sink_x_0_1_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            "", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(True)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
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

        for i in range(1):
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
            "", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
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

        for i in range(1):
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
        self.learning_align_0 = learning.align("packet_num", packet_len, 64, 48)
        self._gain_rx_range = Range(0, 90, .2, 40, 200)
        self._gain_rx_win = RangeWidget(self._gain_rx_range, self.set_gain_rx, 'Amplitude Rx', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._gain_rx_win, 0, 4, 2, 2)
        for r in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(4, 6):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._gain_range = Range(0, 90, .2, 30, 200)
        self._gain_win = RangeWidget(self._gain_range, self.set_gain, 'Amplitude', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._gain_win, 0, 0, 2, 4)
        for r in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.fft_vxx_1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        def _errors_probe():
            while True:

                val = self.learning_ber_bf_0.total_errors()
                try:
                    self.set_errors(val)
                except AttributeError:
                    pass
                time.sleep(1.0 / (10))
        _errors_thread = threading.Thread(target=_errors_probe)
        _errors_thread.daemon = True
        _errors_thread.start()

        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(hdr_format, length_tag_key)
        self.digital_packet_headerparser_b_0 = digital.packet_headerparser_b(header_formatter.base())
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, fft_len//4, False, 0.9)
        self.digital_ofdm_serializer_vcc_payload = digital.ofdm_serializer_vcc(fft_len, [range(-(fft_len//2), fft_len//2)], length_tag_key_0, '', 1, '', True)
        self.digital_ofdm_serializer_vcc_header = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key_0, '', 0, '', True)
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
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(header_mod.base())
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_gr_complex*1, 64)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, length_tag_key, 0)
        self.blocks_tag_gate_0 = blocks.tag_gate(gr.sizeof_gr_complex * 1, False)
        self.blocks_tag_gate_0.set_single_key("")
        self.blocks_sub_xx_0 = blocks.sub_ff(1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0_0_0 = blocks.repack_bits_bb(8, 1, length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_probe_signal_x_0 = blocks.probe_signal_f()
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_cc(0.01, 1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, fft_len+fft_len//4)
        self.blocks_char_to_float_0_0_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_0_0 = blocks.char_to_float(1, 1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 2**bits_per_symbol, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, np.sqrt(10**(np.log10(1/bits_per_symbol)-(mag/10.0)))*1, -1)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(-2.0/fft_len)



        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_packet_headerparser_b_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_random_source_x_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_char_to_float_0_0, 0), (self.blocks_sub_xx_0, 0))
        self.connect((self.blocks_char_to_float_0_0_0, 0), (self.blocks_sub_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.blocks_tag_gate_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.learning_tag_numerotation_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.learning_align_0, 0))
        self.connect((self.blocks_sub_xx_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.blocks_tag_gate_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))
        self.connect((self.blocks_vector_to_stream_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.blocks_vector_to_stream_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_packet_headerparser_b_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.fft_vxx_1, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_header, 0))
        self.connect((self.digital_ofdm_serializer_vcc_header, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_payload, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_repack_bits_bb_0_0_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.blocks_vector_to_stream_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.digital_ofdm_serializer_vcc_payload, 0))
        self.connect((self.learning_align_0, 0), (self.learning_dl_eq_demod_0, 0))
        self.connect((self.learning_align_0, 1), (self.learning_dl_eq_demod_0, 1))
        self.connect((self.learning_ber_bf_0, 0), (self.blocks_probe_signal_x_0, 0))
        self.connect((self.learning_ber_bf_0, 0), (self.qtgui_number_sink_1_0, 0))
        self.connect((self.learning_dl_eq_demod_0, 0), (self.blocks_char_to_float_0_0, 0))
        self.connect((self.learning_dl_eq_demod_0, 1), (self.blocks_char_to_float_0_0_0, 0))
        self.connect((self.learning_dl_eq_demod_0, 0), (self.learning_ber_bf_0, 0))
        self.connect((self.learning_dl_eq_demod_0, 1), (self.learning_ber_bf_0, 1))
        self.connect((self.learning_dl_eq_demod_0, 2), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.learning_dl_eq_demod_0, 3), (self.qtgui_const_sink_x_0_1, 0))
        self.connect((self.learning_sweeper_0, 0), (self.qtgui_vector_sink_f_0_0, 0))
        self.connect((self.learning_sweeper_0, 1), (self.qtgui_vector_sink_f_0_0, 1))
        self.connect((self.learning_tag_numerotation_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.learning_tag_numerotation_0, 0), (self.learning_align_0, 1))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "rldl_mod_demod")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

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
        self.blocks_delay_0.set_dly(self.fft_len+self.fft_len//4)

    def get_errors(self):
        return self.errors

    def set_errors(self, errors):
        self.errors = errors
        self.set_variable_qtgui_label_0(self._variable_qtgui_label_0_formatter(self.errors))
        self.learning_sweeper_0.set_error_count(self.errors)

    def get_bits_per_symbol(self):
        return self.bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol
        self.set_payload_mod(digital.qam_constellation(constellation_points=2**self.bits_per_symbol))
        self.analog_noise_source_x_0.set_amplitude(np.sqrt(10**(np.log10(1/self.bits_per_symbol)-(self.mag/10.0)))*1)
        self.blocks_repack_bits_bb_0_0.set_k_and_l(8,self.bits_per_symbol)

    def get_variable_qtgui_label_0(self):
        return self.variable_qtgui_label_0

    def set_variable_qtgui_label_0(self, variable_qtgui_label_0):
        self.variable_qtgui_label_0 = variable_qtgui_label_0
        Qt.QMetaObject.invokeMethod(self._variable_qtgui_label_0_label, "setText", Qt.Q_ARG("QString", self.variable_qtgui_label_0))

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_tx_lr(self):
        return self.tx_lr

    def set_tx_lr(self, tx_lr):
        self.tx_lr = tx_lr

    def get_tx_explo(self):
        return self.tx_explo

    def set_tx_explo(self, tx_explo):
        self.tx_explo = tx_explo

    def get_training_mod(self):
        return self.training_mod

    def set_training_mod(self, training_mod):
        self.training_mod = training_mod
        Qt.QMetaObject.invokeMethod(self._training_mod_line_edit, "setText", Qt.Q_ARG("QString", str(self.training_mod)))
        self.learning_dl_eq_demod_0.set_alternate(self.training_mod)

    def get_t_state(self):
        return self.t_state

    def set_t_state(self, t_state):
        self.t_state = t_state
        self._t_state_callback(self.t_state)
        self.learning_dl_eq_demod_0.set_training_state(self.t_state)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_stop_button(self):
        return self.stop_button

    def set_stop_button(self, stop_button):
        self.stop_button = stop_button

    def get_start_button(self):
        return self.start_button

    def set_start_button(self, start_button):
        self.start_button = start_button

    def get_snr_stop(self):
        return self.snr_stop

    def set_snr_stop(self, snr_stop):
        self.snr_stop = snr_stop

    def get_snr_step(self):
        return self.snr_step

    def set_snr_step(self, snr_step):
        self.snr_step = snr_step
        self.qtgui_vector_sink_f_0_0.set_x_axis(self.snr_start, self.snr_step)

    def get_snr_start(self):
        return self.snr_start

    def set_snr_start(self, snr_start):
        self.snr_start = snr_start
        self.qtgui_vector_sink_f_0_0.set_x_axis(self.snr_start, self.snr_step)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)

    def get_rx_lr(self):
        return self.rx_lr

    def set_rx_lr(self, rx_lr):
        self.rx_lr = rx_lr
        self.learning_dl_eq_demod_0.set_lr(10**self.rx_lr )

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff

    def get_reset_rx(self):
        return self.reset_rx

    def set_reset_rx(self, reset_rx):
        self.reset_rx = reset_rx

    def get_reset_ber(self):
        return self.reset_ber

    def set_reset_ber(self, reset_ber):
        self.reset_ber = reset_ber

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

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_bits_per_symbol_0(self):
        return self.bits_per_symbol_0

    def set_bits_per_symbol_0(self, bits_per_symbol_0):
        self.bits_per_symbol_0 = bits_per_symbol_0



def main(top_block_cls=rldl_mod_demod, options=None):

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
