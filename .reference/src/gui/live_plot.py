"""
PyQt5 live plotting demo with dummy data
- 4 plots: Voltage vs Time, Flow Rate vs Time, Mass Rate vs Time, Sound (waveform) vs Time
- Simulates streaming data using QTimer and numpy
- Features: Start / Stop / Reset / Save snapshot, interactive matplotlib toolbar
- Enable Prediction button with tissue-name bar and selector (hard/medium/soft tissue)
"""

import sys
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QComboBox,
    QFrame,
)
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class LivePlotsWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Data Display - PyQt5 + Matplotlib")
        self.resize(1100, 700)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout(self.main_widget)

        # Create matplotlib figure and 4 subplots (2x2)
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax_voltage = self.fig.add_subplot(221)
        self.ax_flow = self.fig.add_subplot(222)
        self.ax_mass = self.fig.add_subplot(223)
        self.ax_sound = self.fig.add_subplot(224)

        # Layout: toolbar + canvas
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset")
        self.btn_snapshot = QPushButton("Save Snapshot")
        self.btn_enable_pred = QPushButton("Enable Prediction")

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_enable_pred)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_snapshot)

        self.layout.addLayout(btn_layout)

        # Tissue bar widget (hidden until enabled)
        self.tissue_bar = QWidget()
        tissue_bar_layout = QHBoxLayout(self.tissue_bar)
        self.tissue_label = QLabel("Tissue:")
        self.tissue_name = QLabel("None")
        self.tissue_frame = QFrame()
        self.tissue_frame.setFrameShape(QFrame.StyledPanel)
        self.tissue_frame.setFixedHeight(30)
        tissue_bar_layout.addWidget(self.tissue_label)
        tissue_bar_layout.addWidget(self.tissue_name)
        tissue_bar_layout.addStretch()
        self.tissue_bar.setVisible(False)
        self.layout.addWidget(self.tissue_bar)

        # Tissue selection dropdown
        self.tissue_selector = QComboBox(self)
        self.tissue_selector.addItems(["Hard Tissue", "Medium Tissue", "Soft Tissue"])
        self.tissue_selector.setVisible(False)
        self.layout.addWidget(self.tissue_selector)

        # Connect prediction button and selector
        self.btn_enable_pred.clicked.connect(self.toggle_prediction)
        self.tissue_selector.currentTextChanged.connect(self._update_tissue_bar)

        # Connect control buttons
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_snapshot.clicked.connect(self.save_snapshot)

        # Data buffers (deque for efficient pops)
        self.max_points = 1000  # buffer length
        self.t = deque(maxlen=self.max_points)
        self.voltage = deque(maxlen=self.max_points)
        self.flow = deque(maxlen=self.max_points)
        self.mass = deque(maxlen=self.max_points)
        self.sound = deque(maxlen=self.max_points)

        # Time tracking
        self.start_time = 0.0
        self.current_time = 0.0
        self.dt = 0.01  # 10 ms update for smoother sound waveform

        # Initialize plots
        self._init_plots()

        # Timer for updates
        self.timer = QTimer(self)
        self.timer.setInterval(int(self.dt * 1000))
        self.timer.timeout.connect(self._update_data)

        self.running = False

    def _init_plots(self):
        self.ax_voltage.set_title("Voltage vs Time")
        self.ax_voltage.set_xlabel("Time (s)")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.line_voltage, = self.ax_voltage.plot([], [], color="blue", label="Voltage")
        self.ax_voltage.grid(True)
        self.ax_voltage.legend(loc="upper right")

        self.ax_flow.set_title("Flow Rate vs Time")
        self.ax_flow.set_xlabel("Time (s)")
        self.ax_flow.set_ylabel("Flow Rate (L/min)")
        self.line_flow, = self.ax_flow.plot([], [], color="green", label="Flow Rate")
        self.ax_flow.grid(True)
        self.ax_flow.legend(loc="upper right")

        self.ax_mass.set_title("Mass Rate vs Time")
        self.ax_mass.set_xlabel("Time (s)")
        self.ax_mass.set_ylabel("Mass Rate (kg/s)")
        self.line_mass, = self.ax_mass.plot([], [], color="red", label="Mass Rate")
        self.ax_mass.grid(True)
        self.ax_mass.legend(loc="upper right")

        self.ax_sound.set_title("Sound Waveform vs Time")
        self.ax_sound.set_xlabel("Time (s)")
        self.ax_sound.set_ylabel("Amplitude")
        self.line_sound, = self.ax_sound.plot([], [], color="purple", label="Sound")
        self.ax_sound.grid(True)
        self.ax_sound.legend(loc="upper right")

        self.fig.tight_layout()

    def toggle_prediction(self):
        if not self.tissue_bar.isVisible():
            self.tissue_bar.setVisible(True)
            self.tissue_selector.setVisible(True)
            self.tissue_name.setText(self.tissue_selector.currentText())
            self.btn_enable_pred.setText("Disable Prediction")
        else:
            self.tissue_bar.setVisible(False)
            self.tissue_selector.setVisible(False)
            self.btn_enable_pred.setText("Enable Prediction")

    def _update_tissue_bar(self, text):
        self.tissue_name.setText(text)

    def start(self):
        if not self.running:
            if len(self.t) == 0:
                self.start_time = 0.0
                self.current_time = 0.0
            self.timer.start()
            self.running = True

    def stop(self):
        if self.running:
            self.timer.stop()
            self.running = False

    def reset(self):
        self.timer.stop()
        self.running = False
        for buf in (self.t, self.voltage, self.flow, self.mass, self.sound):
            buf.clear()
        self.current_time = 0.0
        self.start_time = 0.0
        self._redraw_full()

    def save_snapshot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.fig.savefig(file_path)

    def _update_data(self):
        t = self.current_time
        voltage = np.sin(2 * np.pi * 1.0 * t)
        flow = 5.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t + 0.4) + 0.05 * np.random.randn()
        mass = 0.8 + 0.02 * t + 0.05 * np.random.randn()
        sound = 0.6 * np.sin(2 * np.pi * 20 * t) + 0.3 * np.sin(2 * np.pi * 40 * t + 0.2) + 0.05 * np.random.randn()

        self.t.append(t)
        self.voltage.append(voltage)
        self.flow.append(flow)
        self.mass.append(mass)
        self.sound.append(sound)

        self.current_time += self.dt

        x = np.array(self.t)
        if x.size == 0:
            return

        self.line_voltage.set_data(x, np.array(self.voltage))
        self.ax_voltage.relim()
        self.ax_voltage.autoscale_view()

        self.line_flow.set_data(x, np.array(self.flow))
        self.ax_flow.relim()
        self.ax_flow.autoscale_view()

        self.line_mass.set_data(x, np.array(self.mass))
        self.ax_mass.relim()
        self.ax_mass.autoscale_view()

        self.line_sound.set_data(x, np.array(self.sound))
        self.ax_sound.relim()
        self.ax_sound.autoscale_view()

        self.canvas.draw_idle()

    def _redraw_full(self):
        empty_x = []
        self.line_voltage.set_data(empty_x, empty_x)
        self.line_flow.set_data(empty_x, empty_x)
        self.line_mass.set_data(empty_x, empty_x)
        self.line_sound.set_data(empty_x, empty_x)

        for ax in (self.ax_voltage, self.ax_flow, self.ax_mass, self.ax_sound):
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = LivePlotsWindow()
    w.show()
    sys.exit(app.exec_())
