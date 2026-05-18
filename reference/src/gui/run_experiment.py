"""
Unified Experiment Control GUI - Modular Design
- Reuses code from individual sensor scripts
- RealSense camera (color + depth)
- Blackfly cameras 1 & 2
- Microphone (2 channels) with spectrogram
- Individual start/stop controls for each sensor
- Synchronized recording with timestamps
"""

import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from collections import deque
import json

# Fix Qt platform plugin conflict with OpenCV
os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '')

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QCheckBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal

# Import sensor modules
try:
    import pyrealsense2  # noqa: F401
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available")

try:
    import PySpin
    PYSPIN_AVAILABLE = True
except ImportError:
    PYSPIN_AVAILABLE = False
    print("Warning: PySpin not available (Blackfly cameras)")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available")

# Import AudioRecorder from microphone.py
# Both files are in the same directory, so use direct import
gui_dir = Path(__file__).parent
if str(gui_dir) not in sys.path:
    sys.path.insert(0, str(gui_dir))

repo_root = gui_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from microphone import AudioRecorder
from realsense_boot import (
    build_device_records,
    format_device_label,
)
from realsense_capture.stream import (
    DEFAULT_CLIP_DISTANCE_M,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WIDTH,
    RealSenseStream,
    build_display,
    log_status,
)


class RealSenseThread(QThread):
    """Thread for RealSense camera streaming backed by shared capture code."""
    frame_ready = pyqtSignal(object, object, float, int)  # color, depth, timestamp, frame_id
    error_occurred = pyqtSignal(str)
    
    def __init__(self, serial_request):
        super().__init__()
        self.serial_request = serial_request
        self.running = False
        self.stream = None
        self.frame_id = 0
        self.mutex = QMutex()
        
    def run(self):
        if not REALSENSE_AVAILABLE:
            return
            
        try:
            self.stream = RealSenseStream(
                camera_name="run-experiment",
                serial_request=str(self.serial_request or ""),
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                fps=DEFAULT_FPS,
                clip_distance_max=DEFAULT_CLIP_DISTANCE_M,
            )
            self.stream.start()
            self.stream.warm_up(DEFAULT_WARMUP_FRAMES)
            self.running = True
            self.frame_id = 0
            
            while self.running:
                bundle = self.stream.poll_frame()
                if bundle is None:
                    continue
                timestamp = time.time()
                self.mutex.lock()
                current_frame_id = self.frame_id
                self.frame_id += 1
                self.mutex.unlock()
                self.frame_ready.emit(
                    bundle.color_image,
                    bundle.depth_image,
                    timestamp,
                    current_frame_id,
                )
        except Exception as e:
            message = str(e)
            log_status(f"run_experiment: {message}")
            self.error_occurred.emit(message)
        finally:
            if self.stream:
                self.stream.stop()
                self.stream = None
    
    def stop(self):
        self.running = False
        self.wait(1000)


class BlackflyThread(QThread):
    """Thread for Blackfly camera streaming - based on camera_dual.py"""
    frame_ready = pyqtSignal(object, float, int, int)  # image, timestamp, frame_id, camera_id
    
    def __init__(self, camera_id, camera_list, system):
        super().__init__()
        self.camera_id = camera_id
        self.camera_list = camera_list
        self.system = system
        self.camera = None
        self.running = False
        self.frame_id = 0
        self.mutex = QMutex()
        self.pixel_format_is_rgb = False
        
    def run(self):
        if not PYSPIN_AVAILABLE:
            return
            
        try:
            # Initialize camera in thread
            if self.camera_id < self.camera_list.GetSize():
                self.camera = self.camera_list[self.camera_id]
                if not self.camera.IsInitialized():
                    self.camera.Init()
                
                # Setup (from camera_dual.py)
                nodemap = self.camera.GetNodeMap()
                
                # Acquisition mode
                node_acq = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
                node_acq_cont = node_acq.GetEntryByName('Continuous')
                node_acq.SetIntValue(node_acq_cont.GetValue())
                
                # Pixel format (from camera_dual.py)
                node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
                if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
                    node_pixel_format_bgr8 = node_pixel_format.GetEntryByName('BGR8')
                    if PySpin.IsAvailable(node_pixel_format_bgr8) and PySpin.IsReadable(node_pixel_format_bgr8):
                        pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
                        node_pixel_format.SetIntValue(pixel_format_bgr8)
                        self.pixel_format_is_rgb = False
                    else:
                        node_pixel_format_rgb8 = node_pixel_format.GetEntryByName('RGB8')
                        if PySpin.IsAvailable(node_pixel_format_rgb8) and PySpin.IsReadable(node_pixel_format_rgb8):
                            pixel_format_rgb8 = node_pixel_format_rgb8.GetValue()
                            node_pixel_format.SetIntValue(pixel_format_rgb8)
                            self.pixel_format_is_rgb = True
                
                self.camera.BeginAcquisition()
                self.running = True
                self.frame_id = 0
                
                while self.running:
                    image_result = self.camera.GetNextImage(1000)
                    if image_result.IsIncomplete():
                        image_result.Release()
                        continue
                    
                    img_array = image_result.GetNDArray()
                    timestamp = time.time()
                    
                    self.mutex.lock()
                    current_frame_id = self.frame_id
                    self.frame_id += 1
                    self.mutex.unlock()
                    
                    self.frame_ready.emit(img_array, timestamp, current_frame_id, self.camera_id)
                    image_result.Release()
            else:
                print(f"Camera {self.camera_id} not available")
                
        except Exception as e:
            print(f"Blackfly {self.camera_id} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.camera:
                try:
                    if self.camera.IsStreaming():
                        self.camera.EndAcquisition()
                except:
                    pass
    
    def stop(self):
        self.running = False
        self.wait(1000)


class SensorTab(QWidget):
    """Base class for sensor tabs"""
    def __init__(self, sensor_name):
        super().__init__()
        self.sensor_name = sensor_name
        self.recording = False
        self.recorded_data = []
        self.layout = QVBoxLayout(self)
        
        # Control buttons for this sensor
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton(f"Start {sensor_name}")
        self.btn_stop = QPushButton(f"Stop {sensor_name}")
        self.btn_start_recording = QPushButton(f"Start Recording")
        self.btn_stop_recording = QPushButton(f"Stop Recording")
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_start_recording)
        control_layout.addWidget(self.btn_stop_recording)
        control_layout.addStretch()
        
        self.layout.addLayout(control_layout)
        
        # Status
        self.status_label = QLabel("Status: Stopped")
        self.layout.addWidget(self.status_label)
        
    def start_recording(self):
        self.recording = True
        self.recorded_data = []
        
    def stop_recording(self):
        self.recording = False
        
    def save_data(self, save_dir):
        """Override in subclasses"""
        pass


class RealSenseTab(SensorTab):
    """RealSense camera tab"""
    def __init__(self):
        super().__init__("RealSense")
        self.thread = None
        self.current_color = None
        self.current_depth = None
        self.serial_input = QLineEdit()
        self.device_hint_label = QLabel()

        selector_layout = QHBoxLayout()
        selector_label = QLabel("Serial / last 4 digits:")
        self.serial_input.setPlaceholderText("Leave blank if only one RealSense is connected")
        selector_layout.addWidget(selector_label)
        selector_layout.addWidget(self.serial_input)
        self.layout.addLayout(selector_layout)
        self.layout.addWidget(self.device_hint_label)
        self.refresh_device_hints()
        
        # Display
        self.image_label = QLabel("No stream")
        self.image_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.image_label)
        
        # Connect buttons
        self.btn_start.clicked.connect(self.start_streaming)
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)

    def refresh_device_hints(self):
        if not REALSENSE_AVAILABLE:
            self.device_hint_label.setText("Detected devices: pyrealsense2 unavailable")
            return

        try:
            device_records = build_device_records()
        except Exception as exc:
            self.device_hint_label.setText(f"Detected devices: {exc}")
            return

        if not device_records:
            self.device_hint_label.setText("Detected devices: none")
            return

        labels = [format_device_label(record["info"]) for record in device_records]
        self.device_hint_label.setText("Detected devices: " + " || ".join(labels))
        
    def start_streaming(self):
        if not REALSENSE_AVAILABLE:
            QMessageBox.warning(self, "Error", "RealSense not available")
            return
            
        if self.thread and self.thread.isRunning():
            return

        serial_request = self.serial_input.text().strip()
        if not serial_request:
            serial_request = None

        self.thread = RealSenseThread(serial_request)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.error_occurred.connect(self._on_stream_error)
        self.thread.start()
        self.status_label.setText("Status: Streaming...")
        
    def stop_streaming(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.status_label.setText("Status: Stopped")
        
    def _on_frame(self, color, depth, timestamp, frame_id):
        self.current_color = color
        self.current_depth = depth
        
        combined = build_display(color, depth)
        
        # Resize for display
        h, w = combined.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_w, new_h = int(w * scale), int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        
        # Convert to QPixmap
        rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
        # Record if enabled
        if self.recording:
            self.recorded_data.append({
                'color': color.copy(),
                'depth': depth.copy(),
                'timestamp': timestamp,
                'frame_id': frame_id
            })

    def _on_stream_error(self, message):
        self.stop_streaming()
        self.status_label.setText(f"Status: Error ({message})")
        QMessageBox.warning(self, "RealSense Error", message)
            
    def start_recording(self):
        super().start_recording()
        self.status_label.setText("Status: Recording...")
        
    def stop_recording(self):
        super().stop_recording()
        self.status_label.setText(f"Status: Streaming... ({len(self.recorded_data)} frames)")
        
    def save_data(self, save_dir):
        if not self.recorded_data:
            return 0
            
        realsense_dir = Path(save_dir) / "realsense"
        realsense_dir.mkdir(exist_ok=True)
        
        timestamps = []
        for i, frame_data in enumerate(self.recorded_data):
            timestamp = frame_data['timestamp']
            timestamps.append(timestamp)
            
            color_file = realsense_dir / f"color_{i:06d}_{timestamp:.6f}.png"
            depth_file = realsense_dir / f"depth_{i:06d}_{timestamp:.6f}.png"
            
            cv2.imwrite(str(color_file), frame_data['color'])
            cv2.imwrite(str(depth_file), frame_data['depth'])
        
        metadata = {
            'num_frames': len(self.recorded_data),
            'timestamps': timestamps,
            'frame_ids': [f['frame_id'] for f in self.recorded_data]
        }
        with open(realsense_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(self.recorded_data)


class BlackflyTab(SensorTab):
    """Blackfly camera tab"""
    def __init__(self, camera_id, camera_list, system):
        super().__init__(f"Blackfly {camera_id}")
        self.camera_id = camera_id
        self.camera_list = camera_list
        self.system = system
        self.thread = None
        self.current_image = None
        
        # Display
        self.image_label = QLabel("No stream")
        self.image_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.image_label)
        
        # Connect buttons
        self.btn_start.clicked.connect(self.start_streaming)
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        
    def start_streaming(self):
        if not PYSPIN_AVAILABLE:
            QMessageBox.warning(self, "Error", "PySpin not available")
            return
            
        if self.thread and self.thread.isRunning():
            return
            
        self.thread = BlackflyThread(self.camera_id, self.camera_list, self.system)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.start()
        self.status_label.setText("Status: Streaming...")
        
    def stop_streaming(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.status_label.setText("Status: Stopped")
        
    def _on_frame(self, image, timestamp, frame_id, cam_id):
        self.current_image = image
        
        # Convert format (from camera_dual.py)
        display_img = image.copy()
        if len(display_img.shape) == 3 and display_img.shape[2] == 3:
            # Check if RGB format (from thread)
            if hasattr(self.thread, 'pixel_format_is_rgb') and self.thread.pixel_format_is_rgb:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        h, w = display_img.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        # Convert to QPixmap
        if len(display_img.shape) == 2:
            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
        # Record if enabled
        if self.recording:
            self.recorded_data.append({
                'image': image.copy(),
                'timestamp': timestamp,
                'frame_id': frame_id
            })
            
    def start_recording(self):
        super().start_recording()
        self.status_label.setText("Status: Recording...")
        
    def stop_recording(self):
        super().stop_recording()
        self.status_label.setText(f"Status: Streaming... ({len(self.recorded_data)} frames)")
        
    def save_data(self, save_dir):
        if not self.recorded_data:
            return 0
            
        blackfly_dir = Path(save_dir) / f"blackfly_{self.camera_id}"
        blackfly_dir.mkdir(exist_ok=True)
        
        timestamps = []
        for i, frame_data in enumerate(self.recorded_data):
            timestamp = frame_data['timestamp']
            timestamps.append(timestamp)
            
            img_file = blackfly_dir / f"frame_{i:06d}_{timestamp:.6f}.png"
            cv2.imwrite(str(img_file), frame_data['image'])
        
        metadata = {
            'num_frames': len(self.recorded_data),
            'timestamps': timestamps,
            'frame_ids': [f['frame_id'] for f in self.recorded_data],
            'camera_id': self.camera_id
        }
        with open(blackfly_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(self.recorded_data)


class MicrophoneTab(SensorTab):
    """Microphone tab with spectrogram - based on microphone.py"""
    def __init__(self):
        super().__init__("Microphone")
        self.thread = None
        self.sample_rate = 44100
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self._populate_devices()
        device_layout.addWidget(self.device_combo)
        
        # Mic selection
        device_layout.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Both Mics", "Mic 1 Only", "Mic 2 Only"])
        device_layout.addWidget(self.display_combo)
        device_layout.addStretch()
        self.layout.addLayout(device_layout)
        
        # Time series and spectrogram plots
        plot_layout = QHBoxLayout()
        
        # Time series
        self.fig_time = Figure(figsize=(6, 3))
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_time.set_title("Audio Waveform")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.line1, = self.ax_time.plot([], [], 'b-', label='Mic 1', linewidth=0.5)
        self.line2, = self.ax_time.plot([], [], 'r-', label='Mic 2', linewidth=0.5)
        self.ax_time.legend()
        self.ax_time.grid(True, alpha=0.3)
        self.ax_time.set_ylim(-1, 1)
        plot_layout.addWidget(self.canvas_time)
        
        # Spectrogram
        self.fig_spec = Figure(figsize=(6, 3))
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.ax_spec.set_title("Spectrogram (Mic 1)")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.spec_im = None
        plot_layout.addWidget(self.canvas_spec)
        
        self.layout.addLayout(plot_layout)
        
        # Data buffers
        self.time_buffer = deque(maxlen=5000)
        self.data1_buffer = deque(maxlen=5000)
        self.data2_buffer = deque(maxlen=5000)
        
        # Spectrogram buffers
        self.spec_data = deque(maxlen=200)
        self.spec_time = deque(maxlen=200)
        self.spec_window_size = 2048
        self.spec_overlap = 1024
        
        # Connect buttons
        self.btn_start.clicked.connect(self.start_streaming)
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        self.display_combo.currentTextChanged.connect(self._update_display)
        
    def _populate_devices(self):
        if not SOUNDDEVICE_AVAILABLE:
            return
        try:
            devices = sd.query_devices()
            default_device = sd.default.device[0] if sd.default.device[0] is not None else None
            default_idx = 0
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= 1:
                    ch = device['max_input_channels']
                    ch_label = "1 ch (USB mic)" if ch == 1 else f"{ch} ch"
                    device_name = f"{i}: {device['name']} ({ch_label})"
                    idx = self.device_combo.count()
                    self.device_combo.addItem(device_name, i)
                    if i == default_device:
                        default_idx = idx
            
            if self.device_combo.count() > 0:
                self.device_combo.setCurrentIndex(default_idx)
        except Exception as e:
            print(f"Error querying devices: {e}")
    
    def _update_display(self, display_text):
        if display_text == "Mic 1 Only":
            self.line2.set_visible(False)
            self.ax_spec.set_title("Spectrogram (Mic 1)")
        elif display_text == "Mic 2 Only":
            self.line1.set_visible(False)
            self.ax_spec.set_title("Spectrogram (Mic 2)")
        else:
            self.line1.set_visible(True)
            self.line2.set_visible(True)
            self.ax_spec.set_title("Spectrogram (Mic 1)")
        self.canvas_time.draw()
        self.canvas_spec.draw()
        
    def start_streaming(self):
        if not SOUNDDEVICE_AVAILABLE:
            QMessageBox.warning(self, "Error", "sounddevice not available")
            return
            
        if self.thread and self.thread.isRunning():
            return
        
        device_idx = self.device_combo.currentData()
        channels = 2
        if device_idx is not None:
            device_info = sd.query_devices(device_idx)
            self.sample_rate = int(device_info.get('default_samplerate', 44100))
            channels = min(2, device_info['max_input_channels'])
        
        self.thread = AudioRecorder(
            sample_rate=self.sample_rate,
            channels=channels,
            chunk_size=8192,
            device=device_idx
        )
        self.thread.data_ready.connect(self._on_data)
        self.thread.start()
        self.status_label.setText(f"Status: Streaming... ({self.sample_rate} Hz)")
        
    def stop_streaming(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.status_label.setText("Status: Stopped")
        
    def _on_data(self, data, sample_rate):
        if data.shape[1] < 1:
            return
        channel1 = data[:, 0]
        channel2 = data[:, 1] if data.shape[1] >= 2 else data[:, 0]
        
        # Time axis
        chunk_duration = len(channel1) / sample_rate
        if len(self.time_buffer) == 0:
            current_time = 0.0
        else:
            current_time = self.time_buffer[-1] + chunk_duration
        
        time_chunk = np.linspace(
            current_time - chunk_duration,
            current_time,
            len(channel1),
            endpoint=False
        )
        
        # Add to buffers
        self.time_buffer.extend(time_chunk)
        self.data1_buffer.extend(channel1)
        self.data2_buffer.extend(channel2)
        
        # Update time series plot
        if len(self.time_buffer) > 0:
            time_array = np.array(self.time_buffer)
            data1 = np.array(self.data1_buffer)
            data2 = np.array(self.data2_buffer)
            
            self.line1.set_data(time_array, data1)
            self.line2.set_data(time_array, data2)
            self.ax_time.relim()
            self.ax_time.autoscale_view()
            self.canvas_time.draw_idle()
        
        # Compute spectrogram
        display_mode = self.display_combo.currentText()
        if display_mode in ["Both Mics", "Mic 1 Only"]:
            spec_channel = channel1
        else:
            spec_channel = channel2
            
        f, t, Sxx = signal.spectrogram(
            spec_channel,
            fs=sample_rate,
            nperseg=self.spec_window_size,
            noverlap=self.spec_overlap,
            window='hann'
        )
        
        # Store spectrogram data
        for i in range(len(t)):
            self.spec_data.append(10 * np.log10(Sxx[:, i] + 1e-10))
            self.spec_time.append(t[i] + (current_time - chunk_duration))
        
        # Update spectrogram
        if len(self.spec_data) > 0:
            spec_array = np.array(list(self.spec_data))
            time_array = np.array(list(self.spec_time))
            
            if spec_array.size > 0:
                f_axis = np.fft.rfftfreq(self.spec_window_size, 1.0 / sample_rate)
                if spec_array.shape[1] < len(f_axis):
                    f_axis = f_axis[:spec_array.shape[1]]
                
                # Limit to 10kHz
                max_freq_idx = np.searchsorted(f_axis, 10000)
                if max_freq_idx > 0:
                    spec_array = spec_array[:, :max_freq_idx]
                    f_axis = f_axis[:max_freq_idx]
                
                if self.spec_im is None:
                    self.ax_spec.clear()
                    self.ax_spec.set_title(f"Spectrogram ({display_mode})")
                    self.ax_spec.set_xlabel("Time (s)")
                    self.ax_spec.set_ylabel("Frequency (Hz)")
                    extent = [time_array[0], time_array[-1], f_axis[0], f_axis[-1]]
                    self.spec_im = self.ax_spec.imshow(
                        spec_array.T,
                        aspect='auto',
                        origin='lower',
                        extent=extent,
                        cmap='viridis',
                        vmin=-80,
                        vmax=0
                    )
                    self.ax_spec.set_ylim(0, 10000)
                    self.fig_spec.colorbar(self.spec_im, ax=self.ax_spec, label='Power (dB)')
                else:
                    self.spec_im.set_array(spec_array.T)
                    self.spec_im.set_extent([time_array[0], time_array[-1], f_axis[0], f_axis[-1]])
                    self.ax_spec.set_xlim(time_array[0], time_array[-1])
                self.canvas_spec.draw_idle()
        
        # Record if enabled
        if self.recording:
            self.recorded_data.append({
                'data': data.copy(),
                'timestamp': time.time(),
                'chunk_id': len(self.recorded_data)
            })
            
    def start_recording(self):
        super().start_recording()
        self.status_label.setText(f"Status: Recording... ({self.sample_rate} Hz)")
        
    def stop_recording(self):
        super().stop_recording()
        self.status_label.setText(f"Status: Streaming... ({len(self.recorded_data)} chunks, {self.sample_rate} Hz)")
        
    def save_data(self, save_dir):
        if not self.recorded_data:
            return 0
            
        mic_dir = Path(save_dir) / "microphone"
        mic_dir.mkdir(exist_ok=True)
        
        # Concatenate audio
        audio1 = []
        audio2 = []
        timestamps = []
        
        for chunk_data in self.recorded_data:
            data = chunk_data['data']
            audio1.extend(data[:, 0])
            audio2.extend(data[:, 1] if data.shape[1] >= 2 else data[:, 0])
            timestamps.append(chunk_data['timestamp'])
        
        # Save as WAV (stereo; 1-ch devices saved as duplicate L/R)
        import scipy.io.wavfile as wavfile
        stereo_audio = np.column_stack([np.array(audio1), np.array(audio2)])
        wav_file = mic_dir / "audio.wav"
        wavfile.write(str(wav_file), self.sample_rate, stereo_audio)
        
        # Save metadata
        metadata = {
            'num_chunks': len(self.recorded_data),
            'timestamps': timestamps,
            'sample_rate': self.sample_rate,
            'duration': len(audio1) / self.sample_rate
        }
        with open(mic_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(self.recorded_data)


class ExperimentWindow(QMainWindow):
    """Main experiment control window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sonopet Experiment Control")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Global control panel
        control_group = QGroupBox("Global Controls")
        control_layout = QHBoxLayout()
        
        self.btn_start_all = QPushButton("Start All")
        self.btn_stop_all = QPushButton("Stop All")
        self.btn_start_recording_all = QPushButton("Start Recording All")
        self.btn_stop_recording_all = QPushButton("Stop Recording All")
        self.btn_save_data = QPushButton("Save All Data")
        
        control_layout.addWidget(self.btn_start_all)
        control_layout.addWidget(self.btn_stop_all)
        control_layout.addWidget(self.btn_start_recording_all)
        control_layout.addWidget(self.btn_stop_recording_all)
        control_layout.addWidget(self.btn_save_data)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Status
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Initialize sensor tabs
        self.realsense_tab = None
        self.blackfly_tabs = []
        self.microphone_tab = None
        
        self._init_sensors()
        
        # Connect global buttons
        self.btn_start_all.clicked.connect(self.start_all_streams)
        self.btn_stop_all.clicked.connect(self.stop_all_streams)
        self.btn_start_recording_all.clicked.connect(self.start_recording_all)
        self.btn_stop_recording_all.clicked.connect(self.stop_recording_all)
        self.btn_save_data.clicked.connect(self.save_data)
        
        # Recording state
        self.save_directory = None
        
    def _init_sensors(self):
        """Initialize all available sensors"""
        # RealSense
        if REALSENSE_AVAILABLE:
            self.realsense_tab = RealSenseTab()
            self.tabs.addTab(self.realsense_tab, "RealSense")
        
        # Blackfly cameras
        if PYSPIN_AVAILABLE:
            try:
                system = PySpin.System.GetInstance()
                cam_list = system.GetCameras()
                
                self.pyspin_system = system
                self.pyspin_cam_list = cam_list
                
                if cam_list.GetSize() >= 1:
                    tab0 = BlackflyTab(0, cam_list, system)
                    self.blackfly_tabs.append(tab0)
                    self.tabs.addTab(tab0, "Blackfly 1")
                
                if cam_list.GetSize() >= 2:
                    tab1 = BlackflyTab(1, cam_list, system)
                    self.blackfly_tabs.append(tab1)
                    self.tabs.addTab(tab1, "Blackfly 2")
                    
            except Exception as e:
                print(f"Error initializing Blackfly cameras: {e}")
                import traceback
                traceback.print_exc()
        
        # Microphone
        if SOUNDDEVICE_AVAILABLE:
            self.microphone_tab = MicrophoneTab()
            self.tabs.addTab(self.microphone_tab, "Microphone")
            
    def start_all_streams(self):
        """Start all sensor streams"""
        if self.realsense_tab:
            self.realsense_tab.start_streaming()
        for tab in self.blackfly_tabs:
            tab.start_streaming()
        if self.microphone_tab:
            self.microphone_tab.start_streaming()
        self.status_label.setText("Status: All streams started")
        
    def stop_all_streams(self):
        """Stop all sensor streams"""
        if self.realsense_tab:
            self.realsense_tab.stop_streaming()
        for tab in self.blackfly_tabs:
            tab.stop_streaming()
        if self.microphone_tab:
            self.microphone_tab.stop_streaming()
        self.status_label.setText("Status: All streams stopped")
        
    def start_recording_all(self):
        """Start recording on all sensors"""
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_directory = Path(save_dir) / f"experiment_{timestamp}"
        self.save_directory.mkdir(exist_ok=True)
        
        # Save experiment metadata
        metadata = {
            'start_time': time.time(),
            'timestamp': timestamp,
            'sensors': []
        }
        
        # Start recording on all tabs
        if self.realsense_tab:
            self.realsense_tab.start_recording()
            metadata['sensors'].append('realsense')
        for tab in self.blackfly_tabs:
            tab.start_recording()
            metadata['sensors'].append(f'blackfly_{tab.camera_id}')
        if self.microphone_tab:
            self.microphone_tab.start_recording()
            metadata['sensors'].append('microphone')
        
        with open(self.save_directory / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.status_label.setText(f"Status: Recording all to {self.save_directory.name}")
        
    def stop_recording_all(self):
        """Stop recording on all sensors"""
        if self.realsense_tab:
            self.realsense_tab.stop_recording()
        for tab in self.blackfly_tabs:
            tab.stop_recording()
        if self.microphone_tab:
            self.microphone_tab.stop_recording()
        self.status_label.setText("Status: Recording stopped")
        
    def save_data(self):
        """Save all recorded data"""
        if not self.save_directory:
            QMessageBox.warning(self, "Error", "No recording session. Please start recording first.")
            return
            
        # Update metadata
        metadata_file = self.save_directory / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            metadata['end_time'] = time.time()
            metadata['duration'] = metadata['end_time'] - metadata['start_time']
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save data from each sensor
        results = {}
        if self.realsense_tab:
            count = self.realsense_tab.save_data(self.save_directory)
            if count:
                results['RealSense'] = f"{count} frames"
        for tab in self.blackfly_tabs:
            count = tab.save_data(self.save_directory)
            if count:
                results[f'Blackfly {tab.camera_id}'] = f"{count} frames"
        if self.microphone_tab:
            count = self.microphone_tab.save_data(self.save_directory)
            if count:
                results['Microphone'] = f"{count} chunks"
        
        # Show results
        result_text = "\n".join([f"{k}: {v}" for k, v in results.items()])
        QMessageBox.information(
            self,
            "Data Saved",
            f"Data saved to:\n{self.save_directory}\n\n{result_text}"
        )
        
    def closeEvent(self, event):
        """Cleanup on close"""
        self.stop_all_streams()
        self.stop_recording_all()
        event.accept()


if __name__ == "__main__":
    # Fix Qt plugin path
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        plugin_paths = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '').split(':')
        plugin_paths = [p for p in plugin_paths if 'cv2' not in p]
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ':'.join(plugin_paths)
    
    app = QApplication(sys.argv)
    window = ExperimentWindow()
    window.show()
    sys.exit(app.exec_())
