"""
PyQt5 Audio Recording GUI for Focusrite 2i2
- Records from 2 microphones (2 channels)
- Real-time time series plots for both channels
- Real-time spectrograms for both channels
- Start/Stop recording controls
- Save recorded audio to file
"""

import sys
import numpy as np
import sounddevice as sd
from collections import deque
from scipy import signal
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
    QMessageBox,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class AudioRecorder(QThread):
    """Thread for continuous audio recording"""
    data_ready = pyqtSignal(np.ndarray, float)
    
    def __init__(self, sample_rate=44100, channels=2, chunk_size=2048, device=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.running = False
        self.stream = None
        
    def run(self):
        """Start recording in a separate thread"""
        try:
            # Use larger latency and buffer to prevent overflow
            # Calculate latency in seconds (higher for stability - at least 0.2 seconds)
            latency_seconds = max(0.2, self.chunk_size / self.sample_rate * 4)  # At least 4 chunks worth
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                dtype=np.float32,
                device=self.device,
                latency=latency_seconds,  # Explicit latency in seconds
                extra_settings=None
            )
            self.stream.start()
            self.running = True
            self._overflow_count = 0
            
            while self.running:
                try:
                    data, overflowed = self.stream.read(self.chunk_size)
                    if overflowed:
                        # Only print occasionally to avoid spam
                        self._overflow_count += 1
                        if self._overflow_count % 20 == 0:  # Print every 20th overflow
                            print(f"Warning: Audio buffer overflow ({self._overflow_count} total) - latency: {latency_seconds:.3f}s")
                    if data.size > 0:
                        # Convert to numpy array and emit
                        self.data_ready.emit(data, self.sample_rate)
                except Exception as e:
                    if self.running:  # Only print if we're supposed to be running
                        print(f"Error reading audio data: {e}")
                    break
                    
        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.running = False
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    def stop(self):
        """Stop recording"""
        self.running = False


class MicrophoneWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focusrite 2i2 Audio Recorder - Dual Microphone")
        self.resize(1400, 900)
        
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        self.layout = QVBoxLayout(self.main_widget)
        
        # Audio settings - will be set based on device capabilities
        self.sample_rate = 44100  # Default, will be updated
        self.channels = 2
        self.chunk_size = 8192  # Larger chunk to reduce buffer overflow (increased from 4096)
        
        # Initialize time window before device selection (needed by _on_device_changed)
        self.time_window_seconds = 5.0  # Display window in seconds
        self.recording = False  # Initialize before device selection
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Audio Device:"))
        self.device_combo = QComboBox()
        
        # Sample rate display (create before populating devices to avoid AttributeError)
        device_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_label = QLabel("44100 Hz")
        device_layout.addWidget(self.sample_rate_label)
        
        # Now populate devices and connect signal
        self._populate_devices()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addWidget(self.device_combo)
        
        # Microphone display selection
        device_layout.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Both Mics", "Mic 1 Only", "Mic 2 Only"])
        self.display_combo.currentTextChanged.connect(self._update_display_layout)
        device_layout.addWidget(self.display_combo)
        device_layout.addStretch()
        self.layout.addLayout(device_layout)
        
        # Track display mode
        self.display_mode = "both"  # "both", "mic1", "mic2"
        
        # Create matplotlib figure with 4 subplots (2x2)
        # Top row: Time series for mic 1 and mic 2
        # Bottom row: Spectrograms for mic 1 and mic 2
        self.fig = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Subplots: 2 rows, 2 columns
        self.ax_time1 = self.fig.add_subplot(221)
        self.ax_time2 = self.fig.add_subplot(222)
        self.ax_spec1 = self.fig.add_subplot(223)
        self.ax_spec2 = self.fig.add_subplot(224)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop Recording")
        self.btn_reset = QPushButton("Reset")
        self.btn_save = QPushButton("Save Audio")
        self.btn_snapshot = QPushButton("Save Snapshot")
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_snapshot)
        
        self.layout.addLayout(btn_layout)
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        self.layout.addWidget(self.status_label)
        
        # Connect buttons
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_save.clicked.connect(self.save_audio)
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        
        # Data buffers - synchronized time window
        self.time_window_seconds = 5.0  # Display window in seconds
        # Will be updated when sample rate is determined
        self.max_points = int(44100 * self.time_window_seconds)  # Initial buffer length
        self.time_buffer1 = deque(maxlen=self.max_points)
        self.time_buffer2 = deque(maxlen=self.max_points)
        self.time_axis = deque(maxlen=self.max_points)
        
        # Full recording buffer (for saving)
        self.recording_buffer1 = []
        self.recording_buffer2 = []
        self.recording_time = []
        
        # Spectrogram parameters - synchronized with time series window
        # Will be updated when sample rate changes
        self.spec_window_size = 2048  # Larger window for high sample rates
        self.spec_overlap = 1024
        self._update_spectrogram_params()
        
        # Spectrogram data buffers
        self.spec_data1 = deque(maxlen=self.spec_history_length)
        self.spec_data2 = deque(maxlen=self.spec_history_length)
        self.spec_time1 = deque(maxlen=self.spec_history_length)
        self.spec_time2 = deque(maxlen=self.spec_history_length)
        
        # Initialize plots
        self._init_plots()
        
        # Audio recorder thread
        self.recorder = None
        
        # Timer for plot updates - less frequent to reduce CPU load
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # Update every 100ms (reduced from 50ms)
        self.update_timer.timeout.connect(self._update_plots)
        
        # recording already initialized above (before device selection)
        self.start_time = 0.0
        
    def _populate_devices(self):
        """Populate audio device dropdown"""
        try:
            devices = sd.query_devices()
            self.device_combo.clear()
            default_device = sd.default.device[0] if sd.default.device[0] is not None else None
            default_idx = 0
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= 2:  # Need at least 2 input channels
                    max_sr = device.get('default_samplerate', device.get('default_samplerate', 44100))
                    device_name = f"{i}: {device['name']} ({device['max_input_channels']} ch, max {int(max_sr)} Hz)"
                    idx = self.device_combo.count()
                    self.device_combo.addItem(device_name, i)
                    if i == default_device:
                        default_idx = idx
            
            if self.device_combo.count() > 0:
                self.device_combo.setCurrentIndex(default_idx)
                # Update sample rate for selected device
                self._on_device_changed()
        except Exception as e:
            print(f"Error querying devices: {e}")
            QMessageBox.warning(self, "Device Error", f"Could not query audio devices: {e}")
    
    def _on_device_changed(self):
        """Update sample rate when device selection changes"""
        try:
            device_idx = self.device_combo.currentData()
            if device_idx is not None:
                device_info = sd.query_devices(device_idx)
                # Get maximum supported sample rate
                # Try common high sample rates for Focusrite 2i2 (up to 192 kHz)
                test_rates = [192000, 176400, 96000, 88200, 48000, 44100]
                max_rate = 44100
                
                for rate in test_rates:
                    try:
                        # Test if this sample rate is supported
                        test_stream = sd.InputStream(
                            samplerate=rate,
                            channels=2,
                            device=device_idx,
                            dtype=np.float32
                        )
                        test_stream.close()
                        max_rate = rate
                        break  # Found highest supported rate
                    except:
                        continue
                
                self.sample_rate = max_rate
                self.sample_rate_label.setText(f"{max_rate} Hz")
                # Update spectrogram parameters for new sample rate
                self._update_spectrogram_params()
                # Update buffer sizes for new sample rate (only if not recording)
                if not self.recording:
                    self.max_points = int(self.sample_rate * self.time_window_seconds)
                    # Recreate buffers with new size, preserving existing data
                    old_data1 = list(self.time_buffer1)
                    old_data2 = list(self.time_buffer2)
                    old_time = list(self.time_axis)
                    self.time_buffer1 = deque(maxlen=self.max_points)
                    self.time_buffer2 = deque(maxlen=self.max_points)
                    self.time_axis = deque(maxlen=self.max_points)
                    # Restore recent data if available (keep most recent data)
                    if old_data1:
                        keep_count = min(len(old_data1), self.max_points)
                        self.time_buffer1.extend(old_data1[-keep_count:])
                        self.time_buffer2.extend(old_data2[-keep_count:])
                        self.time_axis.extend(old_time[-keep_count:])
                print(f"Selected device: {device_info['name']}, Max sample rate: {max_rate} Hz")
        except Exception as e:
            print(f"Error detecting sample rate: {e}")
            # Fallback to default
            self.sample_rate = 44100
            self.sample_rate_label.setText("44100 Hz")
            self._update_spectrogram_params()
    
    def _update_spectrogram_params(self):
        """Update spectrogram parameters based on current sample rate"""
        # Adjust window size based on sample rate for better frequency resolution
        if self.sample_rate >= 96000:
            self.spec_window_size = 4096  # Larger window for high sample rates
            self.spec_overlap = 2048
        elif self.sample_rate >= 48000:
            self.spec_window_size = 2048
            self.spec_overlap = 1024
        else:
            self.spec_window_size = 1024
            self.spec_overlap = 512
        
        # Calculate spectrogram history to match time window
        spec_time_per_frame = (self.spec_window_size - self.spec_overlap) / self.sample_rate
        self.spec_history_length = int(self.time_window_seconds / spec_time_per_frame) + 10  # Add buffer
    
    def _init_plots(self):
        """Initialize all plots"""
        # Time series plot 1 (Mic 1)
        if self.ax_time1 is not None:
            self.ax_time1.set_title("Microphone 1 - Time Series")
            self.ax_time1.set_xlabel("Time (s)")
            self.ax_time1.set_ylabel("Amplitude")
            self.line_time1, = self.ax_time1.plot([], [], color="blue", linewidth=0.5)
            self.ax_time1.grid(True, alpha=0.3)
            self.ax_time1.set_ylim(-1, 1)
        
        # Time series plot 2 (Mic 2)
        if self.ax_time2 is not None:
            self.ax_time2.set_title("Microphone 2 - Time Series")
            self.ax_time2.set_xlabel("Time (s)")
            self.ax_time2.set_ylabel("Amplitude")
            self.line_time2, = self.ax_time2.plot([], [], color="red", linewidth=0.5)
            self.ax_time2.grid(True, alpha=0.3)
            self.ax_time2.set_ylim(-1, 1)
        
        # Spectrogram plot 1 (Mic 1)
        if self.ax_spec1 is not None:
            self.ax_spec1.set_title("Microphone 1 - Spectrogram")
            self.ax_spec1.set_xlabel("Time (s)")
            self.ax_spec1.set_ylabel("Frequency (Hz)")
            self.spec_im1 = None
            if hasattr(self, 'cbar1'):
                self.cbar1 = None
        
        # Spectrogram plot 2 (Mic 2)
        if self.ax_spec2 is not None:
            self.ax_spec2.set_title("Microphone 2 - Spectrogram")
            self.ax_spec2.set_xlabel("Time (s)")
            self.ax_spec2.set_ylabel("Frequency (Hz)")
            self.spec_im2 = None
            if hasattr(self, 'cbar2'):
                self.cbar2 = None
        
        self.fig.tight_layout()
    
    def _update_display_layout(self, display_text):
        """Update plot layout based on display selection"""
        # Stop recording if active to avoid issues during layout change
        was_recording = self.recording
        if was_recording:
            self.stop_recording()
        
        if display_text == "Both Mics":
            self.display_mode = "both"
            # Recreate subplots in 2x2 layout
            self.fig.clear()
            self.ax_time1 = self.fig.add_subplot(221)
            self.ax_time2 = self.fig.add_subplot(222)
            self.ax_spec1 = self.fig.add_subplot(223)
            self.ax_spec2 = self.fig.add_subplot(224)
        elif display_text == "Mic 1 Only":
            self.display_mode = "mic1"
            # Show only mic 1 plots (2x1 layout)
            self.fig.clear()
            self.ax_time1 = self.fig.add_subplot(211)
            self.ax_spec1 = self.fig.add_subplot(212)
            # Hide mic 2 plots
            self.ax_time2 = None
            self.ax_spec2 = None
        else:  # Mic 2 Only
            self.display_mode = "mic2"
            # Show only mic 2 plots (2x1 layout)
            self.fig.clear()
            self.ax_time2 = self.fig.add_subplot(211)
            self.ax_spec2 = self.fig.add_subplot(212)
            # Hide mic 1 plots
            self.ax_time1 = None
            self.ax_spec1 = None
        
        # Reinitialize plots
        self._init_plots()
        self.canvas.draw()
        
        # Restart recording if it was active
        if was_recording:
            self.start_recording()
    
    def start_recording(self):
        """Start audio recording"""
        if self.recording:
            return
        
        try:
            device_idx = self.device_combo.currentData()
            if device_idx is None:
                device_idx = None  # Use default
            
            # Create and start recorder thread
            self.recorder = AudioRecorder(
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_size=self.chunk_size,
                device=device_idx
            )
            self.recorder.data_ready.connect(self._on_audio_data)
            self.recorder.start()
            
            self.recording = True
            self.start_time = 0.0
            self.update_timer.start()
            
            self.status_label.setText("Status: Recording...")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {e}")
            self.recording = False
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.recorder:
            self.recorder.stop()
            self.recorder.wait(1000)  # Wait up to 1 second
            self.recorder = None
        
        self.update_timer.stop()
        self.status_label.setText("Status: Stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    def reset(self):
        """Reset all buffers and plots"""
        self.stop_recording()
        
        # Clear buffers
        self.time_buffer1.clear()
        self.time_buffer2.clear()
        self.time_axis.clear()
        self.recording_buffer1.clear()
        self.recording_buffer2.clear()
        self.recording_time.clear()
        self.spec_data1.clear()
        self.spec_data2.clear()
        self.spec_time1.clear()
        self.spec_time2.clear()
        
        self.start_time = 0.0
        self._redraw_plots()
    
    def _on_audio_data(self, data, sample_rate):
        """Callback when new audio data is available"""
        if not self.recording:
            return
        
        # data shape: (chunk_size, channels)
        if data.shape[1] < 2:
            return
        
        # Extract channels
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        
        # Create time axis for this chunk
        chunk_duration = len(channel1) / sample_rate
        if len(self.time_axis) == 0:
            current_time = 0.0
        else:
            current_time = self.time_axis[-1] + chunk_duration
        
        time_chunk = np.linspace(
            current_time - chunk_duration,
            current_time,
            len(channel1),
            endpoint=False
        )
        
        # Add to time series buffers
        self.time_buffer1.extend(channel1)
        self.time_buffer2.extend(channel2)
        self.time_axis.extend(time_chunk)
        
        # Add to recording buffers
        self.recording_buffer1.extend(channel1)
        self.recording_buffer2.extend(channel2)
        self.recording_time.extend(time_chunk)
        
        # Compute spectrogram for this chunk
        # Use overlapping windows for smoother spectrograms
        f1, t1, Sxx1 = signal.spectrogram(
            channel1,
            fs=sample_rate,
            nperseg=self.spec_window_size,
            noverlap=self.spec_overlap,
            window='hann'
        )
        f2, t2, Sxx2 = signal.spectrogram(
            channel2,
            fs=sample_rate,
            nperseg=self.spec_window_size,
            noverlap=self.spec_overlap,
            window='hann'
        )
        
        # Adjust time axis to absolute time
        t1_abs = t1 + (current_time - chunk_duration)
        t2_abs = t2 + (current_time - chunk_duration)
        
        # Store spectrogram data (log scale for better visualization)
        # Store as (frequencies, times, power)
        for i in range(len(t1_abs)):
            self.spec_data1.append(10 * np.log10(Sxx1[:, i] + 1e-10))
            self.spec_time1.append(t1_abs[i])
        
        for i in range(len(t2_abs)):
            self.spec_data2.append(10 * np.log10(Sxx2[:, i] + 1e-10))
            self.spec_time2.append(t2_abs[i])
    
    def _update_plots(self):
        """Update all plots with synchronized time windows"""
        if len(self.time_axis) == 0:
            return
        
        time_array = np.array(self.time_axis)
        data1 = np.array(self.time_buffer1)
        data2 = np.array(self.time_buffer2)
        
        # Calculate synchronized time window
        if len(time_array) > 0:
            time_min = time_array[0]
            time_max = time_array[-1]
            # Ensure window matches our display window
            if time_max - time_min > self.time_window_seconds:
                time_min = time_max - self.time_window_seconds
        
        # Update time series plots with synchronized window
        if self.display_mode in ["both", "mic1"] and self.ax_time1 is not None:
            self.line_time1.set_data(time_array, data1)
            self.ax_time1.set_xlim(time_min, time_max)
            self.ax_time1.relim()
            self.ax_time1.autoscale(axis='y')
        
        if self.display_mode in ["both", "mic2"] and self.ax_time2 is not None:
            self.line_time2.set_data(time_array, data2)
            self.ax_time2.set_xlim(time_min, time_max)
            self.ax_time2.relim()
            self.ax_time2.autoscale(axis='y')
        
        # Update spectrograms with synchronized time window
        if self.display_mode in ["both", "mic1"]:
            self._update_spectrogram(1, time_min, time_max)
        if self.display_mode in ["both", "mic2"]:
            self._update_spectrogram(2, time_min, time_max)
        
        self.canvas.draw_idle()
    
    def _update_spectrogram(self, mic_num, time_min=None, time_max=None):
        """Update spectrogram for specified microphone with synchronized time window"""
        if mic_num == 1:
            spec_data = self.spec_data1
            spec_time = self.spec_time1
            ax = self.ax_spec1
            spec_im = self.spec_im1
        else:
            spec_data = self.spec_data2
            spec_time = self.spec_time2
            ax = self.ax_spec2
            spec_im = self.spec_im2
        
        if ax is None or len(spec_data) == 0:
            return
        
        # Convert to 2D array
        spec_list = list(spec_data)
        time_list = list(spec_time)
        
        if len(spec_list) == 0:
            return
        
        # Filter by time window if specified
        if time_min is not None and time_max is not None:
            time_array_temp = np.array(time_list)
            mask = (time_array_temp >= time_min) & (time_array_temp <= time_max)
            if np.any(mask):
                spec_list = [spec_list[i] for i in range(len(spec_list)) if mask[i]]
                time_list = [time_list[i] for i in range(len(time_list)) if mask[i]]
            else:
                return  # No data in time window
        
        # Check if all rows have same length
        if len(spec_list) == 0:
            return
        
        first_len = len(spec_list[0])
        if not all(len(row) == first_len for row in spec_list):
            # Pad or truncate to same length
            spec_list = [row[:first_len] if len(row) >= first_len else np.pad(row, (0, first_len - len(row)), 'constant') 
                        for row in spec_list]
        
        spec_array = np.array(spec_list)
        time_array = np.array(time_list)
        
        if spec_array.size == 0 or len(time_array) == 0:
            return
        
        # Get frequency axis from window size
        f_axis = np.fft.rfftfreq(self.spec_window_size, 1.0 / self.sample_rate)
        if spec_array.shape[1] < len(f_axis):
            f_axis = f_axis[:spec_array.shape[1]]
        elif spec_array.shape[1] > len(f_axis):
            # Pad frequency axis if needed
            f_axis = np.pad(f_axis, (0, spec_array.shape[1] - len(f_axis)), 'edge')
        
        # Limit frequency range for better visualization
        # For high sample rates, show up to 20kHz, otherwise 10kHz
        max_freq_display = 20000 if self.sample_rate >= 96000 else 10000
        max_freq_idx = np.searchsorted(f_axis, max_freq_display)
        if max_freq_idx > 0 and max_freq_idx < len(f_axis):
            spec_array = spec_array[:, :max_freq_idx]
            f_axis = f_axis[:max_freq_idx]
        
        # Use synchronized time window
        if time_min is None:
            time_min = time_array[0] if len(time_array) > 0 else 0
        if time_max is None:
            time_max = time_array[-1] if len(time_array) > 0 else self.time_window_seconds
        
        # Update or create spectrogram image
        if spec_im is None:
            # Create new spectrogram
            ax.clear()
            ax.set_title(f"Microphone {mic_num} - Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            
            if len(time_array) > 0 and len(f_axis) > 0:
                extent = [time_min, time_max, f_axis[0], f_axis[-1]]
                spec_im = ax.imshow(
                    spec_array.T,
                    aspect='auto',
                    origin='lower',
                    extent=extent,
                    cmap='viridis',
                    interpolation='nearest',
                    vmin=-80,
                    vmax=0
                )
                ax.set_xlim(time_min, time_max)
                # Set y-axis limit based on sample rate
                max_freq_display = 20000 if self.sample_rate >= 96000 else 10000
                ax.set_ylim(0, max_freq_display)
                
                if mic_num == 1:
                    self.spec_im1 = spec_im
                    if not hasattr(self, 'cbar1') or self.cbar1 is None:
                        self.cbar1 = self.fig.colorbar(spec_im, ax=ax, label='Power (dB)')
                else:
                    self.spec_im2 = spec_im
                    if not hasattr(self, 'cbar2') or self.cbar2 is None:
                        self.cbar2 = self.fig.colorbar(spec_im, ax=ax, label='Power (dB)')
        else:
            # Update existing image with synchronized time window
            if len(time_array) > 0 and len(f_axis) > 0:
                spec_im.set_array(spec_array.T)
                spec_im.set_extent([time_min, time_max, f_axis[0], f_axis[-1]])
                spec_im.set_clim(vmin=-80, vmax=0)
                ax.set_xlim(time_min, time_max)
                # Set y-axis limit based on sample rate
                max_freq_display = 20000 if self.sample_rate >= 96000 else 10000
                ax.set_ylim(0, max_freq_display)
    
    def _redraw_plots(self):
        """Redraw all plots with empty data"""
        empty = []
        if self.ax_time1 is not None:
            self.line_time1.set_data(empty, empty)
            self.ax_time1.relim()
            self.ax_time1.autoscale_view()
        
        if self.ax_time2 is not None:
            self.line_time2.set_data(empty, empty)
            self.ax_time2.relim()
            self.ax_time2.autoscale_view()
        
        # Clear spectrograms
        if self.ax_spec1 is not None:
            self.ax_spec1.clear()
            self.ax_spec1.set_title("Microphone 1 - Spectrogram")
            self.ax_spec1.set_xlabel("Time (s)")
            self.ax_spec1.set_ylabel("Frequency (Hz)")
            self.spec_im1 = None
            if hasattr(self, 'cbar1'):
                self.cbar1 = None
        
        if self.ax_spec2 is not None:
            self.ax_spec2.clear()
            self.ax_spec2.set_title("Microphone 2 - Spectrogram")
            self.ax_spec2.set_xlabel("Time (s)")
            self.ax_spec2.set_ylabel("Frequency (Hz)")
            self.spec_im2 = None
            if hasattr(self, 'cbar2'):
                self.cbar2 = None
        
        self.canvas.draw_idle()
    
    def save_audio(self):
        """Save recorded audio to file"""
        if len(self.recording_buffer1) == 0:
            QMessageBox.warning(self, "No Data", "No audio data to save. Please record first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio",
            "",
            "WAV Files (*.wav);;NumPy Files (*.npz);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Convert to numpy arrays
            audio1 = np.array(self.recording_buffer1, dtype=np.float32)
            audio2 = np.array(self.recording_buffer2, dtype=np.float32)
            time_array = np.array(self.recording_time)
            
            if file_path.endswith('.wav'):
                # Save as WAV file (stereo)
                import scipy.io.wavfile as wavfile
                stereo_audio = np.column_stack([audio1, audio2])
                wavfile.write(file_path, self.sample_rate, stereo_audio)
                QMessageBox.information(self, "Success", f"Audio saved to {file_path}")
            elif file_path.endswith('.npz'):
                # Save as NumPy compressed file
                np.savez_compressed(
                    file_path,
                    mic1=audio1,
                    mic2=audio2,
                    time=time_array,
                    sample_rate=self.sample_rate
                )
                QMessageBox.information(self, "Success", f"Audio data saved to {file_path}")
            else:
                # Default to npz
                file_path += '.npz'
                np.savez_compressed(
                    file_path,
                    mic1=audio1,
                    mic2=audio2,
                    time=time_array,
                    sample_rate=self.sample_rate
                )
                QMessageBox.information(self, "Success", f"Audio data saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save audio: {e}")
    
    def save_snapshot(self):
        """Save current plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Snapshot saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save snapshot: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_recording()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MicrophoneWindow()
    window.show()
    sys.exit(app.exec_())

