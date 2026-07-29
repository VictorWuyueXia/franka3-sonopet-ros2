#!/usr/bin/env python3
import os
import sys
import sysconfig
import types
import numpy as np
import time
from datetime import datetime
import cv2

# Ensure a usable pkg_resources for flirpy on environments where setuptools'
# pkg_resources is missing. flirpy only uses resource_filename on Windows,
# so on Linux this lightweight stub is sufficient.
try:  # pragma: no cover - environment-specific shim
    import pkg_resources  # type: ignore
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")  # type: ignore

    def resource_filename(package_or_requirement, resource_name):
        raise RuntimeError(
            "pkg_resources.resource_filename is not available in this environment."
        )

    pkg_resources.resource_filename = resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = pkg_resources

# cv2 sets QT_QPA_PLATFORM_PLUGIN_PATH to its own broken bundled Qt plugins.
# Override after cv2 import so both cv2 and matplotlib use the same working Qt.
_pyqt5_plugins = os.path.join(os.path.dirname(sysconfig.get_path("purelib")), "PyQt5", "Qt5", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _pyqt5_plugins

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

class BosonCamera:
    """Class to handle Boson thermal camera operations"""
    
    def __init__(self):
        """Initialize the Boson camera and settings"""
        try:
            from flirpy.camera.boson import Boson
            self.camera = Boson()
            self.camera.set_ffc_manual()  # Set manual FFC to avoid interruptions
            time.sleep(0.1)
            print("Boson camera initialized successfully")
        except ImportError:
            print("Error: flirpy is not installed. Please install using 'pip install flirpy'")
            raise
        except Exception as e:
            print(f"Error initializing Boson camera: {e}")
            raise
            
        # Camera properties
        self.width = 640
        self.height = 512
        
        # Display properties
        self.color_map = cv2.COLORMAP_INFERNO
        self.ref_point = None
        
        # Recording properties
        self.recording = False
        self.video_writer = None
        self.fps = 60
        self.output_folder = "recordings"
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def grab_frame(self):
        """Grab a frame from the Boson camera and convert to displayable format"""
        try:
            # Grab raw frame
            raw = self.camera.grab()
            
            # Convert to 16-bit data
            raw_16 = np.frombuffer(raw.tobytes(), dtype=np.uint16)
            if raw_16.size != self.width * self.height:
                print(f"Warning: Unexpected frame size: {raw_16.size} pixels")
                return None, None
                
            raw_16 = raw_16.reshape((self.height, self.width))
            
            # For debug: print FPA (Focal Plane Array) temperature statistics
            fpa_min = raw_16.min()
            fpa_max = raw_16.max()
            fpa_mean = raw_16.mean()
            print(f"FPA values - Min: {fpa_min}, Max: {fpa_max}, Mean: {fpa_mean:.2f}")
            
            # Normalize for display
            normalized = cv2.normalize(raw_16, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            display_img = cv2.applyColorMap(normalized, self.color_map)
            
            return raw_16, display_img
            
        except Exception as e:
            print(f"Error grabbing frame: {e}")
            return None, None
    
    def start_recording(self):
        """Start recording video"""
        if self.recording:
            print("Already recording")
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_folder, f"boson_{timestamp}.avi")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, self.fps, (self.width, self.height)
        )
        
        if self.video_writer.isOpened():
            self.recording = True
            print(f"Recording started: {filename}")
        else:
            print("Failed to start recording")
    
    def stop_recording(self):
        """Stop recording video"""
        if not self.recording:
            print("Not recording")
            return
            
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("Recording stopped")
    
    def close(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()
        self.camera.close()
        print("Camera resources released")


def on_mouse(event, x, y, flags, param):
    """Mouse callback to handle reference point selection"""
    camera = param
    if event == cv2.EVENT_LBUTTONDOWN:
        camera.ref_point = (x, y)
        print(f"Reference point selected at: ({x}, {y})")


def main():
    """Main function to run a simple Boson camera thermal viewer."""
    try:
        # Initialize camera
        camera = BosonCamera()

        # Use matplotlib for image display so we don't depend on OpenCV HighGUI
        plt.ion()  # Enable interactive mode for matplotlib
        fig_img, ax_img = plt.subplots(num="Boson Thermal Image")
        ax_img.set_title("Boson Thermal Image")
        ax_img.axis("off")
        img_artist = ax_img.imshow(
            np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        )
        fig_img.canvas.draw_idle()

        print("\nControls:")
        print("  Q / Esc - Quit")
        print("  C       - Change color map")
        print("  Left click - Set reference point")

        color_maps = [
            cv2.COLORMAP_INFERNO,
            cv2.COLORMAP_JET,
            cv2.COLORMAP_HOT,
            cv2.COLORMAP_RAINBOW,
        ]
        color_map_index = 0
        quit_requested = False

        def _on_mpl_click(event):
            if event.inaxes != ax_img:
                return
            if event.button != 1:
                return
            if event.xdata is None or event.ydata is None:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            camera.ref_point = (x, y)
            print(f"Reference point selected at: ({x}, {y})")

        def _on_mpl_key(event):
            nonlocal quit_requested, color_map_index
            k = (event.key or "").lower()
            if k in {"q", "escape"}:
                quit_requested = True
                return
            if k == "c":
                color_map_index = (color_map_index + 1) % len(color_maps)
                camera.color_map = color_maps[color_map_index]
                print(f"Changed color map to index {color_map_index}")

        fig_img.canvas.mpl_connect("button_press_event", _on_mpl_click)
        fig_img.canvas.mpl_connect("key_press_event", _on_mpl_key)

        while not quit_requested:
            # Grab and process frame
            raw_frame, display_img = camera.grab_frame()

            if display_img is None:
                # Failed to get a frame; try again shortly
                time.sleep(0.05)
                continue

            # Show reference point if selected
            if camera.ref_point is not None and raw_frame is not None:
                x, y = camera.ref_point
                if 0 <= x < camera.width and 0 <= y < camera.height:
                    ref_value = raw_frame[y, x]
                    cv2.circle(display_img, camera.ref_point, 5, (255, 255, 255), 2)
                    cv2.putText(
                        display_img,
                        f"Value: {ref_value}",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Show the image (convert BGR->RGB for matplotlib)
            img_artist.set_data(display_img[:, :, ::-1])
            fig_img.canvas.draw_idle()
            plt.pause(0.01)  # Allow matplotlib to update

        camera.close()
        plt.close(fig_img)

    except Exception as e:
        print(f"Error in main program: {e}")
        

if __name__ == "__main__":
    main()