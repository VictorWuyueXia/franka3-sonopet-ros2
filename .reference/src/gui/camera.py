import PySpin
import numpy as np
import cv2
import time
import os
from datetime import datetime

# Initialize system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    print("No camera detected!")
    cam_list.Clear()
    system.ReleaseInstance()
    exit()

# Get first camera
cam = cam_list[0]
cam.Init()

# Set continuous acquisition
nodemap = cam.GetNodeMap()
node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

# Check and display current frame rate setting
try:
    node_frame_rate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
    if PySpin.IsAvailable(node_frame_rate) and PySpin.IsReadable(node_frame_rate):
        current_fps = node_frame_rate.GetValue()
        print(f"Camera frame rate setting: {current_fps:.2f} fps")
    else:
        # Try to get frame rate enable/disable and manual setting
        node_frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsAvailable(node_frame_rate_enable) and PySpin.IsReadable(node_frame_rate_enable):
            if node_frame_rate_enable.GetValue():
                node_frame_rate_manual = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsAvailable(node_frame_rate_manual):
                    current_fps = node_frame_rate_manual.GetValue()
                    print(f"Camera frame rate setting: {current_fps:.2f} fps")
            else:
                print("Frame rate: Auto (maximum)")
except Exception as e:
    print(f"Could not read frame rate setting: {e}")

# Set pixel format to color (prefer BGR8 for OpenCV compatibility)
pixel_format_is_rgb = False
try:
    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
        # Try BGR8 first (OpenCV native format, no conversion needed)
        node_pixel_format_bgr8 = node_pixel_format.GetEntryByName('BGR8')
        if PySpin.IsAvailable(node_pixel_format_bgr8) and PySpin.IsReadable(node_pixel_format_bgr8):
            pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
            node_pixel_format.SetIntValue(pixel_format_bgr8)
            print("Set pixel format to BGR8 (OpenCV native)")
            pixel_format_is_rgb = False
        else:
            # Fall back to RGB8
            node_pixel_format_rgb8 = node_pixel_format.GetEntryByName('RGB8')
            if PySpin.IsAvailable(node_pixel_format_rgb8) and PySpin.IsReadable(node_pixel_format_rgb8):
                pixel_format_rgb8 = node_pixel_format_rgb8.GetValue()
                node_pixel_format.SetIntValue(pixel_format_rgb8)
                print("Set pixel format to RGB8 (will convert to BGR for OpenCV)")
                pixel_format_is_rgb = True
            else:
                print("Warning: Could not set color pixel format. Available formats:")
                entries = node_pixel_format.GetEntries()
                for entry in entries:
                    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                        print(f"  - {entry.GetName()}")
    else:
        print("Warning: PixelFormat node is not available or writable")
except Exception as e:
    print(f"Warning: Could not set pixel format: {e}")
    print("Camera may output grayscale images")

# Start acquisition
cam.BeginAcquisition()
print("Camera streaming... Press 'c' to capture, 'q' to quit")

frame_count = 0
capture_count = 0
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0.0

try:
    while True:
        image_result = cam.GetNextImage(1000)
        
        if image_result.IsIncomplete():
            print(f"Frame {frame_count}: Incomplete")
        else:
            img_array = image_result.GetNDArray()
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS every second
            current_time = time.time()
            elapsed = current_time - fps_start_time
            if elapsed >= 1.0:
                fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = current_time
            
            # Print image info on first frame
            if frame_count == 1:
                pixel_format = image_result.GetPixelFormat()
                print(f"Image format: {pixel_format}, Shape: {img_array.shape}, Dtype: {img_array.dtype}")
                # Check if format name contains RGB or BGR
                format_name = str(pixel_format)
                if 'RGB' in format_name.upper():
                    pixel_format_is_rgb = True
                    print("Detected RGB format - will convert to BGR")
                elif 'BGR' in format_name.upper():
                    pixel_format_is_rgb = False
                    print("Detected BGR format - no conversion needed")
            
            # Convert to BGR for OpenCV display - OPTIMIZE: minimal conversions
            if len(img_array.shape) == 2:  # Grayscale
                display_img = img_array  # Don't convert, show grayscale directly
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                if pixel_format_is_rgb:
                    display_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    display_img = img_array  # Already BGR, use directly
            else:
                display_img = img_array
            
            # Add FPS overlay (only update every 10 frames to reduce overhead)
            if frame_count % 10 == 0:
                fps_text = f"FPS: {fps_display:.1f} | Frame: {frame_count} | Press 'c' to capture, 'q' to quit"
                # Create a copy only for text overlay
                display_with_text = display_img.copy()
                cv2.putText(display_with_text, fps_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Blackfly Camera - Live View', display_with_text)
            else:
                cv2.imshow('Blackfly Camera - Live View', display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                # Capture and save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'blackfly_capture_{timestamp}.jpg'
                # Convert to BGR for saving (only convert if needed, no copy for BGR)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    if pixel_format_is_rgb:
                        save_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        save_img = img_array  # Already BGR, cv2.imwrite can handle it directly
                else:
                    save_img = img_array
                cv2.imwrite(filename, save_img)
                capture_count += 1
                print(f"Captured image {capture_count}: {filename} (Frame {frame_count})")
                
                # Show confirmation on display (create copy for modification)
                display_with_capture = display_img.copy()
                cv2.putText(display_with_capture, f"CAPTURED: {filename}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Blackfly Camera - Live View', display_with_capture)
                # Non-blocking confirmation - just show it briefly
                cv2.waitKey(200)  # Reduced from 500ms to 200ms
        
        image_result.Release()
        
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()
    print(f"✓ Camera session complete. Captured {capture_count} image(s).")
