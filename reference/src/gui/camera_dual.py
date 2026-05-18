import PySpin
import numpy as np
import cv2
import time
from datetime import datetime

# Initialize system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() < 2:
    print(f"Error: Need 2 cameras, but only {cam_list.GetSize()} detected!")
    cam_list.Clear()
    system.ReleaseInstance()
    exit()

# Get both cameras
cam0 = cam_list[0]
cam1 = cam_list[1]

try:
    cam0.Init()
    cam1.Init()
except Exception as e:
    print(f"Error initializing cameras: {e}")
    cam0.DeInit()
    cam1.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    exit()

# Set continuous acquisition for both cameras
nodemap0 = cam0.GetNodeMap()
nodemap1 = cam1.GetNodeMap()

# Camera 0 setup
node_acquisition_mode0 = PySpin.CEnumerationPtr(nodemap0.GetNode('AcquisitionMode'))
node_acquisition_mode_continuous0 = node_acquisition_mode0.GetEntryByName('Continuous')
node_acquisition_mode0.SetIntValue(node_acquisition_mode_continuous0.GetValue())

# Camera 1 setup
node_acquisition_mode1 = PySpin.CEnumerationPtr(nodemap1.GetNode('AcquisitionMode'))
node_acquisition_mode_continuous1 = node_acquisition_mode1.GetEntryByName('Continuous')
node_acquisition_mode1.SetIntValue(node_acquisition_mode_continuous1.GetValue())

# Set pixel format to color for both cameras (prefer BGR8)
pixel_format_is_rgb0 = False
pixel_format_is_rgb1 = False

# Camera 0 pixel format
try:
    node_pixel_format0 = PySpin.CEnumerationPtr(nodemap0.GetNode('PixelFormat'))
    if PySpin.IsAvailable(node_pixel_format0) and PySpin.IsWritable(node_pixel_format0):
        node_pixel_format_bgr8_0 = node_pixel_format0.GetEntryByName('BGR8')
        if PySpin.IsAvailable(node_pixel_format_bgr8_0) and PySpin.IsReadable(node_pixel_format_bgr8_0):
            pixel_format_bgr8_0 = node_pixel_format_bgr8_0.GetValue()
            node_pixel_format0.SetIntValue(pixel_format_bgr8_0)
            print("Camera 0: Set pixel format to BGR8")
            pixel_format_is_rgb0 = False
        else:
            node_pixel_format_rgb8_0 = node_pixel_format0.GetEntryByName('RGB8')
            if PySpin.IsAvailable(node_pixel_format_rgb8_0) and PySpin.IsReadable(node_pixel_format_rgb8_0):
                pixel_format_rgb8_0 = node_pixel_format_rgb8_0.GetValue()
                node_pixel_format0.SetIntValue(pixel_format_rgb8_0)
                print("Camera 0: Set pixel format to RGB8")
                pixel_format_is_rgb0 = True
except Exception as e:
    print(f"Warning: Could not set pixel format for camera 0: {e}")

# Camera 1 pixel format
try:
    node_pixel_format1 = PySpin.CEnumerationPtr(nodemap1.GetNode('PixelFormat'))
    if PySpin.IsAvailable(node_pixel_format1) and PySpin.IsWritable(node_pixel_format1):
        node_pixel_format_bgr8_1 = node_pixel_format1.GetEntryByName('BGR8')
        if PySpin.IsAvailable(node_pixel_format_bgr8_1) and PySpin.IsReadable(node_pixel_format_bgr8_1):
            pixel_format_bgr8_1 = node_pixel_format_bgr8_1.GetValue()
            node_pixel_format1.SetIntValue(pixel_format_bgr8_1)
            print("Camera 1: Set pixel format to BGR8")
            pixel_format_is_rgb1 = False
        else:
            node_pixel_format_rgb8_1 = node_pixel_format1.GetEntryByName('RGB8')
            if PySpin.IsAvailable(node_pixel_format_rgb8_1) and PySpin.IsReadable(node_pixel_format_rgb8_1):
                pixel_format_rgb8_1 = node_pixel_format_rgb8_1.GetValue()
                node_pixel_format1.SetIntValue(pixel_format_rgb8_1)
                print("Camera 1: Set pixel format to RGB8")
                pixel_format_is_rgb1 = True
except Exception as e:
    print(f"Warning: Could not set pixel format for camera 1: {e}")

# Start acquisition for both cameras
cam0.BeginAcquisition()
cam1.BeginAcquisition()
print("Both cameras streaming... Press 'c' to capture both, 'q' to quit")

frame_count = 0
capture_count = 0

# FPS tracking for both cameras
fps_start_time0 = time.time()
fps_frame_count0 = 0
fps_display0 = 0.0

fps_start_time1 = time.time()
fps_frame_count1 = 0
fps_display1 = 0.0

try:
    while True:
        # Get images from both cameras
        image_result0 = cam0.GetNextImage(1000)
        image_result1 = cam1.GetNextImage(1000)
        
        if image_result0.IsIncomplete() or image_result1.IsIncomplete():
            if image_result0.IsIncomplete():
                print(f"Frame {frame_count}: Camera 0 incomplete")
            if image_result1.IsIncomplete():
                print(f"Frame {frame_count}: Camera 1 incomplete")
            image_result0.Release()
            image_result1.Release()
            continue
        
        img_array0 = image_result0.GetNDArray()
        img_array1 = image_result1.GetNDArray()
        frame_count += 1
        fps_frame_count0 += 1
        fps_frame_count1 += 1
        
        # Calculate FPS for both cameras
        current_time = time.time()
        elapsed0 = current_time - fps_start_time0
        elapsed1 = current_time - fps_start_time1
        
        if elapsed0 >= 1.0:
            fps_display0 = fps_frame_count0 / elapsed0
            fps_frame_count0 = 0
            fps_start_time0 = current_time
        
        if elapsed1 >= 1.0:
            fps_display1 = fps_frame_count1 / elapsed1
            fps_frame_count1 = 0
            fps_start_time1 = current_time
        
        # Print image info on first frame
        if frame_count == 1:
            pixel_format0 = image_result0.GetPixelFormat()
            pixel_format1 = image_result1.GetPixelFormat()
            print(f"Camera 0: format={pixel_format0}, shape={img_array0.shape}, dtype={img_array0.dtype}")
            print(f"Camera 1: format={pixel_format1}, shape={img_array1.shape}, dtype={img_array1.dtype}")
            
            # Validate images
            if img_array0.size == 0:
                print("ERROR: Camera 0 image is empty!")
            if img_array1.size == 0:
                print("ERROR: Camera 1 image is empty!")
            
            # Use the flags we set during initialization (format 27 is likely BGR8)
            # Since we set BGR8 during init, trust those flags
            print(f"Camera 0: Using {'RGB' if pixel_format_is_rgb0 else 'BGR'} format (from initialization)")
            print(f"Camera 1: Using {'RGB' if pixel_format_is_rgb1 else 'BGR'} format (from initialization)")
        
        # Convert to BGR for OpenCV display - ensure proper format
        # Camera 0
        if len(img_array0.shape) == 2:  # Grayscale - convert to BGR for display
            display_img0 = cv2.cvtColor(img_array0, cv2.COLOR_GRAY2BGR)
        elif len(img_array0.shape) == 3 and img_array0.shape[2] == 3:
            if pixel_format_is_rgb0:
                display_img0 = cv2.cvtColor(img_array0, cv2.COLOR_RGB2BGR)
            else:
                # Already BGR - make a copy to ensure we have our own array
                display_img0 = img_array0.copy()
        else:
            display_img0 = img_array0.copy()
        
        # Camera 1
        if len(img_array1.shape) == 2:  # Grayscale - convert to BGR for display
            display_img1 = cv2.cvtColor(img_array1, cv2.COLOR_GRAY2BGR)
        elif len(img_array1.shape) == 3 and img_array1.shape[2] == 3:
            if pixel_format_is_rgb1:
                display_img1 = cv2.cvtColor(img_array1, cv2.COLOR_RGB2BGR)
            else:
                # Already BGR - make a copy to ensure we have our own array
                display_img1 = img_array1.copy()
        else:
            display_img1 = img_array1.copy()
        
        # Debug: Check image statistics on first frame
        if frame_count == 1:
            print(f"Camera 0 display: shape={display_img0.shape}, min={display_img0.min()}, max={display_img0.max()}, mean={display_img0.mean():.1f}")
            print(f"Camera 1 display: shape={display_img1.shape}, min={display_img1.min()}, max={display_img1.max()}, mean={display_img1.mean():.1f}")
        
        # Resize images to reasonable display size (max 640x480 per camera)
        MAX_DISPLAY_HEIGHT = 480
        MAX_DISPLAY_WIDTH = 640
        
        h0, w0 = display_img0.shape[:2]
        h1, w1 = display_img1.shape[:2]
        
        # Scale down if too large (images are 1200x1920, need significant scaling)
        if h0 > MAX_DISPLAY_HEIGHT or w0 > MAX_DISPLAY_WIDTH:
            scale0 = min(MAX_DISPLAY_HEIGHT / h0, MAX_DISPLAY_WIDTH / w0)
            new_h0 = int(h0 * scale0)
            new_w0 = int(w0 * scale0)
            display_img0 = cv2.resize(display_img0, (new_w0, new_h0), interpolation=cv2.INTER_LINEAR)
            h0, w0 = new_h0, new_w0
            if frame_count == 1:
                print(f"Camera 0: Scaled from {img_array0.shape[:2]} to ({h0}, {w0})")
        
        if h1 > MAX_DISPLAY_HEIGHT or w1 > MAX_DISPLAY_WIDTH:
            scale1 = min(MAX_DISPLAY_HEIGHT / h1, MAX_DISPLAY_WIDTH / w1)
            new_h1 = int(h1 * scale1)
            new_w1 = int(w1 * scale1)
            display_img1 = cv2.resize(display_img1, (new_w1, new_h1), interpolation=cv2.INTER_LINEAR)
            h1, w1 = new_h1, new_w1
            if frame_count == 1:
                print(f"Camera 1: Scaled from {img_array1.shape[:2]} to ({h1}, {w1})")
        
        # Resize to same height for side-by-side display
        if h0 != h1:
            target_height = min(h0, h1)
            if h0 != target_height:
                scale0 = target_height / h0
                new_w0 = int(w0 * scale0)
                display_img0 = cv2.resize(display_img0, (new_w0, target_height), interpolation=cv2.INTER_LINEAR)
                w0 = new_w0
            if h1 != target_height:
                scale1 = target_height / h1
                new_w1 = int(w1 * scale1)
                display_img1 = cv2.resize(display_img1, (new_w1, target_height), interpolation=cv2.INTER_LINEAR)
                w1 = new_w1
        
        # Add FPS overlay (only update every 10 frames to reduce overhead)
        if frame_count % 10 == 0:
            fps_text0 = f"Cam0: {fps_display0:.1f} fps | Frame: {frame_count}"
            fps_text1 = f"Cam1: {fps_display1:.1f} fps"
            
            display_with_text0 = display_img0.copy()
            display_with_text1 = display_img1.copy()
            
            cv2.putText(display_with_text0, fps_text0, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_with_text1, fps_text1, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Combine side by side
            try:
                # Ensure both images have same height
                if display_with_text0.shape[0] != display_with_text1.shape[0]:
                    target_h = min(display_with_text0.shape[0], display_with_text1.shape[0])
                    display_with_text0 = cv2.resize(display_with_text0, 
                        (int(display_with_text0.shape[1] * target_h / display_with_text0.shape[0]), target_h))
                    display_with_text1 = cv2.resize(display_with_text1, 
                        (int(display_with_text1.shape[1] * target_h / display_with_text1.shape[0]), target_h))
                combined = np.hstack((display_with_text0, display_with_text1))
            except Exception as e:
                print(f"Error combining images with text: {e}")
                print(f"  display_with_text0 shape: {display_with_text0.shape}, dtype: {display_with_text0.dtype}")
                print(f"  display_with_text1 shape: {display_with_text1.shape}, dtype: {display_with_text1.dtype}")
                # Fallback: try without text
                if display_img0.shape[0] != display_img1.shape[0]:
                    target_h = min(display_img0.shape[0], display_img1.shape[0])
                    display_img0 = cv2.resize(display_img0, 
                        (int(display_img0.shape[1] * target_h / display_img0.shape[0]), target_h))
                    display_img1 = cv2.resize(display_img1, 
                        (int(display_img1.shape[1] * target_h / display_img1.shape[0]), target_h))
                combined = np.hstack((display_img0, display_img1))
        else:
            # Combine side by side
            try:
                # Ensure both images have same height
                if display_img0.shape[0] != display_img1.shape[0]:
                    target_h = min(display_img0.shape[0], display_img1.shape[0])
                    display_img0 = cv2.resize(display_img0, 
                        (int(display_img0.shape[1] * target_h / display_img0.shape[0]), target_h))
                    display_img1 = cv2.resize(display_img1, 
                        (int(display_img1.shape[1] * target_h / display_img1.shape[0]), target_h))
                combined = np.hstack((display_img0, display_img1))
            except Exception as e:
                print(f"Error combining images: {e}")
                print(f"  display_img0 shape: {display_img0.shape}, dtype: {display_img0.dtype}")
                print(f"  display_img1 shape: {display_img1.shape}, dtype: {display_img1.dtype}")
                continue
        
        # Validate combined image before display
        if combined.size == 0:
            print(f"ERROR: Combined image is empty at frame {frame_count}")
            continue
        
        # Debug: Check combined image on first few frames
        if frame_count <= 3:
            print(f"Frame {frame_count}: Combined shape={combined.shape}, min={combined.min()}, max={combined.max()}, mean={combined.mean():.1f}")
        
        # Ensure combined image is uint8
        if combined.dtype != np.uint8:
            print(f"Warning: Combined image dtype is {combined.dtype}, converting to uint8")
            combined = combined.astype(np.uint8)
        
        # Add instructions
        if frame_count % 10 == 0:
            combined_with_text = combined.copy()
            cv2.putText(combined_with_text, "Press 'c' to capture both, 'q' to quit", 
                       (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Dual Blackfly Cameras - Side by Side', combined_with_text)
        else:
            cv2.imshow('Dual Blackfly Cameras - Side by Side', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('c'):
            # Capture and save both images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename0 = f'blackfly_capture_cam0_{timestamp}.jpg'
            filename1 = f'blackfly_capture_cam1_{timestamp}.jpg'
            
            # Convert to BGR for saving
            if len(img_array0.shape) == 3 and img_array0.shape[2] == 3:
                if pixel_format_is_rgb0:
                    save_img0 = cv2.cvtColor(img_array0, cv2.COLOR_RGB2BGR)
                else:
                    save_img0 = img_array0
            else:
                save_img0 = img_array0
            
            if len(img_array1.shape) == 3 and img_array1.shape[2] == 3:
                if pixel_format_is_rgb1:
                    save_img1 = cv2.cvtColor(img_array1, cv2.COLOR_RGB2BGR)
                else:
                    save_img1 = img_array1
            else:
                save_img1 = img_array1
            
            cv2.imwrite(filename0, save_img0)
            cv2.imwrite(filename1, save_img1)
            capture_count += 1
            print(f"Captured image pair {capture_count}: {filename0}, {filename1}")
            
            # Show confirmation
            combined_with_capture = combined.copy()
            cv2.putText(combined_with_capture, f"CAPTURED: {filename0} & {filename1}", 
                       (10, combined.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Dual Blackfly Cameras - Side by Side', combined_with_capture)
            cv2.waitKey(200)
        
        image_result0.Release()
        image_result1.Release()
        
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Stop acquisition and cleanup
    cam0.EndAcquisition()
    cam1.EndAcquisition()
    cam0.DeInit()
    cam1.DeInit()
    del cam0
    del cam1
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()
    print(f"✓ Dual camera session complete. Captured {capture_count} image pair(s).")

