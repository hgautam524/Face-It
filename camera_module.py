import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable
from queue import Queue
import platform

class CameraModule:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_queue = Queue(maxsize=10)
        self.callback = None
        self.thread = None
        
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Determine platform for backend handling
        self.is_windows = platform.system().lower().startswith('win')
    
    def _create_capture(self, index: int) -> Optional[cv2.VideoCapture]:
        """Create a VideoCapture with backends suitable for the OS."""
        backends_to_try = []
        
        if self.is_windows:
            # Try DirectShow first (more stable on many Windows setups), then MSMF, then default
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            # On non-Windows, default is usually fine
            backends_to_try = [cv2.CAP_ANY]
        
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                try:
                    # Fallback without specifying backend
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        return cap
                    cap.release()
                except Exception:
                    pass
        return None
        
    def start_camera(self) -> bool:
        """Start the camera capture"""
        try:
            # Try to open the requested index
            self.cap = self._create_capture(self.camera_index)
            
            # If failed, try to find any available camera
            if not self.cap or not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}. Scanning for available cameras...")
                available = self.list_available_cameras()
                if available:
                    self.camera_index = available[0]
                    self.cap = self._create_capture(self.camera_index)
                
            if not self.cap or not self.cap.isOpened():
                print("Error: No available camera found")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            
            print(f"Camera started successfully on index {self.camera_index}")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def start_with_video_file(self, file_path: str) -> bool:
        """Start capture from a video file as a fallback/demo source."""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap or not cap.isOpened():
                print(f"Error: Could not open video file {file_path}")
                return False
            
            self.cap = cap
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            print(f"Video file source started: {file_path}")
            return True
        except Exception as e:
            print(f"Error starting video file source: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture"""
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Camera stopped")
    
    def _capture_frames(self):
        """Internal method to capture frames in a separate thread"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret:
                    # Put frame in queue, remove old frames if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    self.frame_queue.put(frame)
                    
                    # Update current frame
                    self.current_frame = frame
                    
                    # Call callback if set
                    if self.callback:
                        try:
                            self.callback(frame)
                        except Exception as e:
                            print(f"Error in camera callback: {e}")
                else:
                    print("Failed to read frame from camera")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(1.0 / self.fps)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame from the camera"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (may be None if camera not started)"""
        return self.current_frame
    
    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """Set a callback function to be called with each new frame"""
        self.callback = callback
    
    def is_camera_active(self) -> bool:
        """Check if camera is currently active"""
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self) -> dict:
        """Get information about the camera"""
        if not self.cap:
            return {}
        
        info = {
            'index': self.camera_index,
            'is_opened': self.cap.isOpened(),
            'frame_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
        return info
    
    def set_camera_properties(self, width: int = None, height: int = None, fps: int = None):
        """Set camera properties"""
        if self.cap and self.cap.isOpened():
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.frame_width = width
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.frame_height = height
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.fps = fps
    
    def capture_single_photo(self) -> Optional[np.ndarray]:
        """Capture a single photo and return the frame"""
        if not self.is_camera_active():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def list_available_cameras(self, max_cameras: int = 10) -> list:
        """List available camera indices"""
        available_cameras = []
        
        for i in range(max_cameras):
            try:
                cap = self._create_capture(i)
                if cap and cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except Exception as e:
                # Silently continue if camera fails
                continue
        
        # If no cameras found, return default indices
        if not available_cameras:
            available_cameras = [0, 1, 2]  # Default indices to try
        
        return available_cameras
