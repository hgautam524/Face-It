import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from typing import List, Optional, Callable

class CameraPreviewWindow:
    def __init__(self, parent, camera_module, face_recognition_module):
        self.parent = parent
        self.camera_module = camera_module
        self.face_recognition_module = face_recognition_module
        
        # Window setup
        self.window = tk.Toplevel(parent)
        self.window.title("Capture Student Photo")
        self.window.geometry("800x600")
        self.window.configure(bg='#f0f0f0')
        self.window.resizable(False, False)
        
        # Center the window
        self.window.transient(parent)
        self.window.grab_set()
        
        # Variables
        self.is_capturing = False
        self.captured_images = []
        self.current_frame = None
        self.update_interval = 50  # milliseconds
        
        # Setup UI
        self.setup_ui()
        
        # Start camera preview
        self.start_preview()
        
        # Start update loop
        self.update_preview()
    
    def setup_ui(self):
        """Setup the camera preview interface"""
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="📸 Capture Student Photo", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Position your face in the camera view and click 'Capture Photo' to take multiple shots from different angles",
                                font=('Arial', 10), wraplength=700)
        instructions.pack(pady=(0, 20))
        
        # Camera preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(preview_frame, bg='black', width=640, height=480)
        self.camera_canvas.pack(pady=10)
        
        # Face detection overlay
        self.face_canvas = tk.Canvas(preview_frame, bg='transparent', width=640, height=480)
        self.face_canvas.place(in_=self.camera_canvas, x=0, y=0)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.capture_btn = ttk.Button(button_frame, text="Capture Photo", 
                                     command=self.capture_photo)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(button_frame, text="Clear All", 
                                   command=self.clear_captures)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.finish_btn = ttk.Button(button_frame, text="Finish & Add Student", 
                                    command=self.finish_capture)
        self.finish_btn.pack(side=tk.RIGHT)
        
        # Captured photos display
        photos_frame = ttk.LabelFrame(main_frame, text="Captured Photos", padding=10)
        photos_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scrollable frame for photos
        canvas = tk.Canvas(photos_frame, height=120)
        scrollbar = ttk.Scrollbar(photos_frame, orient="horizontal", command=canvas.xview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to capture photos", 
                                     font=('Arial', 10, 'italic'))
        self.status_label.pack(pady=(10, 0))
        
        # Configure styles
        self.configure_styles()
    
    def configure_styles(self):
        """Configure custom button styles - minimal design"""
        style = ttk.Style()
        
        # Minimal button styles
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        style.configure('Success.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        style.configure('Warning.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
    
    def start_preview(self):
        """Start camera preview"""
        if not self.camera_module.is_camera_active():
            messagebox.showerror("Error", "Camera is not active. Please start the camera first.")
            self.window.destroy()
            return
        
        # Set camera callback
        self.camera_module.set_callback(self.process_frame)
        self.is_capturing = True
    
    def process_frame(self, frame):
        """Process camera frame for preview"""
        if self.is_capturing:
            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_recognition_module.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            # Draw face rectangles
            frame_with_faces = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_faces, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.current_frame = frame_with_faces
    
    def update_preview(self):
        """Update the camera preview"""
        try:
            if self.current_frame is not None and self.is_capturing:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize to fit canvas
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update canvas
                    self.camera_canvas.delete("all")
                    self.camera_canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
                    self.camera_canvas.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error updating preview: {e}")
        
        # Schedule next update
        if self.is_capturing:
            self.window.after(self.update_interval, self.update_preview)
    
    def capture_photo(self):
        """Capture a photo"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No camera feed available")
            return
        
        # Detect faces in current frame
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_recognition_module.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in the current frame. Please position your face in the camera view.")
            return
        elif len(faces) > 1:
            messagebox.showwarning("Warning", "Multiple faces detected. Please ensure only one face is visible.")
            return
        
        # Add the captured image
        self.captured_images.append(self.current_frame.copy())
        
        # Update status
        self.status_label.config(text=f"Captured {len(self.captured_images)} photo(s). Take more from different angles for better recognition.")
        
        # Add thumbnail to scrollable frame
        self.add_photo_thumbnail(self.current_frame, len(self.captured_images))
        
        # Flash effect
        self.flash_effect()
    
    def add_photo_thumbnail(self, image, photo_num):
        """Add a thumbnail to the captured photos display"""
        # Create thumbnail
        thumbnail = cv2.resize(image, (100, 75))
        thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        pil_thumbnail = Image.fromarray(thumbnail_rgb)
        photo_thumbnail = ImageTk.PhotoImage(pil_thumbnail)
        
        # Create thumbnail frame
        thumb_frame = ttk.Frame(self.scrollable_frame)
        thumb_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Thumbnail label
        thumb_label = ttk.Label(thumb_frame, image=photo_thumbnail)
        thumb_label.image = photo_thumbnail  # Keep reference
        thumb_label.pack()
        
        # Photo number label
        num_label = ttk.Label(thumb_frame, text=f"Photo {photo_num}", font=('Arial', 8))
        num_label.pack()
    
    def flash_effect(self):
        """Create a flash effect when capturing"""
        original_bg = self.camera_canvas['bg']
        self.camera_canvas.configure(bg='white')
        self.window.after(100, lambda: self.camera_canvas.configure(bg=original_bg))
    
    def clear_captures(self):
        """Clear all captured photos"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all captured photos?"):
            self.captured_images.clear()
            
            # Clear thumbnails
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.status_label.config(text="Ready to capture photos")
    
    def finish_capture(self):
        """Finish capturing and return the images"""
        if len(self.captured_images) == 0:
            messagebox.showwarning("Warning", "No photos captured. Please capture at least one photo.")
            return
        
        # Stop capturing
        self.is_capturing = False
        
        # Return the captured images
        self.window.result = self.captured_images
        self.window.destroy()
    
    def on_closing(self):
        """Handle window closing"""
        self.is_capturing = False
        self.window.destroy()
