import cv2
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os

class StudentManagement:
    def __init__(self, database, face_recognition_module):
        self.database = database
        self.face_recognition_module = face_recognition_module
        self.current_student_image = None
        self.current_face_location = None
        
    def capture_student_photo(self, camera_module) -> Optional[np.ndarray]:
        """Capture a photo for a new student"""
        if not camera_module.is_camera_active():
            messagebox.showerror("Error", "Camera is not active. Please start the camera first.")
            return None
        
        # Capture frame
        frame = camera_module.capture_single_photo()
        if frame is None:
            messagebox.showerror("Error", "Failed to capture photo from camera.")
            return None
        
        # Detect faces in the captured frame
        face_locations = self.face_recognition_module.detect_faces_in_image(frame)
        
        if len(face_locations) == 0:
            messagebox.showerror("Error", "No face detected in the captured image. Please try again.")
            return None
        elif len(face_locations) > 1:
            messagebox.showerror("Error", "Multiple faces detected. Please ensure only one face is visible.")
            return None
        
        # Store the face location for later use
        self.current_face_location = face_locations[0]
        self.current_student_image = frame
        
        return frame
    
    def add_student_from_photo(self, name: str, student_id: str, photo: np.ndarray) -> bool:
        """Add a new student using the captured photo"""
        if photo is None or self.current_face_location is None:
            return False
        
        try:
            # Get face encoding from the detected face
            face_encoding = self.face_recognition_module.get_face_encoding(photo, self.current_face_location)
            
            if face_encoding is None:
                messagebox.showerror("Error", "Failed to extract face encoding. Please try again.")
                return False
            
            # Convert numpy array to bytes for storage
            face_encoding_bytes = pickle.dumps(face_encoding)
            
            # Add student to database
            success = self.database.add_student(name, face_encoding_bytes, student_id)
            
            if success:
                # Add to face recognition module
                self.face_recognition_module.add_new_face(photo, name, 
                    self.database.get_all_students()[-1]['id'])
                
                messagebox.showinfo("Success", f"Student {name} added successfully!")
                return True
            else:
                messagebox.showerror("Error", f"Student ID {student_id} already exists in the database.")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add student: {str(e)}")
            return False
    
    def load_student_photo(self, file_path: str) -> Optional[np.ndarray]:
        """Load a student photo from file"""
        try:
            # Read image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image file.")
                return None
            
            # Detect faces in the loaded image
            face_locations = self.face_recognition_module.detect_faces_in_image(image)
            
            if len(face_locations) == 0:
                messagebox.showerror("Error", "No face detected in the loaded image.")
                return None
            elif len(face_locations) > 1:
                messagebox.showerror("Error", "Multiple faces detected. Please ensure only one face is visible.")
                return None
            
            # Store the face location for later use
            self.current_face_location = face_locations[0]
            self.current_student_image = image
            
            return image
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return None
    
    def add_student_from_file(self, name: str, student_id: str, file_path: str) -> bool:
        """Add a new student using a photo file"""
        photo = self.load_student_photo(file_path)
        if photo is not None:
            return self.add_student_from_photo(name, student_id, photo)
        return False
    
    def edit_student(self, student_id: int, new_name: str, new_student_id: str) -> bool:
        """Edit existing student information"""
        try:
            # This would require updating the database schema to support editing
            # For now, we'll just show a message
            messagebox.showinfo("Info", "Student editing functionality will be implemented in future versions.")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to edit student: {str(e)}")
            return False
    
    def delete_student(self, student_id: int) -> bool:
        """Delete a student from the system"""
        try:
            # This would require updating the database schema to support deletion
            # For now, we'll just show a message
            messagebox.showinfo("Info", "Student deletion functionality will be implemented in future versions.")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete student: {str(e)}")
            return False
    
    def get_student_list(self) -> List[Dict]:
        """Get list of all students"""
        return self.database.get_all_students()
    
    def search_student(self, search_term: str) -> List[Dict]:
        """Search for students by name or ID"""
        all_students = self.get_student_list()
        search_results = []
        
        search_term_lower = search_term.lower()
        
        for student in all_students:
            if (search_term_lower in student['name'].lower() or 
                search_term_lower in student['student_id'].lower()):
                search_results.append(student)
        
        return search_results
    
    def validate_student_id(self, student_id: str) -> bool:
        """Validate student ID format"""
        # Basic validation - can be customized based on requirements
        if len(student_id) < 3:
            return False
        
        # Check if student ID already exists
        existing_students = self.get_student_list()
        for student in existing_students:
            if student['student_id'] == student_id:
                return False
        
        return True
    
    def validate_student_name(self, name: str) -> bool:
        """Validate student name"""
        if len(name.strip()) < 2:
            return False
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
        for char in invalid_chars:
            if char in name:
                return False
        
        return True
    
    def export_student_data(self, file_path: str) -> bool:
        """Export student data to CSV file"""
        try:
            students = self.get_student_list()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("ID,Name,Student ID,Created At\n")
                for student in students:
                    f.write(f"{student['id']},{student['name']},{student['student_id']},{student.get('created_at', 'N/A')}\n")
            
            messagebox.showinfo("Success", f"Student data exported to {file_path}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export student data: {str(e)}")
            return False
    
    def import_student_data(self, file_path: str) -> bool:
        """Import student data from CSV file"""
        try:
            # This would require implementing CSV import functionality
            # For now, we'll just show a message
            messagebox.showinfo("Info", "Student data import functionality will be implemented in future versions.")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import student data: {str(e)}")
            return False
    
    def get_student_statistics(self) -> Dict:
        """Get statistics about students"""
        students = self.get_student_list()
        
        stats = {
            'total_students': len(students),
            'students_with_photos': len([s for s in students if s['face_encoding'] is not None]),
            'students_without_photos': len([s for s in students if s['face_encoding'] is None])
        }
        
        return stats
