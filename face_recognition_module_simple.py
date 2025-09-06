import cv2
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict
import time
import os

class SimpleFaceRecognitionModule:
    def __init__(self):
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.face_locations = []
        self.face_names = []
        
        # Tracking variables for attendance
        self.present_students = set()  # Set of student IDs currently present
        self.student_tracking = {}  # Track each student's status
        self.entry_threshold = 3  # Frames to confirm entry
        self.exit_threshold = 5   # Frames to confirm exit
        
        # Simple face matching threshold
        self.matching_threshold = 0.7
        
    def load_known_faces(self, students_data: List[Dict]):
        """Load known faces from database"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        for student in students_data:
            if student['face_encoding']:
                # Convert bytes back to numpy array
                face_encoding = pickle.loads(student['face_encoding'])
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(student['name'])
                self.known_face_ids.append(student['id'])
        
        print(f"Loaded {len(self.known_face_encodings)} known faces")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[Tuple]]:
        """Process a single frame for face detection"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        self.face_locations = []
        self.face_names = []
        
        for (x, y, w, h) in faces:
            # Convert to format expected by the rest of the system
            top, left, bottom, right = y, x, y + h, x + w
            self.face_locations.append((top, right, bottom, left))
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Try to match with known faces
            name = "Unknown"
            student_id = None
            
            if len(self.known_face_encodings) > 0:
                # Resize face ROI to match known face size
                face_roi_resized = cv2.resize(face_roi, (64, 64))
                
                # Simple template matching approach
                best_match = None
                best_score = 0
                
                for i, known_encoding in enumerate(self.known_face_encodings):
                    # Resize known encoding to match
                    if known_encoding.shape != face_roi_resized.shape:
                        known_encoding = cv2.resize(known_encoding, (64, 64))
                    
                    # Calculate similarity (simple correlation)
                    similarity = cv2.matchTemplate(face_roi_resized, known_encoding, cv2.TM_CCOEFF_NORMED)
                    score = similarity[0][0]
                    
                    if score > best_score and score > self.matching_threshold:
                        best_score = score
                        best_match = i
                
                if best_match is not None:
                    name = self.known_face_names[best_match]
                    student_id = self.known_face_ids[best_match]
                    
                    # Track student for attendance
                    if student_id:
                        self.track_student_attendance(student_id, name)
            
            self.face_names.append(name)
        
        return frame, self.face_names, self.face_locations
    
    def track_student_attendance(self, student_id: int, name: str):
        """Track student attendance status"""
        current_time = time.time()
        
        if student_id not in self.student_tracking:
            self.student_tracking[student_id] = {
                'name': name,
                'first_seen': current_time,
                'last_seen': current_time,
                'entry_frames': 0,
                'exit_frames': 0,
                'status': 'unknown'
            }
        else:
            self.student_tracking[student_id]['last_seen'] = current_time
        
        # Update tracking based on current status
        if student_id in self.present_students:
            # Student is already marked present
            self.student_tracking[student_id]['entry_frames'] += 1
            self.student_tracking[student_id]['exit_frames'] = 0
        else:
            # Student is not marked present yet
            self.student_tracking[student_id]['exit_frames'] += 1
            self.student_tracking[student_id]['entry_frames'] = 0
    
    def get_attendance_updates(self) -> List[Dict]:
        """Get attendance updates for students"""
        updates = []
        current_time = time.time()
        
        for student_id, tracking in self.student_tracking.items():
            # Check for entry
            if (tracking['entry_frames'] >= self.entry_threshold and 
                student_id not in self.present_students):
                self.present_students.add(student_id)
                tracking['status'] = 'entered'
                updates.append({
                    'student_id': student_id,
                    'name': tracking['name'],
                    'action': 'entered',
                    'time': current_time
                })
            
            # Check for exit (not seen for several frames)
            elif (tracking['exit_frames'] >= self.exit_threshold and 
                  student_id in self.present_students):
                self.present_students.remove(student_id)
                tracking['status'] = 'exited'
                updates.append({
                    'student_id': student_id,
                    'name': tracking['name'],
                    'action': 'exited',
                    'time': current_time
                })
        
        return updates
    
    def get_current_present_students(self) -> List[int]:
        """Get list of currently present student IDs"""
        return list(self.present_students)
    
    def get_present_count(self) -> int:
        """Get current count of present students"""
        return len(self.present_students)
    
    def reset_tracking(self):
        """Reset tracking data"""
        self.present_students.clear()
        self.student_tracking.clear()
    
    def add_new_face(self, face_image: np.ndarray, name: str, student_id: int) -> bool:
        """Add a new face to the known faces database"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the first face found
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to standard size
                face_encoding = cv2.resize(face_roi, (64, 64))
                
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(student_id)
                
                return True
            else:
                return False
        except Exception as e:
            print(f"Error adding new face: {e}")
            return False
    
    def detect_faces_in_image(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces in an image and return locations"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """Get face encoding for a specific face location"""
        try:
            top, right, bottom, left = face_location
            face_roi = image[top:bottom, left:right]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            face_encoding = cv2.resize(gray, (64, 64))
            
            return face_encoding
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
