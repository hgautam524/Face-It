import cv2
import face_recognition
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict
import time

class FaceRecognitionModule:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.face_detection_interval = 3  # Process every 3rd frame
        self.frame_count = 0
        
        # Tracking variables for attendance
        self.present_students = set()  # Set of student IDs currently present
        self.student_tracking = {}  # Track each student's status
        self.entry_threshold = 3  # Frames to confirm entry
        self.exit_threshold = 5   # Frames to confirm exit
        
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
        """Process a single frame for face recognition"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Only process every nth frame to save time
        if self.process_this_frame:
            # Find all faces in the current frame
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                student_id = None
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    student_id = self.known_face_ids[first_match_index]
                
                self.face_names.append(name)
                
                # Track student for attendance
                if student_id:
                    self.track_student_attendance(student_id, name)
        
        self.process_this_frame = not self.process_this_frame
        
        # Scale back up face locations
        face_locations_scaled = [(top * 4, right * 4, bottom * 4, left * 4) 
                                for (top, right, bottom, left) in self.face_locations]
        
        return frame, self.face_names, face_locations_scaled
    
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
            # Convert BGR to RGB
            rgb_image = face_image[:, :, ::-1]
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if len(face_encodings) > 0:
                # Use the first face found
                face_encoding = face_encodings[0]
                
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
        rgb_image = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_image)
        return face_locations
    
    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """Get face encoding for a specific face location"""
        try:
            rgb_image = image[:, :, ::-1]
            face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                return None
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
