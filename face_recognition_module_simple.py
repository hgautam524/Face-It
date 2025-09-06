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
        
        # Improved face matching threshold
        self.matching_threshold = 0.6  # Lower threshold for better recognition
        self.face_size = (128, 128)  # Standardized face size for better matching
        
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
        
        # Detect faces with improved parameters
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
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
                # Resize face ROI to standardized size
                face_roi_resized = cv2.resize(face_roi, self.face_size)
                
                # Improved template matching approach
                best_match = None
                best_score = 0
                
                for i, known_encoding in enumerate(self.known_face_encodings):
                    # Ensure known encoding is the right size
                    if known_encoding.shape != face_roi_resized.shape:
                        known_encoding = cv2.resize(known_encoding, self.face_size)
                    
                    # Calculate similarity using normalized cross correlation
                    similarity = cv2.matchTemplate(face_roi_resized, known_encoding, cv2.TM_CCOEFF_NORMED)
                    score = similarity[0][0]
                    
                    # Also try histogram comparison for better accuracy
                    hist1 = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([known_encoding], [0], None, [256], [0, 256])
                    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    # Combine both scores for better accuracy
                    combined_score = (score + hist_score) / 2
                    
                    if combined_score > best_score and combined_score > self.matching_threshold:
                        best_score = combined_score
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
            
            # Detect faces in the image with improved parameters
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Use the first face found
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to standardized size for better matching
                face_encoding = cv2.resize(face_roi, self.face_size)
                
                # Apply histogram equalization for better recognition
                face_encoding = cv2.equalizeHist(face_encoding)
                
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(student_id)
                
                print(f"Added face encoding for {name} (ID: {student_id})")
                return True
            else:
                print(f"No face detected in image for {name}")
                return False
        except Exception as e:
            print(f"Error adding new face: {e}")
            return False
    
    def add_multiple_face_angles(self, face_images: List[np.ndarray], name: str, student_id: int) -> bool:
        """Add multiple face angles for better recognition"""
        try:
            successful_encodings = 0
            
            for i, face_image in enumerate(face_images):
                # Convert to grayscale
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Use the first face found
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to standardized size
                    face_encoding = cv2.resize(face_roi, self.face_size)
                    
                    # Apply histogram equalization
                    face_encoding = cv2.equalizeHist(face_encoding)
                    
                    # Add slight variations for better recognition
                    variations = self._create_face_variations(face_encoding)
                    
                    for variation in variations:
                        self.known_face_encodings.append(variation)
                        self.known_face_names.append(name)
                        self.known_face_ids.append(student_id)
                        successful_encodings += 1
                    
                    print(f"Added face encoding {i+1} for {name} (ID: {student_id})")
            
            if successful_encodings > 0:
                print(f"Successfully added {successful_encodings} face encodings for {name}")
                return True
            else:
                print(f"No faces detected in any images for {name}")
                return False
                
        except Exception as e:
            print(f"Error adding multiple face angles: {e}")
            return False
    
    def _create_face_variations(self, face_encoding: np.ndarray) -> List[np.ndarray]:
        """Create variations of a face encoding for better recognition"""
        variations = [face_encoding]  # Original
        
        # Brightness variations
        bright = cv2.convertScaleAbs(face_encoding, alpha=1.2, beta=20)
        dark = cv2.convertScaleAbs(face_encoding, alpha=0.8, beta=-20)
        variations.extend([bright, dark])
        
        # Slight rotation variations
        rows, cols = face_encoding.shape
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(face_encoding, M, (cols, rows))
            variations.append(rotated)
        
        # Gaussian blur variations (slight)
        blurred = cv2.GaussianBlur(face_encoding, (3, 3), 0)
        variations.append(blurred)
        
        return variations
    
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
            
            # Resize to standardized size
            face_encoding = cv2.resize(gray, self.face_size)
            
            # Apply histogram equalization for better recognition
            face_encoding = cv2.equalizeHist(face_encoding)
            
            return face_encoding
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
