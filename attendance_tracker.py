import time
import threading
from typing import List, Dict, Optional
from datetime import datetime, date
import cv2
import numpy as np

class AttendanceTracker:
    def __init__(self, database, face_recognition_module):
        self.database = database
        self.face_recognition_module = face_recognition_module
        self.is_tracking = False
        self.tracking_thread = None
        self.attendance_log = []
        self.current_session = None
        
        # Tracking settings
        self.update_interval = 1.0  # seconds
        self.entry_cooldown = 30    # seconds between entry records
        self.exit_cooldown = 30     # seconds between exit records
        
        # Student tracking state
        self.student_entry_times = {}  # Track when students were last recorded as entering
        self.student_exit_times = {}   # Track when students were last recorded as exiting
        
    def start_tracking(self):
        """Start the attendance tracking system"""
        if self.is_tracking:
            return False
        
        self.is_tracking = True
        self.current_session = {
            'start_time': datetime.now(),
            'date': date.today(),
            'total_entries': 0,
            'total_exits': 0
        }
        
        # Reset tracking state
        self.face_recognition_module.reset_tracking()
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        print("Attendance tracking started")
        return True
    
    def stop_tracking(self):
        """Stop the attendance tracking system"""
        if not self.is_tracking:
            return False
        
        self.is_tracking = False
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
        
        # Update daily summary
        self.database.update_daily_summary()
        
        # Log session end
        if self.current_session:
            self.current_session['end_time'] = datetime.now()
            duration = self.current_session['end_time'] - self.current_session['start_time']
            self.current_session['duration'] = duration
            
            print(f"Attendance tracking stopped. Session duration: {duration}")
        
        return True
    
    def _tracking_loop(self):
        """Main tracking loop that runs in a separate thread"""
        while self.is_tracking:
            try:
                # Get attendance updates from face recognition module
                updates = self.face_recognition_module.get_attendance_updates()
                
                # Process each update
                for update in updates:
                    self._process_attendance_update(update)
                
                # Update current headcount
                current_count = self.face_recognition_module.get_present_count()
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in tracking loop: {e}")
                time.sleep(self.update_interval)
    
    def _process_attendance_update(self, update: Dict):
        """Process a single attendance update"""
        student_id = update['student_id']
        action = update['action']
        current_time = time.time()
        
        if action == 'entered':
            # Check cooldown for entry
            if (student_id not in self.student_entry_times or 
                current_time - self.student_entry_times[student_id] > self.entry_cooldown):
                
                # Record entry in database
                if self.database.record_entry(student_id):
                    self.student_entry_times[student_id] = current_time
                    self.current_session['total_entries'] += 1
                    
                    # Log the entry
                    self._log_attendance_event(student_id, 'entered', update['name'])
                    print(f"Student {update['name']} entered the class")
        
        elif action == 'exited':
            # Check cooldown for exit
            if (student_id not in self.student_exit_times or 
                current_time - self.student_exit_times[student_id] > self.exit_cooldown):
                
                # Record exit in database
                if self.database.record_exit(student_id):
                    self.student_exit_times[student_id] = current_time
                    self.current_session['total_exits'] += 1
                    
                    # Log the exit
                    self._log_attendance_event(student_id, 'exited', update['name'])
                    print(f"Student {update['name']} left the class")
    
    def _log_attendance_event(self, student_id: int, action: str, name: str):
        """Log an attendance event"""
        log_entry = {
            'timestamp': datetime.now(),
            'student_id': student_id,
            'name': name,
            'action': action
        }
        
        self.attendance_log.append(log_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.attendance_log) > 1000:
            self.attendance_log = self.attendance_log[-1000:]
    
    def get_current_attendance_status(self) -> Dict:
        """Get current attendance status"""
        current_present = self.face_recognition_module.get_current_present_students()
        present_count = len(current_present)
        
        # Get total registered students
        all_students = self.database.get_all_students()
        total_students = len(all_students)
        
        # Get today's attendance summary
        today_attendance = self.database.get_today_attendance()
        present_today = len([a for a in today_attendance if a['status'] == 'present'])
        absent_today = total_students - present_today
        
        status = {
            'current_present': present_count,
            'total_students': total_students,
            'present_today': present_today,
            'absent_today': absent_today,
            'attendance_percentage': (present_today / total_students * 100) if total_students > 0 else 0,
            'session_duration': self._get_session_duration(),
            'total_entries_session': self.current_session['total_entries'] if self.current_session else 0,
            'total_exits_session': self.current_session['total_exits'] if self.current_session else 0
        }
        
        return status
    
    def get_attendance_log(self, limit: int = 100) -> List[Dict]:
        """Get recent attendance log entries"""
        return self.attendance_log[-limit:] if self.attendance_log else []
    
    def get_student_attendance_history(self, student_id: int, days: int = 30) -> List[Dict]:
        """Get attendance history for a specific student"""
        # This would require additional database queries
        # For now, return basic info
        return []
    
    def export_attendance_report(self, start_date: date, end_date: date, file_path: str) -> bool:
        """Export attendance report for a date range"""
        try:
            # This would require implementing report generation
            # For now, just show a message
            print(f"Export functionality for {start_date} to {end_date} will be implemented")
            return False
        except Exception as e:
            print(f"Error exporting attendance report: {e}")
            return False
    
    def get_daily_summary(self, target_date: date = None) -> Dict:
        """Get daily attendance summary"""
        if target_date is None:
            target_date = date.today()
        
        # Get attendance for the target date
        attendance = self.database.get_today_attendance()
        
        summary = {
            'date': target_date,
            'total_students': len(attendance),
            'present_count': len([a for a in attendance if a['status'] == 'present']),
            'absent_count': len([a for a in attendance if a['status'] == 'absent']),
            'attendance_percentage': 0
        }
        
        if summary['total_students'] > 0:
            summary['attendance_percentage'] = (summary['present_count'] / summary['total_students']) * 100
        
        return summary
    
    def _get_session_duration(self) -> str:
        """Get current session duration as formatted string"""
        if not self.current_session or not self.is_tracking:
            return "00:00:00"
        
        duration = datetime.now() - self.current_session['start_time']
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def is_tracking_active(self) -> bool:
        """Check if tracking is currently active"""
        return self.is_tracking
    
    def get_tracking_stats(self) -> Dict:
        """Get current tracking statistics"""
        if not self.current_session:
            return {}
        
        stats = {
            'session_start': self.current_session['start_time'],
            'session_duration': self._get_session_duration(),
            'total_entries': self.current_session['total_entries'],
            'total_exits': self.current_session['total_exits'],
            'current_present': self.face_recognition_module.get_present_count()
        }
        
        return stats
    
    def reset_session(self):
        """Reset the current tracking session"""
        if self.is_tracking:
            self.stop_tracking()
        
        self.current_session = None
        self.attendance_log.clear()
        self.student_entry_times.clear()
        self.student_exit_times.clear()
        
        print("Session reset")
    
    def get_realtime_updates(self) -> List[Dict]:
        """Get real-time attendance updates (for UI display)"""
        updates = []
        
        # Get current present students
        present_students = self.face_recognition_module.get_current_present_students()
        
        # Get all students
        all_students = self.database.get_all_students()
        
        for student in all_students:
            is_present = student['id'] in present_students
            status = 'Present' if is_present else 'Absent'
            
            updates.append({
                'id': student['id'],
                'name': student['name'],
                'student_id': student['student_id'],
                'status': status,
                'is_present': is_present
            })
        
        return updates
