import sqlite3
import datetime
from typing import List, Dict, Optional

class AttendanceDatabase:
    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_encoding BLOB,
                student_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date DATE,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                status TEXT DEFAULT 'present',
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        # Create daily_summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_students INTEGER DEFAULT 0,
                present_count INTEGER DEFAULT 0,
                absent_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_student(self, name: str, face_encoding: bytes, student_id: str) -> bool:
        """Add a new student to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO students (name, face_encoding, student_id)
                VALUES (?, ?, ?)
            ''', (name, face_encoding, student_id))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_all_students(self) -> List[Dict]:
        """Get all students from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, face_encoding, student_id FROM students')
        students = []
        
        for row in cursor.fetchall():
            students.append({
                'id': row[0],
                'name': row[1],
                'face_encoding': row[2],
                'student_id': row[3]
            })
        
        conn.close()
        return students
    
    def record_entry(self, student_id: int) -> bool:
        """Record student entry to class"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.date.today()
            current_time = datetime.datetime.now()
            
            # Check if entry already exists for today
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND entry_time IS NOT NULL
            ''', (student_id, today))
            
            if cursor.fetchone() is None:
                cursor.execute('''
                    INSERT INTO attendance (student_id, date, entry_time, status)
                    VALUES (?, ?, ?, 'present')
                ''', (student_id, today, current_time))
                
                conn.commit()
                conn.close()
                return True
            
            conn.close()
            return False
        except Exception as e:
            print(f"Error recording entry: {e}")
            return False
    
    def record_exit(self, student_id: int) -> bool:
        """Record student exit from class"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.date.today()
            current_time = datetime.datetime.now()
            
            # Update exit time for today's entry
            cursor.execute('''
                UPDATE attendance 
                SET exit_time = ? 
                WHERE student_id = ? AND date = ? AND exit_time IS NULL
            ''', (current_time, student_id, today))
            
            if cursor.rowcount > 0:
                conn.commit()
                conn.close()
                return True
            
            conn.close()
            return False
        except Exception as e:
            print(f"Error recording exit: {e}")
            return False
    
    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.date.today()
        
        cursor.execute('''
            SELECT s.name, s.student_id, a.entry_time, a.exit_time, a.status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
            ORDER BY s.name
        ''', (today,))
        
        attendance = []
        for row in cursor.fetchall():
            attendance.append({
                'name': row[0],
                'student_id': row[1],
                'entry_time': row[2],
                'exit_time': row[3],
                'status': row[4] if row[4] else 'absent'
            })
        
        conn.close()
        return attendance
    
    def get_current_headcount(self) -> int:
        """Get current number of students present in class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.date.today()
        
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE date = ? AND entry_time IS NOT NULL AND exit_time IS NULL
        ''', (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def delete_student(self, student_id: int) -> bool:
        """Delete a student from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete student from students table
            cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
            
            if cursor.rowcount > 0:
                # Also delete related attendance records
                cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
                
                conn.commit()
                conn.close()
                return True
            else:
                conn.close()
                return False
        except Exception as e:
            print(f"Error deleting student: {e}")
            return False
    
    def get_student_by_id(self, student_id: int) -> Optional[Dict]:
        """Get a specific student by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, name, face_encoding, student_id FROM students WHERE id = ?', (student_id,))
            row = cursor.fetchone()
            
            if row:
                student = {
                    'id': row[0],
                    'name': row[1],
                    'face_encoding': row[2],
                    'student_id': row[3]
                }
                conn.close()
                return student
            else:
                conn.close()
                return None
        except Exception as e:
            print(f"Error getting student: {e}")
            return None

    def update_daily_summary(self):
        """Update daily summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.date.today()
        
        # Get total students
        cursor.execute('SELECT COUNT(*) FROM students')
        total_students = cursor.fetchone()[0]
        
        # Get present count
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE date = ? AND entry_time IS NOT NULL
        ''', (today,))
        present_count = cursor.fetchone()[0]
        
        absent_count = total_students - present_count
        
        # Insert or update daily summary
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary (date, total_students, present_count, absent_count)
            VALUES (?, ?, ?, ?)
        ''', (today, total_students, present_count, absent_count))
        
        conn.commit()
        conn.close()
