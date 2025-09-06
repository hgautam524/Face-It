import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime, date
import os

# Import our custom modules
from database import AttendanceDatabase
from face_recognition_module_simple import SimpleFaceRecognitionModule as FaceRecognitionModule
from camera_module import CameraModule
from student_management import StudentManagement
from attendance_tracker import AttendanceTracker

class FacialAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Facial Attendance Recognition System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')
        
        # Configure modern styling
        self.configure_styles()
        
        # Reduce OpenCV log noise
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        
        # Initialize modules
        self.database = AttendanceDatabase()
        self.face_recognition_module = FaceRecognitionModule()
        self.camera_module = CameraModule()
        self.student_management = StudentManagement(self.database, self.face_recognition_module)
        self.attendance_tracker = AttendanceTracker(self.database, self.face_recognition_module)
        
        # Load known faces
        self.load_known_faces()
        
        # Variables
        self.is_camera_active = False
        self.is_tracking_active = False
        self.current_frame = None
        self.update_interval = 100  # milliseconds
        
        # Setup UI
        self.setup_ui()
        
        # Start update loop
        self.update_ui()
    
    def configure_styles(self):
        """Configure modern styling for the application"""
        style = ttk.Style()
        
        # Configure notebook style
        style.configure('TNotebook', background='#f5f5f5')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Arial', 10, 'bold'))
        
        # Configure frame styles
        style.configure('Card.TFrame', background='white', relief='solid', borderwidth=1)
        style.configure('Header.TFrame', background='#0078d4', relief='flat')
        
        # Configure label styles
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), background='#0078d4', foreground='white')
        style.configure('Subtitle.TLabel', font=('Arial', 14, 'bold'), background='#0078d4', foreground='white')
        style.configure('Info.TLabel', font=('Arial', 11), background='#0078d4', foreground='white')
        
        # Configure button styles - minimal original design
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        style.configure('Success.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        style.configure('Warning.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        style.configure('Danger.TButton', font=('Arial', 10, 'bold'), padding=[10, 5])
        
        # Configure label frame styles - minimal original design
        style.configure('Card.TLabelframe', background='#f0f0f0', relief='flat', borderwidth=0)
        style.configure('Card.TLabelframe.Label', font=('Arial', 10, 'bold'), background='#f0f0f0')
    
    def load_known_faces(self):
        """Load known faces from database"""
        students = self.database.get_all_students()
        self.face_recognition_module.load_known_faces(students)
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header frame with gradient effect
        header_frame = ttk.Frame(main_container, style='Header.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title with emoji
        title_label = ttk.Label(header_frame, text="🎯 Facial Attendance Recognition System", 
                               style='Title.TLabel', foreground='white')
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="Advanced AI-Powered Student Attendance Management", 
                                  style='Info.TLabel', foreground='white')
        subtitle_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs with emojis
        self.create_dashboard_tab()
        self.create_camera_tab()
        self.create_students_tab()
        self.create_attendance_tab()
        self.create_reports_tab()
        
        # Keep original tab names without emojis
        self.notebook.tab(0, text="Dashboard")
        self.notebook.tab(1, text="Camera View")
        self.notebook.tab(2, text="Students")
        self.notebook.tab(3, text="Attendance")
        self.notebook.tab(4, text="Reports")
        
        # Populate camera list on startup
        try:
            self.refresh_cameras()
        except Exception as e:
            print(f"Error refreshing cameras on startup: {e}")
        
        # Status bar
        self.status_bar = ttk.Label(main_container, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Dashboard content
        # Camera control section
        camera_frame = ttk.LabelFrame(dashboard_frame, text="Camera Control", padding=10)
        camera_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Camera buttons
        camera_btn_frame = ttk.Frame(camera_frame)
        camera_btn_frame.pack(fill=tk.X)
        
        self.start_camera_btn = ttk.Button(camera_btn_frame, text="Start Camera", 
                                          command=self.start_camera)
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_camera_btn = ttk.Button(camera_btn_frame, text="Stop Camera", 
                                         command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Camera info
        self.camera_info_label = ttk.Label(camera_frame, text="Camera: Not Active")
        self.camera_info_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Attendance tracking section
        tracking_frame = ttk.LabelFrame(dashboard_frame, text="Attendance Tracking", padding=10)
        tracking_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Tracking buttons
        tracking_btn_frame = ttk.Frame(tracking_frame)
        tracking_btn_frame.pack(fill=tk.X)
        
        self.start_tracking_btn = ttk.Button(tracking_btn_frame, text="Start Tracking", 
                                            command=self.start_tracking, state=tk.DISABLED)
        self.start_tracking_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_tracking_btn = ttk.Button(tracking_btn_frame, text="Stop Tracking", 
                                           command=self.stop_tracking, state=tk.DISABLED)
        self.stop_tracking_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_session_btn = ttk.Button(tracking_btn_frame, text="Reset Session", 
                                           command=self.reset_session)
        self.reset_session_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Statistics section
        stats_frame = ttk.LabelFrame(dashboard_frame, text="Current Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Stats grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(stats_grid, text="Total Students:").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.total_students_label = ttk.Label(stats_grid, text="0", font=('Arial', 12, 'bold'))
        self.total_students_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(stats_grid, text="Currently Present:").grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.current_present_label = ttk.Label(stats_grid, text="0", font=('Arial', 12, 'bold'))
        self.current_present_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        # Row 2
        ttk.Label(stats_grid, text="Present Today:").grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.present_today_label = ttk.Label(stats_grid, text="0", font=('Arial', 12, 'bold'))
        self.present_today_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(stats_grid, text="Attendance %:").grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.attendance_percent_label = ttk.Label(stats_grid, text="0%", font=('Arial', 12, 'bold'))
        self.attendance_percent_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        # Session info
        session_frame = ttk.LabelFrame(dashboard_frame, text="Session Information", padding=10)
        session_frame.pack(fill=tk.X, padx=10, pady=10)
        
        session_grid = ttk.Frame(session_frame)
        session_grid.pack(fill=tk.X)
        
        ttk.Label(session_grid, text="Session Duration:").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.session_duration_label = ttk.Label(session_grid, text="00:00:00", font=('Arial', 12, 'bold'))
        self.session_duration_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(session_grid, text="Total Entries:").grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.total_entries_label = ttk.Label(session_grid, text="0", font=('Arial', 12, 'bold'))
        self.total_entries_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(session_grid, text="Total Exits:").grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        self.total_exits_label = ttk.Label(session_grid, text="0", font=('Arial', 12, 'bold'))
        self.total_exits_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
    
    def create_camera_tab(self):
        """Create the camera view tab"""
        camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(camera_frame, text="Camera View")
        
        # Camera view section
        camera_view_frame = ttk.LabelFrame(camera_frame, text="Live Camera Feed", padding=10)
        camera_view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(camera_view_frame, bg='black', width=640, height=480)
        self.camera_canvas.pack(pady=10)
        
        # Camera controls
        camera_controls_frame = ttk.Frame(camera_view_frame)
        camera_controls_frame.pack(fill=tk.X)
        
        # Camera selection
        ttk.Label(camera_controls_frame, text="Camera:").pack(side=tk.LEFT, padx=(0, 10))
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(camera_controls_frame, textvariable=self.camera_var, 
                                        values=["0", "1", "2"], width=5, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        # Refresh camera list
        self.refresh_cameras_btn = ttk.Button(camera_controls_frame, text="Refresh Cameras", 
                                             command=self.refresh_cameras)
        self.refresh_cameras_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Camera settings
        settings_frame = ttk.LabelFrame(camera_frame, text="Camera Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.X)
        
        # Resolution
        ttk.Label(settings_grid, text="Resolution:").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(settings_grid, textvariable=self.resolution_var, 
                                       values=["320x240", "640x480", "1280x720"], width=10, state="readonly")
        resolution_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        # FPS
        ttk.Label(settings_grid, text="FPS:").grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.fps_var = tk.StringVar(value="30")
        fps_combo = ttk.Combobox(settings_grid, textvariable=self.fps_var, 
                                values=["15", "30", "60"], width=5, state="readonly")
        fps_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        # Apply settings button
        self.apply_settings_btn = ttk.Button(settings_grid, text="Apply Settings", 
                                            command=self.apply_camera_settings)
        self.apply_settings_btn.grid(row=0, column=4, padx=(20, 0))
    
    def create_students_tab(self):
        """Create the students management tab"""
        students_frame = ttk.Frame(self.notebook)
        self.notebook.add(students_frame, text="Students")
        
        # Add student section
        add_frame = ttk.LabelFrame(students_frame, text="Add New Student", padding=10)
        add_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Student form
        form_frame = ttk.Frame(add_frame)
        form_frame.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.student_name_var = tk.StringVar()
        self.student_name_entry = ttk.Entry(form_frame, textvariable=self.student_name_var, width=30)
        self.student_name_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(form_frame, text="Student ID:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.student_id_var = tk.StringVar()
        self.student_id_entry = ttk.Entry(form_frame, textvariable=self.student_id_var, width=20)
        self.student_id_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Row 2
        self.capture_photo_btn = ttk.Button(form_frame, text="Capture Photo", 
                                           command=self.capture_student_photo)
        self.capture_photo_btn.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        self.load_photo_btn = ttk.Button(form_frame, text="Load Photo", 
                                        command=self.load_student_photo)
        self.load_photo_btn.grid(row=1, column=2, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Add student button
        self.add_student_btn = ttk.Button(form_frame, text="Add Student", 
                                         command=self.add_student)
        self.add_student_btn.grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        # Student list section
        list_frame = ttk.LabelFrame(students_frame, text="Student List", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Search frame
        search_frame = ttk.Frame(list_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 10))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_btn = ttk.Button(search_frame, text="Search", command=self.search_students)
        self.search_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_search_btn = ttk.Button(search_frame, text="Clear", command=self.clear_search)
        self.clear_search_btn.pack(side=tk.LEFT)
        
        # Student treeview
        columns = ('ID', 'Name', 'Student ID', 'Status')
        self.student_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.student_tree.heading(col, text=col)
            self.student_tree.column(col, width=150)
        
        # Scrollbar
        student_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.student_tree.yview)
        self.student_tree.configure(yscrollcommand=student_scrollbar.set)
        
        self.student_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        student_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(list_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.export_students_btn = ttk.Button(action_frame, text="Export Students", 
                                             command=self.export_students)
        self.export_students_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.delete_student_btn = ttk.Button(action_frame, text="Delete Selected", 
                                            command=self.delete_selected_student)
        self.delete_student_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.refresh_students_btn = ttk.Button(action_frame, text="Refresh List", 
                                              command=self.refresh_student_list)
        self.refresh_students_btn.pack(side=tk.LEFT)
        
        # Load initial student list
        self.refresh_student_list()
    
    def create_attendance_tab(self):
        """Create the attendance tracking tab"""
        attendance_frame = ttk.Frame(self.notebook)
        self.notebook.add(attendance_frame, text="Attendance")
        
        # Real-time attendance section
        realtime_frame = ttk.LabelFrame(attendance_frame, text="Real-time Attendance", padding=10)
        realtime_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Attendance treeview
        columns = ('ID', 'Name', 'Student ID', 'Status', 'Last Update')
        self.attendance_tree = ttk.Treeview(realtime_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=150)
        
        # Scrollbar
        attendance_scrollbar = ttk.Scrollbar(realtime_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=attendance_scrollbar.set)
        
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        attendance_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Attendance log section
        log_frame = ttk.LabelFrame(attendance_frame, text="Attendance Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log treeview
        log_columns = ('Timestamp', 'Student', 'Action')
        self.log_tree = ttk.Treeview(log_frame, columns=log_columns, show='headings', height=10)
        
        for col in log_columns:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=200)
        
        # Log scrollbar
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log controls
        log_controls_frame = ttk.Frame(log_frame)
        log_controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.clear_log_btn = ttk.Button(log_controls_frame, text="Clear Log", 
                                       command=self.clear_attendance_log)
        self.clear_log_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_log_btn = ttk.Button(log_controls_frame, text="Export Log", 
                                        command=self.export_attendance_log)
        self.export_log_btn.pack(side=tk.LEFT)
    
    def create_reports_tab(self):
        """Create the reports tab"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="Reports")
        
        # Daily summary section
        summary_frame = ttk.LabelFrame(reports_frame, text="Daily Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Summary display
        summary_display_frame = ttk.Frame(summary_frame)
        summary_display_frame.pack(fill=tk.X)
        
        # Date selection
        ttk.Label(summary_display_frame, text="Date:").pack(side=tk.LEFT, padx=(0, 10))
        self.report_date_var = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        self.report_date_entry = ttk.Entry(summary_display_frame, textvariable=self.report_date_var, width=15)
        self.report_date_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        self.generate_summary_btn = ttk.Button(summary_display_frame, text="Generate Summary", 
                                             command=self.generate_daily_summary)
        self.generate_summary_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        self.export_summary_btn = ttk.Button(summary_display_frame, text="Export Summary", 
                                            command=self.export_daily_summary)
        self.export_summary_btn.pack(side=tk.LEFT)
        
        # Summary results
        self.summary_text = tk.Text(summary_frame, height=8, width=80)
        self.summary_text.pack(fill=tk.X, pady=(10, 0))
        
        # Attendance history section
        history_frame = ttk.LabelFrame(reports_frame, text="Attendance History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Date range selection
        range_frame = ttk.Frame(history_frame)
        range_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(range_frame, text="From:").pack(side=tk.LEFT, padx=(0, 10))
        self.start_date_var = tk.StringVar(value=(date.today().replace(day=1)).strftime("%Y-%m-%d"))
        self.start_date_entry = ttk.Entry(range_frame, textvariable=self.start_date_var, width=15)
        self.start_date_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(range_frame, text="To:").pack(side=tk.LEFT, padx=(0, 10))
        self.end_date_var = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        self.end_date_entry = ttk.Entry(range_frame, textvariable=self.end_date_var, width=15)
        self.end_date_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        self.generate_history_btn = ttk.Button(range_frame, text="Generate Report", 
                                             command=self.generate_attendance_history)
        self.generate_history_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        self.export_history_btn = ttk.Button(range_frame, text="Export Report", 
                                            command=self.export_attendance_history)
        self.export_history_btn.pack(side=tk.LEFT)
        
        # History results
        self.history_text = tk.Text(history_frame, height=10, width=80)
        self.history_text.pack(fill=tk.BOTH, expand=True)
    
    # Camera control methods
    def start_camera(self):
        """Start the camera"""
        camera_index = int(self.camera_var.get())
        self.camera_module = CameraModule(camera_index)
        
        if self.camera_module.start_camera():
            self.is_camera_active = True
            self.start_camera_btn.config(state=tk.DISABLED)
            self.stop_camera_btn.config(state=tk.NORMAL)
            self.start_tracking_btn.config(state=tk.NORMAL)
            self.camera_info_label.config(text=f"Camera: Active (Index {camera_index})")
            self.status_bar.config(text="Camera started successfully")
            
            # Set camera callback
            self.camera_module.set_callback(self.process_camera_frame)
        else:
            # Offer fallback to video file source
            if messagebox.askyesno("Camera Not Found", "No camera could be opened. Would you like to select a video file as a source?"):
                file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[
                    ("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.wmv"),
                    ("All files", "*.*")
                ])
                if file_path:
                    if self.camera_module.start_with_video_file(file_path):
                        self.is_camera_active = True
                        self.start_camera_btn.config(state=tk.DISABLED)
                        self.stop_camera_btn.config(state=tk.NORMAL)
                        self.start_tracking_btn.config(state=tk.NORMAL)
                        self.camera_info_label.config(text=f"Video: {os.path.basename(file_path)}")
                        self.status_bar.config(text="Video source started successfully")
                        self.camera_module.set_callback(self.process_camera_frame)
                        return
            messagebox.showerror("Error", "Failed to start camera")
    
    def stop_camera(self):
        """Stop the camera"""
        if self.is_tracking_active:
            self.stop_tracking()
        
        self.camera_module.stop_camera()
        self.is_camera_active = False
        self.start_camera_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.DISABLED)
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.stop_tracking_btn.config(state=tk.DISABLED)
        self.camera_info_label.config(text="Camera: Not Active")
        self.status_bar.config(text="Camera stopped")
        
        # Clear camera view
        self.camera_canvas.delete("all")
    
    def process_camera_frame(self, frame):
        """Process camera frame for face recognition"""
        if self.is_tracking_active:
            # Process frame for face recognition
            processed_frame, face_names, face_locations = self.face_recognition_module.process_frame(frame)
            
            # Draw face rectangles and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw rectangle
                cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw name
                cv2.putText(processed_frame, name, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            self.current_frame = processed_frame
        else:
            self.current_frame = frame
    
    def refresh_cameras(self):
        """Refresh available camera list"""
        available_cameras = self.camera_module.list_available_cameras()
        self.camera_combo.config(values=available_cameras)
        if available_cameras:
            self.camera_var.set(str(available_cameras[0]))
    
    def apply_camera_settings(self):
        """Apply camera settings"""
        if not self.is_camera_active:
            messagebox.showwarning("Warning", "Camera must be active to apply settings")
            return
        
        # Parse resolution
        width, height = map(int, self.resolution_var.get().split('x'))
        fps = int(self.fps_var.get())
        
        self.camera_module.set_camera_properties(width, height, fps)
        messagebox.showinfo("Success", "Camera settings applied")
    
    # Attendance tracking methods
    def start_tracking(self):
        """Start attendance tracking"""
        if not self.is_camera_active:
            messagebox.showwarning("Warning", "Camera must be active to start tracking")
            return
        
        if self.attendance_tracker.start_tracking():
            self.is_tracking_active = True
            self.start_tracking_btn.config(state=tk.DISABLED)
            self.stop_tracking_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="Attendance tracking started")
        else:
            messagebox.showerror("Error", "Failed to start attendance tracking")
    
    def stop_tracking(self):
        """Stop attendance tracking"""
        if self.attendance_tracker.stop_tracking():
            self.is_tracking_active = False
            self.start_tracking_btn.config(state=tk.NORMAL)
            self.stop_tracking_btn.config(state=tk.DISABLED)
            self.status_bar.config(text="Attendance tracking stopped")
    
    def reset_session(self):
        """Reset the current tracking session"""
        if messagebox.askyesno("Confirm", "Are you sure you want to reset the session?"):
            self.attendance_tracker.reset_session()
            self.status_bar.config(text="Session reset")
    
    # Student management methods
    def capture_student_photo(self):
        """Capture photo for new student"""
        if not self.is_camera_active:
            messagebox.showwarning("Warning", "Camera must be active to capture photo")
            return
        
        # Set parent for the camera preview window
        self.student_management.parent = self.root
        
        photos = self.student_management.capture_student_photo(self.camera_module)
        if photos is not None:
            self.captured_photos = photos
            messagebox.showinfo("Success", f"Captured {len(photos)} photo(s) successfully")
    
    def load_student_photo(self):
        """Load photo from file for new student"""
        file_path = filedialog.askopenfilename(
            title="Select Student Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            photo = self.student_management.load_student_photo(file_path)
            if photo is not None:
                messagebox.showinfo("Success", "Photo loaded successfully")
    
    def add_student(self):
        """Add new student to the system"""
        name = self.student_name_var.get().strip()
        student_id = self.student_id_var.get().strip()
        
        if not name or not student_id:
            messagebox.showwarning("Warning", "Please enter both name and student ID")
            return
        
        if not self.student_management.validate_student_name(name):
            messagebox.showwarning("Warning", "Invalid student name")
            return
        
        if not self.student_management.validate_student_id(student_id):
            messagebox.showwarning("Warning", "Invalid student ID")
            return
        
        # Check if photos are available
        if not hasattr(self, 'captured_photos') or not self.captured_photos:
            if self.student_management.current_student_image is None:
                messagebox.showwarning("Warning", "Please capture or load a photo first")
                return
            # Use single photo if no multiple photos captured
            photos = [self.student_management.current_student_image]
        else:
            photos = self.captured_photos
        
        if self.student_management.add_student_from_photo(name, student_id, photos):
            # Clear form
            self.student_name_var.set("")
            self.student_id_var.set("")
            self.student_management.current_student_image = None
            if hasattr(self, 'captured_photos'):
                self.captured_photos = None
            
            # Refresh student list
            self.refresh_student_list()
            
            # Reload known faces
            self.load_known_faces()
    
    def search_students(self):
        """Search for students"""
        search_term = self.search_var.get().strip()
        if not search_term:
            self.refresh_student_list()
            return
        
        students = self.student_management.search_student(search_term)
        self.update_student_tree(students)
    
    def clear_search(self):
        """Clear search and show all students"""
        self.search_var.set("")
        self.refresh_student_list()
    
    def refresh_student_list(self):
        """Refresh the student list"""
        students = self.student_management.get_student_list()
        self.update_student_tree(students)
    
    def update_student_tree(self, students):
        """Update the student treeview"""
        # Clear existing items
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)
        
        # Add students
        for student in students:
            status = "Has Photo" if student['face_encoding'] else "No Photo"
            self.student_tree.insert('', 'end', values=(
                student['id'],
                student['name'],
                student['student_id'],
                status
            ))
    
    def delete_selected_student(self):
        """Delete the selected student"""
        selected_item = self.student_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a student to delete")
            return
        
        # Get student ID from selected item
        item_values = self.student_tree.item(selected_item[0])['values']
        student_id = int(item_values[0])  # First column is ID
        
        # Delete the student
        if self.student_management.delete_student(student_id):
            # Refresh the student list
            self.refresh_student_list()
            # Reload known faces
            self.load_known_faces()
            self.status_bar.config(text=f"Student deleted successfully")

    def export_students(self):
        """Export student data"""
        file_path = filedialog.asksaveasfilename(
            title="Export Students",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            if self.student_management.export_student_data(file_path):
                self.status_bar.config(text=f"Students exported to {file_path}")
    
    # Attendance methods
    def update_attendance_display(self):
        """Update the attendance display"""
        if not self.is_tracking_active:
            return
        
        # Get real-time updates
        updates = self.attendance_tracker.get_realtime_updates()
        
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Add current status
        for update in updates:
            self.attendance_tree.insert('', 'end', values=(
                update['id'],
                update['name'],
                update['student_id'],
                update['status'],
                datetime.now().strftime("%H:%M:%S")
            ))
    
    def update_attendance_log(self):
        """Update the attendance log display"""
        if not self.is_tracking_active:
            return
        
        # Get recent log entries
        log_entries = self.attendance_tracker.get_attendance_log(50)
        
        # Clear existing items
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        
        # Add log entries
        for entry in log_entries:
            self.log_tree.insert('', 'end', values=(
                entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                entry['name'],
                entry['action']
            ))
    
    def clear_attendance_log(self):
        """Clear the attendance log display"""
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
    
    def export_attendance_log(self):
        """Export attendance log"""
        file_path = filedialog.asksaveasfilename(
            title="Export Attendance Log",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Timestamp,Student,Action\n")
                    for item in self.log_tree.get_children():
                        values = self.log_tree.item(item)['values']
                        f.write(f"{values[0]},{values[1]},{values[2]}\n")
                
                messagebox.showinfo("Success", f"Attendance log exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export log: {str(e)}")
    
    # Report methods
    def generate_daily_summary(self):
        """Generate daily summary report"""
        try:
            target_date = datetime.strptime(self.report_date_var.get(), "%Y-%m-%d").date()
            summary = self.attendance_tracker.get_daily_summary(target_date)
            
            summary_text = f"""
Daily Attendance Summary for {summary['date']}

Total Students: {summary['total_students']}
Present: {summary['present_count']}
Absent: {summary['absent_count']}
Attendance Percentage: {summary['attendance_percentage']:.1f}%

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary_text)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
    
    def export_daily_summary(self):
        """Export daily summary"""
        file_path = filedialog.asksaveasfilename(
            title="Export Daily Summary",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.summary_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Summary exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export summary: {str(e)}")
    
    def generate_attendance_history(self):
        """Generate attendance history report"""
        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d").date()
            
            if start_date > end_date:
                messagebox.showerror("Error", "Start date must be before end date")
                return
            
            # For now, show a simple message
            history_text = f"""
Attendance History Report
From: {start_date} To: {end_date}

This feature will show detailed attendance history for the selected date range.
Currently showing basic information.

Total Students: {len(self.database.get_all_students())}
Date Range: {end_date - start_date} days

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(1.0, history_text)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
    
    def export_attendance_history(self):
        """Export attendance history"""
        file_path = filedialog.asksaveasfilename(
            title="Export Attendance History",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.history_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"History exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export history: {str(e)}")
    
    # UI update methods
    def update_ui(self):
        """Update the UI elements"""
        try:
            # Update camera view
            if self.current_frame is not None and self.is_camera_active:
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
            
            # Update statistics
            if self.is_tracking_active:
                status = self.attendance_tracker.get_current_attendance_status()
                
                self.total_students_label.config(text=str(status['total_students']))
                self.current_present_label.config(text=str(status['current_present']))
                self.present_today_label.config(text=str(status['present_today']))
                self.attendance_percent_label.config(text=f"{status['attendance_percentage']:.1f}%")
                self.session_duration_label.config(text=status['session_duration'])
                self.total_entries_label.config(text=str(status['total_entries_session']))
                self.total_exits_label.config(text=str(status['total_exits_session']))
            
            # Update attendance displays
            if self.is_tracking_active:
                self.update_attendance_display()
                self.update_attendance_log()
            
        except Exception as e:
            print(f"Error updating UI: {e}")
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_ui)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set application icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Create and run the application
    app = FacialAttendanceSystem(root)
    
    # Handle window close
    def on_closing():
        if app.is_camera_active:
            app.stop_camera()
        if app.is_tracking_active:
            app.stop_tracking()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
