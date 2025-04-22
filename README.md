# Attendance AI

A computer vision-based attendance tracking system for university clubs and organizations.

## Features

- **Face Recognition**: Automatically identify and record attendance of members using facial recognition.
- **Member Management**: Add, edit, and delete member profiles with photos.
- **Meeting Management**: Create, track, and end meetings with detailed attendance records.
- **Data Export**: Export meeting attendance data to CSV for analysis.

## Recent Updates

### Meetings Page Enhancements

- **Bulk Delete**: Select and delete multiple meetings at once
- **Active Meeting View**: Specialized view for the currently active meeting
- **CSV Export**: Export attendance data for any meeting to CSV format
- **UI Improvements**: Better status indicators and styling

## Project Structure

- **app/**: Main application folder
  - **camera/**: Camera access and face recognition components
  - **database/**: Database interactions and models
  - **routes/**: Route handlers for different features
  - **static/**: Static assets (CSS, JavaScript, images)
  - **templates/**: HTML templates
  - **utils/**: Utility functions
- **instance/**: Database files (SQLite)
- **run.py**: Application entry point

## Getting Started

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Initialize the database: `flask --app run.py init-db`
4. Run the application: `python run.py`

## Usage

1. First, add members with their photos
2. Create a new meeting
3. Use the camera interface to record attendance
4. View meeting details and export attendance data

## License

[License information]

## Contributors

[Contributors information]
