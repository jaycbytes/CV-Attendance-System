# Attendance-AI
Facial recognition attendance tracking for clubs, events, and organizations

## 📖 Overview
Attendance-AI is an open-source facial recognition attendance system designed for college clubs, classrooms, and small organizations. It provides an easy way to track attendance without manual sign-in sheets, using computer vision to automatically identify and record attendees.

## ✨ Key Features

📸 Real-time facial recognition for attendance tracking
👥 Member management with profiles and photos
📅 Meeting management (create, track, end meetings)
📊 Attendance reporting and data export
🔍 Admin dashboard with analytics

## 🚀 Quick Start
### Prerequisites

Python 3.8 or higher
Webcam for facial recognition
Modern web browser
pip (Python package manager)

### Installation

Clone the repository
bashgit clone https://github.com/yourusername/Attendance-AI.git
cd Attendance-AI

Set up a virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Initialize the database
bashflask --app run.py init-db

Start the application
bashpython run.py

Access the application at http://127.0.0.1:5000

## 🔧 Development
### Project Structure
```bash
Attendance-AI/
├── app/                  # Main application package
│   ├── camera/           # Camera and face recognition
│   ├── database/         # Database models and queries
│   ├── routes/           # Flask route handlers
│   ├── static/           # Static assets (CSS, JS, images)
│   └── templates/        # HTML templates
├── tests/                # Unit and integration tests
├── .gitignore            # Git ignore file
├── LICENSE               # License file
├── README.md             # Project README
├── requirements.txt      # Python dependencies
└── run.py                # Application entry point
```

### Contributing
This project is mainly for SMC AI Club members who want to contribute to do so. If anybody else is interested please fork the project or make pull requests.

**Fork the repository**
Create your feature branch
Commit your changes (git commit -m 'added cool feature discussed in club meeting')
Push to the branch git push origin feature/col-feature)
Open a Pull Request

📜 License
This project is licensed under the MIT License

💬 Contact
Your Name - @yourusername - email@example.com
Project Link: https://github.com/yourusername/Attendance-AI

🙏 Acknowledgements
shout out to Chris and Trent for supporting the idea.

face_recognition - Core facial recognition library
Flask - Web framework
OpenCV - Computer vision library
