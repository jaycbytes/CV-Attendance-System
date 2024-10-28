import sqlite3
from datetime import datetime

"""
This is a very simple implementation of an attendance db which has a lot of flaws.
Currently the database isn't even being used just stored, there's lots of potential here.
"""
def init_db():
    """Initialize the attendance database."""

    # Connect to the SQLite database (it will create the database if it doesn't exist)
    conn = sqlite3.connect('attendance.db') 
    cursor = conn.cursor()

    # Create a table 'attendance' with columns for name and timestamp if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                   (
                   name TEXT, 
                   streak INTEGER DEFAULT 0,
                   total_attendance INTEGER DEFAULT 0,
                   recent_attendance_date TEXT
                   )''')
    

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def mark_attendance(name):

    """Insert a new attendance record into the database."""
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Stores today's date
    today = datetime.now().strftime('%Y-%m-%d')

    # Checks's if student exist in db
    cursor.execute("SELECT streak, total_attendance, recent_attendance_date FROM attendance WHERE name = ?", (name,))
    record = cursor.fetchone()

    # if student alr exist..
    if record:
        streak, total_attendance, recent_attendance_date = record
        
        # Calcualtes days since last meeting (should be 7)
        days_since = (datetime.now() - recent_attendance_date).days

        # checks if it is a streak
        if days_since == 7:
            streak += 1
        else:
            # breaks streak
            streak = 1

        # adds to total attendance
        total_attendance += 1

        # Update the record
        cursor.execute(''' UPDATE attendance 
        SET streak = ?, 
        total_attendance = ?,
        recent_attendance_date = ? 
        WHERE name = ?
        ''', (streak, total_attendance, today, name))

    # add student to db if does not exist alr
    else:
        cursor.execute('''INSERT INTO attendance 
        (name, streak, total_attendance, recent_attendance_date) 
        VALUES (?, 1, 1, ?)
        ''', (name, today))



    # Commit changes and close the connection
    conn.commit()
    conn.close()
