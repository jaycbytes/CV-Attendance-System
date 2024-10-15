import sqlite3
"""
TRIAL
This is a very simple implementation of an attendance db which has a lot of flaws.
Currently the database isn't even being used just stored, there's lots of potential here.
"""
def init_db():
    """Initialize the attendance database."""
    conn = sqlite3.connect('attendance.db') # Connect to the SQLite database (it will create the database if it doesn't exist)
    cursor = conn.cursor()
    # Create a table 'attendance' with columns for name and timestamp if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                      (name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    # Commit changes and close the connection
    conn.commit()
    conn.close()

def mark_attendance(name):
    """Insert a new attendance record into the database."""
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    # Insert a new record into the 'attendance' table with the name and current timestamp
    cursor.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
    # Commit changes and close the connection
    conn.commit()
    conn.close()
