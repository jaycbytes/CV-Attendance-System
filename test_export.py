"""
Test script to verify CSV export functionality for meetings.
Run this script to ensure the CSV export works correctly.
"""

import csv
import sys
import os
import sqlite3
from datetime import datetime
from io import StringIO

# Set up the path to find the app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Connect to the database
def get_db():
    """Get a connection to the database."""
    conn = sqlite3.connect('instance/attendance.sqlite')
    conn.row_factory = sqlite3.Row
    return conn

def get_meeting_attendance(meeting_id):
    """Get all attendance records for a specific meeting."""
    db = get_db()
    attendance = db.execute(
        'SELECT a.id, m.id as member_id, m.name, m.major, m.age, a.timestamp'
        ' FROM attendance a'
        ' JOIN members m ON a.member_id = m.id'
        ' WHERE a.meeting_id = ?'
        ' ORDER BY a.timestamp',
        (meeting_id,)
    ).fetchall()
    return attendance

def get_meeting(meeting_id):
    """Get a meeting by ID."""
    db = get_db()
    meeting = db.execute(
        'SELECT id, title, description, start_time, end_time, created_at'
        ' FROM meetings'
        ' WHERE id = ?',
        (meeting_id,)
    ).fetchone()
    return meeting

def export_attendance_csv(meeting_id):
    """Export meeting attendance to CSV."""
    meeting = get_meeting(meeting_id)
    if not meeting:
        print(f"Error: Meeting ID {meeting_id} not found.")
        return None
    
    attendees = get_meeting_attendance(meeting_id)
    
    # Create CSV in memory
    si = StringIO()
    writer = csv.writer(si)
    
    # Write header
    writer.writerow(['Name', 'Major', 'Age', 'Check-in Time'])
    
    # Write attendee data
    for attendee in attendees:
        writer.writerow([
            attendee['name'],
            attendee['major'] if attendee['major'] else '',
            attendee['age'] if attendee['age'] else '',
            attendee['timestamp']
        ])
    
    return si.getvalue()

def list_meetings():
    """List all meetings in the database."""
    db = get_db()
    meetings = db.execute(
        'SELECT id, title, start_time, end_time FROM meetings ORDER BY start_time DESC'
    ).fetchall()
    
    print("Available meetings:")
    for meeting in meetings:
        status = "Active" if meeting['end_time'] is None else "Ended"
        print(f"ID: {meeting['id']} - {meeting['title']} ({status})")
    
    return meetings

def main():
    """Main function to test CSV export."""
    meetings = list_meetings()
    
    if not meetings:
        print("No meetings found in the database.")
        return
    
    meeting_id = input("\nEnter meeting ID to export (or press Enter to exit): ")
    if not meeting_id:
        return
    
    try:
        meeting_id = int(meeting_id)
    except ValueError:
        print("Invalid meeting ID. Please enter a number.")
        return
    
    csv_content = export_attendance_csv(meeting_id)
    if csv_content:
        meeting = get_meeting(meeting_id)
        filename = f"attendance_{meeting['title'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        with open(filename, 'w') as f:
            f.write(csv_content)
        
        print(f"CSV file '{filename}' has been created successfully.")
        print("\nPreview of CSV content:")
        print("-" * 60)
        print(csv_content[:500] + "..." if len(csv_content) > 500 else csv_content)
        print("-" * 60)

if __name__ == "__main__":
    main()
