from app.database import get_db
from datetime import datetime

def get_all_meetings():
    """Get all meetings from the database."""
    db = get_db()
    meetings = db.execute(
        'SELECT id, title, description, start_time, end_time, created_at'
        ' FROM meetings'
        ' ORDER BY start_time DESC'
    ).fetchall()
    return meetings

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

def get_active_meeting():
    """Get the currently active meeting (started but not ended)."""
    db = get_db()
    meeting = db.execute(
        'SELECT id, title, description, start_time, end_time, created_at'
        ' FROM meetings'
        ' WHERE start_time IS NOT NULL AND end_time IS NULL'
        ' ORDER BY start_time DESC'
        ' LIMIT 1'
    ).fetchone()
    return meeting

def create_meeting(title, description=None):
    """Create a new meeting and set its start time to now."""
    db = get_db()
    start_time = datetime.now()
    cursor = db.execute(
        'INSERT INTO meetings (title, description, start_time)'
        ' VALUES (?, ?, ?)',
        (title, description, start_time)
    )
    db.commit()
    return cursor.lastrowid

def end_meeting(meeting_id):
    """End a meeting by setting its end time."""
    db = get_db()
    end_time = datetime.now()
    db.execute(
        'UPDATE meetings SET end_time = ? WHERE id = ?',
        (end_time, meeting_id)
    )
    db.commit()
    return get_meeting(meeting_id)

def update_meeting(meeting_id, title=None, description=None):
    """Update a meeting's information."""
    db = get_db()
    
    # Get current values
    meeting = get_meeting(meeting_id)
    if not meeting:
        return None
    
    # Update with new values or keep current ones
    title = title if title is not None else meeting['title']
    description = description if description is not None else meeting['description']
    
    db.execute(
        'UPDATE meetings SET title = ?, description = ? WHERE id = ?',
        (title, description, meeting_id)
    )
    db.commit()
    return get_meeting(meeting_id)

def delete_meeting(meeting_id):
    """Delete a meeting."""
    db = get_db()
    db.execute('DELETE FROM meetings WHERE id = ?', (meeting_id,))
    db.commit()

# get_meeting_attendance moved to attendance.py