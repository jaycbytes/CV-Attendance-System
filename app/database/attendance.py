from app.database import get_db
from datetime import datetime
from app.database.members import increment_meeting_count

def record_attendance(member_id, meeting_id=None):
    """Record attendance for a member at a meeting."""
    db = get_db()
    
    # If no meeting ID is provided, use the most recent active meeting
    if meeting_id is None:
        meeting = db.execute(
            'SELECT id FROM meetings WHERE end_time IS NULL ORDER BY start_time DESC LIMIT 1'
        ).fetchone()
        
        if meeting is None:
            return False  # No active meeting
        
        meeting_id = meeting['id']
    
    # Check if this member already has attendance for this meeting
    existing = db.execute(
        'SELECT id FROM attendance WHERE member_id = ? AND meeting_id = ?',
        (member_id, meeting_id)
    ).fetchone()
    
    if existing:
        return False  # Already recorded
    
    # Record attendance
    timestamp = datetime.now()
    db.execute(
        'INSERT INTO attendance (member_id, meeting_id, timestamp)'
        ' VALUES (?, ?, ?)',
        (member_id, meeting_id, timestamp)
    )
    
    # Increment the member's meeting count
    increment_meeting_count(member_id)
    
    db.commit()
    return True

def get_member_attendance(member_id):
    """Get all attendance records for a specific member."""
    db = get_db()
    attendance = db.execute(
        'SELECT a.id, m.title, m.start_time, a.timestamp'
        ' FROM attendance a'
        ' JOIN meetings m ON a.meeting_id = m.id'
        ' WHERE a.member_id = ?'
        ' ORDER BY a.timestamp DESC',
        (member_id,)
    ).fetchall()
    return attendance

def get_meeting_attendance(meeting_id):
    """Get all attendance records for a specific meeting."""
    db = get_db()
    attendance = db.execute(
        'SELECT a.id, m.name, a.timestamp'
        ' FROM attendance a'
        ' JOIN members m ON a.member_id = m.id'
        ' WHERE a.meeting_id = ?'
        ' ORDER BY a.timestamp',
        (meeting_id,)
    ).fetchall()
    return attendance

def delete_attendance(attendance_id):
    """Delete an attendance record."""
    db = get_db()
    
    # Get the member ID to decrement their meeting count
    record = db.execute(
        'SELECT member_id FROM attendance WHERE id = ?',
        (attendance_id,)
    ).fetchone()
    
    if record:
        # Decrement meeting count
        db.execute(
            'UPDATE members SET meeting_count = meeting_count - 1 WHERE id = ?',
            (record['member_id'],)
        )
        
        # Delete the record
        db.execute('DELETE FROM attendance WHERE id = ?', (attendance_id,))
        db.commit()
        return True
    
    return False