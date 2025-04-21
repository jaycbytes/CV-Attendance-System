from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app
)
from werkzeug.exceptions import abort
from app.database import get_db
from app.database.members import get_all_members
from app.database.meetings import get_all_meetings, get_active_meeting
import os

bp = Blueprint('admin', __name__, url_prefix='/admin')

@bp.route('/dashboard')
def dashboard():
    """Admin dashboard with overview statistics."""
    db = get_db()
    
    # Get counts
    member_count = db.execute('SELECT COUNT(*) as count FROM members').fetchone()['count']
    meeting_count = db.execute('SELECT COUNT(*) as count FROM meetings').fetchone()['count']
    attendance_count = db.execute('SELECT COUNT(*) as count FROM attendance').fetchone()['count']
    
    # Get active meeting
    active_meeting = get_active_meeting()
    
    # Get recent attendees
    recent_attendees = db.execute(
        'SELECT m.name, m.major, a.timestamp, mt.title as meeting_title'
        ' FROM attendance a'
        ' JOIN members m ON a.member_id = m.id'
        ' JOIN meetings mt ON a.meeting_id = mt.id'
        ' ORDER BY a.timestamp DESC LIMIT 5'
    ).fetchall()
    
    # Get top attendees
    top_attendees = db.execute(
        'SELECT id, name, major, meeting_count'
        ' FROM members'
        ' ORDER BY meeting_count DESC LIMIT 5'
    ).fetchall()
    
    return render_template('admin/dashboard.html', 
                           member_count=member_count,
                           meeting_count=meeting_count,
                           attendance_count=attendance_count,
                           active_meeting=active_meeting,
                           recent_attendees=recent_attendees,
                           top_attendees=top_attendees)

@bp.route('/database')
def database():
    """Database management page."""
    # Get database file size
    db_path = current_app.config['DATABASE']
    db_size = 0
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path) / (1024 * 1024)  # Size in MB
    
    return render_template('admin/database.html', db_size=db_size)

@bp.route('/reset-database', methods=('POST',))
def reset_database():
    """Reset the database."""
    from app.database import init_db
    
    try:
        init_db()
        flash('Database has been reset successfully.', 'success')
    except Exception as e:
        flash(f'Error resetting database: {e}', 'danger')
    
    return redirect(url_for('admin.database'))