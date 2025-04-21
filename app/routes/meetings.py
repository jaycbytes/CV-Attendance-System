from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from datetime import datetime
from app.database.meetings import (
    get_all_meetings, get_meeting, create_meeting, update_meeting, 
    delete_meeting, end_meeting, get_active_meeting
)
from app.database.attendance import get_meeting_attendance
from app.database.members import get_member

bp = Blueprint('meetings', __name__, url_prefix='/meetings')

@bp.route('/')
def list():
    """Show all meetings."""
    meetings = get_all_meetings()
    return render_template('meetings/list.html', meetings=meetings)

@bp.route('/view/<int:id>')
def view(id):
    """Show a single meeting with attendance."""
    meeting = get_meeting(id)
    if meeting is None:
        abort(404, f"Meeting id {id} doesn't exist.")
        
    # Get attendees for this meeting
    attendees = get_meeting_attendance(id)
    
    return render_template('meetings/view.html', meeting=meeting, attendees=attendees)

@bp.route('/create', methods=('GET', 'POST'))
def create():
    """Create a new meeting."""
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        
        error = None
        
        if not title:
            error = 'Title is required.'
        
        if error is not None:
            flash(error, 'danger')
        else:
            meeting_id = create_meeting(title, description)
            flash(f'Meeting "{title}" was successfully created and started.', 'success')
            return redirect(url_for('meetings.view', id=meeting_id))
            
    return render_template('meetings/create.html')

@bp.route('/edit/<int:id>', methods=('GET', 'POST'))
def edit(id):
    """Edit a meeting."""
    meeting = get_meeting(id)
    if meeting is None:
        abort(404, f"Meeting id {id} doesn't exist.")
        
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        
        error = None
        
        if not title:
            error = 'Title is required.'
        
        if error is not None:
            flash(error, 'danger')
        else:
            update_meeting(id, title, description)
            flash(f'Meeting "{title}" was successfully updated.', 'success')
            return redirect(url_for('meetings.view', id=id))
            
    return render_template('meetings/edit.html', meeting=meeting)

@bp.route('/delete/<int:id>', methods=('POST',))
def delete(id):
    """Delete a meeting."""
    meeting = get_meeting(id)
    if meeting is None:
        abort(404, f"Meeting id {id} doesn't exist.")
        
    delete_meeting(id)
    flash(f'Meeting "{meeting["title"]}" was successfully deleted.', 'success')
    return redirect(url_for('meetings.list'))

@bp.route('/end/<int:id>', methods=('POST',))
def end(id):
    """End a meeting."""
    meeting = get_meeting(id)
    if meeting is None:
        abort(404, f"Meeting id {id} doesn't exist.")
        
    if meeting['end_time'] is not None:
        flash(f'Meeting "{meeting["title"]}" is already ended.', 'info')
    else:
        end_meeting(id)
        flash(f'Meeting "{meeting["title"]}" was successfully ended.', 'success')
        
    return redirect(url_for('meetings.view', id=id))

@bp.route('/active')
def active():
    """Show the currently active meeting, if any."""
    meeting = get_active_meeting()
    
    if meeting is None:
        flash('There is no active meeting.', 'info')
        return redirect(url_for('meetings.list'))
        
    # Get attendees for this meeting
    attendees = get_meeting_attendance(meeting['id'])
    
    return render_template('meetings/active.html', meeting=meeting, attendees=attendees)