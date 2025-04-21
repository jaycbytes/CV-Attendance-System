from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from app.database.meetings import get_active_meeting
from app.database.members import get_all_members

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Home page with active meeting and member stats."""
    active_meeting = get_active_meeting()
    members = get_all_members()
    member_count = len(members)
    
    return render_template('index.html', 
                           active_meeting=active_meeting,
                           member_count=member_count)