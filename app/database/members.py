from app.database import get_db
import numpy as np

def get_all_members():
    """Get all members from the database."""
    db = get_db()
    members = db.execute(
        'SELECT id, name, major, age, bio, image_path, meeting_count, created_at'
        ' FROM members'
        ' ORDER BY name'
    ).fetchall()
    return members

def get_member(member_id):
    """Get a member by ID."""
    db = get_db()
    member = db.execute(
        'SELECT id, name, major, age, bio, image_path, meeting_count, created_at'
        ' FROM members'
        ' WHERE id = ?',
        (member_id,)
    ).fetchone()
    return member

def get_member_by_name(name):
    """Get a member by name."""
    db = get_db()
    member = db.execute(
        'SELECT id, name, major, age, bio, image_path, meeting_count, created_at'
        ' FROM members'
        ' WHERE name = ?',
        (name,)
    ).fetchone()
    return member

def create_member(name, major=None, age=None, bio=None, face_encoding=None, image_path=None):
    """Create a new member."""
    db = get_db()
    cursor = db.execute(
        'INSERT INTO members (name, major, age, bio, face_encoding, image_path)'
        ' VALUES (?, ?, ?, ?, ?, ?)',
        (name, major, age, bio, face_encoding, image_path)
    )
    db.commit()
    return cursor.lastrowid

def update_member(member_id, name=None, major=None, age=None, bio=None, face_encoding=None, image_path=None):
    """Update a member's information."""
    db = get_db()
    
    # Get current values
    member = get_member(member_id)
    if not member:
        return None
    
    # Update with new values or keep current ones
    name = name if name is not None else member['name']
    major = major if major is not None else member['major']
    age = age if age is not None else member['age']
    bio = bio if bio is not None else member['bio']
    image_path = image_path if image_path is not None else member['image_path']
    
    # Get face encoding separately since it's a BLOB
    if face_encoding is None:
        face_encoding_db = db.execute(
            'SELECT face_encoding FROM members WHERE id = ?',
            (member_id,)
        ).fetchone()
        face_encoding = face_encoding_db['face_encoding'] if face_encoding_db else None
    
    db.execute(
        'UPDATE members'
        ' SET name = ?, major = ?, age = ?, bio = ?, face_encoding = ?, image_path = ?'
        ' WHERE id = ?',
        (name, major, age, bio, face_encoding, image_path, member_id)
    )
    db.commit()
    return get_member(member_id)

def delete_member(member_id):
    """Delete a member."""
    db = get_db()
    db.execute('DELETE FROM members WHERE id = ?', (member_id,))
    db.commit()

def increment_meeting_count(member_id):
    """Increment a member's meeting count."""
    db = get_db()
    db.execute(
        'UPDATE members SET meeting_count = meeting_count + 1 WHERE id = ?',
        (member_id,)
    )
    db.commit()

def get_all_face_encodings():
    """Get all face encodings and names for recognition."""
    db = get_db()
    faces = db.execute(
        'SELECT id, name, face_encoding FROM members WHERE face_encoding IS NOT NULL'
    ).fetchall()
    
    encodings = []
    names = []
    member_ids = []
    
    for face in faces:
        encodings.append(face['face_encoding'])
        names.append(face['name'])
        member_ids.append(face['id'])
    
    return encodings, names, member_ids