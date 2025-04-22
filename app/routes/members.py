import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import face_recognition
import cv2
import numpy as np
from app.camera.camera import Camera
from app.database.members import (
    get_all_members, get_member, create_member, update_member, delete_member
)

bp = Blueprint('members', __name__, url_prefix='/members')

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
def list():
    """Show all members."""
    members = get_all_members()
    return render_template('members/list.html', members=members)

@bp.route('/view/<int:id>')
def view(id):
    """Show a single member."""
    member = get_member(id)
    if member is None:
        abort(404, f"Member id {id} doesn't exist.")
    return render_template('members/view.html', member=member)

@bp.route('/create', methods=('GET', 'POST'))
def create():
    """Create a new member."""
    if request.method == 'POST':
        name = request.form['name']
        major = request.form['major']
        age = int(request.form['age']) if request.form['age'] else None
        bio = request.form['bio']
        face_encoding = None
        image_path = None
        
        # Handle profile image upload
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Create a unique filename with member name
                file_ext = os.path.splitext(filename)[1]
                filename = f"{name.replace(' ', '_').lower()}{file_ext}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_path = filename
                
                # Generate face encoding from the uploaded image
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                    else:
                        flash('No face detected in the uploaded image.', 'danger')
                except Exception as e:
                    flash(f'Error processing face: {e}', 'danger')
        
        error = None
        
        if not name:
            error = 'Name is required.'
        
        if error is not None:
            flash(error, 'danger')
        else:
            member_id = create_member(name, major, age, bio, face_encoding, image_path)
            flash(f'Member "{name}" was successfully created.', 'success')
            return redirect(url_for('members.view', id=member_id))
            
    return render_template('members/create.html')

@bp.route('/edit/<int:id>', methods=('GET', 'POST'))
def edit(id):
    """Edit a member."""
    member = get_member(id)
    if member is None:
        abort(404, f"Member id {id} doesn't exist.")
        
    if request.method == 'POST':
        name = request.form['name']
        major = request.form['major']
        age = int(request.form['age']) if request.form['age'] else None
        bio = request.form['bio']
        face_encoding = None
        image_path = member['image_path']
        
        # Check for captured image data first (from webcam)
        if 'captured_image_data' in request.form and request.form['captured_image_data']:
            captured_data = request.form['captured_image_data']
            if captured_data and captured_data.startswith('data:image'):
                try:
                    # Extract the base64 data
                    image_data = captured_data.split(',')[1]
                    import base64
                    decoded_data = base64.b64decode(image_data)
                    
                    # Create a unique filename
                    filename = f"{name.replace(' ', '_').lower()}_webcam.jpg"
                    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    
                    # Save the image file
                    with open(filepath, 'wb') as f:
                        f.write(decoded_data)
                    
                    image_path = filename
                    
                    # Generate face encoding from the captured image
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                    else:
                        flash('No face detected in the captured image.', 'warning')
                    
                except Exception as e:
                    flash(f'Error processing captured image: {e}', 'danger')
        
        # Handle profile image upload - only if no webcam image
        elif 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Create a unique filename with member name
                file_ext = os.path.splitext(filename)[1]
                filename = f"{name.replace(' ', '_').lower()}{file_ext}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_path = filename
                
                # Generate face encoding from the uploaded image
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                    else:
                        flash('No face detected in the uploaded image.', 'warning')
                except Exception as e:
                    flash(f'Error processing face: {e}', 'danger')
        
        error = None
        
        if not name:
            error = 'Name is required.'
        
        if error is not None:
            flash(error, 'danger')
        else:
            update_member(id, name, major, age, bio, face_encoding, image_path)
            flash(f'Member "{name}" was successfully updated.', 'success')
            return redirect(url_for('members.view', id=id))
            
    return render_template('members/edit.html', member=member)

@bp.route('/delete/<int:id>', methods=('POST',))
def delete(id):
    """Delete a member."""
    member = get_member(id)
    if member is None:
        abort(404, f"Member id {id} doesn't exist.")
        
    delete_member(id)
    flash(f'Member "{member["name"]}" was successfully deleted.', 'success')
    return redirect(url_for('members.list'))

@bp.route('/enroll', methods=('GET', 'POST'))
def enroll():
    """Enroll a member from the camera feed."""
    if request.method == 'POST':
        name = request.form['name']
        major = request.form['major']
        age = int(request.form['age']) if request.form['age'] else None
        bio = request.form['bio']
        
        # Process the face data from the recognition result
        face_encoding_str = request.form.get('face_encoding')
        face_encoding = None
        
        if face_encoding_str:
            try:
                # Convert the string representation back to a numpy array
                face_encoding = np.fromstring(face_encoding_str, sep=',')
            except Exception as e:
                flash(f'Error processing face data: {e}', 'danger')
                
        # Save the camera frame as the profile image
        image_path = None
        frame_data = request.form.get('frame_data')
        
        if frame_data:
            try:
                # Create a unique filename with member name
                filename = f"{name.replace(' ', '_').lower()}.jpg"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                
                # Save the image data
                with open(filepath, 'wb') as f:
                    f.write(frame_data.encode('latin1'))
                
                image_path = filename
            except Exception as e:
                flash(f'Error saving image: {e}', 'danger')
        
        error = None
        
        if not name:
            error = 'Name is required.'
        
        if error is not None:
            flash(error, 'danger')
        else:
            member_id = create_member(name, major, age, bio, face_encoding, image_path)
            flash(f'Member "{name}" was successfully enrolled.', 'success')
            return redirect(url_for('members.view', id=member_id))
            
    return render_template('members/enroll.html')