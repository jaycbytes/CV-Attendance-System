from flask import Blueprint, render_template, Response, request, jsonify, current_app
from app.camera.camera import Camera, list_available_cameras
from app.database.members import create_member, get_member_by_name, update_member, get_member
from app.database.attendance import record_attendance
from app.database.meetings import get_active_meeting
import os
import traceback
import time
from datetime import datetime
import cv2
import face_recognition

bp = Blueprint('camera', __name__, url_prefix='/camera')

# Store active camera
active_camera = None

@bp.route('/')
def index():
    """Video streaming home page."""
    # Get active meeting information
    active_meeting = get_active_meeting()
    return render_template('camera/index.html', active_meeting=active_meeting)

@bp.route('/stream')
def stream():
    """Video streaming route."""
    global active_camera
    camera_id = request.args.get('id', default=0, type=int)
    show_faces = request.args.get('show_faces', default='false', type=str).lower() == 'true'
    
    # Initialize camera if not active or different camera selected
    if active_camera is None or active_camera.camera_id != camera_id:
        if active_camera:
            active_camera.stop()
        active_camera = Camera(camera_id=camera_id, recognition_enabled=False)
        try:
            active_camera.start()
        except RuntimeError:
            return "Camera not available", 404
    
    return Response(generate_frames(active_camera, show_faces),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/stop', methods=['POST'])
def stop_camera():
    """Stop the active camera."""
    global active_camera
    if active_camera:
        active_camera.stop()
        active_camera = None
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No active camera"})

@bp.route('/start', methods=['POST'])
def start_camera():
    """Start a camera with the given ID."""
    global active_camera
    camera_id = request.json.get('id', 0)
    
    # Stop current camera if active
    if active_camera:
        active_camera.stop()
    
    # Start new camera
    try:
        active_camera = Camera(camera_id=camera_id, recognition_enabled=False)
        active_camera.start()
        return jsonify({"success": True})
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)})

@bp.route('/recognition/toggle', methods=['POST'])
def toggle_recognition():
    """Toggle face recognition."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    enabled = request.json.get('enabled', None)
    if enabled is not None:
        enabled = bool(enabled)
    
    result = active_camera.toggle_recognition(enabled)
    return jsonify({"recognition_enabled": result})

@bp.route('/recognition/result')
def recognition_result():
    """Get the latest recognition result."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    if not active_camera.recognition_enabled:
        return jsonify({"error": "Face recognition is not enabled"}), 400
    
    result = active_camera.get_recognition_result()
    if result is None:
        return jsonify({"error": "No recognition results available"}), 404
    
    # Get active meeting information
    active_meeting = get_active_meeting()
    meeting_info = None
    if active_meeting:
        meeting_info = {
            "id": active_meeting["id"],
            "title": active_meeting["title"],
            "started": active_meeting["start_time"].strftime('%Y-%m-%d %H:%M')
        }
    
    # Add meeting info to the result
    result["active_meeting"] = meeting_info
    
    return jsonify(result)

# Helper function to record attendance for newly enrolled members
def record_new_member_attendance(member_id):
    """Record attendance for a newly enrolled member if there's an active meeting."""
    active_meeting = get_active_meeting()
    attendance_recorded = False
    if active_meeting:
        try:
            record_attendance(member_id, active_meeting['id'])
            attendance_recorded = True
        except Exception as e:
            print(f"Error recording attendance for new member: {e}")
    return attendance_recorded

@bp.route('/recognition/enroll', methods=['POST'])
def enroll_face():
    """Enroll a new face for recognition."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    # Get all required data for member creation
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    name = data.get('name')
    major = data.get('major', None)
    age = data.get('age', None)
    bio = data.get('bio', None)
    face_index = data.get('face_index')  # This could be an ID now
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    # Check if we already have a member with this name
    existing_member = get_member_by_name(name)
    if existing_member:
        return jsonify({"error": f"A member named '{name}' already exists. Please use a different name."}), 400
    
    try:
        # Get face data from the face processor
        face_processor = active_camera.face_processor
        
        # Look for the specified face ID in the unknown faces
        face_image = None
        face_encoding = None
        
        for face in face_processor.persistent_faces["unknown"]:
            if face["id"] == face_index:
                face_image = face["image"]
                face_encoding = face["encoding"]
                break
                
        # If we couldn't find the face, return an error
        if face_image is None:
            return jsonify({"error": "Face not found. The face may have been removed or the system restarted."}), 404
            
        if face_encoding is None:
            return jsonify({"error": "Face encoding not available. Please ensure the face is in view."}), 400
        
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.replace(' ', '_').lower()}_{timestamp}.jpg"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        cv2.imwrite(filepath, face_image)
        
        # Save to database and get member ID
        member_id = create_member(name, major, age, bio, face_encoding, filename)
        
        # Reload faces from the database to include the new member
        face_processor.load_known_faces_from_db()
        
        # Add to persistent known faces - this will make them appear in recognized members list
        # since they're obviously present in the current session
        face_processor.persistent_faces["known"][name] = {
            "image": face_image,
            "in_view": True,
            "last_seen": time.time(),
            "member_id": member_id,
            "encoding": face_encoding
        }
        
        # Remove from unknown faces
        face_processor.persistent_faces["unknown"] = [
            face for face in face_processor.persistent_faces["unknown"] 
            if face["id"] != face_index
        ]
        
        # Record attendance for the newly enrolled member
        attendance_recorded = record_new_member_attendance(member_id)
        
        # Force an immediate update of the recognition result
        active_camera.process_this_frame = True
        
        return jsonify({
            "success": True, 
            "message": f"Enrolled {name} successfully" + (" and recorded attendance" if attendance_recorded else ""), 
            "member_id": member_id,
            "attendance_recorded": attendance_recorded
        })
    except Exception as e:
        print(f"Error in enroll_face: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Enrollment failed: {str(e)}"}), 500

@bp.route('/list')
def camera_list():
    """List available cameras."""
    start_time = time.time()
    print(f"Camera list requested at {time.strftime('%H:%M:%S')}")
    
    # Use the optimized camera detection with reasonable max index (10)
    cameras = list_available_cameras(max_cameras=10)
    
    elapsed = time.time() - start_time
    print(f"Camera list scan completed in {elapsed:.2f} seconds. Found {len(cameras)} cameras.")
    
    return jsonify({"cameras": cameras, "scan_time": f"{elapsed:.2f} seconds"})

@bp.route('/info')
def camera_info():
    """Get info about the currently active camera."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"})
    
    return jsonify(active_camera.get_camera_properties())

@bp.route('/status')
def camera_status():
    """Check if camera is active."""
    global active_camera
    return jsonify({"active": active_camera is not None})
    
@bp.route('/face_thumbnail/<thumbnail_id>')
def face_thumbnail(thumbnail_id):
    """Get a thumbnail of a specific face by its ID."""
    global active_camera
    if active_camera is None:
        return "No active camera", 404
    
    thumbnail = active_camera.get_face_thumbnail(thumbnail_id)
    if thumbnail:
        response = Response(thumbnail, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response
    
    return "Face thumbnail not found", 404

@bp.route('/member_info/<int:member_id>')
def member_info(member_id):
    """Get member information for the welcome card."""
    try:
        member = get_member(member_id)
        if member:
            # Return member details
            return jsonify({
                "name": member['name'],
                "major": member['major'],
                "age": member['age'],
                "bio": member['bio'],
                "meeting_count": member['meeting_count'],
                "image_path": member['image_path']
            })
        else:
            return jsonify({"error": "Member not found"}), 404
    except Exception as e:
        print(f"Error getting member info: {e}")
        return jsonify({"error": f"Failed to get member info: {str(e)}"}), 500

@bp.route('/member/update', methods=['POST'])
def update_member_info():
    """Update an existing member's information."""
    global active_camera
    
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    # Get all required data for member update
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    member_id = data.get('member_id')
    name = data.get('name')
    major = data.get('major', None)
    age = data.get('age', None)
    bio = data.get('bio', None)
    
    if not member_id or not name:
        return jsonify({"error": "Member ID and name are required"}), 400
    
    try:
        # Update member in database
        updated_member = update_member(
            member_id=member_id,
            name=name,
            major=major,
            age=age,
            bio=bio
        )
        
        if not updated_member:
            return jsonify({"error": "Member not found or could not be updated"}), 404
        
        # Reload faces from database to refresh the data
        active_camera.face_processor.load_known_faces_from_db()
        
        # Also need to update any persistent faces if present
        face_processor = active_camera.face_processor
        
        # Find any known faces with this member ID
        old_name = None
        for face_name in list(face_processor.persistent_faces["known"].keys()):
            face_data = face_processor.persistent_faces["known"][face_name]
            if face_data.get("member_id") == member_id:
                old_name = face_name
                break
                
        # If found and name changed, update the entry
        if old_name and old_name != name:
            face_data = face_processor.persistent_faces["known"].pop(old_name)
            face_processor.persistent_faces["known"][name] = face_data
        
        return jsonify({
            "success": True,
            "message": f"Updated {name} successfully",
            "member_id": member_id
        })
    except Exception as e:
        print(f"Error in update_member: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Update failed: {str(e)}"}), 500

@bp.route('/reset', methods=['POST'])
def reset_camera():
    """Reset the camera recognition state."""
    global active_camera
    if active_camera is None:
        return jsonify({"success": False, "error": "No active camera"}), 400
    
    # Reset recognition state
    active_camera.reset_recognition_state()
    
    return jsonify({"success": True, "message": "Camera recognition state reset"})

@bp.route('/record_all_attendance', methods=['POST'])
def record_all_attendance():
    """Record attendance for all members in the recognized members list."""
    global active_camera
    if active_camera is None:
        return jsonify({"success": False, "error": "No active camera"}), 400
    
    # Get active meeting
    active_meeting = get_active_meeting()
    if not active_meeting:
        return jsonify({"success": False, "error": "No active meeting"}), 400
    
    try:
        # Record attendance for all recognized faces
        recorded_count = active_camera.face_processor.record_attendance(active_meeting['id'])
        
        return jsonify({
            "success": True, 
            "message": f"Recorded attendance for {recorded_count} members",
            "recorded_count": recorded_count
        })
    except Exception as e:
        print(f"Error in record_all_attendance: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/remove_unknown_face/<face_id>', methods=['POST'])
def remove_unknown_face(face_id):
    """Remove an unknown face from the persistent faces list."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    try:
        # Get the face processor
        face_processor = active_camera.face_processor
        
        # Check if the face exists before removing
        before_count = len(face_processor.persistent_faces["unknown"])
        
        # Filter out the face with the given ID
        face_processor.persistent_faces["unknown"] = [
            face for face in face_processor.persistent_faces["unknown"] 
            if face["id"] != face_id
        ]
        
        after_count = len(face_processor.persistent_faces["unknown"])
        
        if before_count == after_count:
            return jsonify({"success": False, "message": "Face not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Face removed from gallery"
        })
    except Exception as e:
        print(f"Error removing unknown face: {e}")
        return jsonify({"error": f"Failed to remove face: {str(e)}"}), 500

def generate_frames(camera, show_faces=False):
    """Generate frames from the camera for streaming."""
    try:
        while True:
            frame = camera.get_frame(show_faces=show_faces)
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {e}")
        # Don't stop the camera here, as it's now managed by the routes
