from flask import Blueprint, render_template, Response, request, jsonify, current_app
from app.camera.camera import Camera, list_available_cameras
from app.database.members import create_member, get_member_by_name
from app.database.attendance import record_attendance
from app.database.meetings import get_active_meeting
import os
import traceback
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
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    # Check if we already have a member with this name
    existing_member = get_member_by_name(name)
    if existing_member:
        return jsonify({"error": f"A member named '{name}' already exists. Please use a different name."}), 400
    
    try:
        # Check if we have a frame
        if active_camera.frame is None:
            return jsonify({"error": "No camera frame available"}), 400
        
        # Create a copy of the current frame
        frame = active_camera.frame.copy()
        
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.replace(' ', '_').lower()}_{timestamp}.jpg"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        
        # Get face encoding from the current frame
        if active_camera.face_encodings and active_camera.face_locations:
            # Use the first detected face (assuming it's the person being enrolled)
            if len(active_camera.face_encodings) > 0:
                encoding = active_camera.face_encodings[0]
                
                # Save to database and get member ID
                member_id = create_member(name, major, age, bio, encoding, filename)
                
                # Add to known faces in memory for immediate recognition
                active_camera.known_face_encodings.append(encoding)
                active_camera.known_face_names.append(name)
                active_camera.known_face_ids.append(member_id)
                
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
            else:
                return jsonify({"error": "No face encodings available"}), 400
        else:
            # If no face detected, try to detect one from the saved image
            try:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if not face_locations:
                    return jsonify({"error": "No faces detected in the current frame"}), 400
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                if not face_encodings:
                    return jsonify({"error": "Could not generate face encoding"}), 400
                
                # Use the first face encoding (assuming it's the person being enrolled)
                encoding = face_encodings[0]
                
                # Save to database and get member ID
                member_id = create_member(name, major, age, bio, encoding, filename)
                
                # Add to known faces in memory for immediate recognition
                active_camera.known_face_encodings.append(encoding)
                active_camera.known_face_names.append(name)
                active_camera.known_face_ids.append(member_id)
                
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
                print(f"Face detection error: {e}")
                traceback.print_exc()
                return jsonify({"error": f"Error processing face: {str(e)}"}), 500
    except Exception as e:
        print(f"Error in enroll_face: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Enrollment failed: {str(e)}"}), 500

@bp.route('/list')
def camera_list():
    """List available cameras."""
    cameras = list_available_cameras()
    return jsonify({"cameras": cameras})

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

def generate_frames(camera, show_faces=False):
    """Generate frames from the camera."""
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