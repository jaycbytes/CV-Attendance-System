from flask import Blueprint, render_template, Response, request, jsonify
from app.camera.camera import Camera, list_available_cameras

bp = Blueprint('camera', __name__, url_prefix='/camera')

# Store active camera
active_camera = None

@bp.route('/')
def index():
    """Video streaming home page."""
    return render_template('camera/index.html')

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
    
    return jsonify(result)

@bp.route('/recognition/enroll', methods=['POST'])
def enroll_face():
    """Enroll a new face for recognition."""
    global active_camera
    if active_camera is None:
        return jsonify({"error": "No active camera"}), 400
    
    # Get the name for the face
    name = request.json.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    # Simple enrollment for testing - in production you'd save this to a database
    result = active_camera.get_recognition_result()
    if not result or not result.get('faces'):
        return jsonify({"error": "No faces detected for enrollment"}), 400
    
    # Get the current face encoding (just use the first face for simplicity)
    face_location = result['faces'][0]['location']
    
    # We need to capture the encoding from the current frame
    if active_camera.face_encodings and active_camera.face_locations:
        # Find the matching encoding based on location
        for i, loc in enumerate(active_camera.face_locations):
            if loc == face_location:
                encoding = active_camera.face_encodings[i]
                
                # Add to known faces
                active_camera.known_face_encodings.append(encoding)
                active_camera.known_face_names.append(name)
                active_camera.known_face_ids.append(None)  # Adding None as we don't save to database in this simple example
                
                # Force an immediate update of the recognition result
                active_camera.process_this_frame = True
                
                return jsonify({"success": True, "message": f"Enrolled {name} successfully"})
    
    return jsonify({"error": "Could not enroll face"}), 400

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