from flask import Blueprint, render_template, Response
from app.camera.camera import Camera

bp = Blueprint('camera', __name__, url_prefix='/camera')

@bp.route('/')
def index():
    """Video streaming home page."""
    return render_template('camera/index.html')

@bp.route('/stream')
def stream():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate frames from the camera."""
    camera = Camera()
    try:
        camera.start()
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.stop()