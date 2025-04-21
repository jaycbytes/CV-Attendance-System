import os
from flask import Flask

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'attendance.sqlite'),
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'member_images'),
    )

    if test_config is None:
        # Load the instance config, if it exists
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass

    # Register database commands
    from app.database import init_app
    init_app(app)
    
    # Register blueprints
    from app.camera import routes as camera_routes
    app.register_blueprint(camera_routes.bp)
    
    return app