from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    
    # Register routes
    from app.camera import routes
    app.register_blueprint(routes.bp)
    
    return app