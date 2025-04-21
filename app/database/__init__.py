import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext
import os
import pickle
import numpy as np

def get_db():
    """Connect to the application's configured database."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e=None):
    """Close the database connection."""
    db = g.pop('db', None)
    
    if db is not None:
        db.close()

def init_db():
    """Initialize the database."""
    db = get_db()
    
    # Register adapter and converter for numpy arrays
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    
    # Create tables
    with current_app.open_resource('database/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

# Define functions to convert between numpy arrays and binary data
def adapt_array(arr):
    """Convert numpy array to binary for SQLite storage."""
    return pickle.dumps(arr)

def convert_array(binary):
    """Convert binary data back to numpy array."""
    return pickle.loads(binary)

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    # Create instance folder if it doesn't exist
    os.makedirs(current_app.instance_path, exist_ok=True)
    
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    """Register database functions with the Flask app."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)