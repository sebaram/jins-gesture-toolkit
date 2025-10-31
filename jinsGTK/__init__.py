from __future__ import annotations

def create_app():
    # Import the Flask app instance defined in `flask_main.py`
    from .flask_main import app as flask_app
    return flask_app

__all__ = ["create_app"]


