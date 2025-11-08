"""
Vercel serverless function handler for Flask app.
This file wraps the Flask application for Vercel deployment.
"""

import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Change to parent directory to ensure relative paths work
os.chdir(parent_dir)

# Import the Flask app
from app import app

# Export the app for Vercel
# Vercel Python runtime expects the handler to be callable
handler = app

