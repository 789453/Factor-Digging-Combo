"""
Pytest configuration and test utilities
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ['PYTHONPATH'] = str(project_root)

# Configure pytest to handle the project structure
def pytest_configure(config):
    """Configure pytest for the project"""
    # Add any pytest configuration here
    pass