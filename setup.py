#!/usr/bin/env python3
"""
Setup script to create the project directory structure
"""

import os
import sys

def create_directory_structure():
    """
    Create the project directory structure.
    """
    # Create main directories
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'utils',
        'experiments',
        'notebooks',
        'results',
        'saved_models',
        'extensions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create an empty __init__.py file in each directory
        with open(os.path.join(directory, '__init__.py'), 'w') as f:
            f.write("# Package initialization file\n")

def main():
    print("Creating directory structure...")
    create_directory_structure()
    print("Directory structure created successfully.")

if __name__ == "__main__":
    main()