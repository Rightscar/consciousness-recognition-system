#!/usr/bin/env python3
"""
Fine-Tune Data Refinement & Review System - Main Entry Point
============================================================

This is the main entry point for the Fine-Tune Data Refinement & Review System.
It automatically detects the environment and runs the appropriate version.

For Render.com deployment, this file serves as the primary application entry point.
"""

import os
import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def main():
    """Main application entry point"""
    
    # Check if we're running in a production environment (like Render)
    is_production = os.getenv('RENDER') or os.getenv('STREAMLIT_SERVER_HEADLESS')
    
    if is_production:
        # Import and run production version
        try:
            from enhanced_app_production import EnhancedUniversalAITrainingDataCreatorProduction
            app = EnhancedUniversalAITrainingDataCreatorProduction()
            app.run()
        except ImportError as e:
            print(f"Production app import failed: {e}")
            # Fallback to standard version
            from enhanced_app import EnhancedUniversalAITrainingDataCreator
            app = EnhancedUniversalAITrainingDataCreator()
            app.run()
    else:
        # Run standard version for local development
        try:
            from enhanced_app_production import EnhancedUniversalAITrainingDataCreatorProduction
            app = EnhancedUniversalAITrainingDataCreatorProduction()
            app.run()
        except ImportError:
            from enhanced_app import EnhancedUniversalAITrainingDataCreator
            app = EnhancedUniversalAITrainingDataCreator()
            app.run()

if __name__ == "__main__":
    main()

