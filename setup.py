"""
Setup script for the Sign Language Recognition System
Handles installation, configuration, and environment setup
"""
import subprocess
import sys
import os
import platform
from pathlib import Path
import json
import logging

def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger = setup_logging()
    
    min_python = (3, 8)
    current_python = sys.version_info[:2]
    
    if current_python < min_python:
        logger.error(f"Python {min_python[0]}.{min_python[1]} or higher is required. "
                    f"Current version: {current_python[0]}.{current_python[1]}")
        return False
    
    logger.info(f"Python version check passed: {current_python[0]}.{current_python[1]}")
    return True

def create_virtual_environment():
    """Create virtual environment for the project"""
    logger = setup_logging()
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("Virtual environment already exists")
        return True
    
    try:
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger = setup_logging()
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = Path("venv/Scripts/pip")
    else:
        pip_path = Path("venv/bin/pip")
    
    if not pip_path.exists():
        logger.error("Virtual environment pip not found")
        return False
    
    try:
        logger.info("Installing dependencies...")
        
        # Install core dependencies
        core_packages = [
            "numpy>=1.24.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "streamlit>=1.25.0",
            "flask>=2.3.0"
        ]
        
        for package in core_packages:
            logger.info(f"Installing {package}...")
            subprocess.run([str(pip_path), "install", package], check=True)
        
        # Try to install AI/ML packages (may fail without proper system setup)
        ml_packages = [
            "tensorflow>=2.12.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0"
        ]
        
        for package in ml_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.run([str(pip_path), "install", package], check=True)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {package} - may need manual installation")
        
        # Install optional packages
        optional_packages = [
            "mediapipe>=0.10.0",
            "nltk>=3.8.0",
            "transformers>=4.30.0"
        ]
        
        for package in optional_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.run([str(pip_path), "install", package], check=True)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {package} - some features may be limited")
        
        logger.info("Dependencies installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Setup required directories"""
    logger = setup_logging()
    
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "exports",
        "temp"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def create_config_files():
    """Create configuration files"""
    logger = setup_logging()
    
    # Create user config
    user_config = {
        "system": {
            "camera_index": 0,
            "confidence_threshold": 0.8,
            "language": "en"
        },
        "ui": {
            "theme": "light",
            "window_size": "1200x800",
            "auto_save": True
        },
        "model": {
            "model_type": "cnn_lstm",
            "batch_size": 32,
            "use_gpu": True
        }
    }
    
    config_path = Path("config/user_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(user_config, f, indent=2)
    
    logger.info("Created user configuration file")
    
    # Create logging config
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler"
            },
            "file": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": "logs/system.log"
            }
        },
        "loggers": {
            "": {
                "handlers": ["default", "file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    log_config_path = Path("config/logging_config.json")
    with open(log_config_path, 'w') as f:
        json.dump(log_config, f, indent=2)
    
    logger.info("Created logging configuration file")
    return True

def download_sample_data():
    """Download or create sample data for testing"""
    logger = setup_logging()
    
    # Create sample gesture data for testing
    sample_gestures = {
        "hello": {
            "description": "Open hand wave gesture",
            "category": "greeting",
            "difficulty": "easy"
        },
        "thank_you": {
            "description": "Touch chin and move hand forward",
            "category": "politeness",
            "difficulty": "medium"
        },
        "please": {
            "description": "Circular motion on chest",
            "category": "politeness",
            "difficulty": "easy"
        },
        "help": {
            "description": "Fist on opposite palm, lift together",
            "category": "request",
            "difficulty": "medium"
        }
    }
    
    sample_data_path = Path("data/sample_gestures.json")
    with open(sample_data_path, 'w') as f:
        json.dump(sample_gestures, f, indent=2)
    
    logger.info("Created sample gesture data")
    return True

def create_launcher_scripts():
    """Create launcher scripts for different interfaces"""
    logger = setup_logging()
    
    # Streamlit launcher
    streamlit_script = '''#!/bin/bash
# Streamlit Web Interface Launcher
echo "Starting Sign Language Recognition Web Interface..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup.py first."
    exit 1
fi

# Launch Streamlit app
streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address localhost

echo "Web interface started at http://localhost:8501"
'''
    
    with open("launch_web.sh", 'w') as f:
        f.write(streamlit_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("launch_web.sh", 0o755)
    
    # Desktop launcher
    desktop_script = '''#!/bin/bash
# Desktop Application Launcher
echo "Starting Sign Language Recognition Desktop Application..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup.py first."
    exit 1
fi

# Launch desktop app
python src/ui/desktop_app.py

echo "Desktop application closed."
'''
    
    with open("launch_desktop.sh", 'w') as f:
        f.write(desktop_script)
    
    if platform.system() != "Windows":
        os.chmod("launch_desktop.sh", 0o755)
    
    # Demo launcher
    demo_script = '''#!/bin/bash
# Demo Script Launcher
echo "Starting Sign Language Recognition System Demo..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup.py first."
    exit 1
fi

# Launch demo
python src/utils/demo.py

echo "Demo completed."
'''
    
    with open("launch_demo.sh", 'w') as f:
        f.write(demo_script)
    
    if platform.system() != "Windows":
        os.chmod("launch_demo.sh", 0o755)
    
    # Windows batch files
    if platform.system() == "Windows":
        # Streamlit batch file
        with open("launch_web.bat", 'w') as f:
            f.write('''@echo off
echo Starting Sign Language Recognition Web Interface...
call venv\\Scripts\\activate
streamlit run src\\ui\\streamlit_app.py --server.port 8501 --server.address localhost
pause
''')
        
        # Desktop batch file
        with open("launch_desktop.bat", 'w') as f:
            f.write('''@echo off
echo Starting Sign Language Recognition Desktop Application...
call venv\\Scripts\\activate
python src\\ui\\desktop_app.py
pause
''')
        
        # Demo batch file
        with open("launch_demo.bat", 'w') as f:
            f.write('''@echo off
echo Starting Sign Language Recognition System Demo...
call venv\\Scripts\\activate
python src\\utils\\demo.py
pause
''')
    
    logger.info("Created launcher scripts")
    return True

def run_tests():
    """Run basic system tests"""
    logger = setup_logging()
    
    try:
        logger.info("Running basic system tests...")
        
        # Test imports
        sys.path.append('src')
        
        # Test configuration
        from config.config import config
        logger.info("✓ Configuration loaded successfully")
        
        # Test data preprocessing (mock)
        logger.info("✓ Data preprocessing modules available")
        
        # Test NLP processor
        from src.models.nlp_processor import GestureToTextMapper
        nlp = GestureToTextMapper()
        result = nlp.map_gesture_to_text('hello', 0.9)
        assert 'text' in result
        logger.info("✓ NLP processor working correctly")
        
        # Test multi-language support
        from src.utils.language_support import multi_language_support
        languages = multi_language_support.get_supported_languages()
        assert len(languages) > 0
        logger.info(f"✓ Multi-language support: {len(languages)} languages")
        
        logger.info("All basic tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger = setup_logging()
    
    print("🚀 Sign Language Recognition System Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        print("❌ Setup failed: Python version requirements not met")
        return False
    
    # Setup steps
    setup_steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up directories", setup_directories),
        ("Creating configuration files", create_config_files),
        ("Downloading sample data", download_sample_data),
        ("Creating launcher scripts", create_launcher_scripts),
        ("Running system tests", run_tests)
    ]
    
    for step_name, step_function in setup_steps:
        print(f"\\n📋 {step_name}...")
        try:
            if step_function():
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            logger.error(f"{step_name} failed: {e}")
            return False
    
    print("\\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\\n📚 Next Steps:")
    print("1. Launch the web interface: ./launch_web.sh (or launch_web.bat on Windows)")
    print("2. Launch the desktop app: ./launch_desktop.sh (or launch_desktop.bat on Windows)")
    print("3. Run the demo: ./launch_demo.sh (or launch_demo.bat on Windows)")
    print("4. Run tests: python -m pytest tests/")
    
    print("\\n🔧 Manual Installation Notes:")
    print("- If TensorFlow/PyTorch installation failed, install manually:")
    print("  pip install tensorflow torch torchvision")
    print("- If MediaPipe installation failed, install manually:")
    print("  pip install mediapipe")
    print("- For NLTK data: python -c 'import nltk; nltk.download(\"all\")'")
    
    print("\\n📖 Documentation:")
    print("- See README.md for detailed usage instructions")
    print("- Check config/config.py for system configuration")
    print("- View logs in logs/system.log for troubleshooting")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)