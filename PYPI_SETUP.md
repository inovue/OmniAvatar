# PyPI Publication Setup for OmniAvatar

This document provides step-by-step instructions for publishing OmniAvatar to PyPI.

## 📦 Package Structure

The package has been configured with the following structure:

```
OmniAvatar/
├── pyproject.toml          # Modern Python package configuration
├── MANIFEST.in            # Files to include in distribution
├── README.md              # Package documentation
├── LICENSE.txt            # License file
├── requirements.txt       # Dependencies
├── OmniAvatar/
│   ├── __init__.py       # Main package init with version info
│   ├── wan_video.py      # Core pipeline implementation
│   ├── base.py           # Base classes
│   ├── models/           # Model implementations
│   ├── prompters/        # Prompt handling
│   ├── schedulers/       # Scheduling algorithms
│   ├── utils/            # Utility functions
│   ├── vram_management/  # Memory management
│   ├── distributed/      # Distributed computing
│   └── configs/          # Configuration files
└── scripts/
    ├── build_and_upload.py  # Automated build/upload script
    └── install_dev.py       # Development installation script
```

## 🚀 Quick Start

### 1. Install Build Dependencies

```bash
pip install build twine setuptools wheel setuptools_scm
```

### 2. Build the Package

```bash
# Using the automated script (recommended)
python scripts/build_and_upload.py --build

# Or manually
python -m build
```

### 3. Upload to PyPI

#### Test on TestPyPI first (recommended):

```bash
# Create account at https://test.pypi.org/
# Generate API token in account settings

# Upload to TestPyPI
python scripts/build_and_upload.py --test

# Or manually
python -m twine upload --repository testpypi dist/*
```

#### Upload to PyPI:

```bash
# Create account at https://pypi.org/
# Generate API token in account settings

# Upload to PyPI  
python scripts/build_and_upload.py --prod

# Or manually
python -m twine upload dist/*
```

## 🔑 Authentication Setup

### Option 1: API Tokens (Recommended)

1. Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. Generate API tokens in account settings
3. Use tokens with twine:

```bash
# For TestPyPI
python -m twine upload --repository testpypi dist/* -u __token__ -p <your-testpypi-token>

# For PyPI
python -m twine upload dist/* -u __token__ -p <your-pypi-token>
```

### Option 2: Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-api-token>

# Then upload without credentials
python -m twine upload dist/*
```

### Option 3: .pypirc Configuration

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-token>
```

## 📋 Pre-Publication Checklist

- [ ] Update version in `OmniAvatar/__init__.py`
- [ ] Update README.md with installation instructions
- [ ] Verify all dependencies in `pyproject.toml`
- [ ] Test package builds without errors: `python -m build`
- [ ] Validate package: `twine check dist/*`
- [ ] Test installation from TestPyPI
- [ ] Verify imports work correctly
- [ ] Update documentation links if needed

## 🔍 Testing Installation

### From TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ omniavatar
```

### From PyPI (after publication):

```bash
pip install omniavatar
```

### Test Import:

```python
import omniavatar
from omniavatar import WanVideoPipeline

print(f"OmniAvatar version: {omniavatar.__version__}")

# Initialize pipeline (requires model weights)
# pipeline = WanVideoPipeline()
```

## 🔧 Development Setup

For development work:

```bash
# Clone the repository
git clone https://github.com/Omni-Avatar/OmniAvatar
cd OmniAvatar

# Install in development mode
python scripts/install_dev.py

# Or manually
pip install -e .[dev]
```

## 📝 Version Management

Update version in `OmniAvatar/__init__.py`:

```python
__version__ = "0.1.1"  # Update this for new releases
```

Consider using semantic versioning:
- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features
- `1.0.0` - Stable release

## 🚨 Common Issues

### Build Errors

1. **Missing files**: Check MANIFEST.in includes all necessary files
2. **Import errors**: Ensure all dependencies are listed in pyproject.toml
3. **Large files**: Model weights should not be included in package

### Upload Errors

1. **Authentication**: Verify API tokens are correct
2. **Version exists**: Cannot upload same version twice, increment version
3. **File size**: PyPI has size limits, large models should be downloaded separately

### Installation Issues

1. **Dependencies**: Users need to install PyTorch separately for GPU support
2. **Model weights**: Provide clear instructions for downloading required models
3. **System requirements**: Document minimum Python version and OS requirements

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [TestPyPI](https://test.pypi.org/)
- [Twine Documentation](https://twine.readthedocs.io/)

## 🎯 Next Steps After Publication

1. Update project README with PyPI installation instructions
2. Create GitHub releases corresponding to PyPI versions
3. Monitor PyPI statistics and user feedback
4. Set up automated CI/CD for future releases
5. Consider publishing to Anaconda/Conda-forge for broader reach