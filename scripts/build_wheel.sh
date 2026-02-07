#!/bin/bash
# Build mlx-unsloth wheel locally
#
# Usage:
#   ./scripts/build_wheel.sh          # Build for current macOS
#   ./scripts/build_wheel.sh 14.0     # Build for macOS 14.0+
#   ./scripts/build_wheel.sh 15.0     # Build for macOS 15.0+
#
# Environment variables:
#   PYPI_RELEASE=1  - Build release version (no dev suffix)
#
# Output: wheelhouse/

set -e

MACOS_TARGET=${1:-$(sw_vers -productVersion | cut -d. -f1).0}
export MACOSX_DEPLOYMENT_TARGET=$MACOS_TARGET

echo "=== Building mlx-unsloth for macOS $MACOS_TARGET ==="
echo "Python: $(python --version)"
echo ""

# Clean previous builds
rm -rf build/ dist/ *.egg-info
mkdir -p wheelhouse

# Install build dependencies
pip install --upgrade pip build setuptools cmake typing_extensions

# Stage 1: Build mlx-unsloth (Python bindings)
echo "=== Stage 1: Building mlx-unsloth ==="
python setup.py clean --all 2>/dev/null || true
MLX_BUILD_STAGE=1 python -m build --wheel
mv dist/*.whl wheelhouse/

# Stage 2: Build mlx-unsloth-metal (Metal backend)
echo ""
echo "=== Stage 2: Building mlx-unsloth-metal ==="
python setup.py clean --all 2>/dev/null || true
MLX_BUILD_STAGE=2 python -m build --wheel
mv dist/*.whl wheelhouse/

echo ""
echo "=== Build complete ==="
echo "Wheels:"
ls -la wheelhouse/*.whl
echo ""
echo "To install locally:"
echo "  pip install wheelhouse/mlx_unsloth_metal-*.whl"
echo "  pip install wheelhouse/mlx_unsloth-*.whl"
echo ""
echo "To upload to PyPI:"
echo "  pip install twine"
echo "  twine upload wheelhouse/*.whl"
