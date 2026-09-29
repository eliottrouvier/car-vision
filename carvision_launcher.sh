#!/usr/bin/env bash
# CarVision Launcher Script

set -e

# Workaround for macOS Qt path containing colon ':'
export QT_PLUGIN_PATH="$HOME/.local/share/carvision/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/.local/share/carvision/plugins/platforms"
export QT_LOGGING_RULES="qt.qpa.fonts=false;*.debug=false"

PROJECT_DIR="/Users/eliottrouvier/Desktop/projets IA:code /car-vision"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

exec "$PYTHON_BIN" "$PROJECT_DIR/main.py" "$@"
