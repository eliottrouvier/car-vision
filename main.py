"""
main.py - Entry point for Car-Vision Autonomous Perception Cockpit.
"""

import os
import sys

# Workaround for macOS Qt path containing colon ':'
user_plugin_dir = os.path.expanduser("~/.local/share/carvision/plugins")
if os.path.exists(user_plugin_dir):
    os.environ["QT_PLUGIN_PATH"] = user_plugin_dir
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(user_plugin_dir, "platforms")

# Suppress harmless font alias warnings on macOS
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false;*.debug=false"

from PySide6 import QtWidgets, QtCore, QtGui
from main_window import MainWindow


def main():
    # High-DPI display scaling
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CarVision")
    app.setOrganizationName("EliottRouvier")

    # Locate sample directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, "samples")

    window = MainWindow(sample_dir=sample_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
