import logging
import os
import sys
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.parent.resolve()


def setup_logger(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level)


def setup_path():
    sys.path.append(str(PROJECT_PATH))
    sys.path.append(str(PROJECT_PATH / "external" / "MiDaS"))
