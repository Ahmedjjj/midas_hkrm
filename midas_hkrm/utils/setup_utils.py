import logging
import sys
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.parent.resolve()


def setup_logger(debug: bool = False) -> None:
    """
    Setup the logger to print to stdout
    Args:
        debug (bool, optional): whether to use debug mode logging. Defaults to False.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level)


def setup_path():
    """
    Setup PATH to include midas_hkrm and the midas code
    """
    sys.path.append(str(PROJECT_PATH))
    sys.path.append(str(PROJECT_PATH / "external" / "MiDaS"))
