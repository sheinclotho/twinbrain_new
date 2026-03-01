"""pytest configuration: add the repository root to sys.path so that tests can
import from the top-level packages (models, utils, etc.) without installing the
project as a package."""

import sys
from pathlib import Path

# Repository root is two levels up from this file (tests/conftest.py → repo root)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
