"""Root conftest — adds the repo root to sys.path so `import kairos` works."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
