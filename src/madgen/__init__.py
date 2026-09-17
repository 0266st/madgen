import warnings

# pyworld 0.3.5 imports pkg_resources at import time; the deprecation notice is noise.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from .cli import main  # noqa: E402

__all__ = ["main"]
