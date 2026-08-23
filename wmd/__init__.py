"""Working Model Downloader -- the ComfyUI-free core.

Nothing in this package imports ComfyUI, so it can be tested, scripted and driven
from the CLI on a machine that has never run ComfyUI. :mod:`wmd.comfy_env` is the
single seam where the real thing is used when it is available.
"""

__version__ = "0.1.0"
