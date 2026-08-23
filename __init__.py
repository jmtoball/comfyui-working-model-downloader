"""Working Model Downloader -- a ComfyUI custom node.

Sort a workflow's models out once in the sidebar panel; the panel pins the result
into a single node, and every run after that -- headless, API, or a fresh machine
months later -- reproduces exactly those downloads.
"""

from __future__ import annotations

import logging

from .wmd_nodes import MODE, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

log = logging.getLogger("working_model_downloader")

# Served by ComfyUI at /extensions/<dirname>/, which is where the sidebar panel
# and its stylesheet are loaded from.
WEB_DIRECTORY = "web"

try:
    from . import routes

    _registered = routes.register()
except Exception:  # noqa: BLE001 - the nodes must still load without the panel
    log.exception("Working Model Downloader: the sidebar panel failed to register")
    _registered = False

if _registered:
    log.info("Working Model Downloader ready (queue-time gate: %s)", MODE)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
