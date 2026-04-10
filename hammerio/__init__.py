"""HammerIO — GPU where it matters. CPU where it doesn't. Zero configuration.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Proprietary License — All Rights Reserved
"""

__version__ = "1.0.8"
__author__ = "Joseph C McGinty Jr"
__copyright__ = "Copyright 2026 ResilientMind AI"
__license__ = "Proprietary"

# Convenience imports for the public API
from hammerio.core.router import JobRouter
from hammerio.core.hardware import detect_hardware, HardwareProfile
from hammerio.core.profiler import profile_file, profile_directory
from hammerio.core.config import load_config

__all__ = [
    "JobRouter",
    "detect_hardware",
    "HardwareProfile",
    "profile_file",
    "profile_directory",
    "load_config",
    "__version__",
]
