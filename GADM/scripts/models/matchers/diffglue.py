# Backward-compatibility alias.
# GADM's matcher extends the DiffGlue architecture (Zhang & Ma, ACM MM 2024);
# older configs and checkpoints reference this module as "matchers.diffglue".
from .gadm import *  # noqa: F401,F403
from .gadm import GADM as DiffGlue  # noqa: F401

__main_model__ = DiffGlue
