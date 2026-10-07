"""StopWatch two-model ObjectNav method."""

import logging
import os

# Set these before Habitat imports, including in spawned render/eval workers.
# Magnum's quiet mode hides the GL startup banner but retains warnings/errors.
# Habitat-Lab expects a numeric level. Preserve explicit diagnostic overrides.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_LAB_LOG", str(logging.WARNING))
