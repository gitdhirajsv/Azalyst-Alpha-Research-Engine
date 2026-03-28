"""
Compatibility entrypoint for the local training-only workflow documented in
SETUP_LOCAL_GPU.md.

This forwards to azalyst_local_gpu.py but forces --year2-only by default so the
command behaves like a "train base models only" step rather than running the
full Year 3 walk-forward loop.
"""

import sys

from azalyst_local_gpu import main


if __name__ == "__main__":
    if "--year2-only" not in sys.argv:
        sys.argv.append("--year2-only")
    main()
