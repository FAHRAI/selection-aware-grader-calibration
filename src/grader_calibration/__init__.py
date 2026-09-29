"""Selection-aware estimation of grader error rates under a targeted audit, in a simulated adaptive tutor."""

import os

# Single-threaded BLAS keeps the quadrature rules, and therefore the recorded outcomes, bit-for-bit reproducible.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "1.0.0"
