#!/usr/bin/env python
"""Does threadpoolctl's BLAS limit, set in the main thread, apply inside joblib worker threads?

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 apptainer exec "$SIF" python python/tests/check_thread_limits.py

Prints the BLAS thread count each library reports in the main thread and inside worker threads,
with and without a ``threadpool_limits(1)`` context around the ``Parallel`` call (as in fit.py).
"""

from __future__ import annotations

from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits


def blas_threads():
    return sorted({(i["internal_api"], i.get("threading_layer"), i["num_threads"])
                   for i in threadpool_info() if i["user_api"] == "blas"})


def main():
    print("main, no limit:      ", blas_threads())
    print("workers, no limit:   ", set(map(tuple, Parallel(n_jobs=3, backend="threading")(
        delayed(blas_threads)() for _ in range(3)))))
    with threadpool_limits(limits=1, user_api="blas"):
        print("main, limit=1:       ", blas_threads())
        print("workers, limit=1:    ", set(map(tuple, Parallel(n_jobs=3, backend="threading")(
            delayed(blas_threads)() for _ in range(3)))))
    # What fit.py does: set the limit inside each worker thread.
    print("workers, limit inside worker:", set(map(tuple, Parallel(n_jobs=3, backend="threading")(
        delayed(_limited)(1) for _ in range(3)))))
    print("main afterwards:     ", blas_threads())


def _limited(n):
    with threadpool_limits(limits=n, user_api="blas"):
        return blas_threads()


if __name__ == "__main__":
    main()
