"""Interruptible, bounded writer exclusion for immutable media cache entries.

Completed entries are read without any lock. This is intentionally separate
from the nonblocking singleton-controller lock in joint_affect_pipeline.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time


@contextmanager
def cache_writer_lock(root, check=lambda: None, *, timeout=60., poll=.05):
    check()
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    stream=(root/".media-write.lock").open("a+b")
    acquired=False
    started=time.monotonic()
    try:
        if os.name=="nt":
            import msvcrt
            # Reading a byte held by another Windows locker itself raises
            # EACCES; inspect length without touching the locked byte instead.
            stream.seek(0,os.SEEK_END)
            if stream.tell()==0:
                stream.write(b"0");stream.flush()
        else:
            import fcntl
        while not acquired:
            check()
            try:
                if os.name=="nt":
                    stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired=True
            except OSError as exc:
                if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
                    raise
                if time.monotonic()-started>=timeout:
                    raise InterruptedError("Media writer lock wait exceeded its fixed timeout") from exc
                time.sleep(poll)
        yield
    finally:
        if acquired:
            if os.name=="nt":
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()
