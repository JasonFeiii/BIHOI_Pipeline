"""Coalesced asynchronous JSON for display-only progress, never checkpoints."""
from pathlib import Path
import atexit,threading,time,sys

class LatestJSONWriter:
    def __init__(self,path):
        self.path=Path(path);self.cv=threading.Condition()
        self.pending=None;self.inflight=False;self.stopping=False;self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='display-progress-writer')
        self.thread.start();atexit.register(self.close)
    def submit(self,payload):
        with self.cv:
            if self.stopping:return False
            self.pending=payload;self.cv.notify_all()
        return True
    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.stopping:self.cv.wait()
                if self.pending is None and self.stopping:return
                payload=self.pending;self.pending=None;self.inflight=True
            try:
                tmp=self.path.with_suffix(self.path.suffix+'.display_tmp')
                tmp.write_text(payload);tmp.replace(self.path);self.error=None
            except OSError as exc:
                self.error=str(exc)
                print('DISPLAY_PROGRESS_IO_ERROR: '+str(exc),file=sys.stderr,flush=True)
            finally:
                with self.cv:self.inflight=False;self.cv.notify_all()
    def flush(self,timeout=5.):
        deadline=time.monotonic()+timeout
        with self.cv:
            while self.pending is not None or self.inflight:
                left=deadline-time.monotonic()
                if left<=0:return False
                self.cv.wait(left)
            return self.error is None
    def close(self,timeout=5.):
        with self.cv:self.stopping=True;self.cv.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive()
