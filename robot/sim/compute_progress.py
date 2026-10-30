"""Small, throttled progress messages, independent of physics and rendering."""
from pathlib import Path
import json,os,time

class ComputeProgress:
    def __init__(self,path,run,**initial):
        self.path=Path(path);self.last_write=0.
        from async_progress import LatestJSONWriter
        self.writer=LatestJSONWriter(self.path)
        self.state=dict(schema_version=1,run=run,pid=os.getpid(),phase='preparing',**initial)
        self.update(force=True)

    def update(self,phase=None,force=False,**fields):
        changed=phase is not None and phase!=self.state.get('phase')
        if phase is not None:self.state['phase']=phase
        self.state.update(fields)
        now=time.monotonic()
        if not force and not changed and now-self.last_write<.3:return
        self.last_write=now;self.state['updated_unix']=time.time()
        self.writer.submit(json.dumps(self.state,ensure_ascii=False))
