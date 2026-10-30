"""Resolve actual checkpoint provenance even after a supervisor handoff."""
import json
from pathlib import Path
def ancestors(root,run,parents):
 root=Path(root);seen=[]
 while run and run not in seen:
  seen.append(run);parent=parents.get(run)
  if not parent:
   rp=root/'physics'/run/'result.json';prefix=run+'_prefix'
   if rp.exists():
    try:prefix=json.loads(rp.read_text()).get('arguments',{}).get('resume_from') or prefix
    except (ValueError,OSError):pass
   pp=root/'physics'/prefix/'provenance.json'
   if pp.exists():
    try:parent=Path(json.loads(pp.read_text())['source']).name
    except (ValueError,KeyError,OSError):pass
  if parent:parents[run]=parent
  run=parent
 return seen
