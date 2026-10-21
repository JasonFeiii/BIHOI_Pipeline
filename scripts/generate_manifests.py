"""Generate exact publishable file list, complete source tree and ignored-local path list."""
import csv,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT).decode()
def main():
 docs=ROOT/'.release-reports';docs.mkdir(exist_ok=True)
 for name in ['upload-manifest.tsv','excluded-manifest.tsv','directory-tree.txt']:(docs/name).touch(exist_ok=True)
 names=sorted(set(p for p in git(['ls-files','--cached','--others','--exclude-standard','-z']).split('\0') if p))
 # Publishable documentation includes its own path; self hash is deliberately left blank.
 with (docs/'upload-manifest.tsv').open('w') as f:
  writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['path','bytes','sha256'])
  for name in names:
   p=ROOT/name
   if p.is_symlink():raise RuntimeError('Unexpected publishable symlink: '+name)
   writer.writerow([name,'' if name=='.release-reports/upload-manifest.tsv' else p.stat().st_size,'' if name=='.release-reports/upload-manifest.tsv' else hashlib.sha256(p.read_bytes()).hexdigest()])
 ignored=sorted(set(p for p in git(['ls-files','--others','--ignored','--exclude-standard','-z']).split('\0') if p))
 # One Git query for all ignored files avoids thousands of child processes.
 ignore_input=b''.join(name.encode()+b'\0' for name in ignored)
 ignore_output=subprocess.run(['git','check-ignore','--verbose','--stdin','-z'],cwd=ROOT,input=ignore_input,stdout=subprocess.PIPE,check=True).stdout
 fields=ignore_output.split(b'\0')
 reasons={}
 for offset in range(0,len(fields)-1,4):
  source,line,pattern,name=(value.decode() for value in fields[offset:offset+4])
  reasons[name]=source+':'+line+':'+pattern
 with (docs/'excluded-manifest.tsv').open('w') as f:
  writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['path','kind','reason'])
  writer.writerow(['.git/','directory','independent local Git metadata; never uploaded as content'])
  for name in ignored:
   p=ROOT/name;reason=reasons[name]
   writer.writerow([name,'symlink' if p.is_symlink() else 'file',reason])
 tree={}
 for name in names:
  cursor=tree
  for part in Path(name).parts:cursor=cursor.setdefault(part,{})
 lines=['BIHOI_Pipeline/']
 def draw(cursor,prefix=''):
  items=sorted(cursor.items(),key=lambda item:(not bool(item[1]),item[0]))
  for i,(name,children) in enumerate(items):
   last=i==len(items)-1;lines.append(prefix+('└── ' if last else '├── ')+name+('/' if children else ''))
   if children:draw(children,prefix+('    ' if last else '│   '))
 draw(tree);(docs/'directory-tree.txt').write_text('\n'.join(lines)+'\n')
 # Rewrite upload list once now the other two reports have their final content.
 with (docs/'upload-manifest.tsv').open('w') as f:
  writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['path','bytes','sha256'])
  for name in names:
   p=ROOT/name;writer.writerow([name,'' if name=='.release-reports/upload-manifest.tsv' else p.stat().st_size,'' if name=='.release-reports/upload-manifest.tsv' else hashlib.sha256(p.read_bytes()).hexdigest()])
 print(json.dumps({'publishable_files':len(names),'publishable_bytes':sum((ROOT/name).stat().st_size for name in names),'ignored_local_entries':len(ignored),'policy':'Only these publishable paths are in scope; all other original repository paths are excluded.'},indent=2))
if __name__=='__main__':main()
