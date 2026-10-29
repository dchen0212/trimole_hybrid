"""Download and verify model artifacts without loading serialized code."""
from pathlib import Path,PurePosixPath
import hashlib
import shutil
import tarfile
import urllib.request
from .registry import checkpoint_manifest

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(root,task=None):
    records=checkpoint_manifest()['records']
    if task:records=[r for r in records if r['task']==task]
    if not records:raise ValueError('No archived checkpoint records for the requested task.')
    errors=[]
    for r in records:
        p=Path(root)/r['id']
        if not p.is_file():errors.append('Missing: '+r['id'])
        elif p.stat().st_size!=r['bytes'] or file_sha256(p)!=r['sha256']:
            errors.append('Mismatch: '+r['id'])
    if errors:raise ValueError('\n'.join(errors))
    return len(records)

def download(root,task=None):
    manifest=checkpoint_manifest();records=manifest['records']
    if task:records=[r for r in records if r['task']==task]
    if not records:raise ValueError('No original checkpoint was archived for this task. Its fixed training settings remain in the task configuration.')
    assets={a['name']:a for a in manifest['assets']}
    required={r['asset'] for r in records};root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cache=root/'.downloads';cache.mkdir(exist_ok=True)
    for name in sorted(required):
        a=assets[name]
        if not a.get('download_url'):raise ValueError('No published download URL for '+name)
        dest=cache/name
        if not dest.is_file() or file_sha256(dest)!=a['sha256']:
            pending=dest.with_suffix(dest.suffix+'.part')
            with urllib.request.urlopen(a['download_url']) as src,pending.open('wb') as out:shutil.copyfileobj(src,out)
            if pending.stat().st_size!=a['bytes'] or file_sha256(pending)!=a['sha256']:
                raise ValueError('Archive hash/size mismatch: '+name)
            pending.replace(dest)
        wanted={r['id'] for r in records if r['asset']==name}
        with tarfile.open(dest,'r:gz') as archive:
            for member in archive:
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not member.isfile():
                    raise ValueError('Unsafe or unexpected archive member: '+member.name)
                if member.name not in wanted:continue
                target=root/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    return verify(root,task)
