#!/usr/bin/env python3
"""Copy a completed experiment, its pretrained dependencies and code snapshot.

Uses only Python's standard library and git. Never deletes or moves the source.
Checksums detect accidental corruption; they are not an authenticity signature.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root):
    return sorted(p for p in root.rglob('*') if p.is_file())


def verify(root):
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest contains a path outside the backup')
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise RuntimeError('Verification failed: ' + item['path'])
    print(f"VERIFIED: {root} ({len(manifest['files'])} files)")


def git(project, *args):
    return subprocess.check_output(['git', '-C', str(project), *args], text=True).strip()


def preserve(project, run, backup_root, code_ref):
    project, run, backup_root = [Path(p).expanduser().resolve() for p in (project, run, backup_root)]
    suite = run.parent
    if run.name != 'full_train':
        raise ValueError('--run-dir must be the completed full_train directory')
    if backup_root.is_relative_to(suite):
        raise ValueError('Backup destination must be outside the source experiment')
    required = ['best.pt', 'last.pt', 'train.csv', 'val.csv', 'config.json', 'perf_summary.json']
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    perf = json.loads((run / 'perf_summary.json').read_text())
    if perf.get('returncode') != 0:
        raise ValueError('Experiment is not recorded as successfully completed')
    config = json.loads((run / 'config.json').read_text())
    inputs = {}
    for key in ('dnet_ckpt', 'fnet_ckpt', 'magnet_ckpt', 'raw_wh_json', 'train_split', 'val_split'):
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = project / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
        inputs[key] = path
    commit = git(project, 'rev-parse', '--verify', code_ref + '^{commit}')
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / (suite.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(exist_ok=False)
    # A failure keeps the partial directory for diagnosis; the original is untouched.
    (destination / 'INCOMPLETE').write_text('Backup and checksum verification not finished.\n')
    originals = []
    for source in files_under(suite):
        rel = source.relative_to(suite)
        target = destination / 'experiment' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        original_hash = sha256(source)
        shutil.copy2(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during backup: {source}')
        if sha256(target) != original_hash:
            raise RuntimeError(f'Copy checksum differs: {source}')
        originals.append({'original': str(source), 'copy': str(target.relative_to(destination)), 'sha256': original_hash})
    for key, source in inputs.items():
        target = destination / 'dependencies' / key / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != original_hash:
            raise RuntimeError(f'Copy checksum differs: {source}')
        originals.append({'original': str(source), 'copy': str(target.relative_to(destination)), 'sha256': original_hash})
    subprocess.run(['git', '-C', str(project), 'archive', '--format=tar.gz',
                    '--output=' + str(destination / 'code.tar.gz'), commit], check=True)
    metadata = {
        'format': 'structmagnet-experiment-backup-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_experiment': str(suite), 'code_commit': commit,
        'code_provenance': 'User-specified training reference; runtime source was not independently hashed.',
        'worktree_head_at_backup': git(project, 'rev-parse', 'HEAD'),
        'worktree_status_at_backup': git(project, 'status', '--porcelain'),
        'input_mapping': originals,
        'files': [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in files_under(destination) if p.name != 'INCOMPLETE'],
    }
    (destination / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    verify(destination)
    (destination / 'INCOMPLETE').unlink()
    print('Source preserved: ' + str(suite))
    print('Backup: ' + str(destination))
    print('Copy this complete directory to another disk or machine for protection against disk failure.')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--run-dir')
    parser.add_argument('--backup-root')
    parser.add_argument('--code-ref', default='bfe125992a332061a055698c6955267a468e7585')
    parser.add_argument('--verify', metavar='BACKUP_DIRECTORY')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        if not args.run_dir or not args.backup_root:
            parser.error('--run-dir and --backup-root are required when creating a backup')
        preserve(args.project_root, args.run_dir, args.backup_root, args.code_ref)


if __name__ == '__main__':
    main()
