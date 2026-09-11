#!/usr/bin/env python3
"""Run inside MaGNet: python start_scannet_training.py

Generate missing ScanNet frame splits from available official scenes, then train.
Training defaults to FP32 because AMP failed the first-batch Gaussian check on
this Orin setup. --amp is an explicit opt-in for future numerical debugging.
Each invocation starts a fresh run from the backbone checkpoints, not a resume.
Existing exported scenes are reused; eligible raw scenes without an exported
folder are exported at stride 5. Pending aria2 files and scene0010_00 are excluded.
This script never downloads .sens data, deletes raw data, or changes official
scene lists. Run --prepare-only to generate splits without starting training.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def scene_list(path):
    if not path.is_file():
        raise FileNotFoundError(f'缺少官方场景列表：{path}')
    rows = [line.split()[0] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    if not rows or any(not re.fullmatch(r'scene\d{4}_\d{2}', s) for s in rows):
        raise ValueError(f'官方场景列表为空或格式错误：{path}')
    return rows


def checked_rows(path, allowed):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in allowed:
            raise ValueError(f'划分格式或场景错误：{line}')
        frame = int(parts[1])
        if frame < 0 or frame % 5:
            raise ValueError(f'帧号不符合 stride 5：{line}')
        rows.append((parts[0], frame))
    if not rows:
        raise ValueError(f'{path.name} 为零样本：检查完整 RGB/depth/pose 窗口和位姿，或补充场景')
    if len(rows) != len(set(rows)):
        raise ValueError(f'{path.name} 包含重复样本')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--dataset-root', default='~/26workspace/datasets/ScanNet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--amp', action='store_true',
                        help='显式启用混合精度；默认关闭，当前 Orin 首步 FP32 已通过')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--include-scene0010-00', action='store_true',
                        help='仅在修复 scene0010_00 下载后使用')
    cli = parser.parse_args()
    if cli.epochs < 1:
        parser.error('--epochs 必须为正整数')
    project = Path(cli.project_root).expanduser().resolve()
    root = Path(cli.dataset_root).expanduser().resolve()
    builder = project / 'tools/scannet/build_scannet_frame_splits.py'
    exporter = project / 'tools/scannet/export_sens_py3.py'
    trainer = project / 'train_StructMaGNet_scannet.py'
    for path in (builder, exporter, trainer):
        if not path.is_file():
            parser.error(f'缺少 {path}；请在 MaGNet 目录运行')
    parent = project / 'exp/STRUCTMAGNET/scannet'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='train_', dir=parent))
    report = {'dataset_root': str(root), 'output_dir': str(output),
              'precision': 'AMP' if cli.amp else 'FP32',
              'skipped': {}, 'export_failed': [], 'split_checks_passed': False,
              'training_started': False, 'training_completed': False}
    code = 1
    with (output / 'console.log').open('w', encoding='utf-8') as logfile:
        def log(message):
            print(message, flush=True)
            logfile.write(message + '\n')
            logfile.flush()

        def run(command):
            log('执行：' + ' '.join(map(str, command)))
            with subprocess.Popen([str(x) for x in command], cwd=project,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace') as child:
                try:
                    for line in child.stdout:
                        log(line.rstrip())
                    return child.wait()
                except KeyboardInterrupt:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
        try:
            log(f'输出目录：{output}')
            lists = {s: scene_list(root / 'official_splits' / f'scannetv2_{s}.txt')
                     for s in ('train', 'val')}
            spaces = {s: {name.split('_')[0] for name in names} for s, names in lists.items()}
            if spaces['train'] & spaces['val']:
                raise ValueError('官方列表存在物理空间交叉，请恢复正确的官方划分')
            # Make dependency errors fatal before interpreting individual exports.
            if run([sys.executable, '-c', 'import numpy; from PIL import Image']):
                raise RuntimeError('当前环境缺少 numpy/Pillow，详情见上方输出')
            available = {}
            for split, names in lists.items():
                available[split] = []
                for scene in names:
                    if scene == 'scene0010_00' and not cli.include_scene0010_00:
                        report['skipped'][scene] = '已知未完成下载，默认排除'
                        continue
                    sens = root / 'raw_sens/scans' / scene / f'{scene}.sens'
                    if Path(str(sens) + '.aria2').exists():
                        report['skipped'][scene] = '.aria2 下载未完成'
                        continue
                    directory = root / 'scans' / scene
                    if not directory.is_dir() and sens.is_file():
                        before = sens.stat()
                        result = run([sys.executable, exporter, '--sens', sens,
                                      '--dataset_root', root, '--stride', '5'])
                        after = sens.stat()
                        changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                        if result or changed or Path(str(sens) + '.aria2').exists():
                            report['export_failed'].append(scene)
                            if directory.exists():
                                backup = Path(tempfile.mkdtemp(prefix='incomplete_export_', dir=root))
                                shutil.move(str(directory), str(backup / scene))
                                log(f'导出失败/下载变化，已隔离：{backup / scene}')
                            continue
                    if directory.is_dir() and all((directory / sub).is_dir() for sub in ('color', 'depth', 'pose')):
                        available[split].append(scene)
                log(f'{split} 可检查场景：{len(available[split])}/{len(names)}')
            report['available'] = available
            for split in ('train', 'val'):
                if not available[split]:
                    candidates = ['scene0019_00'] if split == 'val' and 'scene0019_00' in lists[split] else lists[split][:1]
                    target = candidates[0]
                    log(f'缺少 {split} 数据。请完整下载官方场景 {target} 到：')
                    log(str(root / 'raw_sens/scans' / target / f'{target}.sens'))
                    raise RuntimeError('补齐该场景后重新运行本脚本；将自动导出并继续。不启动训练。')
                (output / f'{split}_scenes.txt').write_text(
                    '\n'.join(available[split]) + '\n', encoding='utf-8')
            staged = {s: output / f'scannet_{s}_stride5.txt' for s in ('train', 'val')}
            command = [sys.executable, builder, '--dataset_root', root,
                       '--train_scenes', output / 'train_scenes.txt',
                       '--val_scenes', output / 'val_scenes.txt',
                       '--train_out', staged['train'], '--val_out', staged['val'],
                       '--reference_stride', '5', '--window_radius', '20', '--num_source_views', '4']
            if run(command):
                raise RuntimeError('生成帧划分失败，见上方日志')
            report['samples'] = {}
            for split, path in staged.items():
                rows = checked_rows(path, set(available[split]))
                report['samples'][split] = len(rows)
                log(f'{split}: {len(rows)} 有效帧样本')
            destination = project / 'data_split'
            destination.mkdir(exist_ok=True)
            for path in staged.values():
                target = destination / path.name
                if target.exists():
                    shutil.copy2(target, output / ('previous_' + path.name))
                shutil.copy2(path, target)
                log(f'已写入：{target}')
            report['split_checks_passed'] = True
            if cli.prepare_only:
                code = 0
                log('划分生成并校验成功，--prepare-only 模式完成')
            else:
                required = [project / 'data/scannet_raw_WH.json'] + [project / 'ckpts' / name for name in
                           ('DNET_scannet.pt', 'FNET_scannet.pt', 'MAGNET_scannet.pt')]
                for path in required:
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError(f'训练所需文件缺失或为空：{path}')
                # Use absolute paths and invoke the executable directly, not an echo.
                # Use per-run copies so another preparation run cannot change this run's inputs.
                command = [sys.executable, '-u', trainer, '--dataset_root', root,
                           '--train_split', staged['train'], '--val_split', staged['val'],
                           '--raw_wh_json', required[0], '--dnet_ckpt', required[1],
                           '--fnet_ckpt', required[2], '--magnet_ckpt', required[3],
                           '--batch_size', '1', '--num_workers', '0', '--epochs', cli.epochs,
                           '--lr', '1e-4', '--gate_mode', 'learned', '--window_radius', '20',
                           '--num_source_views', '4', '--max_train_steps', '0',
                           '--val_max_samples', '32', '--val_degrees', '0', '5', '8',
                           '--seed', '1234', '--output_dir', output]
                if cli.amp:
                    command.append('--amp')
                log('训练精度：' + ('AMP' if cli.amp else 'FP32（AMP 已关闭）'))
                report['training_command'] = list(map(str, command))
                report['training_started'] = True
                code = run(command)
                report['training_completed'] = code == 0
        except KeyboardInterrupt:
            code = 130
            report['error'] = '用户中断'
            log('已中断')
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            log('未完成：' + report['error'])
        finally:
            report['exit_code'] = code
            (output / 'launch_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            log(f'退出码：{code}；日志：{output / "console.log"}')
    return code


if __name__ == '__main__':
    sys.exit(main())
