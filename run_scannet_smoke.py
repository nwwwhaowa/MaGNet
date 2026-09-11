#!/usr/bin/env python3
"""Run from MaGNet: python run_scannet_smoke.py
Requires completed prepare_scannet.py with nonempty official train/val splits.
Checks actual project loader on a few windows, then runs 10 training batches.
Does not download data, install dependencies, change metadata, or resume weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import runpy
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--dataset-root', default='~/26workspace/datasets/ScanNet')
    parser.add_argument('--train-split', default='data_split/scannet_train_stride5.txt')
    parser.add_argument('--val-split', default='data_split/scannet_val_stride5.txt')
    parser.add_argument('--raw-wh-json', default='data/scannet_raw_WH.json')
    parser.add_argument('--dnet-ckpt', default='ckpts/DNET_scannet.pt')
    parser.add_argument('--fnet-ckpt', default='ckpts/FNET_scannet.pt')
    parser.add_argument('--magnet-ckpt', default='ckpts/MAGNET_scannet.pt')
    parser.add_argument('--check-only', action='store_true')
    cli = parser.parse_args()
    project = Path(cli.project_root).expanduser().resolve()
    trainer = project / 'train_StructMaGNet_scannet.py'
    if not trainer.is_file():
        parser.error(f'找不到 {trainer}；请在 MaGNet 目录运行')
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    output_parent = project / 'exp/STRUCTMAGNET/scannet'
    output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='smoke_', dir=output_parent))
    report = {'output_dir': str(output), 'training_completed': False}
    code = 1
    with (output / 'smoke.log').open('w', encoding='utf-8') as logfile:
        def log(text):
            print(text, flush=True)
            logfile.write(text + '\n')
            logfile.flush()
        try:
            flags = ['--dataset_root', str(Path(cli.dataset_root).expanduser().resolve()),
                     '--train_split', cli.train_split, '--val_split', cli.val_split,
                     '--raw_wh_json', cli.raw_wh_json, '--dnet_ckpt', cli.dnet_ckpt,
                     '--fnet_ckpt', cli.fnet_ckpt, '--magnet_ckpt', cli.magnet_ckpt,
                     '--batch_size', '1', '--num_workers', '0', '--epochs', '1',
                     '--lr', '1e-4', '--gate_mode', 'learned', '--window_radius', '20',
                     '--num_source_views', '4', '--input_height', '480', '--input_width', '640',
                     '--dpv_height', '120', '--dpv_width', '160',
                     '--max_train_steps', '10', '--val_max_samples', '8',
                     '--val_degrees', '0', '5', '8', '--amp', '--output_dir', str(output)]
            # Load definitions without invoking the trainer's main().
            module = runpy.run_path(str(trainer), run_name='scannet_smoke_preflight')
            args = module['build_parser']().parse_args(flags)
            module['_check_inputs'](args)
            import numpy as np
            import torch
            from PIL import Image
            from torch.utils.data import DataLoader
            from data.dataloader_scannet_train import ScannetTrainDataset, _read_split
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA 不可用')
            log(f'PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}')
            samples = {s: _read_split(Path(getattr(args, s + '_split')).expanduser())
                       for s in ('train', 'val')}
            scene_sets = {s: {name for name, _ in rows} for s, rows in samples.items()}
            spaces = {s: {name.split('_')[0] for name in names} for s, names in scene_sets.items()}
            if spaces['train'] & spaces['val']:
                raise ValueError('训练/验证包含相同物理空间（sceneXXXX 前缀）')
            root = Path(args.dataset_root)
            for split, names in scene_sets.items():
                official_path = root / 'official_splits' / f'scannetv2_{split}.txt'
                official = {line.split()[0] for line in official_path.read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith('#')}
                if names - official:
                    raise ValueError(f'{split} 包含非官方对应集合的场景：{sorted(names - official)}')
                if len(samples[split]) != len(set(samples[split])):
                    raise ValueError(f'{split} 包含重复样本')
                log(f'{split}: {len(names)} scenes, {len(samples[split])} samples')
            metadata = json.loads(Path(args.raw_wh_json).expanduser().read_text())
            for scene in sorted(scene_sets['train'] | scene_sets['val']):
                scene_dir = root / 'scans' / scene
                size = metadata.get(scene)
                if not isinstance(size, list) or len(size) != 2 or min(size) <= 0:
                    raise ValueError(f'{scene} 缺少有效原始 [W,H] 元数据：{size}')
                intrinsic = np.loadtxt(scene_dir / 'intrinsic/intrinsic_color.txt')
                if intrinsic.shape != (4, 4) or not np.isfinite(intrinsic).all():
                    raise ValueError(f'{scene} 内参不是有限的 4x4 矩阵')
                if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
                    raise ValueError(f'{scene} 焦距无效')
                sample = next(frame for rows in samples.values() for name, frame in rows if name == scene)
                with Image.open(scene_dir / 'color' / f'{sample}.jpg') as image:
                    if list(image.size) != size:
                        raise ValueError(f'{scene}: 实际 RGB 尺寸 {image.size} 与元数据 {size} 不一致')
            log('全部划分场景的 RGB 原始尺寸和内参检查通过')
            model_args = module['build_model_args'](args)
            report['loader_samples'] = {}
            for split in ('train', 'val'):
                dataset = ScannetTrainDataset(model_args, split)
                indices = sorted({0, len(dataset)//2, len(dataset)-1})
                for idx in indices:
                    views, intrinsics = dataset[idx]
                    if len(views) != 5:
                        raise ValueError('窗口不是 5 个视图')
                    for view in views:
                        if not torch.isfinite(view['img']).all() or not np.isfinite(view['extM']).all():
                            raise ValueError(f'{split} 样本 {idx} 图像或逆位姿无效')
                    gt = views[2]['gt_dmap']
                    valid = torch.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
                    if not valid.any():
                        raise ValueError(f'{split} 样本 {idx} 无有效深度像素')
                    if not all(torch.isfinite(t).all() for t in intrinsics.values()):
                        raise ValueError(f'{split} 样本 {idx} 内参/rays 非有限')
                batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
                report['loader_samples'][split] = indices
                log(f'{split} 加载器抽检通过：索引 {indices}；RGB batch {tuple(batch[0][0]["img"].shape)}')
                del batch, dataset
            log('加载器抽检通过；未遍历所有样本，权重兼容性和模型计算将在训练中验证')
            if cli.check_only:
                code = 0
            else:
                command = [sys.executable, '-u', str(trainer)] + flags
                report['command'] = command
                log('开始 10 步试训练；验证每个噪声档最多 8 个样本，共 0/5/8 度三档')
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding='utf-8', errors='replace') as child:
                    try:
                        for line in child.stdout:
                            log(line.rstrip())
                        code = child.wait()
                    except KeyboardInterrupt:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                report['training_completed'] = code == 0
                if code:
                    log(f'试训练失败，退出码 {code}；保留日志以便定位')
        except KeyboardInterrupt:
            code = 130
            report['error'] = '用户中断'
            log('已中断')
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            log('检查失败：' + report['error'])
        finally:
            report['exit_code'] = code
            (output / 'smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            log(f'日志和报告：{output}')
    return code


if __name__ == '__main__':
    sys.exit(main())
