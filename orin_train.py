#!/usr/bin/env python3
"""FP32 ScanNet throughput tuning and monitoring for Jetson Orin.

Place this file in MaGNet. Requires existing valid frame splits and checkpoints.
  python orin_train.py benchmark --train-best
  python orin_train.py train --batch-size 2 --workers 2

Each benchmark starts independently from the same backbone checkpoints. Runs
40 updates, ignores the first 10 for throughput, and validates 1 sample at 0 deg.
Full training uses 10 epochs and 32 validation samples per noise level (0/5/8).
No dataset download or source modification. No AMP or power-mode changes.
Larger batches change optimizer update counts for the same epoch budget.
Timing is end-to-end between completed training CSV rows, not GPU kernel time.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback


def read_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0])
    return {'system_available_gib': values['MemAvailable'] / 1024**2,
            'system_total_gib': values['MemTotal'] / 1024**2,
            'system_swap_used_gib': (values['SwapTotal'] - values['SwapFree']) / 1024**2}


def append_row(path, row):
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def summarize(rows, warmup):
    measured = [r for r in rows if r['epoch_step'] > warmup and r['step_seconds'] > 0]
    if not measured:
        return {'measured_steps': 0}
    seconds = sum(r['step_seconds'] for r in measured)
    count = sum(r['batch_size'] for r in measured)
    return {'measured_steps': len(measured), 'measured_samples': count,
            'samples_per_second': count / seconds,
            'mean_step_seconds': seconds / len(measured),
            'median_step_seconds': statistics.median(r['step_seconds'] for r in measured),
            'min_system_available_gib': min(r['system_available_gib'] for r in measured),
            'max_system_swap_used_gib': max(r['system_swap_used_gib'] for r in measured),
            'peak_torch_allocated_gib': max(r['torch_peak_allocated_gib'] for r in measured),
            'peak_torch_reserved_gib': max(r['torch_peak_reserved_gib'] for r in measured)}


def select_best(results, min_available):
    eligible = [r for r in results if r.get('returncode') == 0
                and r.get('measured_steps', 0) >= 5
                and math.isfinite(r.get('samples_per_second', float('nan')))
                and r.get('min_system_available_gib', 0) >= min_available]
    return max(eligible, key=lambda r: r['samples_per_second']) if eligible else None


def stop_process(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def worker(config):
    project = Path(config['project'])
    output = Path(config['output'])
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = str(config['cpu_threads'])
    tegra, tegra_file = None, None
    perf_rows = []
    returncode = 1
    try:
        import torch
        torch.set_num_threads(config['cpu_threads'])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用')
        torch.backends.cudnn.benchmark = config['cudnn_benchmark']
        # Preserve the working FP32 run's TF32 defaults. No FP16 autocast is enabled.
        info = {'time_utc': datetime.now(timezone.utc).isoformat(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'amp': False,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'memory': read_memory()}
        write_json(output / 'environment.json', info)
        tegra_path = shutil.which('tegrastats')
        if tegra_path:
            tegra_file = (output / 'tegrastats.log').open('w', encoding='utf-8')
            try:
                tegra = subprocess.Popen([tegra_path, '--interval', '1000'],
                                         stdout=tegra_file, stderr=subprocess.STDOUT)
            except OSError as exc:
                print(f'tegrastats 无法启动：{exc}', flush=True)
        else:
            print('未找到 tegrastats；仍记录 /proc 内存和 PyTorch 显存指标', flush=True)
        definitions = runpy.run_path(str(project / 'train_StructMaGNet_scannet.py'),
                                    run_name='orin_instrumented_training')
        globals_dict = definitions['train'].__globals__
        original_append = globals_dict['append_csv']
        last_time = None
        last_epoch = None
        epoch_step = 0
        recent = deque(maxlen=20)

        def monitored_append(path, row):
            nonlocal last_time, last_epoch, epoch_step
            original_append(path, row)
            if Path(path).name != 'train.csv':
                return
            now = time.perf_counter()
            if row['epoch'] != last_epoch:
                last_time = None
                last_epoch = row['epoch']
                epoch_step = 0
                recent.clear()
            epoch_step += 1
            elapsed = 0.0 if last_time is None else now - last_time
            last_time = now
            memory = read_memory()
            record = {'time_utc': datetime.now(timezone.utc).isoformat(),
                      'epoch': row['epoch'], 'step': row['step'], 'epoch_step': epoch_step,
                      'batch_size': config['batch_size'], 'num_workers': config['workers'],
                      'step_seconds': elapsed,
                      'samples_per_second': config['batch_size'] / elapsed if elapsed > 0 else 0,
                      'torch_allocated_gib': torch.cuda.memory_allocated() / 1024**3,
                      'torch_reserved_gib': torch.cuda.memory_reserved() / 1024**3,
                      'torch_peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                      'torch_peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                      **memory, 'depth_nll': row['depth_nll'], 'gate_mean': row['gate_mean'],
                      'update_abs_mean': row['update_abs_mean']}
            append_row(output / 'perf.csv', record)
            perf_rows.append(record)
            if elapsed > 0:
                recent.append(elapsed)
            if epoch_step % 10 == 0 and recent:
                speed = config['batch_size'] * len(recent) / sum(recent)
                print(f"\nPERF step={row['step']} ref_samples/s={speed:.3f} "
                      f"step_s={sum(recent)/len(recent):.3f} "
                      f"torch_peak={record['torch_peak_allocated_gib']:.2f} GiB "
                      f"RAM_available={memory['system_available_gib']:.2f} GiB "
                      f"swap={memory['system_swap_used_gib']:.2f} GiB", flush=True)
                write_json(output / 'perf_summary.json', summarize(perf_rows, config['warmup']))

        globals_dict['append_csv'] = monitored_append
        argv = [str(project / 'train_StructMaGNet_scannet.py'),
                '--dataset_root', config['dataset_root'],
                '--train_split', config['train_split'], '--val_split', config['val_split'],
                '--raw_wh_json', str(project / 'data/scannet_raw_WH.json'),
                '--dnet_ckpt', str(project / 'ckpts/DNET_scannet.pt'),
                '--fnet_ckpt', str(project / 'ckpts/FNET_scannet.pt'),
                '--magnet_ckpt', str(project / 'ckpts/MAGNET_scannet.pt'),
                '--batch_size', str(config['batch_size']), '--num_workers', str(config['workers']),
                '--epochs', str(config['epochs']), '--max_train_steps', str(config['steps']),
                '--val_max_samples', str(config['val_samples']), '--val_degrees',
                *map(str, config['val_degrees']), '--window_radius', '20', '--num_source_views', '4',
                '--lr', '1e-4', '--seed', '1234', '--gate_mode', 'learned',
                '--output_dir', str(output)]
        if config['pin_memory']:
            argv.append('--pin_memory')
        write_json(output / 'command.json', argv)
        sys.argv = argv
        print(f"FP32: batch={config['batch_size']}, workers={config['workers']}, "
              f"cpu_threads={config['cpu_threads']}, cudnn_benchmark={config['cudnn_benchmark']}", flush=True)
        definitions['main']()
        returncode = 0
    except KeyboardInterrupt:
        returncode = 130
    except Exception:
        traceback.print_exc()
    finally:
        stop_process(tegra)
        if tegra_file:
            tegra_file.close()
        summary = summarize(perf_rows, config['warmup'])
        summary.update({'returncode': returncode, 'batch_size': config['batch_size'],
                        'workers': config['workers'], 'output_dir': str(output),
                        'completed_train_steps': len(perf_rows)})
        write_json(output / 'perf_summary.json', summary)
    return returncode


def execute(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    config_file = output / 'launch.json'
    write_json(config_file, config)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '_worker', str(config_file)]
    print(f"\n输出目录：{output}", flush=True)
    child = None
    try:
        with (output / 'console.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace',
                                     start_new_session=True)
            # tqdm may use carriage returns; text universal-newline mode splits them.
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
    except KeyboardInterrupt:
        if child and child.poll() is None:
            import signal
            try:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        raise
    result_file = output / 'perf_summary.json'
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    result.update({'returncode': code, 'batch_size': config['batch_size'],
                   'workers': config['workers'], 'output_dir': str(output)})
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_worker':
        return worker(json.loads(Path(sys.argv[2]).read_text()))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['benchmark', 'train'])
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--dataset-root', default='~/26workspace/datasets/ScanNet')
    parser.add_argument('--train-split', default='data_split/scannet_train_stride5.txt')
    parser.add_argument('--val-split', default='data_split/scannet_val_stride5.txt')
    parser.add_argument('--candidates', default='1:0,1:2,2:2,4:4', help='batch:workers 对比列表')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=40, help='每项基准测试更新步数')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--min-available-gib', type=float, default=8,
                        help='自动选择要求：测量阶段至少保留这些系统可用内存')
    parser.add_argument('--val-max-samples', type=int, default=32)
    parser.add_argument('--train-best', action='store_true', help='基准结束后自动以最快合格配置重新训练')
    parser.add_argument('--pin-memory', action='store_true', help='可选单独测试；默认关闭')
    parser.add_argument('--no-cudnn-benchmark', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads, args.epochs, args.val_max_samples) < 1 or args.workers < 0:
        parser.error('batch/threads/epochs/val samples 必须为正，workers 不能为负')
    if args.warmup < 0 or args.steps < args.warmup + 5:
        parser.error('--steps 至少比 --warmup 多 5')
    if not math.isfinite(args.min_available_gib) or args.min_available_gib < 0:
        parser.error('--min-available-gib 必须非负且有限')
    try:
        candidates = [tuple(map(int, item.split(':'))) for item in args.candidates.split(',')]
        if not candidates or any(len(c) != 2 or c[0] <= 0 or c[1] < 0 for c in candidates):
            raise ValueError()
    except ValueError:
        parser.error('--candidates 格式应为 1:0,1:2,2:2,4:4')
    project = Path(args.project_root).expanduser().resolve()
    def local(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (project / path).resolve()
    train_split, val_split = local(args.train_split), local(args.val_split)
    required = [project / 'train_StructMaGNet_scannet.py', train_split, val_split,
                project / 'data/scannet_raw_WH.json']
    required += [project / 'ckpts' / n for n in ('DNET_scannet.pt', 'FNET_scannet.pt', 'MAGNET_scannet.pt')]
    for path in required:
        if not path.is_file():
            parser.error(f'缺少文件：{path}；先完成数据准备')
    for path in (train_split, val_split):
        if not any(line.strip() and not line.lstrip().startswith('#') for line in path.read_text().splitlines()):
            parser.error(f'划分为空：{path}')
    parent = project / 'exp/STRUCTMAGNET/scannet'
    parent.mkdir(parents=True, exist_ok=True)
    suite = Path(tempfile.mkdtemp(prefix='orin_perf_', dir=parent))
    # Freeze the input lists for the suite and the subsequent full run.
    shutil.copy2(train_split, suite / 'train_split.txt')
    shutil.copy2(val_split, suite / 'val_split.txt')
    base = {'project': str(project), 'dataset_root': str(Path(args.dataset_root).expanduser().resolve()),
            'train_split': str(suite / 'train_split.txt'), 'val_split': str(suite / 'val_split.txt'),
            'cpu_threads': args.cpu_threads, 'cudnn_benchmark': not args.no_cudnn_benchmark,
            'pin_memory': args.pin_memory, 'warmup': args.warmup}
    print(f'性能实验目录：{suite}\n保持 FP32；每个样本是一个参考帧及其 4 个源视图。', flush=True)
    try:
        if args.mode == 'benchmark':
            results = []
            for index, (batch, workers) in enumerate(candidates):
                config = dict(base, batch_size=batch, workers=workers, epochs=1, steps=args.steps,
                              val_samples=1, val_degrees=[0],
                              output=str(suite / f'bench_{index}_b{batch}_w{workers}'))
                results.append(execute(config))
                write_json(suite / 'benchmark_results.json', results)
            best = select_best(results, args.min_available_gib)
            print('\n配置对比（训练参考样本/秒，越高越好）：', flush=True)
            for row in results:
                speed = row.get('samples_per_second', 0)
                print(f"B={row['batch_size']} W={row['workers']} "
                      f"samples/s={speed:.3f} exit={row['returncode']} "
                      f"min_RAM_available={row.get('min_system_available_gib', 0):.2f} GiB", flush=True)
            if not best:
                print('没有完成验证、具有足够计时样本且满足内存余量的配置；请查看各 console.log', flush=True)
                return 1
            write_json(suite / 'best_config.json', best)
            print(f"最快合格配置：batch={best['batch_size']} workers={best['workers']}", flush=True)
            if not args.train_best:
                print('加 --train-best 可在基准测试后自动启动完整训练；也可使用 train 模式手动指定配置', flush=True)
                return 0
            batch, workers = best['batch_size'], best['workers']
        else:
            batch, workers = args.batch_size, args.workers
        config = dict(base, batch_size=batch, workers=workers, epochs=args.epochs, steps=0,
                      val_samples=args.val_max_samples, val_degrees=[0, 5, 8], output=str(suite / 'full_train'))
        result = execute(config)
        return result['returncode'] if result['returncode'] >= 0 else 1
    except KeyboardInterrupt:
        print('\n已停止；日志保留于 ' + str(suite), flush=True)
        return 130


if __name__ == '__main__':
    sys.exit(main())
