#!/usr/bin/env python3
"""Prepare existing ScanNet .sens data using the project's two ScanNet tools.

Run from MaGNet:
    python prepare_scannet.py
Default excluded scene: scene0010_00. After repairing its download:
    python prepare_scannet.py --include-scene0010-00

Uses the current Python environment. Does not install packages or start training.
Raw data is never moved/deleted. Failed scene exports are moved to a unique backup.
Exit codes: 0 = split checks passed, 1 = preparation failed, 130 = interrupted.
A successful run is not proof of complete ScanNet coverage or model readiness.
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
import time
import urllib.request

SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
BASE_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark"


def scene_list(text):
    rows = [line.split()[0] for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(not SCENE_RE.fullmatch(s) for s in rows):
        raise ValueError("场景列表为空或格式错误（应为 scene0000_00）")
    if len(rows) != len(set(rows)):
        raise ValueError("场景列表包含重复场景")
    return rows


def ensure_split(path, split, offline, log):
    if path.is_file():
        return scene_list(path.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"离线模式缺少文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/scannetv2_{split}.txt"
    for attempt in range(3):
        try:
            log(f"下载官方 {split} 场景列表（第 {attempt + 1} 次）")
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8")
            rows = scene_list(text)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=path.parent, delete=False) as f:
                f.write(text)
                tmp = Path(f.name)
            try:
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            return rows
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"下载失败：{url}；可手动下载到 {path}。{exc}") from exc
            time.sleep(1)


def run(command, cwd, log):
    log("执行：" + " ".join(map(str, command)))
    with subprocess.Popen([str(x) for x in command], cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as child:
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


def quarantine(root, scene, report, log):
    path = root / "scans" / scene
    if path.exists() or path.is_symlink():
        backup = Path(tempfile.mkdtemp(prefix="incomplete_export_", dir=root))
        destination = backup / scene
        shutil.move(str(path), str(destination))
        report["quarantined"].append(str(destination))
        log(f"已隔离导出结果：{destination}")


def check_rows(path, allowed, stride):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] not in allowed:
            raise ValueError(f"划分行格式错误或场景不在成功导出列表中：{line}")
        frame = int(fields[1])
        if frame < 0 or frame % stride:
            raise ValueError(f"参考帧号不符合 stride：{line}")
        rows.append((fields[0], frame))
    if not rows:
        raise ValueError(f"{path.name} 没有有效样本；需检查窗口/位姿或补充该集合的数据")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{path.name} 存在重复样本")
    return {s for s, _ in rows}, len(rows)


def inspect_project(project, report, log):
    paths, excerpts = [], []
    excluded = {".git", ".venv", "venv", "__pycache__", "node_modules", "wandb"}
    pattern = re.compile(r"train|pretrain|checkpoint|scannet|weights", re.I)
    for directory, dirs, files in os.walk(project):
        relative = Path(directory).relative_to(project)
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith("."))
        if len(relative.parts) >= 2:
            dirs[:] = []
        for name in sorted(files):
            path = Path(directory) / name
            lower = name.lower()
            if (lower.startswith("readme") or path.suffix in {".yaml", ".yml"}
                    or ("train" in lower and path.suffix in {".py", ".sh"})
                    or ("config" in lower and path.suffix == ".txt")):
                paths.append(str(path.relative_to(project)))
            if lower.startswith("readme") and path.stat().st_size < 500_000:
                for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line) and len(excerpts) < 100:
                        excerpts.append(f"{path.relative_to(project)}:{number}: {line}")
    report["training_files"] = paths
    report["readme_matches"] = excerpts
    log("训练入口/配置文件（仅发现，未验证配置或权重）：")
    for entry in paths + excerpts:
        log(entry)
    commands = [["nvidia-smi"], [sys.executable, "-c",
        'import torch; print("PyTorch:", torch.__version__); '
        'print("CUDA:", torch.version.cuda); '
        'print("CUDA available:", torch.cuda.is_available()); '
        'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); '
        'raise SystemExit(0 if torch.cuda.is_available() else 1)']]
    report["environment_checks"] = []
    for command in commands:
        try:
            code = run(command, project, log)
        except OSError as exc:
            log(f"环境检查无法执行：{exc}")
            code = 127
        report["environment_checks"].append({"command": command, "returncode": code})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".", help="MaGNet 项目目录，默认当前目录")
    parser.add_argument("--dataset-root", default="~/26workspace/datasets/ScanNet")
    parser.add_argument("--skip-scene", action="append", default=[], help="额外跳过的场景，可重复")
    parser.add_argument("--include-scene0010-00", action="store_true", help="修复下载后重新处理 scene0010_00")
    parser.add_argument("--stride", type=int, default=5, help="导出和参考帧步长")
    parser.add_argument("--window-radius", type=int, default=20)
    parser.add_argument("--num-source-views", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="使用已有官方划分列表，禁止下载")
    parser.add_argument("--skip-env-check", action="store_true", help="跳过训练入口和 GPU 环境检查")
    args = parser.parse_args()
    if (args.stride <= 0 or args.window_radius <= 0 or args.num_source_views <= 0
            or args.num_source_views % 2 or args.window_radius % (args.num_source_views // 2)):
        parser.error("步长/窗口必须为正，源视图数必须为正偶数，窗口需被半数源视图整除")
    if (args.window_radius // (args.num_source_views // 2)) % args.stride:
        parser.error("窗口帧间隔必须为导出 stride 的整数倍")
    if any(not SCENE_RE.fullmatch(s) for s in args.skip_scene):
        parser.error("--skip-scene 应为 scene0000_00 格式")
    project = Path(args.project_root).expanduser().resolve()
    root = Path(args.dataset_root).expanduser().resolve()
    exporter = project / "tools/scannet/export_sens_py3.py"
    builder = project / "tools/scannet/build_scannet_frame_splits.py"
    for path in (exporter, builder):
        if not path.is_file():
            parser.error(f"缺少 {path}；请在 MaGNet 目录运行或指定 --project-root")
    raw_files = sorted((root / "raw_sens/scans").glob("*/*.sens"))
    if not raw_files:
        parser.error(f"没有找到 {root}/raw_sens/scans/*/*.sens")
    run_dir = Path(tempfile.mkdtemp(prefix="scannet_check_", dir=project))
    report = {"dataset_root": str(root), "project_root": str(project),
              "arguments": vars(args), "exported": [], "skipped": {}, "failed": {},
              "quarantined": [], "split_checks_passed": False}
    status = 1
    with (run_dir / "check.log").open("w", encoding="utf-8") as logfile:
        def log(message):
            print(message, flush=True)
            logfile.write(message + "\n")
            logfile.flush()
        log(f"日志目录：{run_dir}")
        log("不会自动启动训练。原始 .sens 文件保持不变。")
        try:
            # Fail early on missing exporter dependencies before moving any data.
            if run([sys.executable, "-c", "import numpy; from PIL import Image"], project, log):
                raise RuntimeError("当前 Python 环境缺少 numpy 或 Pillow，请先补齐依赖")
            excluded = set(args.skip_scene)
            if not args.include_scene0010_00:
                excluded.add("scene0010_00")
            for scene in sorted(excluded):
                quarantine(root, scene, report, log)
                report["skipped"][scene] = "显式排除"
            for sens in raw_files:
                scene = sens.stem
                if not SCENE_RE.fullmatch(scene) or sens.parent.name != scene:
                    raise ValueError(f"非标准场景路径：{sens}")
                if scene in excluded:
                    continue
                if Path(str(sens) + ".aria2").exists():
                    report["skipped"][scene] = "存在 .aria2 下载状态文件"
                    quarantine(root, scene, report, log)
                    log(f"跳过未完成下载：{scene}")
                    continue
                before = sens.stat()
                code = run([sys.executable, exporter, "--sens", sens, "--dataset_root", root,
                            "--scans_dir", "scans", "--stride", args.stride], project, log)
                after = sens.stat()
                changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                pending = Path(str(sens) + ".aria2").exists()
                if code or changed or pending:
                    report["failed"][scene] = {"returncode": code, "file_changed": changed,
                                               "download_pending": pending}
                    quarantine(root, scene, report, log)
                    log(f"排除失败/仍在变化的场景：{scene}")
                    continue
                report["exported"].append(scene)
            available = set(report["exported"])
            official = {s: ensure_split(root / "official_splits" / f"scannetv2_{s}.txt",
                                       s, args.offline, log) for s in ("train", "val")}
            if set(official["train"]) & set(official["val"]):
                raise ValueError("官方场景列表出现训练/验证交叉，请检查文件来源")
            selected = {s: set(official[s]) & available for s in official}
            report["coverage"] = {s: {"official": len(official[s]), "available": len(selected[s])}
                                  for s in official}
            for split in selected:
                log(f"{split} 场景覆盖：{len(selected[split])}/{len(official[split])}")
                if not selected[split]:
                    raise ValueError(f"{split} 没有成功导出的官方场景；请补充对应场景数据")
                (run_dir / f"{split}_scenes.txt").write_text(
                    "".join(s + "\n" for s in sorted(selected[split])), encoding="utf-8")
            # Build into this run's directory; publish only if both lists pass.
            outputs = {s: run_dir / f"scannet_{s}_stride{args.stride}.txt" for s in selected}
            command = [sys.executable, builder, "--dataset_root", root,
                       "--train_scenes", run_dir / "train_scenes.txt",
                       "--val_scenes", run_dir / "val_scenes.txt",
                       "--train_out", outputs["train"], "--val_out", outputs["val"],
                       "--reference_stride", args.stride, "--window_radius", args.window_radius,
                       "--num_source_views", args.num_source_views]
            if run(command, project, log):
                raise RuntimeError("帧划分工具执行失败，查看上方输出")
            report["samples"] = {}
            for split, path in outputs.items():
                scenes, count = check_rows(path, selected[split], args.stride)
                report["samples"][split] = {"scenes": len(scenes), "samples": count}
                log(f"{split}: {len(scenes)} scenes, {count} samples")
            destination = project / "data_split"
            destination.mkdir(exist_ok=True)
            for path in outputs.values():
                target = destination / path.name
                if target.exists():
                    shutil.copy2(target, run_dir / ("previous_" + target.name))
                shutil.copy2(path, target)
            report["split_checks_passed"] = True
            log("帧划分检查通过；尚未验证模型前向/反向传播、数据加载器和预训练权重。")
            status = 0
        except KeyboardInterrupt:
            report["error"] = "用户中断；再次运行将重新检查"
            log(report["error"])
            status = 130
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            log("检查未通过：" + report["error"])
        finally:
            if not args.skip_env_check and status != 130:
                try:
                    inspect_project(project, report, log)
                except KeyboardInterrupt:
                    status = 130
                except Exception as exc:
                    report["environment_error"] = str(exc)
                    log(f"环境检查未完成：{exc}")
            report["exit_code"] = status
            (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            log(f"完成：成功导出 {len(report['exported'])}，跳过 {len(report['skipped'])}，"
                f"失败 {len(report['failed'])}；退出码 {status}")
            log(f"请提供这两个文件用于确定训练命令：{run_dir / 'check.log'} 和 {run_dir / 'report.json'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
