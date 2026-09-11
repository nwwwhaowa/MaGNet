#!/usr/bin/env python3
"""Download an official ScanNet validation .sens, then start preparation/training.

Place next to start_scannet_training.py in MaGNet and run:
    python download_scannet_val.py
Download only:
    python download_scannet_val.py --download-only

Uses the existing aria2_batch.txt URL pattern and the installed aria2c executable.
Does not install packages, delete raw data, or modify the official scene lists.
The URL is inferred from your existing download source; resource-specific signed
URLs may not permit substitution and will require a fresh valid download URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
SENS_PATH_PATTERN = re.compile(r"/(scene\d{4}_\d{2})/\1\.sens$")


def derive_url(batch_file: Path, scene: str) -> str:
    if not batch_file.is_file():
        raise FileNotFoundError(f"未找到已有下载列表：{batch_file}")
    for line in batch_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.split():
            if not token.startswith(("http://", "https://")):
                continue
            url = urlsplit(token)
            if url.netloc and SENS_PATH_PATTERN.search(url.path):
                path = SENS_PATH_PATTERN.sub(f"/{scene}/{scene}.sens", url.path)
                return urlunsplit(url._replace(path=path))
    raise ValueError(
        f"无法从 {batch_file} 识别 /sceneXXXX_YY/sceneXXXX_YY.sens 地址。"
        "请提供下载条目，或用 --url 指定该场景的有效下载地址。"
    )


def run(command, cwd):
    # No shell: the training command is actually executed as a subprocess.
    with subprocess.Popen([str(value) for value in command], cwd=str(cwd)) as child:
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".", help="MaGNet 项目目录，默认当前目录")
    parser.add_argument("--dataset-root", default="~/26workspace/datasets/ScanNet")
    parser.add_argument("--scene", default="scene0019_00", help="必须属于本地官方验证列表")
    parser.add_argument("--aria2-batch", help="默认使用 ScanNet/aria2_batch.txt")
    parser.add_argument("--url", help="可选：该场景的有效直接下载地址")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--download-only", action="store_true", help="只下载，不调用训练启动脚本")
    args = parser.parse_args()
    if not SCENE_PATTERN.fullmatch(args.scene):
        parser.error("--scene 必须采用 scene0019_00 格式")
    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")

    project = Path(args.project_root).expanduser().resolve()
    root = Path(args.dataset_root).expanduser().resolve()
    launcher = project / "start_scannet_training.py"
    batch = Path(args.aria2_batch).expanduser().resolve() if args.aria2_batch else root / "aria2_batch.txt"
    try:
        if not project.is_dir():
            raise FileNotFoundError(f"项目目录不存在：{project}")
        if not args.download_only and not launcher.is_file():
            raise FileNotFoundError(
                f"请把 start_scannet_training.py 放到 {project}；"
                "仅下载可加 --download-only"
            )
        official_file = root / "official_splits/scannetv2_val.txt"
        official = {line.split()[0] for line in official_file.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}
        if args.scene not in official:
            raise ValueError(f"{args.scene} 不属于本地官方验证列表")
        aria2 = shutil.which("aria2c")
        if not aria2:
            raise RuntimeError("未找到 aria2c。请先执行 sudo apt install aria2，再重新运行")
        url = args.url if args.url else derive_url(batch, args.scene)
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("下载地址必须是有效的 HTTP/HTTPS URL")
        directory = root / "raw_sens/scans" / args.scene
        directory.mkdir(parents=True, exist_ok=True)
        sens = directory / f"{args.scene}.sens"
        progress_file = Path(str(sens) + ".aria2")
        print(f"验证场景：{args.scene}\n目标文件：{sens}", flush=True)
        print("开始下载/断点续传；成功后检查输出文件。", flush=True)
        code = run([
            aria2, "--continue=true", "--auto-file-renaming=false",
            "--allow-overwrite=false", "--max-connection-per-server=4", "--split=4",
            f"--dir={directory}", f"--out={sens.name}", url,
        ], project)
        if code:
            print(f"下载失败，aria2c 退出码 {code}；没有启动训练。", file=sys.stderr)
            print("可重新运行以续传；HTTP 403/404 时请核对地址，可使用 --url。", file=sys.stderr)
            return code if code > 0 else 1
        if not sens.is_file() or sens.stat().st_size == 0 or progress_file.exists():
            raise RuntimeError("下载输出为空、缺失或仍存在 .aria2 状态；没有启动训练")
        print(f"下载工具已完成：{sens.stat().st_size / (1024 ** 3):.3f} GiB", flush=True)
        if args.download_only:
            print("只下载模式完成。后续执行：python start_scannet_training.py", flush=True)
            return 0
        print("开始调用 start_scannet_training.py：导出、生成划分并训练。", flush=True)
        return run([
            sys.executable, "-u", launcher, "--project-root", project,
            "--dataset-root", root, "--epochs", args.epochs,
        ], project)
    except KeyboardInterrupt:
        print("\n已中断；重新运行可尝试续传。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
