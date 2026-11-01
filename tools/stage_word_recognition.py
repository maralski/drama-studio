#!/usr/bin/env python3
"""Stage the pinned local recognizer beside CLI products or inside an app."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / "Sources/WordRecognitionProtocol/Assets.json").read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--app", type=Path)
    destination.add_argument("--cli", type=Path, help="assemble a relocatable drama CLI directory")
    args = parser.parse_args()
    executable_dir = args.app / "Contents/MacOS" if args.app else args.cli or args.bin
    resources = args.app / "Contents/Resources" if args.app else executable_dir
    frameworks = args.app / "Contents/Frameworks" if args.app else executable_dir
    for directory in (executable_dir, resources, frameworks):
        directory.mkdir(parents=True, exist_ok=True)
    helper = executable_dir / ASSETS["helperName"]
    source_helper = args.bin / ASSETS["helperName"]
    if helper.resolve() != source_helper.resolve():
        shutil.copy2(source_helper, helper)
    if args.app or args.cli:
        source = args.bin / "whisper.framework"
        subprocess.run(["/usr/bin/ditto", str(source), str(frameworks / source.name)], check=True)
    if args.app:
        subprocess.run(["/usr/bin/install_name_tool", "-add_rpath", "@executable_path/../Frameworks", str(helper)], check=True)
    if args.cli:
        shutil.copy2(args.bin / "drama", args.cli / "drama")
        subprocess.run(["/usr/bin/ditto", str(args.bin / "DramaStudio_DramaEngine.bundle"),
                        str(args.cli / "DramaStudio_DramaEngine.bundle")], check=True)
        for notice in ("LICENSE", "NOTICE", "LICENSE-FISH-AUDIO", "THIRD_PARTY_NOTICES"):
            shutil.copy2(ROOT / notice, args.cli / notice)
    models = resources / ASSETS["modelDirectory"]
    models.mkdir(exist_ok=True)
    for asset in ASSETS["models"]:
        source = ROOT / "Vendor/WordRecognition/Models" / asset["fileName"]
        target = models / asset["fileName"]
        if target.exists() and target.stat().st_size == source.stat().st_size and target.stat().st_mtime_ns == source.stat().st_mtime_ns:
            continue
        shutil.copy2(source, target)

if __name__ == "__main__":
    main()
