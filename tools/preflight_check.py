"""Preflight check for DepthCrafter_GUI_Seg.

Checks the large external pieces the app needs, downloads what can be
fetched automatically, skips what is already present, and writes
MISSING_MANUAL.txt for anything that still has to be installed by hand.

Run from the project root with the project's own interpreter:
    .venv\\Scripts\\python.exe tools\\preflight_check.py [--no-download]

Exit code: 0 when nothing is missing, 1 when manual steps remain.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANUAL_FILE = PROJECT_DIR / "MISSING_MANUAL.txt"
FFMPEG_BIN_DIR = PROJECT_DIR / "tools" / "ffmpeg" / "bin"
FFMPEG_ZIP_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_MANUAL_URL = "https://www.gyan.dev/ffmpeg/builds/"
# Triton 3.5.1 matches the Triton that PyTorch 2.9.x uses on Linux; a cp312 win_amd64 wheel exists on PyPI.
TRITON_SPEC = "triton-windows==3.5.1.post24"

# Python packages are installed by `uv sync`; here we only check they import.
REQUIRED_MODULES = [
    "numpy", "PIL", "torch", "diffusers", "transformers", "accelerate",
    "huggingface_hub", "OpenEXR", "Imath", "imageio", "mediapy", "decord",
    "tkinterdnd2", "ttkthemes",
]

# repo id, allow_patterns (None = whole repo), files that must exist after download, description, gated
HF_MODELS = [
    ("tencent/DepthCrafter", None,
     ["config.json", "*.safetensors"],
     "DepthCrafter UNet", False),
    ("stabilityai/stable-video-diffusion-img2vid-xt", ["*.json", "*fp16*.safetensors"],
     ["model_index.json", "unet/*fp16*.safetensors", "vae/*fp16*.safetensors", "image_encoder/*fp16*.safetensors"],
     "Stable Video Diffusion XT base pipeline (fp16 parts only)", True),
]


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_project_ffmpeg_on_path() -> None:
    if FFMPEG_BIN_DIR.is_dir():
        os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")


def download_ffmpeg(dest_bin: Path) -> None:
    """Download the BtbN Windows build and copy ffmpeg.exe and ffprobe.exe into dest_bin."""
    dest_bin.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "ffmpeg.zip"
        urllib.request.urlretrieve(FFMPEG_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                name = os.path.basename(member)
                if "/bin/" in member and name in ("ffmpeg.exe", "ffprobe.exe"):
                    with archive.open(member) as src, open(dest_bin / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        if not (dest_bin / exe).is_file():
            raise RuntimeError(f"{exe} not found in the downloaded archive")


def ensure_triton() -> str:
    """Install triton-windows into the project venv (Windows only). Returns a status message.

    Triton is optional: xformers uses it for some of its kernels when xformers is enabled.
    The install goes through uv because uv-created venvs have no pip.
    """
    if sys.platform != "win32":
        return "skipped (Windows only)"
    if have_module("triton"):
        return "already present"
    uv = shutil.which("uv")
    if not uv:
        return "not installed: uv was not found on PATH"
    result = subprocess.run([uv, "pip", "install", "--python", sys.executable, TRITON_SPEC],
                            capture_output=True, text=True)
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        return "install failed: " + (lines[-1] if lines else "unknown error")
    return f"installed {TRITON_SPEC}"


def check_hf_model(repo: str, patterns, sentinels, download: bool):
    """Return (ok, message). Uses the local cache first, then downloads only what is missing."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub is not installed (run uv sync)"

    def complete(path) -> bool:
        return path is not None and all(any(Path(path).glob(p)) for p in sentinels)

    try:
        local_path = snapshot_download(repo_id=repo, allow_patterns=patterns, local_files_only=True)
        if complete(local_path):
            return True, f"already present ({local_path})"
    except Exception:
        pass  # not in cache yet, or the cache is incomplete

    if not download:
        return False, "not in cache (run without --no-download to fetch it)"

    try:
        path = snapshot_download(repo_id=repo, allow_patterns=patterns)
    except Exception as exc:  # network error, gated repo without license/token, or 401/403
        return False, f"download failed: {type(exc).__name__}: {exc}"
    if complete(path):
        return True, f"downloaded ({path})"
    return False, "download finished but expected files are still missing"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and fetch large dependencies.")
    parser.add_argument("--no-download", action="store_true", help="only check, never download")
    args = parser.parse_args()
    download = not args.no_download
    manual = []  # one entry per item that still needs a manual step

    print("== Python packages")
    missing_modules = [m for m in REQUIRED_MODULES if not have_module(m)]
    if missing_modules:
        print("  missing: " + ", ".join(missing_modules))
        manual.append("Python packages missing: " + ", ".join(missing_modules)
                      + "\n  Fix: run `uv sync` in the project folder (the .bat does this automatically).")
    else:
        print("  all present")

    print("== ffmpeg / ffprobe")
    ensure_project_ffmpeg_on_path()
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("  present")
    elif download:
        print("  not found, downloading the Windows build...")
        try:
            download_ffmpeg(FFMPEG_BIN_DIR)
            ensure_project_ffmpeg_on_path()
            print(f"  downloaded to {FFMPEG_BIN_DIR}")
        except Exception as exc:
            print(f"  download failed: {type(exc).__name__}: {exc}")
            manual.append("ffmpeg / ffprobe not found and the automatic download failed.\n"
                          f"  Manual: get a Windows build from {FFMPEG_MANUAL_URL}\n"
                          f"  and copy ffmpeg.exe and ffprobe.exe into {FFMPEG_BIN_DIR}")
    else:
        print("  not found (run without --no-download to fetch it)")
        manual.append(f"ffmpeg / ffprobe not found. Manual: copy them into {FFMPEG_BIN_DIR}")

    # Optional: a failure here only prints a warning, it does not fail the run.
    print("== Triton (optional, Windows only; used by xformers when xformers is enabled)")
    print("  " + ensure_triton())

    print(f"== Hugging Face models (HF_HOME={os.environ.get('HF_HOME', 'default user cache')})")
    for repo, patterns, sentinels, description, gated in HF_MODELS:
        ok, message = check_hf_model(repo, patterns, sentinels, download)
        print(f"  {repo}: {message}")
        if not ok:
            lines = [f"{description} [{repo}]", f"  Status: {message}",
                     f"  Page: https://huggingface.co/{repo}"]
            if gated:
                lines.append("  This repo is gated: open the page, accept the license, create a")
                lines.append("  read token at https://huggingface.co/settings/tokens, set HF_TOKEN to it,")
                lines.append("  then run the .bat again.")
            else:
                lines.append("  Fix: check the network or proxy, then run the .bat again.")
            manual.append("\n".join(lines))

    if manual:
        MANUAL_FILE.write_text("\n\n".join(manual) + "\n", encoding="utf-8")
        print(f"\n{len(manual)} item(s) still need attention. See {MANUAL_FILE.name}")
        return 1
    if MANUAL_FILE.exists():
        MANUAL_FILE.unlink()
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
