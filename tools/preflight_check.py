"""Preflight check for DepthCrafter_GUI_Seg.

Checks the large external pieces the app needs, downloads what can be
fetched automatically, skips what is already present, and writes
MISSING_MANUAL.txt for anything that still has to be installed by hand.

Run from the project root with the project's own interpreter:
    Windows:        .venv\\Scripts\\python.exe tools\\preflight_check.py [--no-download]
    Linux / macOS:  .venv/bin/python tools/preflight_check.py [--no-download]

Exit code: 0 when nothing is missing, 1 when manual steps remain.
"""

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANUAL_FILE = PROJECT_DIR / "MISSING_MANUAL.txt"
FFMPEG_BIN_DIR = PROJECT_DIR / "tools" / "ffmpeg" / "bin"
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
FFMPEG_MANUAL_URL = "https://www.gyan.dev/ffmpeg/builds/"

# Static ffmpeg builds from BtbN. Other platforms (macOS, non-x86 Linux) have no automatic source.
if sys.platform == "win32":
    FFMPEG_ARCHIVE_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
elif sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
    FFMPEG_ARCHIVE_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
else:
    FFMPEG_ARCHIVE_URL = None

# Triton 3.5.1 matches the Triton that PyTorch 2.9.x uses on Linux; a cp312 win_amd64 wheel exists on PyPI.
# Linux gets Triton from torch itself, so only Windows needs this step.
TRITON_SPEC = "triton-windows==3.5.1.post24"

# Python packages are installed by `uv sync`; here we only check they import.
REQUIRED_MODULES = [
    "numpy", "PIL", "torch", "diffusers", "transformers", "accelerate",
    "huggingface_hub", "OpenEXR", "Imath", "imageio", "mediapy", "tkinterdnd2", "ttkthemes",
]
# decord has no wheel on Apple Silicon or aarch64 Linux; without it the app uses an ffmpeg-based reader.
OPTIONAL_MODULES = ["decord"]

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


def ffmpeg_manual_hint() -> str:
    if sys.platform == "win32":
        return (f"Get a Windows build from {FFMPEG_MANUAL_URL} and copy ffmpeg.exe and "
                f"ffprobe.exe into {FFMPEG_BIN_DIR}")
    if sys.platform == "darwin":
        return f"Install it with Homebrew (brew install ffmpeg), or copy ffmpeg and ffprobe into {FFMPEG_BIN_DIR}"
    return f"Install it with your package manager (e.g. sudo apt install ffmpeg), or copy ffmpeg and ffprobe into {FFMPEG_BIN_DIR}"


def download_ffmpeg(dest_bin: Path) -> None:
    """Download the static build for this platform and copy ffmpeg and ffprobe into dest_bin."""
    if FFMPEG_ARCHIVE_URL is None:
        raise RuntimeError("no automatic ffmpeg download for this platform")
    wanted = {f"ffmpeg{EXE_SUFFIX}", f"ffprobe{EXE_SUFFIX}"}
    dest_bin.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / FFMPEG_ARCHIVE_URL.rsplit("/", 1)[-1]
        urllib.request.urlretrieve(FFMPEG_ARCHIVE_URL, archive_path)
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    name = os.path.basename(member)
                    if "/bin/" in member and name in wanted:
                        with archive.open(member) as src, open(dest_bin / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        else:
            with tarfile.open(archive_path, "r:xz") as archive:
                for member in archive.getmembers():
                    name = os.path.basename(member.name)
                    if member.isfile() and "/bin/" in member.name and name in wanted:
                        with archive.extractfile(member) as src, open(dest_bin / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
    for name in wanted:
        target = dest_bin / name
        if not target.is_file():
            raise RuntimeError(f"{name} not found in the downloaded archive")
        if EXE_SUFFIX == "":
            target.chmod(0o755)


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
                      + "\n  Fix: run `uv sync` in the project folder (run.sh and the .bat do this automatically).")
    else:
        print("  all present")
    missing_optional = [m for m in OPTIONAL_MODULES if not have_module(m)]
    if missing_optional:
        print("  optional, not installed: " + ", ".join(missing_optional) + " (ffmpeg reader is used instead)")

    print("== ffmpeg / ffprobe")
    ensure_project_ffmpeg_on_path()
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("  present")
    elif download and FFMPEG_ARCHIVE_URL:
        print("  not found, downloading a static build...")
        try:
            download_ffmpeg(FFMPEG_BIN_DIR)
            ensure_project_ffmpeg_on_path()
            print(f"  downloaded to {FFMPEG_BIN_DIR}")
        except Exception as exc:
            print(f"  download failed: {type(exc).__name__}: {exc}")
            manual.append("ffmpeg / ffprobe not found and the automatic download failed.\n  " + ffmpeg_manual_hint())
    else:
        print("  not found (run without --no-download to fetch it)" if FFMPEG_ARCHIVE_URL else "  not found")
        manual.append("ffmpeg / ffprobe not found.\n  " + ffmpeg_manual_hint())

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
                lines.append("  then run the launcher again.")
            else:
                lines.append("  Fix: check the network or proxy, then run the launcher again.")
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
