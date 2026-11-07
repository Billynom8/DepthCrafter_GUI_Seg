"""ffmpeg-backed stand-in for decord.VideoReader.

Used only where decord has no wheel (Apple Silicon, aarch64 Linux). It implements
the part of the decord API that DepthCrafter uses: len(), get_avg_fps(),
get_batch(indices) and the .asnumpy() call on the returned batch.
"""

import json
import subprocess
from fractions import Fraction

import numpy as np


def _probe(path: str):
    """Return (width, height, fps, frame_count) for the first video stream, using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", path,
    ]
    stream = json.loads(subprocess.check_output(cmd))["streams"][0]
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if fps <= 0:
        fps = 30.0

    frame_count = stream.get("nb_frames", "N/A")
    if not str(frame_count).isdigit():
        # Some containers do not store the frame count; count the packets instead.
        count_cmd = [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path,
        ]
        frame_count = subprocess.check_output(count_cmd).decode().strip()
    return int(stream["width"]), int(stream["height"]), fps, int(frame_count)


class _FrameBatch:
    """Array-like result of get_batch(), with the .asnumpy() method decord returns."""

    def __init__(self, array: np.ndarray):
        self._array = array
        self.shape = array.shape

    def asnumpy(self) -> np.ndarray:
        return self._array

    def __len__(self):
        return len(self._array)

    def __getitem__(self, key):
        return self._array[key]


class FFmpegVideoReader:
    """Minimal decord.VideoReader replacement that decodes with the ffmpeg binary.

    width and height are optional; when given, frames are scaled to that size.
    Decoding is streamed, so only the requested frames are kept in memory.
    """

    def __init__(self, path, ctx=None, width=None, height=None, **kwargs):
        self._path = str(path)
        self._src_w, self._src_h, self._fps, self._count = _probe(self._path)
        self._out_w = width if width and width > 0 else self._src_w
        self._out_h = height if height and height > 0 else self._src_h

    def __len__(self) -> int:
        return self._count

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices) -> _FrameBatch:
        indices = [int(i) for i in indices]
        frame_bytes = self._out_w * self._out_h * 3
        if not indices:
            return _FrameBatch(np.zeros((0, self._out_h, self._out_w, 3), dtype=np.uint8))

        wanted = set(indices)
        last = max(wanted)
        frames = {}
        cmd = ["ffmpeg", "-v", "error", "-i", self._path]
        if (self._out_w, self._out_h) != (self._src_w, self._src_h):
            cmd += ["-vf", f"scale={self._out_w}:{self._out_h}"]
        cmd += ["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            n = 0
            while n <= last:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                if n in wanted:
                    frames[n] = np.frombuffer(buf, dtype=np.uint8).reshape(self._out_h, self._out_w, 3)
                n += 1
        finally:
            proc.kill()
            proc.stdout.close()
            proc.wait()

        missing = [i for i in indices if i not in frames]
        if missing:
            raise IndexError(f"Frames {missing[:5]} could not be decoded from {self._path}")
        return _FrameBatch(np.stack([frames[i] for i in indices]))
