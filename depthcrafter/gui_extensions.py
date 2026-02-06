"""
GUI Extensions for DepthCrafter
Wraps and extends core depthcrafter.utils functionality for GUI compatibility.
This module provides enhanced versions of upstream functions with additional features.
"""

from typing import Union, List, Optional, Tuple
import tempfile
import numpy as np
import PIL.Image
import torch
import os
import logging

# Import upstream functions that we're extending
from depthcrafter.utils import (
    save_video as _upstream_save_video,
    ColorMapper as _UpstreamColorMapper,
    vis_sequence_depth as _upstream_vis_sequence_depth,
    dataset_res_dict,
)

# Import GUI I/O utilities
from depthcrafter.gui_io_utils import get_video_stream_info

_logger = logging.getLogger(__name__)


class ColorMapper:
    """
    Enhanced ColorMapper with lazy loading and matplotlib fallback.
    Extends upstream ColorMapper with additional functionality.
    """
    
    def __init__(self, colormap: str = "inferno"):
        self.colormap_name = colormap
        self._cmap_data = None

    def _get_cmap_data(self):
        if self._cmap_data is None:
            try:
                import matplotlib.cm as cm_mpl
                self._cmap_data = torch.tensor(cm_mpl.get_cmap(self.colormap_name).colors)
            except ImportError:
                _logger.warning("Matplotlib.cm not found. ColorMapper will use a basic grayscale fallback.")
                # Fallback to a very simple grayscale if matplotlib is not available
                ramp = torch.linspace(0, 1, 256)
                self._cmap_data = torch.stack([ramp, ramp, ramp], dim=1)  # (N, 3)
        return self._cmap_data

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        if image.ndim not in [2, 3]:
            _logger.error(f"ColorMapper.apply: Image must be 2D or 3D, got {image.ndim}D")
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        cmap_data = self._get_cmap_data().to(image.device)
        
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        
        if v_max == v_min:
            image_normalized = torch.zeros_like(image)
        else:
            image_normalized = (image - v_min) / (v_max - v_min)
        
        image_long = (image_normalized * (len(cmap_data) - 1)).long()
        image_long = torch.clamp(image_long, 0, len(cmap_data) - 1)
        colored_image = cmap_data[image_long]
        return colored_image


def vis_sequence_depth(
    depths: np.ndarray,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    colormap: str = "inferno"
) -> np.ndarray:
    """
    Enhanced version with colormap parameter support.
    
    Args:
        depths: Input depth maps (3D array T, H, W)
        v_min: Minimum value for normalization
        v_max: Maximum value for normalization
        colormap: Name of colormap to use (default: "inferno")
    
    Returns:
        np.ndarray: Visualized depth maps with colormap applied
    """
    if not isinstance(depths, np.ndarray):
        depths = np.array(depths)
    if depths.ndim != 3:
        _logger.error(f"vis_sequence_depth: Input depths must be a 3D array (T, H, W), got {depths.ndim}D")
        raise ValueError(f"Input depths must be a 3D array (T, H, W), got {depths.ndim}D")

    visualizer = ColorMapper(colormap=colormap)
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    
    depths_tensor = torch.from_numpy(depths.astype(np.float32))
    colored_sequence_tensor = visualizer.apply(depths_tensor, v_min=v_min, v_max=v_max)
    colored_sequence_np = colored_sequence_tensor.cpu().numpy()
    
    if colored_sequence_np.shape[-1] == 4:
        colored_sequence_np = colored_sequence_np[..., :3]
    return colored_sequence_np


def read_video_frames(
    video_path: str,
    process_length: int = -1,
    target_fps: float = -1.0,
    target_height: int = 512,
    target_width: int = 768,
    start_frame_index: int = 0,
    num_frames_to_load: int = -1,
    cached_ffprobe_info: Optional[dict] = None
) -> Tuple[np.ndarray, float, int, int, int, int, Optional[dict], Optional[str]]:
    """
    GUI-enhanced version with ffprobe integration and extended return values.
    
    This function extends the upstream read_video_frames with:
    - ffprobe integration for metadata extraction
    - Support for partial frame loading (segments)
    - Flexible target height/width specification
    - Cached ffprobe info for performance
    
    Args:
        video_path: Path to the video file
        process_length: Maximum number of frames to process (-1 for all)
        target_fps: Target FPS for output (-1 for original)
        target_height: Target height for processing
        target_width: Target width for processing
        start_frame_index: Starting frame index for partial loading
        num_frames_to_load: Number of frames to load (-1 for all from start)
        cached_ffprobe_info: Pre-fetched ffprobe info to avoid re-running
    
    Returns:
        Tuple of (frames, fps, orig_h, orig_w, proc_h, proc_w, stream_info, ffprobe_raw)
    """
    import gc
    from decord import VideoReader, cpu
    
    _logger.debug(f"Reading video: {os.path.basename(video_path)}")

    # Get video stream info: first try cached, then call ffprobe if not cached
    video_stream_info_actual = None
    ffprobe_raw_stdout_safe = ""

    if cached_ffprobe_info:
        video_stream_info_actual = cached_ffprobe_info
        _logger.debug(f"Reusing cached ffprobe info for {os.path.basename(video_path)}")
    else:
        ffprobe_result_tuple = get_video_stream_info(video_path)
        if ffprobe_result_tuple:
            video_stream_info_actual, raw_stdout_from_func = ffprobe_result_tuple
            ffprobe_raw_stdout_safe = raw_stdout_from_func if raw_stdout_from_func is not None else ""
            _logger.debug(f"FFprobe called for {os.path.basename(video_path)}")
        else:
            _logger.warning(f"Failed to get video stream info for {os.path.basename(video_path)}")
    
    video_stream_info = video_stream_info_actual
    ffprobe_raw_stdout = ffprobe_raw_stdout_safe

    original_height_detected = 0
    original_width_detected = 0
    original_fps_detected = 0.0
    num_total_frames_detected = 0

    if video_stream_info:
        try:
            original_width_detected = int(video_stream_info.get("width", 0))
            original_height_detected = int(video_stream_info.get("height", 0))
            num_total_frames_detected = int(video_stream_info.get("nb_frames", 0))
            
            r_frame_rate_str = video_stream_info.get("r_frame_rate", "0/1").split('/')
            if len(r_frame_rate_str) == 2 and float(r_frame_rate_str[1]) != 0:
                original_fps_detected = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
            elif len(r_frame_rate_str) == 1:
                original_fps_detected = float(r_frame_rate_str[0])
            _logger.debug(f"FFprobe detected: {original_width_detected}x{original_height_detected} @ {original_fps_detected:.2f} FPS, {num_total_frames_detected} frames")
        except (ValueError, TypeError, ZeroDivisionError) as e:
            _logger.warning(f"Failed to parse ffprobe stream info: {e}. Falling back to Decord.")
            video_stream_info = None
    
    # If ffprobe failed, use Decord as fallback
    if not video_stream_info:
        try:
            temp_reader = VideoReader(video_path, ctx=cpu(0))
            num_total_frames_detected = len(temp_reader)
            if num_total_frames_detected > 0:
                first_frame_shape = temp_reader.get_batch([0]).shape
                original_height_detected = first_frame_shape[1]
                original_width_detected = first_frame_shape[2]
            original_fps_detected = temp_reader.get_avg_fps()
            del temp_reader
            gc.collect()
        except Exception as e:
            _logger.error(f"Failed to get metadata from Decord: {e}")
            return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, 0, 0, 0, 0, None, None

    if num_total_frames_detected == 0:
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, 0, 0, video_stream_info, None

    if original_height_detected == 0 or original_width_detected == 0:
        original_height_detected = 128
        original_width_detected = 128

    # Round to multiples of 64
    final_height_for_decord = max(64, round(target_height / 64) * 64)
    final_width_for_decord = max(64, round(target_width / 64) * 64)

    try:
        vid_reader = VideoReader(video_path, ctx=cpu(0), width=final_width_for_decord, height=final_height_for_decord)
    except Exception as e:
        _logger.error(f"Failed to initialize Decord: {e}")
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, 0, 0, video_stream_info, None

    actual_output_fps = original_fps_detected if target_fps == -1.0 else target_fps
    if actual_output_fps <= 0:
        actual_output_fps = 23.976

    stride = 1
    if original_fps_detected > 0 and actual_output_fps > 0:
        stride = max(round(original_fps_detected / actual_output_fps), 1)

    end_frame_exclusive = num_total_frames_detected
    if num_frames_to_load != -1:
        end_frame_exclusive = min(start_frame_index + num_frames_to_load, num_total_frames_detected)

    frames_idx = list(range(start_frame_index, end_frame_exclusive, stride))

    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    
    if not frames_idx:
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, final_height_for_decord, final_width_for_decord, video_stream_info, None

    frames_batch = vid_reader.get_batch(frames_idx)
    frames_numpy = frames_batch.asnumpy().astype("float32") / 255.0

    del vid_reader
    gc.collect()

    actual_processed_height = frames_numpy.shape[1]
    actual_processed_width = frames_numpy.shape[2]

    return frames_numpy, actual_output_fps, original_height_detected, original_width_detected, actual_processed_height, actual_processed_width, video_stream_info, ffprobe_raw_stdout


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray],
    output_video_path: str = None,
    fps: Union[int, float] = 10.0,
    crf: int = 18,
    output_format: Optional[str] = None
) -> str:
    """
    GUI-enhanced version with multi-format support (HEVC, 10-bit).
    
    This function extends the upstream save_video with:
    - HEVC (H.265) encoding support
    - 10-bit color depth support
    - Multiple output formats (mp4, main10_mp4)
    
    Args:
        video_frames: List of frames or numpy array
        output_video_path: Path to save the video
        fps: Frames per second
        crf: Constant Rate Factor for quality
        output_format: Output format ("mp4", "main10_mp4", etc.)
    
    Returns:
        str: Path to saved video
    """
    import mediapy
    
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    elif output_format == "main10_mp4" and not output_video_path.lower().endswith(".mp4"):
        _logger.warning(f"Saving HEVC to .mp4 container, but path doesn't end with .mp4")

    # Determine scaling factor and dtype based on output format
    target_max_val = 255.0
    target_dtype = np.uint8
    
    if output_format == "main10_mp4":
        target_max_val = 65535.0
        target_dtype = np.uint16

    # Frame conversion logic
    if isinstance(video_frames, np.ndarray):
        if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
            video_frames = (video_frames * target_max_val).astype(target_dtype)
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], np.ndarray):
        processed_frames = []
        for frame in video_frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                processed_frames.append((frame * target_max_val).astype(target_dtype))
            elif frame.dtype == np.uint8 or frame.dtype == np.uint16:
                processed_frames.append(frame)
            else:
                raise ValueError(f"Unsupported dtype in list: {frame.dtype}")
        video_frames = processed_frames
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    elif isinstance(video_frames, list) and len(video_frames) == 0:
        _logger.warning("Empty list of frames. Cannot save video.")
        return output_video_path
    else:
        raise ValueError("video_frames must be list/array of np.ndarray or PIL.Image")

    mediapy_kwargs = {'fps': fps}
    ffmpeg_custom_args = []

    if output_format == "main10_mp4" and output_video_path.lower().endswith(".mp4"):
        mediapy_kwargs['codec'] = 'libx265'
        ffmpeg_custom_args.extend(['-pix_fmt', 'yuv420p10le'])
        ffmpeg_custom_args.extend(['-tag:v', 'hvc1'])
        mediapy_kwargs['crf'] = crf
    else:
        if output_video_path.lower().endswith(".mp4"):
            if output_format == "mp4" or output_format is None:
                mediapy_kwargs['codec'] = 'libx264'
                ffmpeg_custom_args.extend(['-pix_fmt', 'yuv420p'])
        mediapy_kwargs['crf'] = crf

    if ffmpeg_custom_args:
        mediapy_kwargs['ffmpeg_args'] = ffmpeg_custom_args

    try:
        mediapy.write_video(output_video_path, video_frames, **mediapy_kwargs)
    except Exception as e:
        _logger.error(f"Error writing video: {e}")
        raise
    
    return output_video_path
