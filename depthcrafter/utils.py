from typing import Union, List, Optional, Callable, Tuple
import tempfile
import numpy as np
import PIL.Image
# import matplotlib.cm as cm # No longer directly used here, ColorMapper will import it
import mediapy # Ensure mediapy is installed: pip install mediapy
import torch
from decord import VideoReader, cpu # Ensure decord is installed: pip install decord
import os
import shutil
import imageio # Added, as it's used for PNG/EXR saving
import time # Added for get_formatted_timestamp (though message_catalog has its own)
import json # Added for JSON utilities
import gc # Added for define_video_segments

# Import from the new message catalog
from message_catalog import (
    log_message,
    INFO, DEBUG, WARNING, ERROR, CRITICAL # For direct level checks if ever needed
)

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}


try:
    import OpenEXR
    import Imath
    _OPENEXR_AVAILABLE_IN_UTILS = True
except ImportError:
    _OPENEXR_AVAILABLE_IN_UTILS = False
    log_message("OPENEXR_UNAVAILABLE", context="utils.py") # Log using the new system

# --- NEW UTILITY FUNCTIONS ---

def format_duration(seconds: float) -> str:
    """Converts seconds to H:MM:SS.s format."""
    if seconds < 0:
        return "0:00:00.0"
    
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours}:{minutes:02}:{seconds:04.1f}"

def get_formatted_timestamp_utils() -> str: # Renamed to avoid clash with message_catalog's internal one
    """Generates a timestamp string in HH:MM:SS.s format for use within utils if needed directly."""
    current_time_val = time.time()
    time_struct = time.localtime(current_time_val)
    milliseconds_tenths = int((current_time_val - int(current_time_val)) * 10)
    return f"{time_struct.tm_hour:02d}:{time_struct.tm_min:02d}:{time_struct.tm_sec:02d}.{milliseconds_tenths}"

def get_segment_output_folder_name(original_video_basename: str) -> str:
    """Returns the standard name for a segment subfolder."""
    return f"{original_video_basename}_seg"

def get_segment_npz_output_filename(original_video_basename: str, segment_id: int, total_segments: int) -> str:
    """Returns the standard NPZ filename for a segment."""
    return f"{original_video_basename}_depth_{segment_id + 1}of{total_segments}.npz"

def get_full_video_output_filename(original_video_basename: str, extension: str = "mp4") -> str:
    """Returns the standard filename for a full video output."""
    return f"{original_video_basename}_depth.{extension}"


def get_sidecar_json_filename(base_filepath_with_ext: str) -> str:
    """Returns the corresponding .json sidecar filename for a given base file."""
    return os.path.splitext(base_filepath_with_ext)[0] + ".json"


def define_video_segments(
    video_path: str,
    original_basename: str,
    gui_target_fps_setting: int,
    gui_process_length_overall: int,
    gui_segment_output_window_frames: int,
    gui_segment_output_overlap_frames: int
    # log_func: Callable[[str], None] # Removed, will use global log_message
) -> Tuple[List[dict], Optional[dict]]:
    """
    Defines video segments based on input parameters.
    Returns:
        A tuple containing:
        - A list of segment job dictionaries.
        - A base job info dictionary (common details for the video).
          Returns None for base_job_info if video metadata read fails.
    """
    segment_jobs = []
    base_job_info_for_video = {}

    total_raw_frames_in_original_video = 0
    original_video_fps = 30.0
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_raw_frames_in_original_video = len(vr)
        original_video_fps = vr.get_avg_fps()
        del vr
        gc.collect()
        if original_video_fps <= 0:
            log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Invalid original FPS ({original_video_fps}). Assuming 30 FPS.")
            original_video_fps = 30.0
    except Exception as e:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Error getting metadata of {video_path}: {e}")
        return [], None

    base_job_info_for_video = {
        "video_path": video_path,
        "original_basename": original_basename,
        "original_video_raw_frame_count": total_raw_frames_in_original_video,
        "original_video_fps": original_video_fps
    }

    fps_for_stride_calc = original_video_fps if gui_target_fps_setting == -1 else gui_target_fps_setting
    if fps_for_stride_calc <= 0:
        fps_for_stride_calc = original_video_fps
    
    stride_for_fps_adjustment = max(round(original_video_fps / fps_for_stride_calc), 1)
    
    max_possible_output_frames_after_fps = (total_raw_frames_in_original_video + stride_for_fps_adjustment - 1) // stride_for_fps_adjustment
    
    effective_total_output_frames_to_target_for_video = max_possible_output_frames_after_fps
    if gui_process_length_overall != -1 and gui_process_length_overall < effective_total_output_frames_to_target_for_video:
        effective_total_output_frames_to_target_for_video = gui_process_length_overall
    
    if effective_total_output_frames_to_target_for_video <= 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="Effective output frames is zero or less.")
        return [], base_job_info_for_video

    log_message("SEGMENT_DEFINE_PROGRESS", video_name=original_basename, 
                output_frames=effective_total_output_frames_to_target_for_video, 
                raw_frames=min(effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment, total_raw_frames_in_original_video))


    if gui_segment_output_window_frames <= 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame count ({gui_segment_output_window_frames}) must be positive.")
        return [], base_job_info_for_video
    if gui_segment_output_overlap_frames < 0 or gui_segment_output_overlap_frames >= gui_segment_output_window_frames:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame overlap ({gui_segment_output_overlap_frames}) invalid for window {gui_segment_output_window_frames}.")
        return [], base_job_info_for_video

    segment_def_window_raw = gui_segment_output_window_frames * stride_for_fps_adjustment
    segment_def_overlap_raw = gui_segment_output_overlap_frames * stride_for_fps_adjustment
    advance_per_segment_raw = segment_def_window_raw - segment_def_overlap_raw
    
    effective_raw_video_length_to_consider = min(
        effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment,
        total_raw_frames_in_original_video
    )

    if advance_per_segment_raw <= 0 and effective_raw_video_length_to_consider > segment_def_window_raw :
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment raw advance ({advance_per_segment_raw}) is not positive. Check window/overlap/FPS.")
        return [], base_job_info_for_video

    current_raw_frame_idx = 0
    segment_id_counter = 0
    temp_segment_jobs = []

    while current_raw_frame_idx < effective_raw_video_length_to_consider:
        num_raw_frames_for_this_segment_def = min(
            segment_def_window_raw,
            effective_raw_video_length_to_consider - current_raw_frame_idx
        )
        if num_raw_frames_for_this_segment_def <= 0:
            break 
        
        segment_job = {
            **base_job_info_for_video,
            "start_frame_raw_index": current_raw_frame_idx,
            "num_frames_to_load_raw": num_raw_frames_for_this_segment_def,
            "segment_id": segment_id_counter,
            "is_segment": True,
            "gui_desired_output_window_frames": gui_segment_output_window_frames,
            "gui_desired_output_overlap_frames": gui_segment_output_overlap_frames,
        }
        temp_segment_jobs.append(segment_job)
        segment_id_counter += 1

        if current_raw_frame_idx + num_raw_frames_for_this_segment_def >= effective_raw_video_length_to_consider:
            break 
        current_raw_frame_idx += advance_per_segment_raw
        if current_raw_frame_idx >= effective_raw_video_length_to_consider:
            break
            
    total_segments_for_this_vid = len(temp_segment_jobs)
    if total_segments_for_this_vid == 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="No segments defined after loop.")
    else:
        for i_job in range(total_segments_for_this_vid):
            temp_segment_jobs[i_job]["total_segments"] = total_segments_for_this_vid
        segment_jobs.extend(temp_segment_jobs)
        log_message("SEGMENT_DEFINE_SUCCESS", num_segments=total_segments_for_this_vid, video_name=original_basename)
        
    return segment_jobs, base_job_info_for_video


def normalize_video_data(
    video_data: np.ndarray,
    use_percentile_norm: bool,
    low_perc: float,
    high_perc: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Normalizes video data to the 0-1 range."""
    if video_data is None or video_data.size == 0:
        log_message("UTIL_NORMALIZE_EMPTY_VIDEO_ERROR") # New ID
        raise ValueError("Cannot normalize empty video array.")

    log_message("UTIL_NORMALIZE_VIDEO_START", shape=video_data.shape) # New ID
    
    normalized_video = video_data.copy().astype(np.float32)
    min_val_for_norm, max_val_for_norm = np.min(normalized_video), np.max(normalized_video)
    method_str = "percentile"

    if use_percentile_norm:
        if normalized_video.ndim > 0 and normalized_video.shape[0] > 2 and normalized_video.flatten().size > 20:
            min_val_for_norm = np.percentile(normalized_video.flatten(), low_perc)
            max_val_for_norm = np.percentile(normalized_video.flatten(), high_perc)
        else:
            log_message("UTIL_NORMALIZE_PERCENTILE_FALLBACK", low=low_perc, high=high_perc) # New ID
            method_str = "absolute (percentile fallback)"
    else:
        method_str = "absolute"

    log_message("UTIL_NORMALIZE_VIDEO", shape=video_data.shape, method=method_str, 
                min_val=min_val_for_norm, max_val=max_val_for_norm)

    if abs(max_val_for_norm - min_val_for_norm) < 1e-6:
        log_message("UTIL_NORMALIZE_FLAT_VIDEO_WARN") # New ID
        flat_value = 0.5
        if (0.0 <= min_val_for_norm <= 1.0 and 0.0 <= max_val_for_norm <= 1.0 and abs(max_val_for_norm - min_val_for_norm) < 1e-7):
            flat_value = np.clip(min_val_for_norm, 0.0, 1.0)
        
        normalized_video = np.full_like(normalized_video, flat_value, dtype=np.float32)
        log_message("UTIL_NORMALIZE_FLAT_VIDEO_RESULT", value=flat_value) # New ID
    else:
        normalized_video = (normalized_video - min_val_for_norm) / (max_val_for_norm - min_val_for_norm)
    
    normalized_video = np.clip(normalized_video, 0.0, 1.0)
    log_message("UTIL_NORMALIZE_FINAL_RANGE", min_val=np.min(normalized_video), max_val=np.max(normalized_video)) # New ID
    return normalized_video


def apply_gamma_correction_to_video(
    video_data: np.ndarray,
    gamma_value: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Applies gamma correction to video data."""
    processed_video = video_data.copy()
    actual_gamma = max(0.1, gamma_value)

    if abs(actual_gamma - 1.0) > 1e-3:
        log_message("UTIL_GAMMA_CORRECTION", gamma_val=actual_gamma)
        processed_video = np.power(np.clip(processed_video, 0, 1), 1.0 / actual_gamma)
        processed_video = np.clip(processed_video, 0, 1)
    else:
        log_message("UTIL_GAMMA_CORRECTION_SKIPPED", gamma_val=actual_gamma) # New ID
    return processed_video


def apply_dithering_to_video(
    video_data: np.ndarray,
    dither_strength_factor: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Applies dithering to video data."""
    processed_video = video_data.copy()
    log_message("UTIL_DITHERING_START") # New ID
    
    dither_range = (1.0 / 255.0) * dither_strength_factor
    noise = np.random.uniform(-dither_range, dither_range, processed_video.shape).astype(np.float32)
    processed_video = np.clip(processed_video + noise, 0, 1)
    
    log_message("UTIL_DITHERING", strength_factor=dither_strength_factor, dither_range=dither_range)
    return processed_video


def load_json_file(filepath: str) -> Optional[dict]: # log_func removed
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_message("FILE_LOAD_SUCCESS", filepath=filepath)
        return data
    except FileNotFoundError:
        log_message("FILE_NOT_FOUND", filepath=filepath)
    except json.JSONDecodeError as e:
        log_message("JSON_DECODE_ERROR", filepath=filepath, reason=str(e))
    except Exception as e:
        log_message("GENERAL_ERROR", message=f"loading JSON from {filepath}: {e}")
    return None

def save_json_file(data: dict, filepath: str, indent: int = 4) -> bool: # log_func removed
    """Saves data to a JSON file."""
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        log_message("FILE_SAVE_SUCCESS", filepath=os.path.basename(filepath)) # Log only basename for brevity in repeated calls
        return True
    except TypeError as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=f"Data not JSON serializable: {e}")
    except (IOError, OSError) as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=str(e))
    except Exception as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=f"Unexpected error: {e}")
    return False

# --- END OF NEW UTILITY FUNCTIONS ---


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open",
                      start_frame_index=0, num_frames_to_load=-1):
    original_height, original_width = 0, 0
    try:
        temp_vid_for_meta = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = temp_vid_for_meta.get_batch([0]).shape[1:3]
        original_video_fps = temp_vid_for_meta.get_avg_fps()
        total_frames_in_video = len(temp_vid_for_meta)
        del temp_vid_for_meta
    except Exception as e:
        log_message("VIDEO_READ_METADATA_ERROR", video_path=video_path, error=str(e)) # New ID
        return np.array([]), target_fps if target_fps != -1 else 30

    # Resolution calculation logic (unchanged, assuming it's correct)
    if dataset == "open":
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res and max_res > 0 :
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        if dataset in dataset_res_dict:
            height = dataset_res_dict[dataset][0]
            width = dataset_res_dict[dataset][1]
        else:
            log_message("VIDEO_UNKNOWN_DATASET_WARN", dataset_name=dataset) # New ID
            height = round(original_height / 64) * 64
            width = round(original_width / 64) * 64
            if max(height, width) > max_res and max_res > 0:
                scale = max_res / max(original_height, original_width)
                height = round(original_height * scale / 64) * 64
                width = round(original_width * scale / 64) * 64
    try:
        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    except Exception as e:
        log_message("VIDEO_READER_INIT_ERROR", video_path=video_path, error=str(e)) # New ID
        return np.array([]), target_fps if target_fps != -1 else original_video_fps

    actual_fps_for_save = original_video_fps if target_fps == -1 else target_fps
    stride = round(original_video_fps / actual_fps_for_save) if actual_fps_for_save > 0 else 1
    stride = max(stride, 1)
    
    effective_num_frames_in_source_segment = num_frames_to_load
    if num_frames_to_load == -1:
        effective_num_frames_in_source_segment = total_frames_in_video - start_frame_index
    
    segment_end_frame_exclusive = min(start_frame_index + effective_num_frames_in_source_segment, total_frames_in_video)
    
    if start_frame_index >= segment_end_frame_exclusive :
        log_message("VIDEO_SEGMENT_EMPTY_WARN", video_path=video_path, start_index=start_frame_index, num_frames=num_frames_to_load) # New ID
        return np.array([]), actual_fps_for_save

    source_indices_for_segment = list(range(start_frame_index, segment_end_frame_exclusive))

    if not source_indices_for_segment:
        log_message("VIDEO_NO_SOURCE_INDICES_WARN", video_path=video_path) # New ID
        return np.array([]), actual_fps_for_save

    final_indices_to_read = [source_indices_for_segment[i] for i in range(0, len(source_indices_for_segment), stride)]

    if process_length != -1 and process_length < len(final_indices_to_read):
        final_indices_to_read = final_indices_to_read[:process_length]
    
    # Optional debug logging for frame indices (can be a specific message ID)
    # log_message("VIDEO_FRAME_INDICES_DEBUG", video_path=os.path.basename(video_path), 
    #             source_start=source_indices_for_segment[0], source_end=source_indices_for_segment[-1],
    #             stride=stride, target_fps=actual_fps_for_save, num_to_read=len(final_indices_to_read),
    #             final_start=final_indices_to_read[0] if final_indices_to_read else -1,
    #             final_end=final_indices_to_read[-1] if final_indices_to_read else -1)


    if not final_indices_to_read:
        log_message("VIDEO_NO_FRAMES_TO_READ_WARN", video_path=video_path) # New ID
        return np.array([]), actual_fps_for_save

    try:
        frames = vid.get_batch(final_indices_to_read).asnumpy().astype("float32") / 255.0
    except Exception as e:
        log_message("VIDEO_GET_BATCH_ERROR", video_path=video_path, error=str(e)) # New ID
        return np.array([]), actual_fps_for_save
        
    del vid
    return frames, actual_fps_for_save


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray],
    output_video_path: str = None,
    fps: Union[int, float] = 10.0,
    crf: int = 18,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # Frame conversion logic (unchanged, assuming correct)
    if isinstance(video_frames, np.ndarray):
        if video_frames.ndim == 3:
            if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
                 video_frames = (video_frames * 255).astype(np.uint8)
        elif video_frames.ndim == 4:
            if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
                video_frames = (video_frames * 255).astype(np.uint8)
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], np.ndarray):
        processed_frames = []
        for frame in video_frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                processed_frames.append((frame * 255).astype(np.uint8))
            elif frame.dtype == np.uint8:
                processed_frames.append(frame)
            else:
                log_message("VIDEO_SAVE_UNSUPPORTED_DTYPE_ERROR", dtype=str(frame.dtype)) # New ID
                raise ValueError(f"Unsupported numpy array dtype in list: {frame.dtype}")
        video_frames = processed_frames
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    elif isinstance(video_frames, list) and len(video_frames) == 0:
        log_message("VIDEO_SAVE_EMPTY_FRAMES_WARN") # New ID
        return output_video_path # Or raise error, or return None
    else:
        log_message("VIDEO_SAVE_INVALID_FRAMES_TYPE_ERROR") # New ID
        raise ValueError("video_frames must be a list/array of np.ndarray or a list of PIL.Image.Image")

    try:
        mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
        # log_message("FILE_SAVE_SUCCESS", filepath=output_video_path) # This might be too verbose if called many times
    except Exception as e:
        log_message("VIDEO_SAVE_MEDIAPY_ERROR", filepath=output_video_path, error=str(e)) # New ID
        raise
    return output_video_path


class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        self.colormap_name = colormap
        self._cmap_data = None

    def _get_cmap_data(self):
        if self._cmap_data is None:
            try:
                import matplotlib.cm as cm_mpl
                self._cmap_data = torch.tensor(cm_mpl.get_cmap(self.colormap_name).colors)
            except ImportError:
                log_message("COLORMAP_MPL_IMPORT_ERROR") # New ID
                # Fallback to a very simple grayscale if matplotlib is not available
                # This is a basic fallback, not a full replacement.
                ramp = torch.linspace(0, 1, 256)
                self._cmap_data = torch.stack([ramp, ramp, ramp], dim=1) # (N, 3)
        return self._cmap_data


    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        if image.ndim not in [2,3]:
            log_message("COLORMAP_INVALID_INPUT_DIMS_ERROR", ndim=image.ndim) # New ID
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        cmap_data = self._get_cmap_data().to(image.device)
        
        if v_min is None: v_min = image.min()
        if v_max is None: v_max = image.max()
        
        if v_max == v_min:
            image_normalized = torch.zeros_like(image)
        else:
            image_normalized = (image - v_min) / (v_max - v_min)
        
        image_long = (image_normalized * (len(cmap_data) -1) ).long()
        image_long = torch.clamp(image_long, 0, len(cmap_data) - 1)
        colored_image = cmap_data[image_long]
        return colored_image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None, colormap: str = "inferno"):
    if not isinstance(depths, np.ndarray):
        depths = np.array(depths)
    if depths.ndim != 3:
        log_message("VIS_SEQ_INVALID_INPUT_DIMS_ERROR", ndim=depths.ndim) # New ID
        raise ValueError(f"Input depths must be a 3D array (T, H, W), got {depths.ndim}D")

    visualizer = ColorMapper(colormap=colormap)
    if v_min is None: v_min = depths.min()
    if v_max is None: v_max = depths.max()
    
    depths_tensor = torch.from_numpy(depths.astype(np.float32))
    colored_sequence_tensor = visualizer.apply(depths_tensor, v_min=v_min, v_max=v_max)
    colored_sequence_np = colored_sequence_tensor.cpu().numpy()
    
    if colored_sequence_np.shape[-1] == 4:
        colored_sequence_np = colored_sequence_np[..., :3]
    return colored_sequence_np

def save_depth_visual_as_mp4_util(
    depth_frames_normalized: np.ndarray, 
    output_filepath: str, 
    fps: Union[int, float]
) -> Tuple[Optional[str], Optional[str]]:
    try:
        save_video(depth_frames_normalized, output_filepath, fps=fps) 
        # log_message("FILE_SAVE_SUCCESS", filepath=output_filepath) # Potentially too verbose
        return output_filepath, None
    except Exception as e:
        log_message("VIDEO_SAVE_MP4_UTIL_ERROR", filepath=output_filepath, error=str(e)) # New ID
        return None, str(e)

def save_depth_visual_as_png_sequence_util(
    depth_frames_normalized: np.ndarray, 
    output_dir_base: str,
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    try:
        visual_dirname = f"{base_filename_no_ext}_visual_png_seq"
        png_dir_path = os.path.join(output_dir_base, visual_dirname)
        if os.path.exists(png_dir_path): 
            shutil.rmtree(png_dir_path)
        os.makedirs(png_dir_path, exist_ok=True)
        for i, frame_float in enumerate(depth_frames_normalized):
            frame_uint16 = (np.clip(frame_float, 0, 1) * 65535.0).astype(np.uint16)
            frame_filename = os.path.join(png_dir_path, f"frame_{i:05d}.png")
            imageio.imwrite(frame_filename, frame_uint16)
        # log_message("FILE_SAVE_SUCCESS", filepath=png_dir_path) # Potentially too verbose
        return png_dir_path, None
    except Exception as e:
        log_message("IMAGE_SAVE_PNG_SEQ_UTIL_ERROR", dir_path=png_dir_path, error=str(e)) # New ID
        return None, str(e)

def save_depth_visual_as_exr_sequence_util(
    depth_frames_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        log_message("OPENEXR_UNAVAILABLE", context="save_depth_visual_as_exr_sequence_util")
        return None, "OpenEXR libraries not available in utils.py"
    try:
        if depth_frames_normalized.ndim != 3:
             err_msg = f"EXR sequence expects 3D array (T,H,W), got {depth_frames_normalized.ndim}D"
             log_message("IMAGE_SAVE_EXR_SEQ_UTIL_ERROR", error=err_msg) # New ID (generic for this func)
             return None, err_msg
        
        sequence_subfolder_name = f"{base_filename_no_ext}_visual_exr_seq"
        exr_sequence_output_dir = os.path.join(output_dir_base, sequence_subfolder_name)
        
        if os.path.exists(exr_sequence_output_dir): 
            shutil.rmtree(exr_sequence_output_dir)
        os.makedirs(exr_sequence_output_dir, exist_ok=True)

        for i, frame_data_normalized in enumerate(depth_frames_normalized):
            frame_float32 = frame_data_normalized.astype(np.float32) 
            output_exr_filepath = os.path.join(exr_sequence_output_dir, f"frame_{i:05d}.exr")
            try:
                imageio.imwrite(output_exr_filepath, frame_float32, format='EXR-FI')
            except Exception:
                imageio.imwrite(output_exr_filepath, frame_float32)
        # log_message("FILE_SAVE_SUCCESS", filepath=exr_sequence_output_dir) # Potentially too verbose
        return exr_sequence_output_dir, None 
    except Exception as e:
        log_message("IMAGE_SAVE_EXR_SEQ_UTIL_ERROR", dir_path=exr_sequence_output_dir if 'exr_sequence_output_dir' in locals() else "unknown_path", error=str(e))
        return None, str(e)

def save_depth_visual_as_single_exr_util(
    first_depth_frame_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        log_message("OPENEXR_UNAVAILABLE", context="save_depth_visual_as_single_exr_util")
        return None, "OpenEXR libraries not available in utils.py"
    try:
        if first_depth_frame_normalized is None or first_depth_frame_normalized.size == 0:
            err_msg = "No frame data to save for single EXR"
            log_message("IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR", error=err_msg) # New ID (generic for this func)
            return None, err_msg
        
        frame_float32 = first_depth_frame_normalized.astype(np.float32)
        os.makedirs(output_dir_base, exist_ok=True)
        output_exr_filepath = os.path.join(output_dir_base, f"{base_filename_no_ext}_visual.exr")
        
        try:
            imageio.imwrite(output_exr_filepath, frame_float32, format='EXR-FI')
        except Exception:
            imageio.imwrite(output_exr_filepath, frame_float32)
        # log_message("FILE_SAVE_SUCCESS", filepath=output_exr_filepath) # Potentially too verbose
        return output_exr_filepath, None
    except Exception as e:
        log_message("IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR", filepath=output_exr_filepath if 'output_exr_filepath' in locals() else "unknown_path", error=str(e))
        return None, str(e)