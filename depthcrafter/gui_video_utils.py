"""
GUI Video Processing Utilities
Pure additions (not present in upstream) for video processing in GUI context.
"""

from typing import Union, List, Optional, Tuple
import numpy as np
import os
import logging

_logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SINGLE_IMAGE_CLIP_FRAMES = 5


def format_duration(seconds: float) -> str:
    """
    Converts seconds to H:MM:SS.s format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted duration string
    """
    if seconds < 0:
        return "0:00:00.0"
    
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours}:{minutes:02}:{seconds:04.1f}"


def get_segment_output_folder_name(original_video_basename: str) -> str:
    """
    Returns the standard name for a segment subfolder.
    
    Args:
        original_video_basename: Base name of the original video
    
    Returns:
        str: Segment folder name
    """
    return f"{original_video_basename}_seg"


def get_segment_npz_output_filename(original_video_basename: str, segment_id: int, total_segments: int) -> str:
    """
    Returns the standard NPZ filename for a segment.
    
    Args:
        original_video_basename: Base name of the original video
        segment_id: ID of the segment (0-indexed)
        total_segments: Total number of segments
    
    Returns:
        str: NPZ filename
    """
    return f"{original_video_basename}_depth_{segment_id + 1}of{total_segments}.npz"


def get_full_video_output_filename(original_video_basename: str, extension: str = "mp4") -> str:
    """
    Returns the standard filename for a full video output.
    
    Args:
        original_video_basename: Base name of the original video
        extension: File extension (default: mp4)
    
    Returns:
        str: Output filename
    """
    return f"{original_video_basename}_depth.{extension}"


def define_video_segments(
    video_path_or_folder: str,
    original_basename: str,
    gui_target_fps_setting: int,
    gui_process_length_overall: int,
    gui_segment_output_window_frames: int,
    gui_segment_output_overlap_frames: int,
    source_type: str,
    gui_target_height_setting: int,
    gui_target_width_setting: int
) -> Tuple[List[dict], Optional[dict]]:
    """
    Defines video segments based on input parameters.
    
    This function calculates how to split a video into segments for processing,
    taking into account FPS adjustments, overlap settings, and target frame counts.
    
    Args:
        video_path_or_folder: Path to video file or image folder
        original_basename: Base name of the original file
        gui_target_fps_setting: Target FPS from GUI (-1 for original)
        gui_process_length_overall: Overall frame limit (-1 for all)
        gui_segment_output_window_frames: Frames per segment window
        gui_segment_output_overlap_frames: Overlap frames between segments
        source_type: Type of source ("video_file", "image_sequence_folder", "single_image_file")
        gui_target_height_setting: Target height from GUI
        gui_target_width_setting: Target width from GUI
    
    Returns:
        Tuple of (list of segment job dicts, base job info dict)
    """
    from depthcrafter.gui_io_utils import get_image_sequence_metadata, get_single_image_metadata
    from depthcrafter.gui_extensions import read_video_frames
    import gc
    
    segment_jobs = []
    base_job_info_for_video = {}

    total_raw_frames_in_original_video = 0
    original_video_fps = 30.0
    original_h_detected, original_w_detected = 0, 0
    video_stream_info_from_ffprobe = None

    if source_type == "video_file":
        # Get metadata without loading all frames
        frames_numpy_dummy, fps_detected, h_orig, w_orig, _, _, video_stream_info_from_ffprobe, _ = read_video_frames(
            video_path_or_folder,
            process_length=-1,
            target_fps=-1.0,
            target_height=128,  # Dummy values for metadata
            target_width=128,
            start_frame_index=0,
            num_frames_to_load=-1
        )
        total_raw_frames_in_original_video = frames_numpy_dummy.shape[0] if frames_numpy_dummy is not None else 0
        original_video_fps = fps_detected if fps_detected is not None and fps_detected > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig

        if original_video_fps <= 0:
            _logger.warning(f"Invalid original FPS ({original_video_fps}). Assuming 30 FPS.")
            original_video_fps = 30.0
            
    elif source_type == "image_sequence_folder":
        count, fps, h_orig, w_orig, _ = get_image_sequence_metadata(video_path_or_folder, gui_target_fps_setting)
        if count is None:
            _logger.error(f"Error getting metadata for image sequence folder {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps if fps is not None and fps > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig
        
    elif source_type == "single_image_file":
        count, fps, h_orig, w_orig, _ = get_single_image_metadata(
            video_path_or_folder,
            gui_target_fps_setting
        )
        if count is None:
            _logger.error(f"Error getting metadata for single image {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps if fps is not None and fps > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig
    else:
        _logger.error(f"Unknown source_type: {source_type}")
        return [], None

    base_job_info_for_video = {
        "video_path": video_path_or_folder,
        "source_type": source_type,
        "gui_fps_setting_at_definition": gui_target_fps_setting,
        "original_basename": original_basename,
        "original_video_raw_frame_count": total_raw_frames_in_original_video,
        "original_video_fps": original_video_fps,
        "original_height": original_h_detected,
        "original_width": original_w_detected,
        "video_stream_ffprobe_info": video_stream_info_from_ffprobe if source_type == "video_file" else None,
        "gui_target_height_setting": gui_target_height_setting,
        "gui_target_width_setting": gui_target_width_setting,
    }

    fps_for_stride_calc = original_video_fps
    if source_type == "video_file":
        fps_for_stride_calc = original_video_fps if gui_target_fps_setting == -1 else gui_target_fps_setting
    
    if fps_for_stride_calc <= 0:
        fps_for_stride_calc = original_video_fps if original_video_fps > 0 else 24.0
    
    stride_for_fps_adjustment = max(round(original_video_fps / fps_for_stride_calc), 1)
    
    max_possible_output_frames_after_fps = (total_raw_frames_in_original_video + stride_for_fps_adjustment - 1) // stride_for_fps_adjustment
    
    effective_total_output_frames_to_target_for_video = max_possible_output_frames_after_fps
    if gui_process_length_overall != -1 and gui_process_length_overall < effective_total_output_frames_to_target_for_video:
        effective_total_output_frames_to_target_for_video = gui_process_length_overall
    
    if effective_total_output_frames_to_target_for_video <= 0:
        _logger.warning(f"Effective output frames is zero or less. Skipping segment definition.")
        return [], base_job_info_for_video

    _logger.debug(f"Defining segments for {original_basename} (up to {effective_total_output_frames_to_target_for_video} output frames)")

    if gui_segment_output_window_frames <= 0:
        _logger.error(f"Segment output frame count must be positive.")
        return [], base_job_info_for_video
    if gui_segment_output_overlap_frames < 0 or gui_segment_output_overlap_frames >= gui_segment_output_window_frames:
        _logger.error(f"Segment overlap invalid for window {gui_segment_output_window_frames}.")
        return [], base_job_info_for_video

    segment_def_window_raw = gui_segment_output_window_frames * stride_for_fps_adjustment
    segment_def_overlap_raw = gui_segment_output_overlap_frames * stride_for_fps_adjustment
    advance_per_segment_raw = segment_def_window_raw - segment_def_overlap_raw
    
    effective_raw_video_length_to_consider = min(
        effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment,
        total_raw_frames_in_original_video
    )

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
        
        if advance_per_segment_raw <= 0:
            _logger.warning(f"Segment raw advance is zero or negative. Breaking after first segment.")
            break

        current_raw_frame_idx += advance_per_segment_raw
        if current_raw_frame_idx >= effective_raw_video_length_to_consider:
            break
            
    total_segments_for_this_vid = len(temp_segment_jobs)
    if total_segments_for_this_vid == 0 and effective_total_output_frames_to_target_for_video > 0:
        _logger.warning(f"No segments defined after loop, but frames were expected.")
    elif total_segments_for_this_vid > 0:
        for i_job in range(total_segments_for_this_vid):
            temp_segment_jobs[i_job]["total_segments"] = total_segments_for_this_vid
        segment_jobs.extend(temp_segment_jobs)
        _logger.debug(f"Defined {total_segments_for_this_vid} segments for {original_basename}.")
        
    return segment_jobs, base_job_info_for_video


def normalize_video_data(
    video_data: np.ndarray,
    use_percentile_norm: bool,
    low_perc: float,
    high_perc: float
) -> np.ndarray:
    """
    Normalizes video data to the 0-1 range.
    
    Args:
        video_data: Input video array
        use_percentile_norm: Whether to use percentile-based normalization
        low_perc: Lower percentile for normalization
        high_perc: Upper percentile for normalization
    
    Returns:
        np.ndarray: Normalized video data
    """
    if video_data is None or video_data.size == 0:
        _logger.critical("Cannot normalize empty video array.")
        raise ValueError("Cannot normalize empty video array.")

    _logger.debug(f"Normalizing video data. Shape: {video_data.shape}")
    
    normalized_video = video_data.copy().astype(np.float32)
    min_val_for_norm, max_val_for_norm = np.min(normalized_video), np.max(normalized_video)
    method_str = "percentile"

    if use_percentile_norm:
        if normalized_video.ndim > 0 and normalized_video.shape[0] > 2 and normalized_video.flatten().size > 20:
            min_val_for_norm = np.percentile(normalized_video.flatten(), low_perc)
            max_val_for_norm = np.percentile(normalized_video.flatten(), high_perc)
        else:
            _logger.debug(f"Array too small for percentile, using min/max.")
            method_str = "absolute (percentile fallback)"
    else:
        method_str = "absolute"

    _logger.debug(f"Method: {method_str}. Range: {min_val_for_norm:.4f}-{max_val_for_norm:.4f}")

    if abs(max_val_for_norm - min_val_for_norm) < 1e-6:
        _logger.warning("Range very small. Video appears flat.")
        flat_value = 0.5
        normalized_video = np.full_like(normalized_video, flat_value, dtype=np.float32)
    else:
        normalized_video = (normalized_video - min_val_for_norm) / (max_val_for_norm - min_val_for_norm)
    
    normalized_video = np.clip(normalized_video, 0.0, 1.0)
    return normalized_video


def apply_gamma_correction_to_video(
    video_data: np.ndarray,
    gamma_value: float
) -> np.ndarray:
    """
    Applies gamma correction to video data.
    
    Args:
        video_data: Input video array (0-1 range)
        gamma_value: Gamma correction value
    
    Returns:
        np.ndarray: Gamma-corrected video data
    """
    processed_video = video_data.copy()
    actual_gamma = max(0.1, gamma_value)

    if abs(actual_gamma - 1.0) > 1e-3:
        _logger.debug(f"Applying Gamma ({actual_gamma:.2f}) to video.")
        processed_video = np.power(np.clip(processed_video, 0, 1), 1.0 / actual_gamma)
        processed_video = np.clip(processed_video, 0, 1)
    else:
        _logger.debug(f"Gamma value {actual_gamma:.2f} effectively 1.0, no transform applied.")
    return processed_video


def apply_dithering_to_video(
    video_data: np.ndarray,
    dither_strength_factor: float
) -> np.ndarray:
    """
    Applies dithering to video data.
    
    Args:
        video_data: Input video array (0-1 range)
        dither_strength_factor: Strength of dithering effect
    
    Returns:
        np.ndarray: Dithered video data
    """
    processed_video = video_data.copy()
    _logger.debug("Applying dithering...")
    
    dither_range = (1.0 / 255.0) * dither_strength_factor
    noise = np.random.uniform(-dither_range, dither_range, processed_video.shape).astype(np.float32)
    processed_video = np.clip(processed_video + noise, 0, 1)
    
    _logger.debug(f"Dithering applied. Strength: {dither_strength_factor:.2f}, Range: {dither_range:.4f}")
    return processed_video
