"""
GUI I/O Utilities
File operations, metadata handling, EXR support, and JSON utilities.
Pure additions not present in upstream.
"""

from typing import Optional, Tuple, Union
import json
import os
import glob
import shutil
import numpy as np
import imageio
import PIL.Image
import logging
import subprocess

_logger = logging.getLogger(__name__)

# OpenEXR availability check
try:
    import OpenEXR
    import Imath
    _OPENEXR_AVAILABLE = True
except ImportError:
    _OPENEXR_AVAILABLE = False
    _logger.warning("OpenEXR/Imath libraries not found. EXR features will be limited.")


# ---------------------------------------------------------------------------
# EXR input + color transforms
# ---------------------------------------------------------------------------
# Chế độ biến đổi EXR (giá trị linear float -> sRGB-ish 0..1 cho model):
#   "rec709"      : Rec.709 OETF (gamma chuẩn), clip 0..1  [mặc định]
#   "aces_rec709" : ACES filmic tonemap (fitted curve) rồi Rec.709 OETF, cho HDR
#   "clip"        : chỉ clip 0..1, không gamma
EXR_DEFAULT_TRANSFORM = "rec709"
EXR_TRANSFORMS = ("rec709", "aces_rec709", "clip")


def set_exr_default_transform(mode: str) -> None:
    """Đổi chế độ EXR mặc định (GUI gọi trước khi chạy job)."""
    global EXR_DEFAULT_TRANSFORM
    if mode not in EXR_TRANSFORMS:
        raise ValueError(f"exr_transform '{mode}' không hợp lệ. Chọn một trong {EXR_TRANSFORMS}")
    EXR_DEFAULT_TRANSFORM = mode


def read_exr_rgb(path: str) -> np.ndarray:
    """Đọc EXR -> float32 HxWx3 linear, giữ nguyên giá trị HDR."""
    if not _OPENEXR_AVAILABLE:
        raise RuntimeError("OpenEXR/Imath chưa cài, không đọc được EXR.")
    f = OpenEXR.InputFile(path)
    hdr = f.header()
    dw = hdr["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    names = list(hdr["channels"].keys())

    def ch(name):
        return np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(h, w)

    if all(c in names for c in ("R", "G", "B")):
        rgb = np.stack([ch("R"), ch("G"), ch("B")], axis=-1)
    else:  # EXR kiểu depth/Y/mono: lặp 1 kênh thành RGB
        base = "Y" if "Y" in names else names[0]
        rgb = np.repeat(ch(base)[..., None], 3, axis=-1)
    f.close()
    return np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


# EXR output bit depth: "float32" (default, full precision) or "half" (float16, ~half the size)
EXR_BIT_DEPTH = "float32"
EXR_BIT_DEPTHS = ("float32", "half")


def set_exr_bit_depth(mode: str) -> None:
    """Set the EXR output bit depth used by all EXR writers."""
    global EXR_BIT_DEPTH
    if mode not in EXR_BIT_DEPTHS:
        raise ValueError(f"exr_bit_depth '{mode}' is invalid. Choose one of {EXR_BIT_DEPTHS}")
    EXR_BIT_DEPTH = mode


def exr_pixel_type():
    """Imath pixel type for the current EXR output bit depth."""
    kind = Imath.PixelType.HALF if EXR_BIT_DEPTH == "half" else Imath.PixelType.FLOAT
    return Imath.PixelType(kind)


def exr_pixel_bytes(arr: np.ndarray) -> bytes:
    """Cast a float array to the current EXR bit depth and return its raw bytes."""
    dtype = np.float16 if EXR_BIT_DEPTH == "half" else np.float32
    return np.ascontiguousarray(arr, dtype=dtype).tobytes()


def read_exr_size(path: str) -> Tuple[int, int]:
    """Lấy (h, w) từ header EXR, không đọc pixel."""
    if not _OPENEXR_AVAILABLE:
        raise RuntimeError("OpenEXR/Imath chưa cài, không đọc được EXR.")
    f = OpenEXR.InputFile(path)
    dw = f.header()["dataWindow"]
    f.close()
    return dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1


def _rec709_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(np.maximum(x, 1e-12), 0.45) - 0.099)


def apply_exr_transform(rgb_linear: np.ndarray, mode: str) -> np.ndarray:
    """Biến đổi linear float -> display 0..1 theo chế độ đã chọn."""
    if mode not in EXR_TRANSFORMS:
        raise ValueError(f"exr_transform '{mode}' không hợp lệ. Chọn một trong {EXR_TRANSFORMS}")
    x = np.clip(rgb_linear, 0.0, None)
    if mode == "clip":
        out = x
    elif mode == "rec709":
        out = _rec709_oetf(x)
    else:  # aces_rec709: fitted ACES filmic (Narkowicz) rồi Rec.709
        out = _rec709_oetf((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _image_shape(path: str) -> Tuple[int, int]:
    """(h, w) của ảnh bất kỳ được hỗ trợ."""
    if path.lower().endswith(".exr"):
        return read_exr_size(path)
    img = imageio.v2.imread(path)
    return img.shape[0], img.shape[1]


def read_any_image_float01(path: str, exr_transform: Optional[str] = None) -> np.ndarray:
    """Đọc ảnh bất kỳ -> float32 HxWx3 trong 0..1. EXR đi qua color transform."""
    if path.lower().endswith(".exr"):
        mode = exr_transform or EXR_DEFAULT_TRANSFORM
        return apply_exr_transform(read_exr_rgb(path), mode)
    img = imageio.v2.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return img.astype(np.float32) / 255.0


def _resize_float_hwc(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize float HxWxC giữ precision (PIL mode F, từng kênh)."""
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    chans = []
    for c in range(img.shape[2]):
        pil = PIL.Image.fromarray(img[..., c].astype(np.float32))
        chans.append(np.array(pil.resize((target_w, target_h), PIL.Image.LANCZOS), dtype=np.float32))
    # Lanczos overshoot ra ngoài 0..1 trên float -> clip lại (uint8 cũ tự làm việc này)
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0).astype(np.float32)


def load_json_file(filepath: str) -> Optional[dict]:
    """
    Loads data from a JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict or None: Loaded data or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _logger.debug(f"Successfully loaded: {filepath}")
        return data
    except FileNotFoundError:
        _logger.error(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        _logger.error(f"Could not decode JSON from: {filepath}. Reason: {e}")
    except Exception as e:
        _logger.error(f"ERROR loading JSON from {filepath}: {e}")
    return None


def save_json_file(data: dict, filepath: str, indent: int = 4) -> bool:
    """
    Saves data to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Target file path
        indent: JSON indentation level
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        _logger.debug(f"Successfully saved: {os.path.basename(filepath)}")
        return True
    except TypeError as e:
        _logger.error(f"Failed to save: {filepath}. Data not JSON serializable: {e}")
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save: {filepath}. Reason: {e}")
    except Exception as e:
        _logger.error(f"Failed to save: {filepath}. Unexpected error: {e}")
    return False


def get_sidecar_json_filename(base_filepath_with_ext: str) -> str:
    """
    Returns the corresponding .json sidecar filename for a given base file.
    
    Args:
        base_filepath_with_ext: Path to file (e.g., "video.mp4" or "data.npz")
    
    Returns:
        str: Path with .json extension
    """
    return os.path.splitext(base_filepath_with_ext)[0] + ".json"


def get_image_sequence_metadata(folder_path: str, target_fps_from_gui: int) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Gets metadata (frame count, fps, H, W) for an image sequence.
    
    Args:
        folder_path: Path to folder containing images
        target_fps_from_gui: Target FPS setting from GUI
    
    Returns:
        Tuple of (frame_count, fps, height, width, None)
    """
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr")
    frames_found = []
    for ext in supported_exts:
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    frames = sorted(list(set(frames_found)))

    if not frames:
        _logger.warning(f"No compatible image files found in '{folder_path}'. Supported: {supported_exts}")
        return None, None, None, None, None

    try:
        h, w = _image_shape(frames[0])
    except Exception as e:
        _logger.error(f"Could not read image file '{frames[0]}'. Error: {e}")
        return None, None, None, None, None

    total_frames = len(frames)
    effective_fps = float(target_fps_from_gui) if target_fps_from_gui != -1 else 24.0
    _logger.debug(f"Image Sequence: {total_frames} frames, {effective_fps:.2f} FPS, {h}x{w}")
    return total_frames, effective_fps, h, w, None


def get_single_image_metadata(
    image_path: str,
    gui_target_fps_setting: int
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[dict]]:
    """
    Gets metadata for a single image.
    Frame count uses DEFAULT_SINGLE_IMAGE_CLIP_FRAMES.
    
    Args:
        image_path: Path to image file
        gui_target_fps_setting: Target FPS setting from GUI
    
    Returns:
        Tuple of (frame_count, fps, height, width, stream_info)
    """
    try:
        h, w = _image_shape(image_path)
    except Exception as e:
        _logger.error(f"Could not read image file '{image_path}'. Error: {e}")
        return None, None, None, None, None

    effective_fps = float(gui_target_fps_setting) if gui_target_fps_setting != -1 else 24.0
    num_generated_frames = 5  # DEFAULT_SINGLE_IMAGE_CLIP_FRAMES

    _logger.debug(f"Single Image: {num_generated_frames} frames for 1s clip at {effective_fps:.2f} FPS, {h}x{w}")
    return num_generated_frames, effective_fps, h, w, None


def get_video_stream_info(video_path: str) -> Optional[Tuple[dict, str]]:
    """
    Extracts comprehensive video stream metadata using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (stream_info_dict, raw_stdout) or None if failed
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,pix_fmt,color_primaries,transfer_characteristics,color_space,r_frame_rate,width,height,nb_frames,duration,side_data_list",
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
        raw_stdout = result.stdout
        data = json.loads(raw_stdout)
        
        stream_info = {}
        
        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0]
            
            stream_info["codec_name"] = s.get("codec_name")
            stream_info["profile"] = s.get("profile")
            stream_info["pix_fmt"] = s.get("pix_fmt")
            stream_info["color_primaries"] = s.get("color_primaries")
            stream_info["transfer_characteristics"] = s.get("transfer_characteristics")
            stream_info["color_space"] = s.get("color_space")
            stream_info["r_frame_rate"] = s.get("r_frame_rate")
            stream_info["width"] = s.get("width")
            stream_info["height"] = s.get("height")
            stream_info["nb_frames"] = s.get("nb_frames")
            stream_info["duration"] = s.get("duration")
            
            if s.get("side_data_list"):
                for sd in s["side_data_list"]:
                    if sd.get("side_data_type") == "Mastering display metadata":
                        stream_info["mastering_display_metadata"] = sd.get("mastering_display_metadata")
                    if sd.get("side_data_type") == "Content light level metadata":
                        stream_info["max_content_light_level"] = sd.get("max_content_light_level")

        # Guess nb_frames from duration * r_frame_rate if still missing
        if not stream_info.get("nb_frames") or int(stream_info.get("nb_frames", "0")) == 0:
            if stream_info.get("duration") and stream_info.get("r_frame_rate"):
                try:
                    duration_f = float(stream_info["duration"])
                    r_frame_rate_str = stream_info["r_frame_rate"].split('/')
                    if len(r_frame_rate_str) == 2 and float(r_frame_rate_str[1]) != 0:
                        fps_val = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
                    else:
                        fps_val = float(r_frame_rate_str[0])
                    if duration_f > 0 and fps_val > 0:
                        stream_info["nb_frames"] = str(round(duration_f * fps_val))
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

        # Filter out empty/invalid values
        filtered_info = {k: v for k, v in stream_info.items() if v is not None and str(v).strip() not in ["N/A", "und", "unknown", "0", "0.0"]}
        
        return (filtered_info, raw_stdout) if filtered_info else (None, raw_stdout)

    except subprocess.CalledProcessError as e:
        _logger.error(f"ffprobe failed for {video_path}: {e.stderr}")
    except subprocess.TimeoutExpired:
        _logger.error(f"ffprobe timed out for {video_path}")
    except json.JSONDecodeError as e:
        _logger.error(f"Failed to parse ffprobe output: {e}")
    except Exception as e:
        _logger.error(f"Unexpected error with ffprobe: {e}")
    return None


def read_image_sequence_as_frames(
    folder_path: str,
    num_frames_to_load: int,
    target_height: int,
    target_width: int,
    start_index: int = 0,
    exr_transform: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    Reads a segment of an image sequence from a folder into a NumPy array.
    
    Args:
        folder_path: Path to folder containing images
        num_frames_to_load: Number of frames to load (-1 for all remaining)
        target_height: Target height for resizing
        target_width: Target width for resizing
        start_index: Starting frame index
    
    Returns:
        Tuple of (frames_array, original_h, original_w)
    """
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr")
    frames_found = []
    for ext in supported_exts:
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    all_image_paths_sorted = sorted(list(set(frames_found)))

    if not all_image_paths_sorted:
        _logger.warning(f"No compatible image files found in '{folder_path}'")
        return None, None, None

    if start_index < 0 or start_index >= len(all_image_paths_sorted):
        _logger.error(f"Invalid start_index {start_index} for folder with {len(all_image_paths_sorted)} images")
        return None, None, None

    if num_frames_to_load == -1:
        end_index = len(all_image_paths_sorted)
    else:
        end_index = start_index + num_frames_to_load
    
    image_paths_for_segment = all_image_paths_sorted[start_index:end_index]

    if not image_paths_for_segment:
        _logger.warning(f"No frames found for segment (start: {start_index}, load: {num_frames_to_load})")
        return None, None, None

    loaded_frames_list = []
    try:
        original_h, original_w = _image_shape(all_image_paths_sorted[0])
        target_h = max(64, round(target_height / 64) * 64)
        target_w = max(64, round(target_width / 64) * 64)
    except Exception as e:
        _logger.error(f"Error reading reference frame: {e}")
        return None, None, None

    for frame_path in image_paths_for_segment:
        try:
            img = read_any_image_float01(frame_path, exr_transform)
            img = _resize_float_hwc(img, target_w, target_h)
            loaded_frames_list.append(img.astype(np.float32))
        except Exception as e:
            _logger.warning(f"Error reading frame '{frame_path}': {e}. Skipping.")
            continue

    if not loaded_frames_list:
        _logger.error(f"Failed to load any frames from '{folder_path}'")
        return None, None, None

    frames_array = np.stack(loaded_frames_list, axis=0)
    _logger.debug(f"Loaded {frames_array.shape[0]} frames from '{folder_path}' ({frames_array.shape[1]}x{frames_array.shape[2]})")
    return frames_array, original_h, original_w


def create_frames_from_single_image(
    image_path: str,
    num_frames_to_generate: int,
    target_height: int,
    target_width: int,
    exr_transform: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    Creates a sequence of identical frames from a single image.
    
    Args:
        image_path: Path to image file
        num_frames_to_generate: Number of frames to create
        target_height: Target height for resizing
        target_width: Target width for resizing
    
    Returns:
        Tuple of (frames_array, original_h, original_w)
    """
    try:
        original_h, original_w = _image_shape(image_path)
        target_h = max(64, round(target_height / 64) * 64)
        target_w = max(64, round(target_width / 64) * 64)

        frame_float = _resize_float_hwc(read_any_image_float01(image_path, exr_transform), target_w, target_h)
        frames_array = np.stack([frame_float] * num_frames_to_generate, axis=0)
        _logger.info(f"Single Image: Generated {frames_array.shape[0]} frames from '{image_path}' ({frames_array.shape[1]}x{frames_array.shape[2]})")
        return frames_array, original_h, original_w
    except Exception as e:
        _logger.error(f"Error generating frames from '{image_path}': {e}")
        return None, None, None


def save_depth_visual_as_mp4_util(
    depth_frames_normalized: np.ndarray,
    output_filepath: str,
    fps: Union[int, float],
    output_format: str = "mp4"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Saves depth visualization as MP4.
    
    Args:
        depth_frames_normalized: Normalized depth frames (0-1 range)
        output_filepath: Output file path
        fps: Frames per second
        output_format: Format ("mp4" or "main10_mp4")
    
    Returns:
        Tuple of (output_path, error_message)
    """
    try:
        from depthcrafter.gui_extensions import save_video
        save_video(depth_frames_normalized, output_filepath, fps=fps, output_format=output_format)
        return output_filepath, None
    except Exception as e:
        _logger.error(f"Error saving MP4 to {output_filepath}: {e}")
        return None, str(e)


def save_depth_visual_as_png_sequence_util(
    depth_frames_normalized: np.ndarray,
    output_dir_base: str,
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Saves depth visualization as PNG sequence.
    
    Args:
        depth_frames_normalized: Normalized depth frames (0-1 range)
        output_dir_base: Base output directory
        base_filename_no_ext: Base filename without extension
    
    Returns:
        Tuple of (output_dir_path, error_message)
    """
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
        
        _logger.debug(f"Saved PNG sequence to {png_dir_path}")
        return png_dir_path, None
    except Exception as e:
        _logger.error(f"Error saving PNG sequence: {e}")
        return None, str(e)


def save_depth_visual_as_exr_sequence_util(
    depth_frames_normalized: np.ndarray,
    output_dir_base: str,
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Saves depth visualization as EXR sequence.
    
    Args:
        depth_frames_normalized: Normalized depth frames (0-1 range)
        output_dir_base: Base output directory
        base_filename_no_ext: Base filename without extension
    
    Returns:
        Tuple of (output_dir_path, error_message)
    """
    if not _OPENEXR_AVAILABLE:
        return None, "OpenEXR libraries not available."

    try:
        if depth_frames_normalized.ndim != 3:
            return None, f"EXR sequence expects 3D array (T,H,W), got {depth_frames_normalized.ndim}D"
        
        num_frames, height, width = depth_frames_normalized.shape
        if num_frames == 0:
            return None, "No frames to save"

        sequence_subfolder_name = f"{base_filename_no_ext}_visual_exr_seq"
        exr_sequence_output_dir = os.path.join(output_dir_base, sequence_subfolder_name)
        
        if os.path.exists(exr_sequence_output_dir):
            shutil.rmtree(exr_sequence_output_dir)
        os.makedirs(exr_sequence_output_dir, exist_ok=True)

        for i in range(num_frames):
            frame_data_float32 = depth_frames_normalized[i].astype(np.float32)
            output_exr_filepath = os.path.join(exr_sequence_output_dir, f"frame_{i:05d}.exr")

            try:
                header = OpenEXR.Header(width, height)
                if frame_data_float32.ndim == 2:
                    header['channels'] = {'Z': Imath.Channel(exr_pixel_type())}
                    pixel_data = {'Z': exr_pixel_bytes(frame_data_float32)}
                elif frame_data_float32.ndim == 3 and frame_data_float32.shape[-1] == 1:
                    header['channels'] = {'Z': Imath.Channel(exr_pixel_type())}
                    pixel_data = {'Z': exr_pixel_bytes(frame_data_float32.squeeze())}
                elif frame_data_float32.ndim == 3 and frame_data_float32.shape[-1] == 3:
                    header['channels'] = {
                        'R': Imath.Channel(exr_pixel_type()),
                        'G': Imath.Channel(exr_pixel_type()),
                        'B': Imath.Channel(exr_pixel_type())
                    }
                    pixel_data = {
                        'R': exr_pixel_bytes(frame_data_float32[:, :, 0]),
                        'G': exr_pixel_bytes(frame_data_float32[:, :, 1]),
                        'B': exr_pixel_bytes(frame_data_float32[:, :, 2])
                    }
                else:
                    _logger.error(f"Unsupported frame shape: {frame_data_float32.shape}")
                    continue
                
                exr_file = OpenEXR.OutputFile(output_exr_filepath, header)
                exr_file.writePixels(pixel_data)
                exr_file.close()
            except Exception as frame_ex:
                _logger.error(f"Error saving frame to EXR: {frame_ex}")
                
        _logger.debug(f"Saved EXR sequence to {exr_sequence_output_dir}")
        return exr_sequence_output_dir, None
    except Exception as e:
        _logger.error(f"Error saving EXR sequence: {e}")
        return None, str(e)


def save_depth_visual_as_single_exr_util(
    first_depth_frame_normalized: np.ndarray,
    output_dir_base: str,
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Saves single depth frame as EXR.
    
    Args:
        first_depth_frame_normalized: Single normalized depth frame
        output_dir_base: Base output directory
        base_filename_no_ext: Base filename without extension
    
    Returns:
        Tuple of (output_path, error_message)
    """
    if not _OPENEXR_AVAILABLE:
        return None, "OpenEXR libraries not available."

    try:
        if first_depth_frame_normalized is None or first_depth_frame_normalized.size == 0:
            return None, "No frame data to save"
        
        frame_float32 = first_depth_frame_normalized.astype(np.float32)
        
        if frame_float32.ndim == 2:
            height, width = frame_float32.shape
        elif frame_float32.ndim == 3 and (frame_float32.shape[-1] == 1 or frame_float32.shape[-1] == 3):
            height, width = frame_float32.shape[:2]
        else:
            return None, f"Unsupported frame shape: {frame_float32.shape}"

        os.makedirs(output_dir_base, exist_ok=True)
        output_exr_filepath = os.path.join(output_dir_base, f"{base_filename_no_ext}_visual.exr")
        
        header = OpenEXR.Header(width, height)
        pixel_data = {}

        if frame_float32.ndim == 2:
            header['channels'] = {'Z': Imath.Channel(exr_pixel_type())}
            pixel_data = {'Z': exr_pixel_bytes(frame_float32)}
        elif frame_float32.ndim == 3 and frame_float32.shape[-1] == 1:
            header['channels'] = {'Z': Imath.Channel(exr_pixel_type())}
            pixel_data = {'Z': exr_pixel_bytes(frame_float32.squeeze())}
        elif frame_float32.ndim == 3 and frame_float32.shape[-1] == 3:
            header['channels'] = {
                'R': Imath.Channel(exr_pixel_type()),
                'G': Imath.Channel(exr_pixel_type()),
                'B': Imath.Channel(exr_pixel_type())
            }
            pixel_data = {
                'R': exr_pixel_bytes(frame_float32[:, :, 0]),
                'G': exr_pixel_bytes(frame_float32[:, :, 1]),
                'B': exr_pixel_bytes(frame_float32[:, :, 2])
            }
        
        exr_file = OpenEXR.OutputFile(output_exr_filepath, header)
        exr_file.writePixels(pixel_data)
        exr_file.close()
        
        _logger.debug(f"Saved single EXR to {output_exr_filepath}")
        return output_exr_filepath, None
    except Exception as e:
        _logger.error(f"Error saving single EXR: {e}")
        return None, str(e)
