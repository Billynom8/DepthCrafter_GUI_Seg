import os
import gc
import numpy as np
import torch
import time # For perf_counter, strftime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

# Import from the new message catalog
from message_catalog import (
    log_message,
    INFO, DEBUG, WARNING, ERROR, CRITICAL
)

# --- MODIFIED IMPORTS from depthcrafter.utils ---
from depthcrafter.utils import (
    save_video, read_video_frames,
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
    format_duration,
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    save_json_file # Now uses global log_message
)
# --- END MODIFIED IMPORTS ---

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_LOGIC = True
except ImportError:
    OPENEXR_AVAILABLE_LOGIC = False
    log_message("OPENEXR_UNAVAILABLE", context="depth_crafter_logic.py")


warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from typing import Optional, Tuple, List, Dict

class DepthCrafterDemo:
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model", use_cudnn_benchmark: bool = True):
        torch.backends.cudnn.benchmark = use_cudnn_benchmark
        try:
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            self.pipe = DepthCrafterPipeline.from_pretrained(
                pre_train_path,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                log_message("MODEL_CPU_OFFLOAD_UNKNOWN_WARN", option=cpu_offload) # New ID
                self.pipe.enable_model_cpu_offload() # Defaulting
            self.pipe.enable_attention_slicing()
            log_message("MODEL_INIT_SUCCESS", component="DepthCrafterPipeline") # Updated existing ID
        except Exception as e:
            log_message("MODEL_INIT_FAILURE", component="DepthCrafterPipeline", reason=str(e)) # Updated existing ID
            raise # Re-raise after logging

    def _setup_paths(self, base_output_folder: str, original_video_basename: str,
                     segment_job_info: Optional[dict]) -> Tuple[str, str, str]:
        actual_save_folder_for_output = base_output_folder
        output_filename_for_meta = ""

        if segment_job_info:
            segment_subfolder_name = get_segment_output_folder_name(original_video_basename)
            actual_save_folder_for_output = os.path.join(base_output_folder, segment_subfolder_name)
            output_filename_for_meta = get_segment_npz_output_filename(
                original_video_basename,
                segment_job_info['segment_id'],
                segment_job_info['total_segments']
            )
        else:
            output_filename_for_meta = get_full_video_output_filename(original_video_basename)

        full_save_path = os.path.join(actual_save_folder_for_output, output_filename_for_meta)
        os.makedirs(actual_save_folder_for_output, exist_ok=True)
        return actual_save_folder_for_output, output_filename_for_meta, full_save_path

    def _initialize_job_metadata(self, guidance_scale: float, num_denoising_steps: int,
                                    user_max_res_for_read: int, seed_val: int,
                                    target_fps_for_read: float, segment_job_info: Optional[dict],
                                    output_filename_for_meta: str, pipe_call_window_size: int,
                                    pipe_call_overlap: int,
                                    original_video_basename: str) -> dict:
        job_specific_metadata = {
            "original_video_basename": original_video_basename, 
            "guidance_scale": float(guidance_scale),
            "inference_steps": int(num_denoising_steps),
            "max_res_during_process": int(user_max_res_for_read),
            "seed": int(seed_val),
            "target_fps_setting": float(target_fps_for_read),
            "status": "pending",
            "_individual_metadata_path": None
        }

        if segment_job_info:
            job_specific_metadata.update({
                "segment_id": int(segment_job_info["segment_id"]),
                "source_start_frame_raw_index": int(segment_job_info["start_frame_raw_index"]),
                "source_num_frames_raw_for_segment": int(segment_job_info["num_frames_to_load_raw"]),
                "output_segment_filename": output_filename_for_meta,
                "output_segment_format": "npz",
                "segment_definition_window_setting": int(pipe_call_window_size),
                "segment_definition_overlap_setting": int(pipe_call_overlap)
            })
        else:
            job_specific_metadata.update({
                "output_video_filename": output_filename_for_meta,
                "pipeline_window_size_used_for_full_video_pass": int(pipe_call_window_size),
                "pipeline_overlap_used_for_full_video_pass": int(pipe_call_overlap)
            })
        return job_specific_metadata

    def _load_frames(self, video_path_for_read_or_none: Optional[str],
                     frames_array_if_provided: Optional[np.ndarray],
                     process_length_for_read: int, target_fps_for_read: float,
                     user_max_res_for_read: int, segment_job_info: Optional[dict],
                     job_specific_metadata: dict) -> Tuple[Optional[np.ndarray], float]:
        actual_frames_to_process = None
        actual_fps_for_save = target_fps_for_read

        if frames_array_if_provided is not None:
            actual_frames_to_process = frames_array_if_provided
            actual_fps_for_save = target_fps_for_read if target_fps_for_read > 0 else 30.0
            log_message("FRAMES_LOAD_FROM_ARRAY_INFO", num_frames=len(actual_frames_to_process), fps=actual_fps_for_save) # New ID
        elif video_path_for_read_or_none:
            start_frame_idx = 0
            num_frames_to_load_for_seg = -1
            if segment_job_info:
                start_frame_idx = segment_job_info["start_frame_raw_index"]
                num_frames_to_load_for_seg = segment_job_info["num_frames_to_load_raw"]

            loaded_frames, fps_from_read = read_video_frames(
                video_path_for_read_or_none, process_length_for_read,
                target_fps_for_read, user_max_res_for_read, "open",
                start_frame_index=start_frame_idx, num_frames_to_load=num_frames_to_load_for_seg
            )
            actual_frames_to_process = loaded_frames
            actual_fps_for_save = fps_from_read
            log_message("FRAMES_LOAD_FROM_VIDEO_INFO", video_path=video_path_for_read_or_none, num_frames=len(actual_frames_to_process) if actual_frames_to_process is not None else 0, fps=actual_fps_for_save) # New ID
        else:
            job_specific_metadata["status"] = "failure_no_input_source"
            log_message("FRAMES_LOAD_NO_SOURCE_ERROR") # New ID
            return None, 0.0 

        return actual_frames_to_process, actual_fps_for_save

    def _handle_no_frames_failure(self, job_specific_metadata: dict, full_save_path: str,
                                  infer_start_time: float, actual_fps_for_save: float,
                                  segment_job_info: Optional[dict],
                                  save_final_output_json_config_passed_in: bool) -> Tuple[None, dict]:
        video_basename_for_log = job_specific_metadata.get("original_video_basename", "unknown_video")
        log_message("PROCESSING_NO_FRAMES", item_name=video_basename_for_log) # Using existing ID

        job_specific_metadata["status"] = "failure_no_frames"
        job_specific_metadata["frames_in_output_video"] = 0
        job_specific_metadata["processed_at_fps"] = float(actual_fps_for_save if actual_fps_for_save is not None and actual_fps_for_save > 0 else 0)
        
        infer_duration_sec_noframes = time.perf_counter() - infer_start_time
        job_specific_metadata["internal_processing_duration_seconds"] = round(infer_duration_sec_noframes, 2)
        job_specific_metadata["internal_processing_duration_formatted"] = format_duration(infer_duration_sec_noframes)
        job_specific_metadata["processing_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        should_save_failure_json = (segment_job_info is not None) or \
                                   (not segment_job_info and save_final_output_json_config_passed_in)
        
        if should_save_failure_json and full_save_path:
            individual_metadata_json_path_noframes = get_sidecar_json_filename(full_save_path)
            if save_json_file(job_specific_metadata, individual_metadata_json_path_noframes): # log_func removed
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path_noframes)
                log_message("METADATA_SAVE_NOFRAMES_JSON_SUCCESS", filepath=individual_metadata_json_path_noframes) # New ID
            else:
                job_specific_metadata["_individual_metadata_path"] = None
                # save_json_file now logs its own errors via log_message
        else:
            job_specific_metadata["_individual_metadata_path"] = None
        return None, job_specific_metadata

    def _perform_inference(self, actual_frames_to_process: np.ndarray,
                           guidance_scale: float, num_denoising_steps: int,
                           pipe_call_window_size: int, pipe_call_overlap: int,
                           segment_job_info: Optional[dict]) -> np.ndarray:
        current_pipe_window_for_call = pipe_call_window_size
        current_pipe_overlap_for_call = pipe_call_overlap
        if segment_job_info: 
            current_pipe_window_for_call = actual_frames_to_process.shape[0]
            current_pipe_overlap_for_call = 0

        log_message("INFERENCE_START", num_frames=actual_frames_to_process.shape[0], 
                    height=actual_frames_to_process.shape[1], width=actual_frames_to_process.shape[2],
                    guidance=guidance_scale, steps=num_denoising_steps, 
                    window=current_pipe_window_for_call, overlap=current_pipe_overlap_for_call) # New ID
        with torch.inference_mode():
            res = self.pipe(
                actual_frames_to_process,
                height=actual_frames_to_process.shape[1],
                width=actual_frames_to_process.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=current_pipe_window_for_call,
                overlap=current_pipe_overlap_for_call,
            ).frames[0]
        log_message("INFERENCE_COMPLETE", result_shape=res.shape) # New ID

        if res.ndim == 4 and res.shape[-1] > 1: 
            res = res.sum(-1) / res.shape[-1]
            log_message("INFERENCE_CHANNEL_AVERAGED", final_shape=res.shape) # New ID
        return res

    def _save_segment_npz(self, res: np.ndarray, full_save_path: str, job_specific_metadata: dict) -> bool:
        try:
            np.savez_compressed(full_save_path, frames=res)
            job_specific_metadata["npz_segment_path"] = os.path.abspath(full_save_path)
            log_message("FILE_SAVE_SUCCESS", filepath=full_save_path) # Using existing ID
            return True
        except Exception as e_save_npz:
            log_message("FILE_SAVE_FAILURE", filepath=full_save_path, reason=f"NPZ segment save error: {e_save_npz}") # Using existing ID
            job_specific_metadata["status"] = "failure_npz_save"
            return False

    def _save_intermediate_visual_for_segment(self, res_normalized_for_visual: np.ndarray,
                                               actual_save_folder_for_output: str,
                                               output_filename_for_meta: str,
                                               intermediate_visual_format_to_save: str,
                                               actual_fps_for_save: float,
                                               job_specific_metadata: dict):
        base_filename_no_ext_for_visual = os.path.splitext(os.path.basename(output_filename_for_meta))[0]
        
        visual_save_path_or_dir = None
        visual_save_error = None 
        target_fps_for_visual_float = actual_fps_for_save if actual_fps_for_save > 0 else 30.0

        save_func = None
        save_args = []
        
        if intermediate_visual_format_to_save == "mp4":
            mp4_path = os.path.join(actual_save_folder_for_output, f"{base_filename_no_ext_for_visual}_visual.mp4")
            save_func = save_depth_visual_as_mp4_util
            save_args = [res_normalized_for_visual, mp4_path, target_fps_for_visual_float]
        elif intermediate_visual_format_to_save == "png_sequence":
            save_func = save_depth_visual_as_png_sequence_util
            save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
        elif intermediate_visual_format_to_save == "exr_sequence":
            save_func = save_depth_visual_as_exr_sequence_util
            save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
        elif intermediate_visual_format_to_save == "exr":
            first_frame_to_save = res_normalized_for_visual[0] if len(res_normalized_for_visual) > 0 else None
            if first_frame_to_save is None: 
                visual_save_error = "Cannot save single EXR from empty or invalid visual data."
            else: 
                save_func = save_depth_visual_as_single_exr_util
                save_args = [first_frame_to_save, actual_save_folder_for_output, base_filename_no_ext_for_visual]
        elif intermediate_visual_format_to_save == "none":
            pass 
        else:
            visual_save_error = f"Unknown intermediate visual format: {intermediate_visual_format_to_save}"

        if save_func and not visual_save_error:
            visual_save_path_or_dir, visual_save_error = save_func(*save_args)

        if visual_save_path_or_dir:
            job_specific_metadata["intermediate_visual_path"] = os.path.abspath(visual_save_path_or_dir)
            job_specific_metadata["intermediate_visual_format_saved"] = intermediate_visual_format_to_save
            log_filename_or_dirname = os.path.basename(visual_save_path_or_dir)
            log_message("VISUAL_SAVE_SEGMENT_SUCCESS", format=intermediate_visual_format_to_save, name=log_filename_or_dirname) # New ID
        
        if visual_save_error: 
            job_specific_metadata["intermediate_visual_save_error"] = visual_save_error 
            log_message("VISUAL_SAVE_SEGMENT_ERROR", format=intermediate_visual_format_to_save, error=visual_save_error) # New ID

    def _save_full_video_output(self, res: np.ndarray, full_save_path: str,
                                actual_fps_for_save: float, job_specific_metadata: dict) -> bool:
        res_min_full, res_max_full = res.min(), res.max()
        if res_max_full != res_min_full:
            res_normalized_for_mp4 = (res - res_min_full) / (res_max_full - res_min_full)
        else:
            res_normalized_for_mp4 = np.zeros_like(res)
        res_normalized_for_mp4 = np.clip(res_normalized_for_mp4, 0, 1)

        try:
            save_video_fps_full = int(round(actual_fps_for_save))
            if save_video_fps_full <= 0: save_video_fps_full = 30
            save_video(res_normalized_for_mp4, full_save_path, fps=save_video_fps_full)
            log_message("FILE_SAVE_SUCCESS", filepath=full_save_path) # Using existing ID
            return True
        except Exception as e_save_mp4:
            log_message("FILE_SAVE_FAILURE", filepath=full_save_path, reason=f"Full video MP4 save error: {e_save_mp4}") # Using existing ID
            job_specific_metadata["status"] = "failure_mp4_save"
            return False

    def _finalize_job_metadata_and_save_json(self, job_specific_metadata: dict, infer_start_time: float,
                                           actual_fps_for_save: float, frames_processed_count: int,
                                           saved_output_successfully: bool, full_save_path: Optional[str],
                                           segment_job_info: Optional[dict],
                                           save_final_output_json_config_passed_in: bool):
        if "internal_processing_duration_seconds" not in job_specific_metadata: 
            infer_duration_sec = time.perf_counter() - infer_start_time
            job_specific_metadata["internal_processing_duration_seconds"] = round(infer_duration_sec, 2)
            job_specific_metadata["internal_processing_duration_formatted"] = format_duration(infer_duration_sec)

        job_specific_metadata["processed_at_fps"] = float(actual_fps_for_save)
        job_specific_metadata["frames_in_output_video"] = frames_processed_count
        
        if saved_output_successfully and job_specific_metadata["status"] == "pending":
            job_specific_metadata["status"] = "success"
        elif job_specific_metadata["status"] == "pending": 
            job_specific_metadata["status"] = "failure_at_finalize" 
            
        if "processing_timestamp_utc" not in job_specific_metadata: 
            job_specific_metadata["processing_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        should_save_this_job_json = (segment_job_info is not None) or \
                                    (not segment_job_info and save_final_output_json_config_passed_in)
        
        if should_save_this_job_json and full_save_path:
            individual_metadata_json_path = get_sidecar_json_filename(full_save_path)
            if save_json_file(job_specific_metadata, individual_metadata_json_path): # log_func removed
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path)
                # log_message itself will log success from save_json_file
            else:
                if job_specific_metadata["status"] == "success": 
                    job_specific_metadata["status"] = "failure_metadata_save" 
                job_specific_metadata["_individual_metadata_path"] = None
                # save_json_file logs its own errors
        elif job_specific_metadata.get("_individual_metadata_path") is None : 
            job_specific_metadata["_individual_metadata_path"] = None

    def _internal_infer(self,
                        video_path_for_read_or_none: Optional[str],
                        frames_array_if_provided: Optional[np.ndarray],
                        num_denoising_steps: int, guidance_scale: float,
                        base_output_folder: str, user_max_res_for_read: int,
                        seed_val: int, original_video_basename: str,
                        process_length_for_read: int, target_fps_for_read: float,
                        pipe_call_window_size: int, pipe_call_overlap: int,
                        segment_job_info: Optional[dict] = None,
                        should_save_intermediate_visuals: bool = False,
                        intermediate_visual_format_to_save: str = "none",
                        save_final_output_json_config_passed_in: bool = False
                        ) -> Tuple[Optional[str], dict]:

        infer_start_time = time.perf_counter()
        set_seed(seed_val)
        log_message("INFERENCE_JOB_START", basename=original_video_basename, seed=seed_val, 
                    is_segment=bool(segment_job_info), segment_id=segment_job_info.get('segment_id', -1) if segment_job_info else -1) # New ID

        actual_save_folder_for_output, output_filename_for_meta, full_save_path = \
            self._setup_paths(base_output_folder, original_video_basename, segment_job_info)

        job_specific_metadata = self._initialize_job_metadata(
            guidance_scale, num_denoising_steps, user_max_res_for_read, seed_val,
            target_fps_for_read, segment_job_info, output_filename_for_meta,
            pipe_call_window_size, pipe_call_overlap, original_video_basename
        )

        actual_frames_to_process, actual_fps_for_save = self._load_frames(
            video_path_for_read_or_none, frames_array_if_provided,
            process_length_for_read, target_fps_for_read, user_max_res_for_read,
            segment_job_info, job_specific_metadata
        )

        if job_specific_metadata["status"] == "failure_no_input_source":
            self._finalize_job_metadata_and_save_json(
                job_specific_metadata, infer_start_time,
                0.0, 0, False, 
                full_save_path, segment_job_info, save_final_output_json_config_passed_in
            )
            return None, job_specific_metadata

        if actual_frames_to_process is None or actual_frames_to_process.shape[0] == 0:
            return self._handle_no_frames_failure(
                job_specific_metadata, full_save_path, infer_start_time,
                actual_fps_for_save if actual_fps_for_save is not None else 0.0,
                segment_job_info, save_final_output_json_config_passed_in
            )

        inference_result = self._perform_inference(
            actual_frames_to_process, guidance_scale, num_denoising_steps,
            pipe_call_window_size, pipe_call_overlap, segment_job_info
        )

        saved_output_successfully = False
        if segment_job_info:
            saved_output_successfully = self._save_segment_npz(
                inference_result, full_save_path, job_specific_metadata 
            )
            if saved_output_successfully and should_save_intermediate_visuals and \
               intermediate_visual_format_to_save != "none" and inference_result.size > 0:
                
                res_min_seg, res_max_seg = inference_result.min(), inference_result.max()
                if res_max_seg != res_min_seg:
                    res_normalized_for_visual = (inference_result - res_min_seg) / (res_max_seg - res_min_seg)
                else:
                    res_normalized_for_visual = np.zeros_like(inference_result)
                res_normalized_for_visual = np.clip(res_normalized_for_visual, 0, 1)

                self._save_intermediate_visual_for_segment(
                    res_normalized_for_visual, actual_save_folder_for_output, 
                    output_filename_for_meta, 
                    intermediate_visual_format_to_save,
                    actual_fps_for_save, job_specific_metadata
                )
        else: 
            saved_output_successfully = self._save_full_video_output(
                inference_result, full_save_path, actual_fps_for_save, job_specific_metadata 
            )

        self._finalize_job_metadata_and_save_json(
            job_specific_metadata, infer_start_time,
            actual_fps_for_save, actual_frames_to_process.shape[0],
            saved_output_successfully, full_save_path, 
            segment_job_info, save_final_output_json_config_passed_in
        )
        
        log_message("INFERENCE_JOB_COMPLETE", basename=original_video_basename, 
                    status=job_specific_metadata["status"],
                    duration_fmt=job_specific_metadata["internal_processing_duration_formatted"],
                    output_path=full_save_path if saved_output_successfully else "N/A") # New ID
        return full_save_path if saved_output_successfully else None, job_specific_metadata
    
    def run(self, video_path_or_frames, num_denoising_steps, guidance_scale, 
            base_output_folder, gui_window_size, gui_overlap,    
            process_length_for_read_full_video, max_res, seed,
            original_video_basename_override=None, 
            target_fps_for_read_and_save=-1, 
            segment_job_info_param=None,       
            keep_intermediate_npz_config=False,
            intermediate_segment_visual_format_config="none",
            save_final_json_for_this_job_config=False
            ): 
        video_path_for_read = None; frames_array_input = None
        original_basename = original_video_basename_override
        
        if isinstance(video_path_or_frames, str):
            video_path_for_read = video_path_or_frames
            if not original_basename: original_basename = os.path.splitext(os.path.basename(video_path_for_read))[0]
        elif isinstance(video_path_or_frames, np.ndarray):
            frames_array_input = video_path_or_frames
            if not original_basename:
                log_message("RUN_MISSING_BASENAME_ERROR") # New ID
                raise ValueError("original_video_basename_override needed for np.ndarray input.")
        else:
            log_message("RUN_INVALID_INPUT_TYPE_ERROR", type=type(video_path_or_frames).__name__) # New ID
            raise ValueError("video_path_or_frames must be str or np.ndarray.")
        if not original_basename: original_basename = "unknown_video"

        should_save_visuals_for_infer = False
        intermediate_visual_fmt_for_infer = "none"

        if segment_job_info_param and keep_intermediate_npz_config: 
            should_save_visuals_for_infer = True
            intermediate_visual_fmt_for_infer = intermediate_segment_visual_format_config

        save_path, job_metadata_dict = self._internal_infer(
            video_path_for_read_or_none=video_path_for_read, frames_array_if_provided=frames_array_input,
            num_denoising_steps=num_denoising_steps, guidance_scale=guidance_scale, base_output_folder=base_output_folder,
            user_max_res_for_read=max_res, seed_val=seed, original_video_basename=original_basename,
            process_length_for_read=process_length_for_read_full_video, 
            target_fps_for_read=target_fps_for_read_and_save, 
            pipe_call_window_size=gui_window_size, pipe_call_overlap=gui_overlap,         
            segment_job_info=segment_job_info_param,   
            should_save_intermediate_visuals=should_save_visuals_for_infer,
            intermediate_visual_format_to_save=intermediate_visual_fmt_for_infer,
            save_final_output_json_config_passed_in=save_final_json_for_this_job_config
        )
        gc.collect(); torch.cuda.empty_cache()
        return save_path, job_metadata_dict