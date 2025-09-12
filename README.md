## ___***DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos***___
<div align="center">
<img src='https://depthcrafter.github.io/img/logo.png' style="height:140px"></img>



![Version](https://img.shields.io/badge/version-1.0.1-blue) &nbsp;
 <a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
 <a href='https://depthcrafter.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/tencent/DepthCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;


_**[Wenbo Hu<sup>1* &dagger;</sup>](https://wbhu.github.io), 
[Xiangjun Gao<sup>2*</sup>](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en), 
[Xiaoyu Li<sup>1* &dagger;</sup>](https://xiaoyu258.github.io), 
[Sijie Zhao<sup>1</sup>](https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en), 
[Xiaodong Cun<sup>1</sup>](https://vinthony.github.io/academic), <br>
[Yong Zhang<sup>1</sup>](https://yzhang2016.github.io), 
[Long Quan<sup>2</sup>](https://home.cse.ust.hk/~quan), 
[Ying Shan<sup>3, 1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**_
<br><br>
<sup>1</sup>Tencent AI Labv
<sup>2</sup>The Hong Kong University of Science and Technology
<sup>3</sup>ARC Lab, Tencent PCG

CVPR 2025， **Highlight**


</div>


## GUI Update

*   `[25-09-12]` 

**Added Features:**

1.  **GUI Status Bar:** Added dynamic text updates below progress bar.
2.  **Current Processing Info Frame:** Display for filename, resolution, frames of current job.
3.  **"Local Models Only" Menu Option:** Checkbox to force local model loading only.
4.  **"Restore Finished/Failed Input Files" Menu:** Restores original source videos.
5.  **Independent Height/Width GUI Inputs:** Replaced `max_res` control.
6.  **Robust FPS/Frame Count:** Integrated `ffprobe` for accurate frame rate read from source.
7.  **Tooltip Help System:** Replaced "❓" icons with mouse-over tooltips.

**Removed Features:**

1.  **Custom `message_catalog.py`:** Entirely deleted.
2.  **GUI Log Window (`tk.Text` widget):** Removed internal log display.
3.  **GUI Log Verbosity Control:** Removed combobox for setting log verbosity.
4.  **"❓" Help Icons:** Replaced by new tooltip system.

Also moved helper files into depthcrafter sub-folder

*   `[25-06-01]` 

    *   Image sequence loading.
    *   Single image loading to generate 5 frame depth video or image.
    *   Single video loading and process (not batch).
    *   Save 10bit x265 MP4

## GUI for Enhanced Workflow (Community Contribution)

<div align="center">
  <img src="https://raw.githubusercontent.com/Billynom8/DepthCrafter_GUI_Seg/main/tools/gui_image_25.09.12.png" alt="DepthCrafter GUI" width="600" />
</div>

A graphical user interface (GUI) has been added to DepthCrafter to simplify batch processing, manage videos requiring segmentation (e.g., for low VRAM environments or very long videos), and streamline the output process.

**Key Features Added:**

*   **Intuitive Graphical Interface:**
    *   Easily select input video folders and output directories.
    *   Adjust core DepthCrafter parameters (guidance scale, inference steps, resolution, seed, etc.) through a user-friendly panel.
*   **Video Segmentation for Long/Large Videos:**
    *   **Automatic Segmentation:** Process videos in manageable segments with configurable window sizes and overlap, ideal for systems with limited VRAM or for extremely long input videos.
    *   **Segment Management:** Handles the creation and processing of individual video segments.
    *   **Resume Capability:** If processing is interrupted, the GUI can identify and re-process only missing or failed segments, saving significant time.
*   **Segment Merging & Post-Processing:**
    *   **Automated Merging:** After segments are processed, the GUI can automatically merge them back into a full-length depth map video.
    *   **Alignment Options:** Choose between "Compute Shift & Scale" or "Linear Blend" methods for aligning segment values during the merge.
    *   **Post-Processing for MP4:** Apply optional gamma adjustment and dithering to merged MP4 outputs to enhance visual quality.
    *   **Flexible Output Formats for Merged Video:** Save the final merged depth video as MP4, or as a sequence of PNG or EXR frames. Single frame EXR output is also supported for the first frame of a merged sequence.
    *   **Re-merging:** Re-merge final video without re processing. (If NPZ files are saved)
*   **Job Management & Configuration:**
    *   **Batch Processing:** Process all videos within a selected input folder.
    *   **Settings Persistence:** Save and load GUI settings (including all processing parameters) to/from JSON configuration files for repeatable workflows.
*   **Intermediate Output Control:**
    *   Optionally keep intermediate NPZ segment files.
    *   Generate visual representations (MP4, PNG/EXR sequence) of individual depth segments for review *after* initial NPZ processing.
*   **Sidecar Metadata:**
    *   Generates detailed JSON metadata files for both individual segments and final processed/merged videos, capturing all settings and processing information(required for resuming).

This GUI aims to make DepthCrafter more accessible and efficient for users dealing with multiple videos or those needing to break down large processing tasks.

[GUI based on enoky/DepthCrafter](https://github.com/enoky/DepthCrafter)
[Main code contributer Gemini 2.5](https://aistudio.google.com/)

---

# 📦 DepthCrafter_GUI_Seg Installer

> ⚠️ **Do not run the installer from inside the cloned repository.**  
> The installer itself is responsible for cloning the repository correctly.
[more](https://github.com/Billynom8/DepthCrafter_GUI_Seg/blob/main/_install/READ_BEFORE_CLONING.md)
---

## 🔆 Introduction
🤗 If you find DepthCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

🔥 DepthCrafter can generate temporally consistent long-depth sequences with fine-grained details for open-world videos, 
without requiring additional information such as camera poses or optical flow.

- `[25-04-05]` 🔥🔥🔥 Its upgraded work, [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter), is released now, for **video to point cloud**!
- `[25-04-05]` 🎉🎉🎉 DepthCrafter is selected as **Highlight** in CVPR‘25.
- `[24-12-10]` 🌟🌟🌟 EXR output format is supported now, with --save_exr option.
- `[24-11-26]` 🚀🚀🚀 DepthCrafter v1.0.1 is released now, with improved quality and speed
- `[24-10-19]` 🤗🤗🤗 DepthCrafter now has been integrated into [ComfyUI](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes)!
- `[24-10-08]` 🤗🤗🤗 DepthCrafter now has been integrated into [Nuke](https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter), have a try!
- `[24-09-28]` Add full dataset inference and evaluation scripts for better comparison use. :-)
- `[24-09-25]` 🤗🤗🤗 Add huggingface online demo [DepthCrafter](https://huggingface.co/spaces/tencent/DepthCrafter). 
- `[24-09-19]` Add scripts for preparing benchmark datasets. 
- `[24-09-18]` Add point cloud sequence visualization.
- `[24-09-14]` 🔥🔥🔥 **DepthCrafter** is released now, have fun!


## 📦 Release Notes
- **DepthCrafter v1.0.1**:
    - Quality and speed improvement
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>ms/frame&#x2193; @1024&#xD7;576 </th>
              <th colspan="2">Sintel (~50 frames)</th>
              <th colspan="2">Scannet (90 frames)</th>
              <th colspan="2">KITTI (110 frames)</th>
              <th colspan="2">Bonn (110 frames)</th>
            </tr>
            <tr>
                <th></th>
                <th></th>
              <th>AbsRel&#x2193;</th>
              <th>&delta;&#x2081; &#x2191;</th>
              <th>AbsRel&#x2193;</th>
              <th>&delta;&#x2081; &#x2191;</th>
              <th>AbsRel&#x2193;</th>
              <th>&delta;&#x2081; &#x2191;</th>
              <th>AbsRel&#x2193;</th>
              <th>&delta;&#x2081; &#x2191;</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Marigold</td>
              <td>1070.29</td>
              <td>0.532</td>
              <td>0.515</td>
              <td>0.166</td>
              <td>0.769</td>
              <td>0.149</td>
              <td>0.796</td>
              <td>0.091</td>
              <td>0.931</td>
            </tr>
            <tr>
              <td>Depth-Anything-V2</td>
              <td><strong>180.46</strong></td>
              <td>0.367</td>
              <td>0.554</td>
              <td>0.135</td>
              <td>0.822</td>
              <td>0.140</td>
              <td>0.804</td>
              <td>0.106</td>
              <td>0.921</td>
            </tr>
            <tr>
              <td>DepthCrafter previous</td>
              <td>1913.92</td>
                <td><u>0.292</u></td>
                <td><strong>0.697</strong></td>
                <td><u>0.125</u></td>
                <td><u>0.848</u></td>
                <td><u>0.110</u></td>
                <td><u>0.881</u></td>
                <td><u>0.075</u></td>
                <td><u>0.971</u></td>
            </tr>
            <tr>
              <td>DepthCrafter v1.0.1</td>
              <td><u>465.84</u></td>
                <td><strong>0.270</strong></td>
                <td><strong>0.697</strong></td>
                <td><strong>0.123</strong></td>
                <td><strong>0.856</strong></td>
                <td><strong>0.104</strong></td>
                <td><strong>0.896</strong></td>
                <td><strong>0.071</strong></td>
                <td><strong>0.972</strong></td>
            </tr>
          </tbody>
        </table>

    

## 🎥 Visualization
We provide demos of unprojected point cloud sequences, with reference RGB and estimated depth videos. 
For more details, please refer to our [project page](https://depthcrafter.github.io).


https://github.com/user-attachments/assets/62141cc8-04d0-458f-9558-fe50bc04cc21




## 🚀 Quick Start

### 🤖 Gradio Demo
- Online demo: [DepthCrafter](https://huggingface.co/spaces/tencent/DepthCrafter) 
- Local demo:
    ```bash
    gradio app.py
    ``` 

### 🌟 Community Support
- [NukeDepthCrafter](https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter): 
    a plugin allows you to generate temporally consistent Depth sequences inside Nuke, 
    which is widely used in the VFX industry.
- [ComfyUI-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes): creating consistent depth maps for your videos using DepthCrafter in ComfyUI.


### 🛠️ Installation
1. Clone this repo:
```bash
git clone https://github.com/Tencent/DepthCrafter.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```



### 🤗 Model Zoo
[DepthCrafter](https://huggingface.co/tencent/DepthCrafter) is available in the Hugging Face Model Hub.

### 🏃‍♂️ Inference
#### 1. High-resolution inference, requires a GPU with ~26GB memory for 1024x576 resolution:
- ~2.1 fps on A100, recommended for high-quality results:

    ```bash
    python run.py  --video-path examples/example_01.mp4
    ```

#### 2. Low-resolution inference requires a GPU with ~9GB memory for 512x256 resolution:
- ~8.6 fps on A100:

    ```bash
    python run.py  --video-path examples/example_01.mp4 --max-res 512
    ```

## 🚀 Dataset Evaluation
Please check the `benchmark` folder. 
- To create the dataset we use in the paper, you need to run `dataset_extract/dataset_extract_${dataset_name}.py`.
- Then you will get the `csv` files that save the relative root of extracted RGB video and depth npz files. We also provide these csv files.
- Inference for all datasets scripts:
  ```bash
  bash benchmark/infer/infer.sh
  ```
  (Remember to replace the `input_rgb_root` and `saved_root` with your path.)
- Evaluation for all datasets scripts:
  ```bash
  bash benchmark/eval/eval.sh
  ```
   (Remember to replace the `pred_disp_root` and `gt_disp_root` with your wpath.)
####

## 🤝🍻 Contributing
- Welcome to open issues and pull requests.
- Welcome to optimize the inference speed and memory usage, e.g., through model quantization, distillation, or other acceleration techniques.

    ### Contributors
    <a href="https://github.com/Tencent/DepthCrafter/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Tencent/DepthCrafter" />
    </a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tencent/DepthCrafter&type=Date)](https://star-history.com/#Tencent/DepthCrafter&Date)


## 📜 Citation
If you find this work helpful, please consider citing:
```BibTeXw
@inproceedings{hu2025-DepthCrafter,
            author      = {Hu, Wenbo and Gao, Xiangjun and Li, Xiaoyu and Zhao, Sijie and Cun, Xiaodong and Zhang, Yong and Quan, Long and Shan, Ying},
            title       = {DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos},
            booktitle   = {CVPR},
            year        = {2025}
    }
```
