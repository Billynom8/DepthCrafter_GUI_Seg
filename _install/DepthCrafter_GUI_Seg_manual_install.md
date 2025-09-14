# 🛠️ Manual Installation Guide for DepthCrafter_GUI_Seg

This guide walks you through manually installing the **DepthCrafter_GUI_Seg** environment without using the batch script.

---

## 📋 Prerequisites

Ensure the following tools are installed and available in your system's PATH:

- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/)
- [CUDA Toolkit 12.8 or 12.9](https://developer.nvidia.com/cuda-toolkit)
- [FFMPEG](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/)

---

## 🚀 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Billynom8/DepthCrafter_GUI_Seg.git
```

> If the folder `DepthCrafter_GUI_Seg` already exists, delete or rename it before proceeding.

---

### 2. Navigate to the Project Directory

```bash
cd DepthCrafter_GUI_Seg
```

---

### 3. Check for CUDA Toolkit (12.8 or 12.9)

Verify that `nvcc` is available:

```bash
nvcc --version
```

Extract the version number from the output. You should see something like:

```
Cuda compilation tools, release 12.8, V12.8.89
```

> If `nvcc` is not found or the version is not 12.8 or 12.9, install the correct version from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit).

---

### 4. Verify Python Installation and Version

Check Python version:

```bash
python --version
```

> Ensure the version is **3.8 or higher**.

---

### 5. Create a Virtual Environment

```bash
python -m venv venv
```

---

### 6. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

---

### 7. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 8. Install Dependencies

Ensure `requirements.txt` exists in the project directory, then run:

```bash
python -m pip install --upgrade -r requirements.txt
```

> This will install PyTorch with CUDA support via the extra index URL defined in the file.

---

### 9. Install xformers

```bash
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

---

## ✅ Final Notes

- If any step fails, check your environment variables and permissions.
- You can refer to `install_log.txt` (if generated during script-based install) for troubleshooting.
- CUDA support is critical for GPU acceleration. Ensure your drivers and toolkit are correctly installed.

