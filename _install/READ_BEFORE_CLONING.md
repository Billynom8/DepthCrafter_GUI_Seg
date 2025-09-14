# 📦 How to Use the DepthCrafter_GUI_Seg Installer

> ⚠️ **Do not run the installer from inside the cloned repository.**  
> The installer itself is responsible for cloning the repository correctly.

---

## 🧭 Usage Instructions

### 1. Download the Installer

- Obtain the `DepthCrafter_GUI_Seg_Installer.bat` file from the release package or distribution source.
- Place it in any folder **outside** of `DepthCrafter_GUI_Seg`.

---

### 2. Run the Installer

- Double-click the `.bat` file, or run it from a terminal:

```cmd
DepthCrafter_GUI_Seg_Installer.bat
```

> The installer will:
> - Check for required tools (Git, Python, CUDA, FFMPEG)
> - Clone the `DepthCrafter_GUI_Seg` repository
> - Set up a virtual environment
> - Install all dependencies

---

### 3. Launch the Application

Once installation completes successfully:

- Navigate to the newly created `DepthCrafter_GUI_Seg` folder.
- Follow the project’s usage instructions (e.g., run the GUI or script entry point).

---

## 🛠️ Troubleshooting

- Check `install_log.txt` for detailed logs if anything fails.
- Ensure Git, Python 3.8+, and CUDA 12.8/12.9 are installed and available in your system's PATH.

---