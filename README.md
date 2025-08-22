# TGC-Designer-Tools (2k25-friendly fork)

This is a community fork of Chad Rockey's TGC-Designer-Tools. It keeps the original features (Lidar LAZ + OpenStreetMap pipeline) and adds fixes so empty PGA2K25 templates work.

This includes updates from HiCamino that provide optional smoothing facilities.


## What is new in this fork

- Fix: first-tab render no longer crashes with `KeyError: 'surfaceBrushes'` on 2K25 templates.
- Fix: "Shift Features" works across legacy and 2K25 format files.

Tested on Windows 11, Python 3.11.



## Quick start (Windows, run from source as a developer)

1) Install Python 3.11 (x64) and Git for Windows.

2) Create a venv (virtual environment) and install dependencies:
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install -U pip setuptools wheel
   pip install -r requirements.txt
   
3) Run:
   python tgc_gui.py

   If OpenCV DLL issues occur, try:
      pip uninstall -y opencv-python
      pip install opencv-python-headless==4.10.0.84   
      
      
## Troubleshooting

### Pip fails on laspy with "No module named 'numpy'"

The legacy laspy fork imports numpy at build time. Install numpy first, then the rest:

   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   pip install -U pip setuptools wheel
   pip install "numpy==1.26.4"
   pip install -r requirements.txt
   
Make sure the laspy line in requirements.txt uses https, not git:
   git+https://github.com/chadrockey/laspy@14_fix#egg=laspy   
   
   
### VS Code shows yellow squiggles for cv2 / numpy / PIL but the app runs

That is the editor not seeing your venv, not a runtime error.

   Ctrl+Shift+P -> Python: Select Interpreter -> choose .venv\Scripts\python.exe
   Ctrl+Shift+P -> Python: Restart Language Server

   If needed, add to .vscode/settings.json:

   {
     "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
     "python.terminal.activateEnvironment": true,
     "terminal.integrated.defaultProfile.windows": "Command Prompt",
     "python.analysis.extraPaths": [
       "${workspaceFolder}\\.venv\\Lib\\site-packages"
     ]
   }   
   
   
### OpenCV DLL issues on some machines

If you hit a DLL load error and you do not need GUI windows from OpenCV, use the headless wheel:

   pip uninstall -y opencv-python
   pip install opencv-python-headless==4.10.0.84   