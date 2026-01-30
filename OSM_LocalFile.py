#
#   Override querying overpass API when user chooses to supply a .OSM file locally.
#   Provides UI elements for selecting a local .OSM file and enabling/disabling the override.
#   execute_local_override() is called by overpass_query_retry() if is_local_override_enabled() returns True instead of querying the Overpass API.
import tkinter as tk
import os
import overpy

_osm_local_override_enabled = None # type: tk.BooleanVar
_osm_local_override_filename = None # type: tk.StringVar
_osm_local_override_errmsg = None # type: tk.StringVar
_osm_local_err_frame = None # type: tk.Frame

def is_local_override_enabled():
    """
    Check if the local OSM file override is enabled.
    :return: True if enabled, False otherwise.
    """
    global _osm_local_override_enabled
    if(_osm_local_override_enabled is None):
        return False
    return _osm_local_override_enabled.get()

def execute_local_override(api, query, printf=print, name="OSM"):
    """
    Executed when local override is enabled. Instead of querying (with retries) the Overpass API, this function reads the local .OSM file and parses it using overpy's built in parse_from_xml function. The actual api query parameters are discarded for now, as we don't seem to need it; the rest of TGC Designer Tool functions seem to manage the data correctly once we return the overpy.Result object, even if the overpy.Result may be larger than what we would have gotten from the server.
    """
    global _osm_local_override_filename
    global _osm_local_override_enabled
    if(_osm_local_override_filename is None):
        printf("Local OSM override filename variable not initialized.")
        show_error("Internal error: filename variable not initialized.")
        _osm_local_override_enabled.set(False)
        return None
    filename = _osm_local_override_filename.get()
    if(len(filename) == 0 or filename == "File selected: <none>"):
        printf("Local OSM override enabled but no file selected.")
        show_error("Please select a valid .OSM file before enabling the override.")
        _osm_local_override_enabled.set(False)
        return None
    
    printf("[{name}] using local OSM file override: {filename}".format(name=name, filename=filename))
    with open(filename, "r") as file:
        osm_data = file.read()
    printf("[{name}] parsing local OSM file...".format(name=name))
    result = overpy.Result.from_xml(osm_data)
    return result

def add_local_override_ui(parent_frame, bg="grey25", fg="grey90", check_button_bg="grey60", check_button_fg="black"):
    """
    Add UI elements to allow user to select a local .OSM file as an override. Called after the course button in tgc_gui.py. 
    :param parent_frame: The parent Tkinter frame to which the UI elements will be added.
    :return: None
    """
    global _osm_local_override_enabled
    global _osm_local_override_filename
    global _osm_local_override_errmsg
    global _osm_local_err_frame

    if(_osm_local_override_enabled is None):
        _osm_local_override_enabled = tk.BooleanVar()
    if(_osm_local_override_filename is None):
        _osm_local_override_filename = tk.StringVar(value="File selected: <none>")
    if(_osm_local_override_errmsg is None):
        _osm_local_override_errmsg = tk.StringVar(value="")

    frame = tk.Frame(parent_frame, bg=bg)
    
    top_frame = tk.Frame(frame, bg=bg)
    bottom_frame = tk.Frame(frame, bg=bg)

    if(_osm_local_err_frame is None):
        _osm_local_err_frame = tk.Frame(bottom_frame, bg=bg)

    fp_button = tk.Button(top_frame, text="Select .OSM File", command=on_filepick)
    checkbox = tk.Checkbutton(top_frame, variable=_osm_local_override_enabled,bg=check_button_bg, fg=check_button_fg, text="Use local OSM instead of querying Overpass API", command=on_toggle)

    filename_label = tk.Label(bottom_frame, textvariable=_osm_local_override_filename, wraplength=500, justify="left", bg=bg, fg=fg)

    err_label = tk.Label(_osm_local_err_frame, textvariable=_osm_local_override_errmsg, justify="left", fg="red", bg="black", wraplength=500)
    ok_err_button = tk.Button(_osm_local_err_frame, text=" OK ", fg="white", bg="red", cursor="hand2", command=lambda: show_error(""))

    frame.pack(pady=5)

    top_frame.pack(side="top")
    bottom_frame.pack(side="bottom")

    fp_button.pack(padx=10, pady=5, side="left")
    checkbox.pack(padx=10, pady=5, side="right")

    filename_label.pack(side="top")
    err_label.pack(side="left")
    ok_err_button.pack(side="right")

def show_error(msg):
    """
    Shows an error message relating to the local OSM file selection.
    """
    global _osm_local_override_errmsg
    global _osm_local_err_frame
    if(msg == None or len(msg) == 0):
        _osm_local_override_errmsg.set("")
        if(_osm_local_err_frame is not None):
            _osm_local_err_frame.pack_forget()
    else:
        _osm_local_override_errmsg.set(msg)
        if(_osm_local_err_frame is not None):
            _osm_local_err_frame.pack(side="top")

def on_filepick():
    """
    Callback for file picker button.
    """
    global _osm_local_override_filename
    global _osm_local_override_enabled
    if(_osm_local_override_filename is None):
        show_error("Internal error: filename variable not initialized.")
        return
    file = tk.filedialog.askopenfilename(title="Select OSM File", filetypes=[("OSM Files", "*.osm"), ("All Files", "*.*")])
    if file:
        _osm_local_override_filename.set(file)
    if _osm_local_override_enabled is not None:
        if(_osm_local_override_enabled.get()):
            on_checked()

def on_checked():
    """
    Callback for checkbox toggle to true. Also called when the file name is changed while the checkbox is checked. Validates the file, only allowing _osm_local_override_enabled to end up true if the file is valid.

    Does create a overpy.Result object to validate the file contents, but this seems quick enough to be worth the validation.
    """
    global _osm_local_override_enabled
    global _osm_local_override_filename
    if(_osm_local_override_enabled is None):
        show_error("Internal error: override enabled variable not initialized.")
        return
    if(_osm_local_override_filename is None):
        show_error("Internal error: filename variable not initialized.")
        _osm_local_override_enabled.set(False)
        return
    if(_osm_local_override_enabled.get()):
        if(len(_osm_local_override_filename.get()) == 0 or _osm_local_override_filename.get() == "File selected: <none>"):
            show_error("Please select a valid .OSM file before enabling the override.")
            _osm_local_override_enabled.set(False)
        else:
            # validate file exists
            if(not os.path.isfile(_osm_local_override_filename.get())):
                show_error("The selected file does not exist. Please select a valid .OSM file.")
                _osm_local_override_enabled.set(False)
            else:                
                try:
                    with open(_osm_local_override_filename.get(), "r") as file:
                        osm_data = file.read()
                    result = overpy.Result.from_xml(osm_data)
                    show_error("")
                except Exception as e:
                    print("Error reading or parsing local OSM file: {}".format(e))
                    show_error("Failed to read or parse the selected .OSM file. Inspect the file in a text editor to ensure it is a valid OSM XML file.")
                    _osm_local_override_enabled.set(False)

def on_toggle():
    global _osm_local_override_enabled
    if(_osm_local_override_enabled is not None):
        if(_osm_local_override_enabled.get()):
            on_checked()
    