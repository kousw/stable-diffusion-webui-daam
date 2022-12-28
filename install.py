import launch
if not launch.is_installed('matplotlib'):
    launch.run_pip('install matplotlib==3.6.2', desc='Installing matplotlib==3.6.2')