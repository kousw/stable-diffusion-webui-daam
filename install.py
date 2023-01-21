import launch


def check_matplotlib():
    if not launch.is_installed("matplotlib"):
        return False

    try:
        import matplotlib
    except ImportError:
        return False

    if hasattr(matplotlib, "__version_info__"):
        version = matplotlib.__version_info__
        version = (version.major, version.minor, version.micro)
        return version >= (3, 6, 2)
    return False


if not check_matplotlib():
    launch.run_pip("install matplotlib==3.6.2", desc="Installing matplotlib==3.6.2")
