import os
import sys
import platform
import subprocess
import shutil
from locales.i18n_extended import translate


def _is_wsl():
    return "WSL_DISTRO_NAME" in os.environ or "microsoft" in platform.release().lower()


def play_completion_sound():
    """Play a platform-appropriate notification sound.

    Returns:
        bool: True if a sound was played, False otherwise.
    """
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            return True
        if _is_wsl() and shutil.which("powershell.exe"):
            subprocess.Popen(["powershell.exe", "-c", "[console]::beep(1000,500)"])
            return True
        if sys.platform == "darwin" and shutil.which("afplay"):
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
            return True
        if sys.platform.startswith("linux"):
            if shutil.which("paplay") and os.path.exists("/usr/share/sounds/freedesktop/stereo/complete.oga"):
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
                return True
            if shutil.which("aplay") and os.path.exists("/usr/share/sounds/alsa/Front_Center.wav"):
                subprocess.Popen(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"])
                return True
        print("\a", end="")
        return True
    except Exception as e:
        print(translate("完了通知音の再生に失敗しました: {0}").format(e))
        return False
