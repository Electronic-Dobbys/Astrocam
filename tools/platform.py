from dataclasses import dataclass
import platform, os

@dataclass(frozen=True)
class PlatformInfo:
    os_name:str
    arch:str
    is_rpi:bool
    is_arm:bool

def detect_platform():
    sys=platform.system().lower()
    os_name='windows' if 'windows'in sys else 'linux' if 'linux'in sys else 'macos'
    arch=platform.machine().lower()
    is_arm='arm' in arch
    is_rpi=False
    if os_name=='linux' and is_arm:
        try:
            with open('/proc/device-tree/model','rb') as f:
                if b'raspberry pi' in f.read().lower(): is_rpi=True
        except: pass
    return PlatformInfo(os_name,arch,is_rpi,is_arm)
