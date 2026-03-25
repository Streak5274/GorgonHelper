"""Global keyboard hotkey via Win32 WH_KEYBOARD_LL hook.

Same low-level hook approach as InventoryClickWatcher (WH_MOUSE_LL).
Must be started from the thread that runs the Qt/Windows message pump
(i.e. the qasync main thread).
"""
import ctypes
import ctypes.wintypes as wt
from ctypes import CFUNCTYPE, c_int, c_longlong, POINTER

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
HC_ACTION = 0


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode",      wt.DWORD),
        ("scanCode",    wt.DWORD),
        ("flags",       wt.DWORD),
        ("time",        wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


HOOKPROC = CFUNCTYPE(c_longlong, c_int, wt.WPARAM, POINTER(KBDLLHOOKSTRUCT))


class KeyboardHotkey:
    """Fires *callback* (no args) whenever *vk_code* is pressed, globally.

    Usage::

        hotkey = KeyboardHotkey(0x75, my_callback)  # 0x75 = F6
        hotkey.start()
        # ... later ...
        hotkey.stop()
    """

    def __init__(self, vk_code: int, callback):
        self._vk = vk_code
        self._callback = callback
        self._hook = None
        self._proc_ref = None   # keep reference so ctypes doesn't GC the function

    # ------------------------------------------------------------------

    def start(self):
        if self._hook:
            return

        def _low_level_proc(nCode, wParam, lParam):
            if nCode == HC_ACTION and wParam == WM_KEYDOWN:
                try:
                    if lParam.contents.vkCode == self._vk:
                        self._callback()
                except Exception:
                    pass
            return ctypes.windll.user32.CallNextHookEx(None, nCode, wParam, lParam)

        self._proc_ref = HOOKPROC(_low_level_proc)
        self._hook = ctypes.windll.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc_ref, None, 0
        )

    def stop(self):
        if self._hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self._proc_ref = None

    def update_vk(self, vk_code: int):
        """Change the trigger key without restarting the hook."""
        self._vk = vk_code
