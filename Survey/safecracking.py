"""Safecracking puzzle solver for Project Gorgon.

Mastermind-variant: 4 slots, up to 12 symbols, repetition allowed.
Feedback: (exact_matches, misplaced_matches)

Symbols are referred to by 1-based index (1..N) matching the display order.
"""
import base64
import io
import logging
import re
from collections import Counter
from itertools import product
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window layout constants (fractions of whole Minotaur Lock window)
# Tune these if the game window proportions change.
# ---------------------------------------------------------------------------

# (left, top, width, height) as fraction of window size
SC_SYMBOL_GRID    = (0.08, 0.37, 0.66, 0.24)   # 6×2 rune symbol grid
SC_SOLUTION_SLOTS = (0.08, 0.23, 0.43, 0.10)   # 4 solution slots (current guess)
SC_GUESS_HISTORY  = (0.14, 0.64, 0.83, 0.34)   # entire guess history area

SC_GUESS_COLS = 3
SC_GUESS_ROWS = 4
SC_FEEDBACK_X_FRAC = 0.72   # feedback text starts at this fraction of each panel width

# Orange rune pixel detection (R>150, G<120, B<80)
_ORANGE_R_MIN, _ORANGE_G_MAX, _ORANGE_B_MAX = 150, 120, 80
_ORANGE_THRESHOLD = 30   # minimum orange pixels to consider a slot filled

# MSE threshold for symbol identification (lower = stricter match)
_MSE_THRESHOLD = 2000.0

# Optional: pytesseract for feedback OCR
try:
    import pytesseract  # type: ignore
    _TESS_OK = True
except ImportError:
    _TESS_OK = False


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_guess(guess: Tuple[int, ...], secret: Tuple[int, ...]) -> Tuple[int, int]:
    """Return (exact, misplaced) feedback for a guess vs. the secret."""
    exact = sum(g == s for g, s in zip(guess, secret))
    total = sum((Counter(guess) & Counter(secret)).values())
    return exact, total - exact


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SafecrackingSolver:
    """Constraint-based solver tracking remaining consistent candidates.

    Strategy
    --------
    Phase 1 — AABB: test symbol pairs (s0,s0,s1,s1), (s2,s2,s3,s3), …
              until the accumulated (exact+misplaced) across AABB guesses
              reaches 4 (all solution symbols identified) or all pairs used.
    Phase 2 — Return the first remaining candidate (fine for the small search
              space: 6^4 = 1 296, 12^4 = 20 736 max).
    """

    SLOTS = 4

    def __init__(self, symbols: List[int]) -> None:
        self.symbols: List[int] = list(symbols)
        self._all: List[Tuple[int, ...]] = list(product(self.symbols, repeat=self.SLOTS))
        self._candidates: List[Tuple[int, ...]] = list(self._all)
        self._history: List[Tuple[List[int], int, int]] = []   # (guess, exact, misplaced)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def candidates_count(self) -> int:
        return len(self._candidates)

    @property
    def history(self) -> List[dict]:
        return [{"guess": list(g), "exact": e, "misplaced": m}
                for g, e, m in self._history]

    @property
    def solved(self) -> bool:
        return len(self._candidates) == 1

    @property
    def impossible(self) -> bool:
        return len(self._candidates) == 0

    @property
    def guess_count(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> Optional[List[int]]:
        """Return next recommended guess as a list of symbol indices."""
        if self.impossible:
            return None
        if self.solved:
            return list(self._candidates[0])

        aabb = self._next_aabb()
        if aabb is not None:
            return list(aabb)

        return list(self._candidates[0])

    def record(self, guess: List[int], exact: int, misplaced: int) -> None:
        """Record feedback and prune candidates."""
        t = tuple(guess)
        self._history.append((guess, exact, misplaced))
        self._candidates = [
            c for c in self._candidates
            if score_guess(t, c) == (exact, misplaced)
        ]

    def undo(self) -> bool:
        """Remove the last recorded guess and recompute candidates."""
        if not self._history:
            return False
        self._history.pop()
        self._candidates = list(self._all)
        for guess, exact, misplaced in self._history:
            t = tuple(guess)
            self._candidates = [c for c in self._candidates
                                 if score_guess(t, c) == (exact, misplaced)]
        return True

    def reset(self, symbols: Optional[List[int]] = None) -> None:
        if symbols is not None:
            self.symbols = list(symbols)
            self._all = list(product(self.symbols, repeat=self.SLOTS))
        self._candidates = list(self._all)
        self._history = []

    def to_dict(self) -> dict:
        """Serialise state for broadcast."""
        suggestion = self.suggest()
        return {
            "symbols": self.symbols,
            "candidates": self.candidates_count,
            "suggestion": suggestion,
            "history": self.history,
            "solved": self.solved,
            "impossible": self.impossible,
            "guess_count": self.guess_count,
        }

    # ------------------------------------------------------------------
    # AABB phase helpers
    # ------------------------------------------------------------------

    def _next_aabb(self) -> Optional[Tuple[int, ...]]:
        """Return the next AABB pair to test, or None if AABB phase is done."""
        n = len(self.symbols)
        max_pairs = n // 2

        pairs_tested = 0
        total_hits = 0

        for guess, exact, misplaced in self._history:
            g = tuple(guess)
            if self._is_aabb(g):
                pairs_tested += 1
                total_hits += exact + misplaced
            else:
                return None   # non-AABB guess made — phase over

        if total_hits >= 4 or pairs_tested >= max_pairs:
            return None

        a = self.symbols[pairs_tested * 2]
        b = self.symbols[pairs_tested * 2 + 1]
        return (a, a, b, b)

    @staticmethod
    def _is_aabb(g: Tuple[int, ...]) -> bool:
        return len(g) == 4 and g[0] == g[1] and g[2] == g[3] and g[0] != g[2]


# ---------------------------------------------------------------------------
# Screen capture helpers
# ---------------------------------------------------------------------------

def _grab_window(x: int, y: int, w: int, h: int):
    """Grab the whole window region. Returns a PIL Image or raises."""
    from PIL import ImageGrab  # type: ignore
    try:
        return ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
    except TypeError:
        return ImageGrab.grab(bbox=(x, y, x + w, y + h))


def _sub_crop(img, window_w: int, window_h: int, fracs: Tuple[float, float, float, float]):
    """Crop a sub-region from a whole-window image using fractional coordinates."""
    fl, ft, fw, fh = fracs
    left   = int(fl * window_w)
    top    = int(ft * window_h)
    right  = left + int(fw * window_w)
    bottom = top  + int(fh * window_h)
    return img.crop((left, top, right, bottom))


def _img_to_b64(img, thumb: int = 52) -> str:
    """Resize image to thumb×thumb and return base64-encoded PNG."""
    from PIL import Image  # type: ignore
    img = img.resize((thumb, thumb), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_arr(b64: str):
    """Decode a base64 PNG to a numpy uint8 array (H×W×3 RGB)."""
    import numpy as np
    from PIL import Image  # type: ignore
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _detect_orange(arr) -> bool:
    """Return True if the image array contains enough orange rune pixels."""
    import numpy as np
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mask = (r > _ORANGE_R_MIN) & (g < _ORANGE_G_MAX) & (b < _ORANGE_B_MAX)
    return int(mask.sum()) >= _ORANGE_THRESHOLD


def _identify_symbol(cell_img, known_b64s: List[str]) -> int:
    """Return 1-based index of the best-matching known symbol, or 0 if no match.

    Uses normalised MSE over the resized thumbnail.
    """
    import numpy as np
    from PIL import Image  # type: ignore

    if not known_b64s:
        return 0

    thumb = 52
    cell_arr = np.array(cell_img.resize((thumb, thumb), Image.LANCZOS), dtype=np.float32)

    best_idx = 0
    best_mse = _MSE_THRESHOLD

    for i, b64 in enumerate(known_b64s):
        try:
            ref_arr = _b64_to_arr(b64).astype(np.float32)
            if ref_arr.shape != cell_arr.shape:
                ref_img = Image.fromarray(ref_arr.astype(np.uint8))
                ref_arr = np.array(ref_img.resize((thumb, thumb), Image.LANCZOS),
                                   dtype=np.float32)
            mse = float(np.mean((cell_arr - ref_arr) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_idx = i + 1   # 1-based
        except Exception:
            continue

    return best_idx


def _ocr_feedback(img) -> Optional[Tuple[int, int]]:
    """OCR a small feedback region and parse 'X, Y'. Returns None if unavailable."""
    if not _TESS_OK:
        return None
    try:
        import numpy as np
        from PIL import Image, ImageFilter  # type: ignore

        # Pre-process: scale up, grayscale, threshold for cleaner OCR
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        gray = img.convert("L")
        # Invert if background is light (game UI is light gray)
        arr = np.array(gray)
        if float(arr.mean()) > 128:
            gray = gray.point(lambda p: 255 - p)
        txt = pytesseract.image_to_string(  # type: ignore[name-defined]
            gray, config="--psm 7 -c tessedit_char_whitelist=0123456789, "
        ).strip()
        m = re.search(r'(\d)\s*,\s*(\d)', txt)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception as exc:
        log.debug("OCR feedback failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public capture functions
# ---------------------------------------------------------------------------

def capture_symbols(x: int, y: int, w: int, h: int,
                    cols: int = 6, rows: int = 2,
                    thumb: int = 52) -> List[str]:
    """Capture the whole Minotaur Lock window and extract 12 symbol thumbnails.

    (x, y, w, h) is now the whole window.  The symbol grid sub-region is
    located automatically using SC_SYMBOL_GRID fractions.

    Returns a flat list of ``cols * rows`` base64-encoded PNG strings,
    ordered left-to-right, top-to-bottom.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        log.error("Pillow not installed — cannot capture symbols")
        return []

    try:
        window_img = _grab_window(x, y, w, h)
        grid_img = _sub_crop(window_img, w, h, SC_SYMBOL_GRID)
        gw, gh = grid_img.size

        cw = gw // cols
        ch = gh // rows
        result: List[str] = []

        for r in range(rows):
            for c in range(cols):
                cell = grid_img.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
                result.append(_img_to_b64(cell, thumb))

        return result

    except Exception as exc:
        log.warning("Symbol capture failed: %s", exc)
        return []


def capture_current_slots(x: int, y: int, w: int, h: int,
                          known_b64s: List[str]) -> List[int]:
    """Read the 4 solution slots and identify which known symbols are placed.

    Returns a list of 4 ints (1-based symbol index, or 0 = empty / unknown).
    """
    try:
        window_img = _grab_window(x, y, w, h)
        slots_img = _sub_crop(window_img, w, h, SC_SOLUTION_SLOTS)
        sw, sh = slots_img.size
        slot_w = sw // 4
        result: List[int] = []

        for i in range(4):
            cell = slots_img.crop((i * slot_w, 0, (i + 1) * slot_w, sh))
            import numpy as np
            arr = np.array(cell.convert("RGB"), dtype=np.uint8)
            if not _detect_orange(arr):
                result.append(0)   # empty slot
            else:
                result.append(_identify_symbol(cell, known_b64s))

        return result

    except Exception as exc:
        log.warning("Slot capture failed: %s", exc)
        return [0, 0, 0, 0]


def capture_guess_history(x: int, y: int, w: int, h: int) -> List[dict]:
    """Scan the guess history area and return detected rows.

    Each entry:
        {
            "filled": bool,
            "symbol_b64s": [str, str, str, str],   # thumbnails (empty if not filled)
            "feedback": (exact, misplaced) | None,  # None if OCR unavailable/failed
        }

    Rows are ordered top-to-bottom, left-to-right (same as game display).
    """
    try:
        window_img = _grab_window(x, y, w, h)
        hist_img = _sub_crop(window_img, w, h, SC_GUESS_HISTORY)
        hw, hh = hist_img.size

        panel_w = hw // SC_GUESS_COLS
        panel_h = hh // SC_GUESS_ROWS
        results: List[dict] = []

        for row in range(SC_GUESS_ROWS):
            for col in range(SC_GUESS_COLS):
                panel = hist_img.crop((
                    col * panel_w, row * panel_h,
                    (col + 1) * panel_w, (row + 1) * panel_h,
                ))
                pw, ph = panel.size

                import numpy as np
                arr = np.array(panel.convert("RGB"), dtype=np.uint8)
                filled = _detect_orange(arr)

                symbol_b64s: List[str] = []
                feedback: Optional[Tuple[int, int]] = None

                if filled:
                    # Left portion: 4 symbol thumbnails
                    sym_w = int(pw * SC_FEEDBACK_X_FRAC)
                    sym_area = panel.crop((0, 0, sym_w, ph))
                    sym_cell_w = sym_w // 4
                    for s in range(4):
                        cell = sym_area.crop((s * sym_cell_w, 0, (s + 1) * sym_cell_w, ph))
                        symbol_b64s.append(_img_to_b64(cell))

                    # Right portion: feedback text
                    fb_area = panel.crop((sym_w, 0, pw, ph))
                    feedback = _ocr_feedback(fb_area)

                results.append({
                    "filled": filled,
                    "symbol_b64s": symbol_b64s,
                    "feedback": list(feedback) if feedback else None,
                })

        return results

    except Exception as exc:
        log.warning("Guess history capture failed: %s", exc)
        return []
