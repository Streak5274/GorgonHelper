"""Safecracking puzzle solver for Project Gorgon.

Mastermind-variant: 4 slots, up to 12 symbols, repetition allowed.
Feedback: (exact_matches, misplaced_matches)

Symbols are referred to by 1-based index (1..N) matching the display order.
"""
import base64
import io
import logging
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

# Orange rune pixel detection
_ORANGE_R_MIN, _ORANGE_G_MAX, _ORANGE_B_MAX = 150, 120, 80
_ORANGE_THRESHOLD = 30

# MSE threshold for symbol identification
_MSE_THRESHOLD = 2000.0

# Normalised size for digit templates
_DIGIT_SIZE = 24

# Persistence path for learned digit templates
_TEMPLATES_PATH = Path(__file__).parent / "data" / "sc_digit_templates.npz"

# In-memory digit templates: digit_value (0-4) → averaged float32 array (_DIGIT_SIZE×_DIGIT_SIZE)
_digit_templates: Dict[int, np.ndarray] = {}


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
    """Constraint-based solver tracking remaining consistent candidates."""

    SLOTS = 4

    def __init__(self, symbols: List[int]) -> None:
        self.symbols: List[int] = list(symbols)
        self._all: List[Tuple[int, ...]] = list(product(self.symbols, repeat=self.SLOTS))
        self._candidates: List[Tuple[int, ...]] = list(self._all)
        self._history: List[Tuple[List[int], int, int]] = []

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

    def suggest(self) -> Optional[List[int]]:
        if self.impossible:
            return None
        if self.solved:
            return list(self._candidates[0])
        aabb = self._next_aabb()
        if aabb is not None:
            return list(aabb)
        return list(self._candidates[0])

    def record(self, guess: List[int], exact: int, misplaced: int) -> None:
        t = tuple(guess)
        self._history.append((guess, exact, misplaced))
        self._candidates = [c for c in self._candidates
                            if score_guess(t, c) == (exact, misplaced)]

    def undo(self) -> bool:
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

    def _next_aabb(self) -> Optional[Tuple[int, ...]]:
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
                return None
        if total_hits >= 4 or pairs_tested >= max_pairs:
            return None
        a = self.symbols[pairs_tested * 2]
        b = self.symbols[pairs_tested * 2 + 1]
        return (a, a, b, b)

    @staticmethod
    def _is_aabb(g: Tuple[int, ...]) -> bool:
        return len(g) == 4 and g[0] == g[1] and g[2] == g[3] and g[0] != g[2]


# ---------------------------------------------------------------------------
# Digit template learning & matching (replaces pytesseract)
# ---------------------------------------------------------------------------

def _feedback_img_to_binary(img) -> Optional[np.ndarray]:
    """Convert a feedback PIL image to an inverted binary array (text = white)."""
    try:
        import cv2  # type: ignore
        gray = np.array(img.convert("L"), dtype=np.uint8)
        # Game UI uses dark text on light gray — invert so text is white
        if float(gray.mean()) > 100:
            gray = 255 - gray
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        return binary
    except Exception as exc:
        log.debug("Binary conversion failed: %s", exc)
        return None


def _extract_digit_images(feedback_img) -> List[np.ndarray]:
    """Extract up to 2 digit images (float32, _DIGIT_SIZE×_DIGIT_SIZE) from a feedback crop.

    Uses OpenCV connected components to find the two largest blobs (the digits),
    ignoring the comma/space separator.
    """
    try:
        import cv2  # type: ignore
        binary = _feedback_img_to_binary(feedback_img)
        if binary is None:
            return []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return []

        # Collect non-background components
        components = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x    = int(stats[i, cv2.CC_STAT_LEFT])
            y    = int(stats[i, cv2.CC_STAT_TOP])
            w    = int(stats[i, cv2.CC_STAT_WIDTH])
            h    = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < 4:   # skip single-pixel noise
                continue
            components.append((area, x, y, w, h))

        if not components:
            return []

        # Take the 2 largest blobs (digits), sort left→right
        components.sort(key=lambda c: -c[0])
        digit_comps = sorted(components[:2], key=lambda c: c[1])

        digit_imgs: List[np.ndarray] = []
        for area, x, y, w, h in digit_comps:
            pad = 2
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)
            crop = binary[y1:y2, x1:x2]
            resized = cv2.resize(crop, (_DIGIT_SIZE, _DIGIT_SIZE),
                                 interpolation=cv2.INTER_AREA).astype(np.float32)
            digit_imgs.append(resized)

        return digit_imgs

    except Exception as exc:
        log.debug("Digit extraction failed: %s", exc)
        return []


def calibrate_digit(digit_arr: np.ndarray, digit_value: int) -> None:
    """Store or refine a template for digit_value using exponential moving average."""
    if digit_value in _digit_templates:
        # Blend: 70% existing, 30% new observation
        _digit_templates[digit_value] = (
            _digit_templates[digit_value] * 0.7 + digit_arr * 0.3
        )
    else:
        _digit_templates[digit_value] = digit_arr.copy()
    log.info("Calibrated digit %d  (known: %s)", digit_value,
             sorted(_digit_templates.keys()))


def _match_digit(digit_arr: np.ndarray) -> Optional[int]:
    """Return the best-matching digit value (0-4) or None if below threshold."""
    best_val: Optional[int] = None
    best_mse = 1500.0   # MSE threshold — tune if needed
    for val, tmpl in _digit_templates.items():
        mse = float(np.mean((digit_arr - tmpl) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_val = val
    return best_val


def match_feedback(feedback_img) -> Optional[Tuple[int, int]]:
    """Try to recognise (exact, misplaced) from a feedback crop using stored templates.

    Returns None if templates are insufficient or recognition fails.
    """
    if len(_digit_templates) < 2:
        return None
    digits = _extract_digit_images(feedback_img)
    if len(digits) < 2:
        return None
    d0 = _match_digit(digits[0])
    d1 = _match_digit(digits[1])
    if d0 is None or d1 is None:
        return None
    return d0, d1


def calibrated_digit_count() -> int:
    return len(_digit_templates)


# ---------------------------------------------------------------------------
# Template persistence
# ---------------------------------------------------------------------------

def save_digit_templates() -> None:
    """Persist learned digit templates to disk."""
    if not _digit_templates:
        return
    try:
        _TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(_TEMPLATES_PATH),
                 **{str(k): v for k, v in _digit_templates.items()})
        log.info("Saved digit templates: %s", sorted(_digit_templates.keys()))
    except Exception as exc:
        log.warning("Could not save digit templates: %s", exc)


def load_digit_templates() -> None:
    """Load previously learned digit templates from disk."""
    if not _TEMPLATES_PATH.exists():
        return
    try:
        data = np.load(str(_TEMPLATES_PATH))
        for k in data.files:
            _digit_templates[int(k)] = data[k]
        log.info("Loaded digit templates: %s", sorted(_digit_templates.keys()))
    except Exception as exc:
        log.warning("Could not load digit templates: %s", exc)


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


def _sub_crop(img, window_w: int, window_h: int,
              fracs: Tuple[float, float, float, float]):
    """Crop a sub-region using fractional coordinates."""
    fl, ft, fw, fh = fracs
    left   = int(fl * window_w)
    top    = int(ft * window_h)
    right  = left + int(fw * window_w)
    bottom = top  + int(fh * window_h)
    return img.crop((left, top, right, bottom))


def _img_to_b64(img, thumb: int = 52) -> str:
    from PIL import Image  # type: ignore
    img = img.resize((thumb, thumb), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _detect_orange(arr: np.ndarray) -> bool:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mask = (r > _ORANGE_R_MIN) & (g < _ORANGE_G_MAX) & (b < _ORANGE_B_MAX)
    return int(mask.sum()) >= _ORANGE_THRESHOLD


def _identify_symbol(cell_img, known_b64s: List[str]) -> int:
    """Return 1-based index of the best-matching known symbol, or 0."""
    from PIL import Image  # type: ignore
    if not known_b64s:
        return 0
    thumb = 52
    cell_arr = np.array(
        cell_img.resize((thumb, thumb), Image.LANCZOS), dtype=np.float32
    )
    best_idx = 0
    best_mse = _MSE_THRESHOLD
    for i, b64 in enumerate(known_b64s):
        try:
            data = base64.b64decode(b64)
            ref = np.array(
                Image.open(io.BytesIO(data)).convert("RGB").resize(
                    (thumb, thumb), Image.LANCZOS
                ),
                dtype=np.float32,
            )
            mse = float(np.mean((cell_arr - ref) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_idx = i + 1
        except Exception:
            continue
    return best_idx


# ---------------------------------------------------------------------------
# Public capture functions
# ---------------------------------------------------------------------------

def capture_symbols(x: int, y: int, w: int, h: int,
                    cols: int = 6, rows: int = 2,
                    thumb: int = 52) -> List[str]:
    """Capture the whole Minotaur Lock window and return 12 symbol thumbnails (base64 PNG)."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        log.error("Pillow not installed")
        return []
    try:
        window_img = _grab_window(x, y, w, h)
        grid_img = _sub_crop(window_img, w, h, SC_SYMBOL_GRID)
        gw, gh = grid_img.size
        cw, ch = gw // cols, gh // rows
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
    """Detect which known symbols are in the 4 solution slots. Returns 4 ints (0 = empty)."""
    try:
        window_img = _grab_window(x, y, w, h)
        slots_img = _sub_crop(window_img, w, h, SC_SOLUTION_SLOTS)
        sw, sh = slots_img.size
        slot_w = sw // 4
        result: List[int] = []
        for i in range(4):
            cell = slots_img.crop((i * slot_w, 0, (i + 1) * slot_w, sh))
            arr = np.array(cell.convert("RGB"), dtype=np.uint8)
            if not _detect_orange(arr):
                result.append(0)
            else:
                result.append(_identify_symbol(cell, known_b64s))
        return result
    except Exception as exc:
        log.warning("Slot capture failed: %s", exc)
        return [0, 0, 0, 0]


def capture_guess_history(x: int, y: int, w: int, h: int) -> List[dict]:
    """Scan the guess history area. Returns one entry per panel (SC_GUESS_ROWS × SC_GUESS_COLS).

    Each entry:
        { "filled": bool, "symbol_b64s": [...], "feedback": [e, m] | None }
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
                arr = np.array(panel.convert("RGB"), dtype=np.uint8)
                filled = _detect_orange(arr)

                symbol_b64s: List[str] = []
                feedback = None

                if filled:
                    sym_w = int(pw * SC_FEEDBACK_X_FRAC)
                    sym_area = panel.crop((0, 0, sym_w, ph))
                    sym_cell_w = sym_w // 4
                    for s in range(4):
                        cell = sym_area.crop(
                            (s * sym_cell_w, 0, (s + 1) * sym_cell_w, ph)
                        )
                        symbol_b64s.append(_img_to_b64(cell))

                    fb_area = panel.crop((sym_w, 0, pw, ph))
                    fb = match_feedback(fb_area)
                    feedback = list(fb) if fb else None

                results.append({
                    "filled": filled,
                    "symbol_b64s": symbol_b64s,
                    "feedback": feedback,
                })

        return results

    except Exception as exc:
        log.warning("Guess history capture failed: %s", exc)
        return []


def capture_and_calibrate(x: int, y: int, w: int, h: int,
                          guess_index: int, exact: int, misplaced: int) -> bool:
    """Capture the feedback region for a specific guess row and learn digit templates.

    guess_index is 0-based (0 = first guess submitted).
    Returns True if templates were successfully extracted and stored.
    """
    try:
        grid_row = guess_index // SC_GUESS_COLS
        grid_col = guess_index % SC_GUESS_COLS

        window_img = _grab_window(x, y, w, h)
        hist_img = _sub_crop(window_img, w, h, SC_GUESS_HISTORY)
        hw, hh = hist_img.size
        panel_w = hw // SC_GUESS_COLS
        panel_h = hh // SC_GUESS_ROWS

        panel = hist_img.crop((
            grid_col * panel_w, grid_row * panel_h,
            (grid_col + 1) * panel_w, (grid_row + 1) * panel_h,
        ))
        pw, ph = panel.size
        fb_x = int(pw * SC_FEEDBACK_X_FRAC)
        fb_area = panel.crop((fb_x, 0, pw, ph))

        digits = _extract_digit_images(fb_area)
        if len(digits) < 2:
            log.debug("calibrate: only found %d digit blobs for guess %d", len(digits), guess_index)
            return False

        calibrate_digit(digits[0], exact)
        calibrate_digit(digits[1], misplaced)
        save_digit_templates()
        return True

    except Exception as exc:
        log.warning("Calibration failed for guess %d: %s", guess_index, exc)
        return False
