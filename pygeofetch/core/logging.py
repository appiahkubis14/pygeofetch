"""
PyGeoFetch logging & display layer.

Usage::

    from pygeofetch.core.logging import configure_logging, get_logger
    from pygeofetch.core.logging import print_search_results, DownloadProgress

    configure_logging(level="INFO")
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import re
import sys
import time
import threading
from typing import Optional, List, Any

# ── ANSI codes ────────────────────────────────────────────────────────────────

_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "magenta": "\033[35m",
    "blue":    "\033[34m",
    "white":   "\033[97m",
    "grey":    "\033[90m",
    "bg_blue": "\033[44m",
    "bg_green":"\033[42m",
}

_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    """Apply an ANSI code only when writing to a real terminal."""
    if not _TTY:
        return text
    return f"{_C.get(code, '')}{text}{_C['reset']}"

def _cb(text: str) -> str:
    return _c("bold", text)

def _cg(text: str) -> str:
    return _c("green", text)

def _cy(text: str) -> str:
    return _c("yellow", text)

def _cd(text: str) -> str:
    return _c("dim", text)

def _cr(text: str) -> str:
    return _c("red", text)

# ── Credential redaction ──────────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset({
    "password", "passwd", "secret", "api_key", "apikey", "token",
    "access_key", "secret_key", "client_secret", "credentials",
    "authorization", "auth",
})

def _redact(msg: str) -> str:
    for key in _SENSITIVE_KEYS:
        msg = re.sub(
            r"(?i)(" + re.escape(key) + r"[\s]*[=:][\s]*)[^\s,}\]\n\"']+",
            r"\1***REDACTED***",
            msg,
        )
    return msg


# ── Log formatter ─────────────────────────────────────────────────────────────

class _PGFFormatter(logging.Formatter):
    _LEVEL_STYLE = {
        "DEBUG":    ("dim",     "DEBG"),
        "INFO":     ("cyan",    "INFO"),
        "WARNING":  ("yellow",  "WARN"),
        "ERROR":    ("red",     "ERRR"),
        "CRITICAL": ("magenta", "CRIT"),
    }

    def __init__(self, use_colour: bool = True, show_module: bool = True):
        super().__init__()
        self._use_colour  = use_colour and _TTY
        self._show_module = show_module

    def format(self, record: logging.LogRecord) -> str:
        ts    = self.formatTime(record, datefmt="%H:%M:%S")
        style, badge = self._LEVEL_STYLE.get(record.levelname, ("", record.levelname[:4]))
        if self._use_colour:
            badge = _c(style, badge)
            ts    = _cd(ts)

        module = record.name.split(".")[-1] if self._show_module else ""
        mod_str = _cd(f"[{module:>12}]") if (module and self._use_colour) else \
                  (f"[{module:>12}]" if module else "")

        msg = _redact(record.getMessage())
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return f"{ts} {badge} {mod_str} {msg}"


# ── configure_logging / get_logger ───────────────────────────────────────────

def configure_logging(
    level:       str  = "INFO",
    log_file:    Optional[str] = None,
    use_colour:  bool = True,
    show_module: bool = True,
) -> None:
    """Configure PyGeoFetch logging. Call once at startup."""
    root = logging.getLogger("pygeofetch")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_PGFFormatter(use_colour=use_colour, show_module=show_module))
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    for lib in ("urllib3", "requests", "httpx", "botocore", "azure",
                "pystac", "planetary_computer"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("pygeofetch"):
        name = f"pygeofetch.{name}"
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_search_header(
    providers: List[str],
    bbox: Any,
    start_date: Any,
    end_date: Any,
    cloud_max: float,
    product: str = "any",
) -> None:
    """Print a clean search header box."""
    w = 72
    bbox_str = (f"[{bbox.min_lon:.3f}, {bbox.min_lat:.3f}, "
                f"{bbox.max_lon:.3f}, {bbox.max_lat:.3f}]") if bbox else "—"
    prov_str = ", ".join(providers)

    print()
    print(_cb(_cg("┌" + "─" * (w - 2) + "┐")))
    print(_cb(_cg("│")) + _cb(f"  🛰  PYGEOFETCH SEARCH".center(w - 2)) + _cb(_cg("│")))
    print(_cb(_cg("├" + "─" * (w - 2) + "┤")))

    def row(label: str, value: str) -> None:
        val_w = w - 18
        print(_cb(_cg("│")) + f"  {_cb(label):<22} {value[:val_w]:<{val_w}}" + _cg(_cb("│")))

    row("Providers",  prov_str)
    row("BBox",       bbox_str)
    row("Date range", f"{start_date}  →  {end_date}")
    row("Cloud max",  f"{cloud_max}%")
    row("Product",    product)
    print(_cb(_cg("└" + "─" * (w - 2) + "┘")))
    print()


def print_provider_progress(provider: str, status: str, count: int = 0,
                             duration: float = 0.0, error: str = "") -> None:
    """Print a single provider search result line."""
    if error:
        icon = _cr("✗")
        line = f"  {icon}  {provider:<28}  {_cr(error[:45])}"
    elif count == 0:
        icon = _cy("○")
        line = f"  {icon}  {provider:<28}  {_cy('no results')}"
    else:
        icon = _cg("✓")
        dur  = _cd(f"{duration:.1f}s")
        cnt  = _cb(_cg(f"{count:>4} scenes"))
        line = f"  {icon}  {provider:<28}  {cnt}   {dur}"
    print(line)


def _pad(text: str, width: int, align: str = "<") -> str:
    """
    Pad *text* to *width* using visible character count (strips ANSI codes first).
    This ensures columns align correctly even when text contains colour escapes.
    """
    ansi_re = re.compile(r"\033\[[\d;]*m")
    visible = len(ansi_re.sub("", text))
    padding = max(0, width - visible)
    if align == "<":
        return text + " " * padding
    if align == ">":
        return " " * padding + text
    # centre
    left  = padding // 2
    right = padding - left
    return " " * left + text + " " * right


def print_search_results(results: List[Any], elapsed: float = 0.0) -> None:
    """Print a perfectly aligned table of search results."""
    if not results:
        print(f"\n  {_cy('No scenes found.')}  Try a wider date range or higher cloud cover.\n")
        return

    n = len(results)

    # ── column definitions: (header, data_width) ─────────────────────────────
    # data_width = max visible chars for that column's data
    COLS = [
        ("SCENE ID",      42),
        ("DATE",          10),
        ("SATELLITE",     14),
        ("CLOUD",          6),
        ("PRODUCT",        7),
        ("POLARISATION",  11),
        ("PASS",          11),
        ("ORBIT",          5),
        ("SCORE",          5),
        ("PROVIDER",      22),
    ]

    # ── total table width ─────────────────────────────────────────────────────
    # 2 leading spaces per col separator + 2 prefix for │
    inner = sum(w for _, w in COLS) + len(COLS) * 2
    bar   = "─" * inner

    # ── header ───────────────────────────────────────────────────────────────
    print()
    print(_cb(_cg(f"  ┌{bar}┐")))
    hdr_cells = "".join(
        "  " + _pad(_cb(h), w)
        for h, w in COLS
    )
    print(_cb(_cg("  │")) + hdr_cells + _cb(_cg("│")))
    print(_cb(_cg(f"  ├{bar}┤")))

    # ── rows ─────────────────────────────────────────────────────────────────
    for r in results[:30]:
        scene_id = (r.id or "—")[:COLS[0][1]]
        date     = (str(r.datetime)[:10] if r.datetime else "—")
        sat      = (r.satellite or "—")[:COLS[2][1]]
        ptype    = (r.product_type or "—")[:COLS[4][1]]
        pol      = (r.polarisation or "—")[:COLS[5][1]]
        pdir     = (r.pass_direction or "—")[:COLS[6][1]]
        orbit    = str(r.relative_orbit or "—")[:COLS[7][1]]
        score    = f"{r.score:.2f}" if r.score else "—"
        provider = (r.provider or "—")[:COLS[9][1]]

        # Cloud cover with colour
        if r.cloud_cover is not None:
            raw = f"{r.cloud_cover:.0f}%"
            if r.cloud_cover <= 10:
                cloud = _cg(raw)
            elif r.cloud_cover <= 30:
                cloud = _cy(raw)
            else:
                cloud = _cr(raw)
        else:
            cloud = "—"

        # Build each cell using _pad (ANSI-aware)
        cells = [
            _pad(_cd(scene_id), COLS[0][1]),
            _pad(date,          COLS[1][1]),
            _pad(_cb(sat),      COLS[2][1]),
            _pad(cloud,         COLS[3][1]),
            _pad(ptype,         COLS[4][1]),
            _pad(pol,           COLS[5][1]),
            _pad(pdir,          COLS[6][1]),
            _pad(orbit,         COLS[7][1]),
            _pad(score,         COLS[8][1]),
            _pad(_cd(provider), COLS[9][1]),
        ]
        row = "".join("  " + c for c in cells)
        print(_cb(_cg("  │")) + row + _cb(_cg("│")))

    # ── footer ────────────────────────────────────────────────────────────────
    overflow    = f"  (+{n - 30} more)" if n > 30 else ""
    elapsed_str = f"  ·  {elapsed:.1f}s" if elapsed else ""
    print(_cb(_cg(f"  ├{bar}┤")))
    summary = (f"  {_cb(_cg(str(n)))} scene{'s' if n != 1 else ''} found"
               f"{elapsed_str}{overflow}")
    print(_cb(_cg("  │")) + summary)
    print(_cb(_cg(f"  └{bar}┘")))
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class DownloadProgress:
    """
    Live download progress display.

    Usage::

        with DownloadProgress(total=5) as dp:
            for item in items:
                with dp.item(item.id, expected_bytes=2_000_000_000):
                    # do download …
                    dp.update(bytes_done=500_000_000, speed_bps=18_000_000)
                dp.complete_item(success=True, bytes_total=2_000_000_000, duration=17.8)
    """

    def __init__(self, total: int, destination: str = "") -> None:
        self.total       = total
        self.destination = destination
        self._completed  = 0
        self._failed     = 0
        self._start      = time.time()
        self._lock       = threading.Lock()
        self._spin_i     = 0
        self._current_id = ""
        self._bytes_done = 0
        self._bytes_total= 0
        self._speed      = 0.0
        self._lines      = 0     # lines printed so far in current block

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self._print_header()
        return self

    def __exit__(self, *_):
        self._print_footer()

    # ── public API ───────────────────────────────────────────────────────────

    def start_item(self, scene_id: str, expected_bytes: int = 0) -> None:
        with self._lock:
            self._current_id  = scene_id
            self._bytes_done  = 0
            self._bytes_total = expected_bytes
            self._speed       = 0.0
            self._spin_i      = 0
            self._render()

    def update(self, bytes_done: int, speed_bps: float = 0.0) -> None:
        with self._lock:
            self._bytes_done = bytes_done
            self._speed      = speed_bps
            self._spin_i     = (self._spin_i + 1) % len(_SPINNER)
            self._render()

    def complete_item(self, success: bool, bytes_total: int = 0,
                      duration: float = 0.0) -> None:
        with self._lock:
            if success:
                self._completed += 1
            else:
                self._failed += 1
            self._print_item_done(success, bytes_total, duration)
            self._current_id = ""

    # ── internal rendering ───────────────────────────────────────────────────

    def _fmt_size(self, b: int) -> str:
        if b <= 0:       return "—"
        if b < 1024:     return f"{b} B"
        if b < 1 << 20:  return f"{b/1024:.0f} KB"
        if b < 1 << 30:  return f"{b/1048576:.1f} MB"
        return               f"{b/1073741824:.2f} GB"

    def _fmt_speed(self, bps: float) -> str:
        if bps <= 0:         return ""
        if bps < 1 << 20:    return f"{bps/1024:.0f} KB/s"
        return                   f"{bps/1048576:.1f} MB/s"

    def _bar(self, filled_pct: float, width: int = 28) -> str:
        done  = int(width * max(0.0, min(1.0, filled_pct)))
        empty = width - done
        return _cg("█" * done) + _cd("░" * empty)

    def _print_header(self) -> None:
        dest = f"  → {self.destination}" if self.destination else ""
        print()
        print(_cb(_cg(f"  ┌{'─'*70}┐")))
        print(_cb(_cg("  │")) + _cb(f"  ⬇  DOWNLOADING  {self.total} scene{'s' if self.total != 1 else ''}{dest}".ljust(70)) + _cb(_cg("│")))
        print(_cb(_cg(f"  └{'─'*70}┘")))
        print()

    def _render(self) -> None:
        """Overwrite the current line with live progress."""
        if not _TTY:
            return

        spin  = _cg(_SPINNER[self._spin_i])
        idx   = self._completed + self._failed + 1
        id_   = (self._current_id or "")[:42]

        pct   = (self._bytes_done / self._bytes_total) if self._bytes_total > 0 else 0.0
        bar   = self._bar(pct)
        done_ = self._fmt_size(self._bytes_done)
        tot_  = self._fmt_size(self._bytes_total)
        spd_  = self._fmt_speed(self._speed)

        size_str = f"{done_} / {tot_}" if self._bytes_total > 0 else done_
        spd_str  = f"  {_cy(spd_)}" if spd_ else ""

        line = (
            f"  {spin}  [{_cd(f'{idx}/{self.total}')}]  "
            f"{_cb(id_):<52}  "
            f"[{bar}]  "
            f"{_cd(size_str):<20}"
            f"{spd_str}"
        )
        sys.stdout.write(f"\r{line}  ")
        sys.stdout.flush()

    def _print_item_done(self, success: bool, bytes_total: int, duration: float) -> None:
        """Print a completed item line (replaces the spinner line)."""
        # Clear spinner line
        if _TTY:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

        idx      = self._completed + self._failed
        id_      = (self._current_id or "")[:42]
        size_str = self._fmt_size(bytes_total)
        dur_str  = f"{duration:.1f}s"

        if success:
            icon = _cg("✓")
            line = (
                f"  {icon}  [{_cd(f'{idx}/{self.total}')}]  "
                f"{_cb(id_):<52}  "
                f"{_cg(size_str):<12}  "
                f"{_cd(dur_str)}"
            )
        else:
            icon = _cr("✗")
            line = (
                f"  {icon}  [{_cd(f'{idx}/{self.total}')}]  "
                f"{_cr(id_):<52}  "
                f"{_cr('FAILED')}"
            )
        print(line)

    def _print_footer(self) -> None:
        elapsed = time.time() - self._start
        total_done = self._completed + self._failed
        print()
        if self._failed == 0:
            status = _cg(_cb(f"✓  All {self._completed} scenes downloaded"))
        else:
            status = (_cb(_cg(f"✓ {self._completed} succeeded")) +
                      "  " + _cb(_cr(f"✗ {self._failed} failed")))

        print(f"  {status}   {_cd(f'total time: {elapsed:.1f}s')}")
        print()


# ── Backward compat ───────────────────────────────────────────────────────────

def _render_progress_bar(completed, total, filename, bytes_done,
                         bytes_total, speed_bps, bar_width=20) -> str:
    """Legacy helper kept for backward compatibility."""
    filled   = int(bar_width * min(completed, total) / max(total, 1))
    bar      = "█" * filled + "░" * (bar_width - filled)
    done_gb  = bytes_done  / 1e9
    tot_gb   = bytes_total / 1e9 if bytes_total else 0.0
    spd_mbs  = speed_bps   / 1e6
    name     = filename[:40] + "…" if len(filename) > 40 else filename
    size_str = (f"{done_gb:.1f} GB / {tot_gb:.1f} GB"
                if bytes_total > 0 else f"{done_gb:.1f} GB")
    return (f"\r  [{bar}]  {completed}/{total}  "
            f"{name:<42}  {size_str}  {spd_mbs:.1f} MB/s")