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
import threading
import time
from typing import Any

# ── ANSI codes ────────────────────────────────────────────────────────────────

_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "white": "\033[97m",
    "grey": "\033[90m",
    "bg_blue": "\033[44m",
    "bg_green": "\033[42m",
}

# Detect execution environment
_TTY = sys.stdout.isatty()


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return "ZMQInteractiveShell" in type(ip).__name__  # Jupyter kernel
    except ImportError:
        return False


_JUPYTER = _in_jupyter()


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

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "token",
        "access_key",
        "secret_key",
        "client_secret",
        "credentials",
        "authorization",
        "auth",
    }
)


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
        "DEBUG": ("dim", "DEBG"),
        "INFO": ("cyan", "INFO"),
        "WARNING": ("yellow", "WARN"),
        "ERROR": ("red", "ERRR"),
        "CRITICAL": ("magenta", "CRIT"),
    }

    def __init__(self, use_colour: bool = True, show_module: bool = True):
        super().__init__()
        self._use_colour = use_colour and _TTY
        self._show_module = show_module

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        style, badge = self._LEVEL_STYLE.get(
            record.levelname, ("", record.levelname[:4])
        )
        if self._use_colour:
            badge = _c(style, badge)
            ts = _cd(ts)

        module = record.name.split(".")[-1] if self._show_module else ""
        mod_str = (
            _cd(f"[{module:>12}]")
            if (module and self._use_colour)
            else (f"[{module:>12}]" if module else "")
        )

        msg = _redact(record.getMessage())
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return f"{ts} {badge} {mod_str} {msg}"


# ── configure_logging / get_logger ───────────────────────────────────────────


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    use_colour: bool = True,
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
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)

    for lib in (
        "urllib3",
        "requests",
        "httpx",
        "botocore",
        "azure",
        "pystac",
        "planetary_computer",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("pygeofetch"):
        name = f"pygeofetch.{name}"
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════


def print_search_header(
    providers: list[str],
    bbox: Any,
    start_date: Any,
    end_date: Any,
    cloud_max: float,
    product: str = "any",
) -> None:
    """Print a clean search header box."""
    w = 72
    bbox_str = (
        (
            f"[{bbox.min_lon:.3f}, {bbox.min_lat:.3f}, {bbox.max_lon:.3f}, {bbox.max_lat:.3f}]"
        )
        if bbox
        else "—"
    )
    prov_str = ", ".join(providers)

    def row(label: str, value: str) -> None:
        w - 18

    row("Providers", prov_str)
    row("BBox", bbox_str)
    row("Date range", f"{start_date}  →  {end_date}")
    row("Cloud max", f"{cloud_max}%")
    row("Product", product)


def print_provider_progress(
    provider: str, status: str, count: int = 0, duration: float = 0.0, error: str = ""
) -> None:
    """Print a single provider search result line."""
    if error:
        icon = _cr("✗")
        f"  {icon}  {provider:<28}  {_cr(error[:45])}"
    elif count == 0:
        icon = _cy("○")
        f"  {icon}  {provider:<28}  {_cy('no results')}"
    else:
        icon = _cg("✓")
        _cd(f"{duration:.1f}s")
        _cb(_cg(f"{count:>4} scenes"))


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
    left = padding // 2
    right = padding - left
    return " " * left + text + " " * right


def print_search_results(results: list[Any], elapsed: float = 0.0) -> None:
    """Print a perfectly aligned table of search results."""
    if not results:
        return

    n = len(results)

    # ── column definitions: (header, data_width) ─────────────────────────────
    # data_width = max visible chars for that column's data
    COLS = [
        ("SCENE ID", 42),
        ("DATE", 10),
        ("SATELLITE", 14),
        ("CLOUD", 6),
        ("PRODUCT", 7),
        ("POLARISATION", 11),
        ("PASS", 11),
        ("ORBIT", 5),
        ("SCORE", 5),
        ("PROVIDER", 22),
    ]

    # ── total table width ─────────────────────────────────────────────────────
    # 2 leading spaces per col separator + 2 prefix for │
    inner = sum(w for _, w in COLS) + len(COLS) * 2
    "─" * inner

    # ── header ───────────────────────────────────────────────────────────────
    "".join("  " + _pad(_cb(h), w) for h, w in COLS)

    # ── rows ─────────────────────────────────────────────────────────────────
    for r in results[:30]:
        scene_id = (r.id or "—")[: COLS[0][1]]
        date = str(r.datetime)[:10] if r.datetime else "—"
        sat = (r.satellite or "—")[: COLS[2][1]]
        ptype = (r.product_type or "—")[: COLS[4][1]]
        pol = (r.polarisation or "—")[: COLS[5][1]]
        pdir = (r.pass_direction or "—")[: COLS[6][1]]
        orbit = str(r.relative_orbit or "—")[: COLS[7][1]]
        score = f"{r.score:.2f}" if r.score else "—"
        provider = (r.provider or "—")[: COLS[9][1]]

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
            _pad(date, COLS[1][1]),
            _pad(_cb(sat), COLS[2][1]),
            _pad(cloud, COLS[3][1]),
            _pad(ptype, COLS[4][1]),
            _pad(pol, COLS[5][1]),
            _pad(pdir, COLS[6][1]),
            _pad(orbit, COLS[7][1]),
            _pad(score, COLS[8][1]),
            _pad(_cd(provider), COLS[9][1]),
        ]
        "".join("  " + c for c in cells)

    # ── footer ────────────────────────────────────────────────────────────────
    overflow = f"  (+{n - 30} more)" if n > 30 else ""
    elapsed_str = f"  ·  {elapsed:.1f}s" if elapsed else ""
    f"  {_cb(_cg(str(n)))} scene{'s' if n != 1 else ''} found{elapsed_str}{overflow}"


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class DownloadProgress:
    """
    Live download progress — works in both terminal and Jupyter notebook.

    In a terminal: spinner + in-place updating progress bar.
    In Jupyter:    clean printed lines with IPython display updates.
    """

    def __init__(self, total: int, destination: str = "") -> None:
        self.total = total
        self.destination = destination
        self._completed = 0
        self._failed = 0
        self._start = time.time()
        self._lock = threading.Lock()
        self._spin_i = 0
        self._current_id = ""
        self._bytes_done = 0
        self._bytes_total = 0
        self._speed = 0.0
        # Jupyter display handle
        self._jup_handle = None

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self._print_header()
        return self

    def __exit__(self, *_):
        self._print_footer()

    # ── public API ───────────────────────────────────────────────────────────

    def start_item(self, scene_id: str, expected_bytes: int = 0) -> None:
        with self._lock:
            self._current_id = scene_id
            self._bytes_done = 0
            self._bytes_total = expected_bytes
            self._speed = 0.0
            self._spin_i = 0
            self._render()

    def update(self, bytes_done: int, speed_bps: float = 0.0) -> None:
        with self._lock:
            self._bytes_done = bytes_done
            self._speed = speed_bps
            self._spin_i = (self._spin_i + 1) % len(_SPINNER)
            self._render()

    def complete_item(
        self, success: bool, bytes_total: int = 0, duration: float = 0.0
    ) -> None:
        with self._lock:
            if success:
                self._completed += 1
            else:
                self._failed += 1
            self._print_item_done(success, bytes_total, duration)
            self._current_id = ""

    # ── formatting helpers ────────────────────────────────────────────────────

    def _fmt_size(self, b: int) -> str:
        if b <= 0:
            return "—"
        if b < 1 << 20:
            return f"{b / 1024:.0f} KB"
        if b < 1 << 30:
            return f"{b / 1048576:.1f} MB"
        return f"{b / 1073741824:.2f} GB"

    def _fmt_speed(self, bps: float) -> str:
        if bps <= 0:
            return ""
        if bps < 1 << 20:
            return f"{bps / 1024:.0f} KB/s"
        return f"{bps / 1048576:.1f} MB/s"

    def _bar_ansi(self, pct: float, width: int = 28) -> str:
        done = int(width * max(0.0, min(1.0, pct)))
        return _cg("█" * done) + _cd("░" * (width - done))

    def _bar_plain(self, pct: float, width: int = 28) -> str:
        done = int(width * max(0.0, min(1.0, pct)))
        return "█" * done + "░" * (width - done)

    # ── Jupyter rendering ────────────────────────────────────────────────────

    def _jup_html_item(
        self,
        scene_id: str,
        success: bool | None,
        bytes_total: int,
        duration: float,
        bytes_done: int = 0,
        speed: float = 0,
    ) -> str:
        """Build an HTML row for a single download item."""
        sid = (scene_id or "")[:50]
        size = self._fmt_size(bytes_total or bytes_done)
        dur = f"{duration:.1f}s" if duration else ""
        spd = self._fmt_speed(speed)

        if success is None:
            # In-progress
            pct = (bytes_done / bytes_total * 100) if bytes_total > 0 else 0
            pct_str = f"{pct:.0f}%" if bytes_total > 0 else "…"
            bar_fill = f"{pct:.1f}%" if bytes_total > 0 else "0%"
            spd_str = f"<span style='color:#888'>{spd}</span>" if spd else ""
            size_str = (
                (
                    f"<span style='color:#888'>{self._fmt_size(bytes_done)} / {size}</span>"
                )
                if bytes_total
                else ""
            )
            return (
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:4px 0;font-family:monospace;font-size:13px'>"
                f"<span style='color:#f0c040'>⟳</span>"
                f"<span style='min-width:360px;color:#ddd'>{sid}</span>"
                f"<div style='flex:1;background:#333;border-radius:4px;"
                f"height:8px;min-width:140px'>"
                f"<div style='background:#4caf50;width:{bar_fill};"
                f"height:8px;border-radius:4px;transition:width 0.3s'></div>"
                f"</div>"
                f"<span style='min-width:60px;color:#aaa'>{pct_str}</span>"
                f"{size_str} {spd_str}"
                f"</div>"
            )
        if success:
            return (
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:4px 0;font-family:monospace;font-size:13px'>"
                f"<span style='color:#4caf50'>✓</span>"
                f"<span style='min-width:360px;color:#eee'>{sid}</span>"
                f"<span style='color:#4caf50;min-width:80px'>{size}</span>"
                f"<span style='color:#888'>{dur}</span>"
                f"</div>"
            )
        return (
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"padding:4px 0;font-family:monospace;font-size:13px'>"
            f"<span style='color:#f44336'>✗</span>"
            f"<span style='min-width:360px;color:#f44336'>{sid}</span>"
            f"<span style='color:#f44336'>FAILED</span>"
            f"</div>"
        )

    def _jup_render_full(self) -> None:
        """Re-render the complete Jupyter progress widget."""
        if not _JUPYTER:
            return
        try:
            from IPython.display import HTML, display
        except ImportError:
            return

        done = self._completed + self._failed
        total = self.total
        pct = done / total * 100 if total else 0
        overall_bar = f"{pct:.0f}%"

        rows = "".join(self._jup_rows)
        # Add in-progress row for current item
        if self._current_id:
            rows += self._jup_html_item(
                self._current_id,
                None,
                self._bytes_total,
                0,
                self._bytes_done,
                self._speed,
            )

        elapsed = time.time() - self._start
        html = (
            f"<div style='background:#1e1e2e;border-radius:8px;padding:16px 20px;"
            f"font-family:monospace;margin:8px 0'>"
            # Header
            f"<div style='color:#7c7cff;font-weight:bold;font-size:14px;"
            f"margin-bottom:12px'>⬇ DOWNLOADING  {total} scene{'s' if total != 1 else ''}"
            f"{'  →  ' + self.destination if self.destination else ''}</div>"
            # Overall progress bar
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>"
            f"<span style='color:#888;font-size:12px'>{done}/{total}</span>"
            f"<div style='flex:1;background:#333;border-radius:4px;height:10px'>"
            f"<div style='background:linear-gradient(90deg,#4caf50,#81c784);"
            f"width:{pct:.1f}%;height:10px;border-radius:4px;"
            f"transition:width 0.3s'></div></div>"
            f"<span style='color:#888;font-size:12px'>{overall_bar}</span>"
            f"<span style='color:#555;font-size:12px'>{elapsed:.0f}s</span>"
            f"</div>"
            # Item rows
            f"<div style='border-top:1px solid #333;padding-top:10px'>{rows}</div>"
            f"</div>"
        )

        if self._jup_handle is None:
            self._jup_handle = display(HTML(html), display_id=True)
        else:
            self._jup_handle.update(HTML(html))

    # ── header / footer ──────────────────────────────────────────────────────

    def _print_header(self) -> None:
        self._jup_rows: list[str] = []

        if _JUPYTER:
            self._jup_render_full()
            return

    def _render(self) -> None:
        """Live in-progress render — terminal spinner or Jupyter bar."""
        if _JUPYTER:
            self._jup_render_full()
            return
        if not _TTY:
            return

        spin = _cg(_SPINNER[self._spin_i])
        idx = self._completed + self._failed + 1
        id_ = (self._current_id or "")[:42]
        pct = (self._bytes_done / self._bytes_total) if self._bytes_total > 0 else 0.0
        bar = self._bar_ansi(pct)
        done_ = self._fmt_size(self._bytes_done)
        tot_ = self._fmt_size(self._bytes_total)
        spd_ = self._fmt_speed(self._speed)
        size_str = f"{done_} / {tot_}" if self._bytes_total > 0 else done_
        spd_str = f"  {_cy(spd_)}" if spd_ else ""

        line = (
            f"  {spin}  [{_cd(f'{idx}/{self.total}')}]  "
            f"{_cb(id_):<52}  [{bar}]  "
            f"{_cd(size_str):<20}{spd_str}"
        )
        sys.stdout.write("\r" + line + "  ")
        sys.stdout.flush()

    def _print_item_done(
        self, success: bool, bytes_total: int, duration: float
    ) -> None:
        """Finalise a completed item."""
        idx = self._completed + self._failed
        id_ = (self._current_id or "")[:42]
        size_str = self._fmt_size(bytes_total)
        dur_str = f"{duration:.1f}s"

        if _JUPYTER:
            row = self._jup_html_item(self._current_id, success, bytes_total, duration)
            self._jup_rows.append(row)
            self._jup_render_full()
            return

        # Terminal
        if _TTY:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

        if success:
            icon = _cg("✓")
            (
                f"  {icon}  [{_cd(f'{idx}/{self.total}')}]  "
                f"{_cb(id_):<52}  {_cg(size_str):<12}  {_cd(dur_str)}"
            )
        else:
            icon = _cr("✗")
            f"  {icon}  [{_cd(f'{idx}/{self.total}')}]  {_cr(id_):<52}  {_cr('FAILED')}"

    def _print_footer(self) -> None:
        time.time() - self._start

        if _JUPYTER:
            self._jup_render_full()
            return

        if self._failed == 0:
            _cg(_cb(f"✓  All {self._completed} scenes downloaded"))
        else:
            (
                _cb(_cg(f"✓ {self._completed} succeeded"))
                + "  "
                + _cb(_cr(f"✗ {self._failed} failed"))
            )


# ── Backward compat ───────────────────────────────────────────────────────────


def _render_progress_bar(
    completed, total, filename, bytes_done, bytes_total, speed_bps, bar_width=20
) -> str:
    """Legacy helper kept for backward compatibility."""
    filled = int(bar_width * min(completed, total) / max(total, 1))
    bar = "█" * filled + "░" * (bar_width - filled)
    done_gb = bytes_done / 1e9
    tot_gb = bytes_total / 1e9 if bytes_total else 0.0
    spd_mbs = speed_bps / 1e6
    name = filename[:40] + "…" if len(filename) > 40 else filename
    size_str = (
        f"{done_gb:.1f} GB / {tot_gb:.1f} GB"
        if bytes_total > 0
        else f"{done_gb:.1f} GB"
    )
    return (
        f"\r  [{bar}]  {completed}/{total}  {name:<42}  {size_str}  {spd_mbs:.1f} MB/s"
    )
