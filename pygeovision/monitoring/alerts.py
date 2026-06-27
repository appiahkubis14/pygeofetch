"""Alert management for drift detection and performance monitoring."""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


class AlertManager:
    """Manage and route drift + performance alerts."""

    def __init__(self, channels: Optional[List[str]] = None) -> None:
        self.channels = channels or ["log"]
        self._handlers: Dict[str, Callable] = {"log": self._log_alert}
        self._history: List[Dict] = []

    def register_handler(self, name: str, fn: Callable) -> None:
        self._handlers[name] = fn

    def trigger(self, severity: str, message: str, data: Optional[Dict] = None) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "message": message,
            "data": data or {},
        }
        self._history.append(entry)
        for channel in self.channels:
            handler = self._handlers.get(channel)
            if handler:
                handler(entry)

    def _log_alert(self, entry: Dict) -> None:
        level = {"info": logger.info, "warning": logger.warning, "critical": logger.critical}
        fn = level.get(entry["severity"], logger.warning)
        fn("[Alert] %s: %s", entry["severity"].upper(), entry["message"])

    def check_drift_report(self, report: Dict) -> None:
        if report.get("drift_detected"):
            for alert in report.get("performance_drift", {}).get("alerts", []):
                self.trigger("warning", f"Performance drift: {alert}", report)
            dd = report.get("data_drift", {})
            if dd.get("drift_level") == "major":
                self.trigger("critical", "Major data distribution shift detected!", dd)
            elif dd.get("drift_level") == "minor":
                self.trigger("warning", "Minor data drift detected.", dd)

    @property
    def history(self) -> List[Dict]:
        return self._history
