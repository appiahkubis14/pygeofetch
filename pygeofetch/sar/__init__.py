"""
PyGeoFetch SAR — optional SAR processing layer.

Lightweight backend (sarxarray): pip install "pygeofetch[sar]"
Heavy backend (SNAP via OST):    pip install "pygeofetch[ost]"

Usage::

    from pygeofetch.sar import SARProcessor

    proc   = SARProcessor(backend="sarxarray")  # lightweight
    result = proc.calibrate("sentinel1.tif", output_type="sigma0")
    result = proc.despeckle("calibrated.tif", filter="lee")
"""

from pygeofetch.sar.processor import SARProcessor

__all__ = ["SARProcessor"]
