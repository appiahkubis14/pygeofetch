"""
pygeovision.preprocess
======================

Complete satellite imagery preprocessing with 100+ operations.
Every method validates its output before returning.

Quick-start::

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    # Stack → clip → SCL-mask → normalise in one call
    result = client.preprocess.pipeline(
        input_path  = "./downloads/S2C_20240628/",
        output_path = "ready.tif",
        stack_bands = ["B02","B03","B04","B08","B11","B12"],
        bbox        = (-74.1, 40.6, -73.7, 40.9),
        scl_path    = "./downloads/S2C_20240628/SCL.tif",
        normalise   = "scale_factor",
    )

    # Or use individual steps
    pre = client.preprocess
    pre.stack_from_dir("./scene/", ["B02","B03","B04","B08"], "stack.tif")
    pre.clip_to_bbox("stack.tif", (-74.1,40.6,-73.7,40.9))
    pre.normalise("stack_clipped.tif", method="scale_factor")
"""

from pygeovision.preprocess.core import Preprocessor

__all__ = ["Preprocessor"]
