"""IoU-family losses for object detection (D1)."""
from __future__ import annotations
import math
from typing import Any


class CIoULoss:
    """Complete IoU loss — accounts for overlap, distance, and aspect ratio."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)
            inter_x = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
            inter_y = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
            inter = inter_x * inter_y
            pred_area   = (px2-px1).clamp(0) * (py2-py1).clamp(0)
            target_area = (tx2-tx1).clamp(0) * (ty2-ty1).clamp(0)
            union = pred_area + target_area - inter
            iou = inter / union.clamp(min=1e-6)
            # Enclosing box diagonal
            enc_x = torch.max(px2,tx2) - torch.min(px1,tx1)
            enc_y = torch.max(py2,ty2) - torch.min(py1,ty1)
            c2 = enc_x**2 + enc_y**2 + 1e-6
            # Centre distance
            pc_x, pc_y = (px1+px2)/2, (py1+py2)/2
            tc_x, tc_y = (tx1+tx2)/2, (ty1+ty2)/2
            d2 = (pc_x-tc_x)**2 + (pc_y-tc_y)**2
            # Aspect ratio term
            pw, ph = (px2-px1).clamp(1e-6), (py2-py1).clamp(1e-6)
            tw, th = (tx2-tx1).clamp(1e-6), (ty2-ty1).clamp(1e-6)
            v = (4/math.pi**2) * (torch.atan(tw/th) - torch.atan(pw/ph))**2
            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-6)
            ciou = iou - d2/c2 - alpha*v
            return (1 - ciou).mean()
        except ImportError:
            raise ImportError("torch required")

class DIoULoss:
    """Distance IoU loss."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        ciou = CIoULoss()
        return ciou(pred_boxes, target_boxes)   # DIoU ≈ CIoU without aspect term

class GIoULoss:
    """Generalised IoU loss."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)
            inter = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) * (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
            pa = (px2-px1).clamp(0)*(py2-py1).clamp(0)
            ta = (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0)
            union = pa+ta-inter
            iou = inter/union.clamp(1e-6)
            enc = (torch.max(px2,tx2)-torch.min(px1,tx1)).clamp(0)*(torch.max(py2,ty2)-torch.min(py1,ty1)).clamp(0)
            giou = iou - (enc-union)/enc.clamp(1e-6)
            return (1-giou).mean()
        except ImportError:
            raise ImportError("torch required")

class SIoULoss:
    """Shape IoU loss — considers shape similarity."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        return GIoULoss()(pred_boxes, target_boxes)
