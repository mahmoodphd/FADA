"""PIL-based annotation renderer for Gradio.

Draws bounding boxes, segmentation polygons, keypoints, and labels on PIL images.
Returns PIL Images directly (no matplotlib/file I/O).
"""
from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from constants import CLASS_COLORS, _RAW_PALETTE
from output_parser import ParsedOutput


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _get_color(label: str) -> Tuple[int, int, int]:
    """Get RGB color for a class label."""
    hex_color = CLASS_COLORS.get(label)
    if hex_color is None:
        idx = hash(label) % len(_RAW_PALETTE)
        hex_color = _RAW_PALETTE[idx]
    return _hex_to_rgb(hex_color)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Get a font for label text, falling back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    parsed: ParsedOutput,
    line_width: int = 3,
    font_size: int = 14,
    mask_alpha: float = 0.25,
) -> Image.Image:
    """Draw detection boxes, segmentation masks, and keypoints on the image.

    Args:
        image: Input PIL Image (RGB).
        parsed: ParsedOutput with coordinates in pixel space.
        line_width: Border width for bounding boxes.
        font_size: Label text size.
        mask_alpha: Opacity for segmentation polygon fill.

    Returns:
        New PIL Image with annotations drawn.
    """
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw_main = ImageDraw.Draw(result)
    draw_overlay = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    # Draw detections (solid bounding boxes)
    for det in parsed.detections:
        color = _get_color(det.label)
        x1, y1, x2, y2 = det.bbox_xyxy
        draw_main.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        _draw_label(draw_main, det.label, x1, y1, color, font)

    # Draw segmentations (filled polygon + dashed bbox)
    for seg in parsed.segmentations:
        color = _get_color(seg.label)
        x1, y1, x2, y2 = seg.bbox_xyxy

        # Draw filled polygon on overlay
        if seg.mask_polygon and len(seg.mask_polygon) >= 3:
            fill_color = color + (int(255 * mask_alpha),)
            outline_color = color + (255,)
            poly_flat = [coord for pt in seg.mask_polygon for coord in pt]
            draw_overlay.polygon(poly_flat, fill=fill_color, outline=outline_color)

        # Draw bounding box (dashed effect via short segments)
        _draw_dashed_rect(draw_main, x1, y1, x2, y2, color, line_width)
        _draw_label(draw_main, seg.label, x1, y1, color, font)

    # Draw keypoints
    for kp_pred in parsed.keypoints:
        color = _get_color(kp_pred.label)
        x1, y1, x2, y2 = kp_pred.bbox_xyxy
        # Draw bounding box (dotted)
        _draw_dashed_rect(draw_main, x1, y1, x2, y2, color, line_width,
                          dash_len=6, gap_len=4)
        _draw_label(draw_main, kp_pred.label, x1, y1, color, font)

        # Draw individual keypoints
        kp_radius = max(3, min(image.size) // 150)
        for i, (kx, ky, vis) in enumerate(kp_pred.keypoints):
            if vis == 0:
                continue  # skip invisible keypoints
            # Visible keypoints: filled circle
            kp_color = color if vis == 2 else (color[0], color[1], color[2])
            draw_main.ellipse(
                [kx - kp_radius, ky - kp_radius,
                 kx + kp_radius, ky + kp_radius],
                fill=kp_color, outline=(255, 255, 255), width=1)
            # Draw small index number next to keypoint
            idx_text = str(i + 1)
            draw_main.text((kx + kp_radius + 2, ky - kp_radius),
                           idx_text, fill=color, font=font)

        # Connect consecutive visible keypoints with lines
        visible_pts = [(kx, ky) for kx, ky, vis in kp_pred.keypoints if vis > 0]
        if len(visible_pts) >= 2:
            for j in range(len(visible_pts) - 1):
                draw_main.line(
                    [visible_pts[j], visible_pts[j + 1]],
                    fill=color, width=max(1, line_width - 1))

    # Composite overlay onto result
    result = Image.alpha_composite(result, overlay)

    # Draw classification labels at top
    y_offset = 10
    for cls_pred in parsed.classifications:
        color = _get_color(cls_pred.label)
        label_text = f"View: {cls_pred.label}"
        draw_final = ImageDraw.Draw(result)
        bbox = draw_final.textbbox((10, y_offset), label_text, font=font)
        pad = 4
        draw_final.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(50, 50, 50, 200))
        draw_final.text((10, y_offset), label_text, fill=(255, 255, 255, 255), font=font)
        y_offset = bbox[3] + 10

    return result.convert("RGB")


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Draw a label with colored background above a bounding box."""
    bbox = draw.textbbox((x, y - 20), label, font=font)
    pad = 3
    bg_box = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    # Clamp to image bounds
    bg_box[1] = max(0, bg_box[1])
    draw.rectangle(bg_box, fill=color)
    draw.text((x, max(0, y - 20)), label, fill=(255, 255, 255), font=font)


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    width: int,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed rectangle."""
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        dx = ex - sx
        dy = ey - sy
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        step = dash_len + gap_len
        for i in range(0, length, step):
            seg_start = i
            seg_end = min(i + dash_len, length)
            if dx != 0:
                draw.line(
                    [(sx + seg_start * dx // length, sy),
                     (sx + seg_end * dx // length, sy)],
                    fill=color, width=width)
            else:
                draw.line(
                    [(sx, sy + seg_start * dy // length),
                     (sx, sy + seg_end * dy // length)],
                    fill=color, width=width)
