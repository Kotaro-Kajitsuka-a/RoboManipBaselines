import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# =========================================================
# 1x3 ArUco Board PDF Generator
#
# Dictionary:
#   DICT_5X5_250
#
# Marker IDs:
#   100, 101, 102
#
# Marker size:
#   30 mm
#
# Gap:
#   5 mm
#
# Margin:
#   5 mm
#
# Output:
#   aruco_1x3_5x5.pdf
# =========================================================

# =========================================================
# Parameters
# =========================================================

MARKER_SIZE_MM = 30
GAP_MM = 5
MARGIN_MM = 5

DPI = 600

MARKER_IDS = [100, 101, 102]

OUTPUT_PDF = "aruco_1x3_5x5.pdf"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# =========================================================
# Utility
# =========================================================


def mm_to_px(mm, dpi):
    return int(mm / 25.4 * dpi)


def mm_to_pt(mm):
    return mm / 25.4 * 72


# =========================================================
# Convert physical size -> pixels
# =========================================================

marker_px = mm_to_px(MARKER_SIZE_MM, DPI)
gap_px = mm_to_px(GAP_MM, DPI)
margin_px = mm_to_px(MARGIN_MM, DPI)

# =========================================================
# Canvas size
# =========================================================

board_width_px = marker_px * len(MARKER_IDS) + gap_px * (len(MARKER_IDS) - 1)

board_height_px = marker_px

canvas_width_px = board_width_px + margin_px * 2
canvas_height_px = board_height_px + margin_px * 2

# =========================================================
# Create white canvas
# =========================================================

img = np.ones((canvas_height_px, canvas_width_px), dtype=np.uint8) * 255

# =========================================================
# Draw markers
# =========================================================

for i, marker_id in enumerate(MARKER_IDS):
    marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_px)

    x = margin_px + i * (marker_px + gap_px)
    y = margin_px

    img[y : y + marker_px, x : x + marker_px] = marker_img

# =========================================================
# Convert to PIL image
# =========================================================

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
pil_img = Image.fromarray(img_rgb)

# =========================================================
# Physical board size (mm)
# =========================================================

board_width_mm = (
    MARKER_SIZE_MM * len(MARKER_IDS) + GAP_MM * (len(MARKER_IDS) - 1) + MARGIN_MM * 2
)

board_height_mm = MARKER_SIZE_MM + MARGIN_MM * 2

# =========================================================
# Convert physical size -> PDF points
# =========================================================

board_width_pt = mm_to_pt(board_width_mm)
board_height_pt = mm_to_pt(board_height_mm)

# =========================================================
# Create PDF
# =========================================================

pdf = canvas.Canvas(OUTPUT_PDF, pagesize=A4)

a4_width_pt, a4_height_pt = A4

# center on page
x_pt = (a4_width_pt - board_width_pt) / 2
y_pt = (a4_height_pt - board_height_pt) / 2

pdf.drawImage(
    ImageReader(pil_img), x_pt, y_pt, width=board_width_pt, height=board_height_pt
)

pdf.save()

# =========================================================
# Print info
# =========================================================

print(f"Saved: {OUTPUT_PDF}")

print()
print("=== Board Info ===")
print("Dictionary     : DICT_5X5_250")
print(f"Marker IDs     : {MARKER_IDS}")
print(f"Marker size    : {MARKER_SIZE_MM} mm")
print(f"Gap             : {GAP_MM} mm")
print(f"Margin          : {MARGIN_MM} mm")

print()
print("=== Physical Size ===")
print(f"Width  : {board_width_mm:.1f} mm")
print(f"Height : {board_height_mm:.1f} mm")

print()
print("=== Print Settings ===")
print("Scale = 100%")
print("Disable 'Fit to page'")
