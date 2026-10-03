"""QA rules for shoebox.

A photo lands in the review queue when something about the restoration is
suspect. The rules, from the brief:

  * any inference exception during face restoration
  * fewer faces detected in the restored output than in the input
  * sharpness (Laplacian variance) dropped more than 15%
  * a grayscale photo that was asked to be colorized but came back gray

Each rule returns a stable reason string; the gallery shows them verbatim.
"""

import cv2
import numpy as np

SHARPNESS_DROP = 0.15


def sharpness(gray_bgr_image):
    """Variance of the Laplacian on the grayscale image (classic focus metric)."""
    if gray_bgr_image.ndim == 3:
        gray = cv2.cvtColor(gray_bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_bgr_image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def chroma_variance(rgb_or_bgr_image):
    """Mean of per-pixel channel spread: ~0 for grayscale, >0 for color."""
    img = rgb_or_bgr_image.astype(np.int16)
    if img.ndim == 2 or img.shape[2] == 1:
        return 0.0
    d1 = (img[:, :, 0] - img[:, :, 1]).var()
    d2 = (img[:, :, 1] - img[:, :, 2]).var()
    d3 = (img[:, :, 2] - img[:, :, 0]).var()
    return float((d1 + d2 + d3) / 3.0)


FACE_FAILED = 'face restoration failed - review'
FACE_LOST = 'fewer faces detected after restoration'
SHARPNESS_LOST = 'sharpness dropped more than 15%'
STAYED_GRAY = 'colorization ran but the photo is still grayscale'


def evaluate(face_errors, face_count_in, face_count_out,
             sharpness_in, sharpness_out,
             was_gray, colorize_requested, restored_still_gray):
    """Return the list of flag reasons for one photo (empty when clean)."""
    flags = []
    if face_errors:
        flags.append(FACE_FAILED)
    if face_count_out < face_count_in:
        flags.append(FACE_LOST)
    if sharpness_in > 0 and sharpness_out < sharpness_in * (1 - SHARPNESS_DROP):
        flags.append(SHARPNESS_LOST)
    if was_gray and colorize_requested and restored_still_gray:
        flags.append(STAYED_GRAY)
    return flags
