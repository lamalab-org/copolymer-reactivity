import base64
import io
import os
import time
from pathlib import Path
from typing import Tuple, Union

import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV (BGR) NumPy array.

    Args:
        image: PIL image in RGB mode.

    Returns:
        The image as a BGR NumPy array, as expected by OpenCV functions.
    """
    np_image = np.array(image)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return cv2_image


def correct_text_orientation(
    image: Union[Image.Image, np.ndarray],
    save_directory: Union[str, Path],
    file_path: Union[str, Path],
    i: int,
) -> np.ndarray:
    """Detect and correct the rotation of a scanned page using Tesseract's OSD, saving the result.

    Args:
        image: Page image, either a PIL image or an OpenCV (BGR) NumPy array.
        save_directory: Directory the corrected image is written to.
        file_path: Original file path, used to derive the output filename.
        i: Zero-based page index, used to derive the output filename.

    Returns:
        The rotation-corrected image as an OpenCV (BGR) NumPy array.
    """
    if isinstance(image, Image.Image):
        image = pil_to_cv2(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

    rotated = imutils.rotate_bound(image, angle=results["rotate"])

    base_filename = os.path.basename(file_path)
    name_without_ext, _ = os.path.splitext(base_filename)
    new_filename = os.path.join(save_directory, f"corrected_{name_without_ext}_page{i+1}.png")

    cv2.imwrite(new_filename, rotated)
    return rotated


def resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    """Downscale an image to fit within `max_dimension` and convert it to grayscale.

    Args:
        image: PIL image to resize. Palette images are converted to RGB/RGBA first.
        max_dimension: Maximum allowed width/height in pixels; the image is only
            resized (preserving aspect ratio) if it exceeds this size.

    Returns:
        The grayscale, resized PIL image.
    """
    width, height = image.size

    # Check if the image has a palette and convert it to true color mode
    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    # convert to black and white
    image = image.convert("L")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)

        timestamp = time.time()

    return image


def convert_to_jpeg(image: Image.Image) -> bytes:
    """Encode a PIL image as JPEG bytes.

    Args:
        image: PIL image to encode.

    Returns:
        The JPEG-encoded image bytes.
    """
    with io.BytesIO() as output:
        image.save(output, format="jpeg")
        return output.getvalue()


def convert_to_jpeg2(cv2_image: np.ndarray) -> Union[bytes, None]:
    """Encode an OpenCV image as JPEG bytes.

    Args:
        cv2_image: Image as a NumPy array (OpenCV format).

    Returns:
        The JPEG-encoded image bytes, or None if encoding failed.
    """
    retval, buffer = cv2.imencode(".jpg", cv2_image)
    if retval:
        return buffer


def process_image(
    image: Image.Image,
    max_size: int,
    output_folder: Union[str, Path],
    file_path: Union[str, Path],
    i: int,
) -> Tuple[str, int]:
    """Resize, deskew and JPEG/base64-encode a page image for downstream LLM consumption.

    Args:
        image: PIL page image to process.
        max_size: Maximum width/height in pixels passed to `resize_image`.
        output_folder: Directory the deskewed intermediate image is written to.
        file_path: Original file path, used to derive the intermediate filename.
        i: Zero-based page index, used to derive the intermediate filename.

    Returns:
        A (base64_encoded_jpeg, original_max_dimension) tuple, where
        `original_max_dimension` is the larger of the original image's width/height.
    """
    width, height = image.size
    resized_image = resize_image(image, max_size)
    rotate_image = correct_text_orientation(resized_image, output_folder, file_path, i)
    jpeg_image = convert_to_jpeg2(rotate_image)
    base64_encoded_image = base64.b64encode(jpeg_image).decode("utf-8")
    return (
        base64_encoded_image,
        max(width, height),  # same tuple metadata
    )
