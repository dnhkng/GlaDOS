"""
High-Performance Image Preprocessor for Vision Transformers

This module provides a Numba-optimized implementation of image preprocessing operations
specifically designed for Vision Transformer models. It offers efficient image resizing,
normalization, and patch splitting capabilities with a focus on performance.

Key Features:
    - Numba-accelerated image processing operations
    - Configurable patch-based image splitting
    - Memory-efficient processing pipeline
    - Automatic aspect ratio preservation
    - Support for both global and local patch processing

Main Components:
    ImagePreprocessor: Core class handling all preprocessing operations
        - Configurable through a dictionary-based initialization
        - Supports customizable normalization parameters
        - Implements efficient patch-based processing
        - Provides both single-image and multi-patch outputs

    Optimized Functions:
        - numba_resize_image: Parallel image resizing with boundary handling
        - numba_process_patch: Fast patch normalization and channel reordering

Performance Notes:
    - First run includes JIT compilation overhead
    - Subsequent runs benefit from cached compiled code
    - Parallel processing optimized for large image resizing operations
    - Memory-efficient implementation for handling large images

Usage Example:
    config = {
        'image_mean': [0.485, 0.456, 0.406],
        'image_std': [0.229, 0.224, 0.225],
        'rescale_factor': 1/255.0,
        'size': {'longest_edge': 224},
        'max_image_size': {'longest_edge': 1024}
    }
    processor = ImagePreprocessor(config)
    result = processor.preprocess_image(image)

Dependencies:
    - numba: For JIT compilation and parallel processing
    - numpy: For efficient array operations
    - math: For ceiling and rounding operations

References:
    Based on the Idefics3ImageProcessor implementation from HuggingFace Transformers:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py
"""

import math
from typing import Any

from numba import jit, prange
import numpy as np
from numpy.typing import NDArray


@jit(nopython=True, parallel=True, cache=True)
def numba_resize_image(image: NDArray, dst_h: int, dst_w: int) -> NDArray:
    """
    Perform fast image resizing using Numba parallel processing.

    This function implements a nearest-neighbor interpolation algorithm optimized
    with Numba's parallel processing capabilities. It includes manual boundary
    clipping for robustness.

    Args:
        image (NDArray): Input image array of shape (height, width, channels).
        dst_h (int): Desired output height.
        dst_w (int): Desired output width.

    Returns:
        NDArray: Resized image array of shape (dst_h, dst_w, channels).

    Notes:
        - Uses parallel processing for row-wise operations
        - Implements manual boundary clipping for robustness
        - Optimized for float32 dtype operations
    """
    src_h, src_w = image.shape[:2]
    resized = np.empty((dst_h, dst_w, 3), dtype=np.float32)

    x_ratio = float(src_w - 1) / dst_w
    y_ratio = float(src_h - 1) / dst_h

    for i in prange(dst_h):
        src_y = int(i * y_ratio)
        # Manual clipping
        if src_y < 0:
            src_y = 0
        elif src_y >= src_h:
            src_y = src_h - 1

        for j in range(dst_w):
            src_x = int(j * x_ratio)
            # Manual clipping
            if src_x < 0:
                src_x = 0
            elif src_x >= src_w:
                src_x = src_w - 1

            resized[i, j] = image[src_y, src_x]

    return resized


@jit(nopython=True, cache=True)
def numba_process_patch(
    patch: NDArray[np.float32], scale_and_norm: NDArray[np.float32], scaled_mean: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Process an image patch with combined rescaling and normalization.

    This function applies normalization and rescaling to an image patch using
    pre-computed normalization factors. The operation is optimized using Numba
    for performance.

    Args:
        patch (NDArray[np.float32]): Input image patch of shape (height, width, 3).
        scale_and_norm (NDArray[np.float32]): Pre-computed normalization factors for each channel.
        scaled_mean (NDArray[np.float32]): Pre-computed scaled mean values for each channel.

    Returns:
        NDArray[np.float32]: Processed patch of shape (3, height, width).

    Notes:
        - Combines rescaling and normalization into a single operation
        - Performs channel-first conversion during processing
        - Optimized for float32 dtype operations
    """
    result = np.empty((3, patch.shape[0], patch.shape[1]), dtype=np.float32)

    # Combined rescale and normalization
    for c in range(3):
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                result[c, i, j] = patch[i, j, c] * scale_and_norm[c] - scaled_mean[c]

    return result


class ImagePreprocessor:
    """
    A high-performance image preprocessing class optimized with Numba for vision transformer models.

    This class implements image preprocessing operations including resizing, normalization,
    and patch splitting. It is specifically designed for preparing images for vision
    transformer models, with a focus on performance through Numba optimization.

    Attributes:
        image_mean (np.ndarray): Mean values for image normalization, shape (3, 1, 1).
        image_std (np.ndarray): Standard deviation values for normalization, shape (3, 1, 1).
        rescale_factor (float): Factor for rescaling pixel values.
        do_image_splitting (bool): Whether to split images into patches.
        max_size (int): Maximum allowed size for the longest edge during initial resize.
        patch_size (int): Size of patches for splitting operation.
        scale_and_norm (np.ndarray): Pre-computed normalization factors.
        scaled_mean (np.ndarray): Pre-computed scaled mean values.

    Notes:
        - All operations are optimized using Numba for high-performance processing
        - The class supports both single-image processing and patch-based processing
        - Image splitting is optional and controlled by do_image_splitting flag
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.image_mean = np.array(config["image_mean"]).reshape(3, 1, 1)
        self.image_std = np.array(config["image_std"]).reshape(3, 1, 1)
        self.rescale_factor = config["rescale_factor"]
        self.do_image_splitting = config.get("do_image_splitting", True)
        self.max_size = config["size"]["longest_edge"]
        self.patch_size = config["max_image_size"]["longest_edge"]

        # Pre-compute normalization factors
        self.scale_and_norm = (self.rescale_factor / self.image_std).astype(np.float32)
        self.scaled_mean = (self.image_mean / self.image_std).astype(np.float32)

    def get_split_params(self, height: int, width: int) -> tuple:
        """
        Calculate optimal parameters for splitting an image into patches.

        This method computes the optimal splitting parameters to ensure uniform
        coverage of the input image while maintaining the specified patch size
        constraints.

        Args:
            height (int): Height of the input image.
            width (int): Width of the input image.

        Returns:
            tuple: A tuple containing:
                - starts_y (np.ndarray): Starting y-coordinates for each patch
                - starts_x (np.ndarray): Starting x-coordinates for each patch
                - ends_y (np.ndarray): Ending y-coordinates for each patch
                - ends_x (np.ndarray): Ending x-coordinates for each patch

        Notes:
            - Ensures optimal coverage with minimum overlap between patches
            - Handles edge cases where image dimensions aren't perfectly divisible
        """
        n_patches_h = (height + self.patch_size - 1) // self.patch_size
        n_patches_w = (width + self.patch_size - 1) // self.patch_size
        optimal_height = math.ceil(height / n_patches_h)
        optimal_width = math.ceil(width / n_patches_w)

        starts_y = np.arange(n_patches_h) * optimal_height
        starts_x = np.arange(n_patches_w) * optimal_width
        ends_y = np.minimum(starts_y + optimal_height, height)
        ends_x = np.minimum(starts_x + optimal_width, width)

        return starts_y, starts_x, ends_y, ends_x

    def split_and_process_patches(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], int, int]:
        """
        Split an image into patches and process them for model input.

        This method handles both the splitting of large images into patches and
        the processing of each patch. It also generates a global view of the
        entire image.

        Args:
            image (NDArray[np.uint8]): Input image as uint8 array of shape (height, width, 3).

        Returns:
            tuple: A tuple containing:
                - patches_array (NDArray[np.float32]): Processed patches including global view
                - pixel_attention_mask (NDArray[np.bool_]): Attention mask for the patches
                - num_splits_h (int): Number of splits in height dimension
                - num_splits_w (int): Number of splits in width dimension

        Notes:
            - Automatically handles images smaller than patch_size
            - Includes a global view as the last patch
            - Applies both resizing and normalization to each patch
        """
        height, width = image.shape[:2]

        num_splits_h: int = 0
        num_splits_w: int = 0

        if height > self.patch_size or width > self.patch_size:
            # Calculate splits
            num_splits_h = math.ceil(height / self.patch_size)
            num_splits_w = math.ceil(width / self.patch_size)

            n_patches = (num_splits_h * num_splits_w) + 1  # +1 for global view
            patches_array = np.empty((n_patches, 3, self.patch_size, self.patch_size), dtype=np.float32)

            # Calculate optimal patch sizes
            optimal_height = math.ceil(height / num_splits_h)
            optimal_width = math.ceil(width / num_splits_w)

            idx = 0
            for h in range(num_splits_h):
                for w in range(num_splits_w):
                    start_y = h * optimal_height
                    start_x = w * optimal_width
                    end_y = min(start_y + optimal_height, height)
                    end_x = min(start_x + optimal_width, width)

                    patch = image[start_y:end_y, start_x:end_x]
                    if patch.shape[:2] != (self.patch_size, self.patch_size):
                        patch = numba_resize_image(patch, self.patch_size, self.patch_size)
                    patches_array[idx] = numba_process_patch(
                        patch, self.scale_and_norm[:, 0, 0], self.scaled_mean[:, 0, 0]
                    )
                    idx += 1

            # Add global view
            global_view = numba_resize_image(image, self.patch_size, self.patch_size)
            patches_array[-1] = numba_process_patch(
                global_view, self.scale_and_norm[:, 0, 0], self.scaled_mean[:, 0, 0]
            )
            pixel_attention_mask = np.ones((1, 1, self.patch_size, self.patch_size), dtype=np.bool_)
            return patches_array[None], pixel_attention_mask, num_splits_h, num_splits_w

        else:
            processed = numba_process_patch(image, self.scale_and_norm[:, 0, 0], self.scaled_mean[:, 0, 0])
            return processed[None, None], np.ones((1, 1, self.patch_size, self.patch_size), dtype=np.bool_), 0, 0

    def preprocess_image(self, image: NDArray[np.uint8]) -> dict[str, Any]:
        """
        Execute the complete image preprocessing pipeline.

        This method implements the full preprocessing pipeline including initial
        resizing, optional patch splitting, and normalization. It handles various
        image sizes and aspect ratios while maintaining aspect ratio during resizing.

        Args:
            image (NDArray[np.uint8]): Input image as uint8 array of shape (height, width, 3).

        Returns:
            dict[str, Any]: A dictionary containing:
                - pixel_values (NDArray): Processed image values
                - pixel_attention_mask (NDArray): Attention mask for the processed values
                - rows (list): Number of row splits
                - cols (list): Number of column splits

        Notes:
            - Maintains aspect ratio during initial resizing
            - Ensures dimensions are even numbers
            - Handles both patch-based and single-image processing
            - Returns nested structure compatible with transformer models
        """
        # Initial resize to max size
        height, width = image.shape[:2]
        aspect_ratio = width / height

        if width >= height:
            new_width = min(self.max_size, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(self.max_size, height)
            new_width = int(new_height * aspect_ratio)

        # Make dimensions even
        new_height = new_height + (new_height % 2)
        new_width = new_width + (new_width % 2)

        if (new_height, new_width) != image.shape[:2]:
            image = numba_resize_image(image, new_height, new_width)

        if self.do_image_splitting:
            # Resize to multiple of patch size
            if new_height % self.patch_size != 0 or new_width % self.patch_size != 0:
                height = math.ceil(new_height / self.patch_size) * self.patch_size
                width = math.ceil(new_width / self.patch_size) * self.patch_size
                image = numba_resize_image(image, height, width)

            # Split and process patches
            pixel_values, attention_mask, rows, cols = self.split_and_process_patches(image)
        else:
            # Process single image
            processed = numba_process_patch(image, self.scale_and_norm[:, 0, 0], self.scaled_mean[:, 0, 0])
            pixel_values = processed[None, None]
            attention_mask = np.ones((1, 1), dtype=np.bool_)
            rows, cols = 0, 0

        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": attention_mask,
            "rows": [[rows]],
            "cols": [[cols]],
        }
