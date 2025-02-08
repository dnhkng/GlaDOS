'''
Minimal replication of the Idefics3ImageProcessor by HuggingFace, with Numba optimization.

https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py

The original implementation is a part of the Idefics3 model, which is a part of the HuggingFace Transformers library.
The Idefics3ImageProcessor is used to preprocess images for the Idefics3 model, which is a vision transformer model.

The ImageProcessor class has the following public methods:
- get_split_params(height: int, width: int) -> tuple: Pre-calculate split parameters.
- split_and_process_patches(image: np.ndarray) -> tuple: Split and process patches with row/col tracking.
- preprocess_image(image: np.ndarray) -> dict: Main preprocessing pipeline including row/col information.

The ImageProcessor class has the following private methods:
- numba_resize_image(image: NDArray, dst_h: int, dst_w: int) -> NDArray: Fast image resize using Numba with
manual clipping.
- numba_process_patch(patch: NDArray, scale_and_norm: NDArray, scaled_mean: NDArray) -> NDArray: Process
patch with Numba optimization.

The ImageProcessor class has the following attributes:
- image_mean: Mean values for image normalization.
- image_std: Standard deviation values for image normalization.
- rescale_factor: Rescale factor for image normalization.
- do_image_splitting: Flag to enable image splitting.
- max_size: Maximum size for image resizing.
- patch_size: Patch size for image splitting.
- scale_and_norm: Pre-computed normalization factors for image processing.
- scaled_mean: Pre-computed scaled mean values for image processing.

'''


import math

from numba import jit, prange
import numpy as np
from numpy.typing import NDArray


@jit(nopython=True, parallel=True)
def numba_resize_image(image: NDArray, dst_h: int, dst_w: int) -> NDArray:
    """Fast image resize using Numba with manual clipping."""
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

@jit(nopython=True)
def numba_process_patch(patch: NDArray, scale_and_norm: NDArray, scaled_mean: NDArray) -> NDArray:
    """Process patch with Numba optimization."""
    result = np.empty((3, patch.shape[0], patch.shape[1]), dtype=np.float32)
    
    # Combined rescale and normalization
    for c in range(3):
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                result[c, i, j] = patch[i, j, c] * scale_and_norm[c] - scaled_mean[c]
    
    return result

class ImagePreprocessor:
    def __init__(self, config: dict[str, any]) -> None:
        self.image_mean = np.array(config['image_mean']).reshape(3, 1, 1)
        self.image_std = np.array(config['image_std']).reshape(3, 1, 1)
        self.rescale_factor = config['rescale_factor']
        self.do_image_splitting = config.get('do_image_splitting', True)
        self.max_size = config['size']['longest_edge']
        self.patch_size = config['max_image_size']['longest_edge']

        # Pre-compute normalization factors
        self.scale_and_norm = (self.rescale_factor / self.image_std).astype(np.float32)
        self.scaled_mean = (self.image_mean / self.image_std).astype(np.float32)

    def get_split_params(self, height: int, width: int) -> tuple:
        """Pre-calculate split parameters."""
        n_patches_h = (height + self.patch_size - 1) // self.patch_size
        n_patches_w = (width + self.patch_size - 1) // self.patch_size
        optimal_height = math.ceil(height / n_patches_h)
        optimal_width = math.ceil(width / n_patches_w)
        
        starts_y = np.arange(n_patches_h) * optimal_height
        starts_x = np.arange(n_patches_w) * optimal_width
        ends_y = np.minimum(starts_y + optimal_height, height)
        ends_x = np.minimum(starts_x + optimal_width, width)
        
        return starts_y, starts_x, ends_y, ends_x

    def split_and_process_patches(self, image: np.ndarray) -> tuple:
        """Split and process patches with row/col tracking."""
        height, width = image.shape[:2]
        num_splits_h, num_splits_w = 0, 0
        
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
                    patches_array[idx] = numba_process_patch(patch, self.scale_and_norm[:, 0, 0],
                                                           self.scaled_mean[:, 0, 0])
                    idx += 1
            
            # Add global view
            global_view = numba_resize_image(image, self.patch_size, self.patch_size)
            patches_array[-1] = numba_process_patch(global_view, self.scale_and_norm[:, 0, 0],
                                                  self.scaled_mean[:, 0, 0])
            pixel_attention_mask = np.ones((1,1,self.patch_size,self.patch_size), dtype=np.bool_)
            return patches_array[None], pixel_attention_mask, num_splits_h, num_splits_w

        else:
            processed = numba_process_patch(image, self.scale_and_norm[:, 0, 0],
                                         self.scaled_mean[:, 0, 0])
            return processed[None, None], np.ones((1, 1, self.patch_size, self.patch_size), dtype=np.bool_), 0, 0

    def preprocess_image(self, image: np.ndarray) -> dict:
        """Main preprocessing pipeline including row/col information."""
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
            processed = numba_process_patch(image, self.scale_and_norm[:, 0, 0],
                                         self.scaled_mean[:, 0, 0])
            pixel_values = processed[None, None]
            attention_mask = np.ones((1, 1), dtype=np.bool_)
            rows, cols = 0, 0
            
        return {
            'pixel_values': pixel_values,
            'pixel_attention_mask': attention_mask,
            'rows': [[rows]],
            'cols': [[cols]]
        }