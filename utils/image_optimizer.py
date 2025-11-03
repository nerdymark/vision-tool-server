"""
Image optimization for LLM token management
Resizes images to prevent token overflow with smart estimation
"""
import cv2
import math
from pathlib import Path
from typing import Tuple, Optional
import tempfile


# Token estimation constants
# Based on typical vision model tokenization:
# - Each 512x512 tile ≈ 256-512 tokens depending on model
# - qwen2-vl uses about 0.5-1 token per pixel patch
TOKENS_PER_TILE_512 = 400  # Conservative estimate
MAX_TOKENS_TARGET = 3500   # Leave room for text context (target 3.5k of 4k)
MIN_IMAGE_SIZE = 224       # Minimum size for vision models


def estimate_image_tokens(width: int, height: int) -> int:
    """
    Estimate token count for an image based on dimensions

    Vision models typically process images in tiles/patches:
    - Image is divided into fixed-size patches (e.g., 14x14, 16x16)
    - Each patch becomes a token
    - Additional tokens for positional encoding

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count
    """
    # Calculate number of 512x512 tiles needed to cover image
    tiles_x = math.ceil(width / 512)
    tiles_y = math.ceil(height / 512)
    total_tiles = tiles_x * tiles_y

    # Estimate tokens (including overhead for multi-tile images)
    estimated_tokens = total_tiles * TOKENS_PER_TILE_512

    # Add overhead for very large images (positional encodings, etc)
    if total_tiles > 4:
        estimated_tokens = int(estimated_tokens * 1.2)

    return estimated_tokens


def calculate_target_dimensions(current_width: int, current_height: int,
                                target_tokens: int) -> Tuple[int, int]:
    """
    Calculate optimal image dimensions to fit within token budget

    Uses geometric scaling to maintain aspect ratio while reducing tokens.
    Formula: new_area / old_area = target_tokens / current_tokens

    Args:
        current_width: Current image width
        current_height: Current image height
        target_tokens: Target token count

    Returns:
        Tuple of (new_width, new_height)
    """
    current_tokens = estimate_image_tokens(current_width, current_height)

    if current_tokens <= target_tokens:
        return current_width, current_height

    # Calculate scaling factor based on token ratio
    # Since tokens scale roughly with area: scale = sqrt(target/current)
    scale_factor = math.sqrt(target_tokens / current_tokens)

    # Apply scaling
    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)

    # Ensure minimum size
    if new_width < MIN_IMAGE_SIZE or new_height < MIN_IMAGE_SIZE:
        # Scale up to minimum while maintaining aspect ratio
        aspect_ratio = current_width / current_height
        if aspect_ratio > 1:
            new_width = MIN_IMAGE_SIZE
            new_height = int(MIN_IMAGE_SIZE / aspect_ratio)
        else:
            new_height = MIN_IMAGE_SIZE
            new_width = int(MIN_IMAGE_SIZE * aspect_ratio)

    # Round to multiples of 16 for better model compatibility
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    return max(new_width, MIN_IMAGE_SIZE), max(new_height, MIN_IMAGE_SIZE)


def resize_image_for_tokens(image_path: str, max_tokens: int = MAX_TOKENS_TARGET,
                            output_path: Optional[str] = None) -> Tuple[str, dict]:
    """
    Resize an image to fit within token budget

    Args:
        image_path: Path to input image
        max_tokens: Maximum token budget (default 3500)
        output_path: Optional output path (creates temp file if None)

    Returns:
        Tuple of (resized_image_path, metadata_dict)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_height, original_width = image.shape[:2]
    original_tokens = estimate_image_tokens(original_width, original_height)

    metadata = {
        "original_size": (original_width, original_height),
        "original_tokens_estimated": original_tokens,
        "resized": False
    }

    # Check if resizing needed
    if original_tokens <= max_tokens:
        metadata["message"] = "Image within token budget, no resize needed"
        return image_path, metadata

    # Calculate target dimensions
    new_width, new_height = calculate_target_dimensions(
        original_width, original_height, max_tokens
    )

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height),
                               interpolation=cv2.INTER_AREA)

    # Save resized image
    if output_path is None:
        # Create temp file
        suffix = Path(image_path).suffix or '.jpg'
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix,
            dir=Path(image_path).parent
        )
        output_path = temp_file.name
        temp_file.close()

    cv2.imwrite(output_path, resized_image)

    # Update metadata (ensure all values are JSON-serializable)
    new_tokens = estimate_image_tokens(new_width, new_height)
    metadata.update({
        "resized": True,
        "new_size": [new_width, new_height],  # List instead of tuple for JSON
        "new_tokens_estimated": int(new_tokens),
        "scale_factor": float(new_width / original_width),
        "token_reduction": f"{((original_tokens - new_tokens) / original_tokens * 100):.1f}%",
        "output_path": str(output_path)  # Convert Path to string
    })

    return str(output_path), metadata


def resize_with_retry(image_path: str, max_attempts: int = 3) -> Tuple[str, dict]:
    """
    Resize image with exponential backoff if token budget exceeded

    Attempts progressively smaller token targets:
    - Attempt 1: 3500 tokens (87.5% of 4k context)
    - Attempt 2: 2800 tokens (70% of 4k context)
    - Attempt 3: 2000 tokens (50% of 4k context)

    Args:
        image_path: Path to input image
        max_attempts: Maximum number of resize attempts

    Returns:
        Tuple of (resized_image_path, metadata_dict)
    """
    # Token targets for each attempt (exponential backoff)
    token_targets = [
        3500,  # 87.5% of 4k
        2800,  # 70% of 4k
        2000,  # 50% of 4k
    ]

    all_metadata = []

    for attempt in range(max_attempts):
        target = token_targets[min(attempt, len(token_targets) - 1)]

        try:
            resized_path, metadata = resize_image_for_tokens(
                image_path, max_tokens=target
            )

            metadata["attempt"] = attempt + 1
            metadata["target_tokens"] = target

            # If we resized successfully, return with retry history
            if metadata.get("resized") or metadata.get("original_tokens_estimated", 0) <= target:
                # Create a shallow copy without retry_history to avoid circular reference
                metadata_copy = {
                    "attempt": metadata.get("attempt"),
                    "target_tokens": metadata.get("target_tokens"),
                    "original_size": metadata.get("original_size"),
                    "original_tokens_estimated": metadata.get("original_tokens_estimated"),
                    "resized": metadata.get("resized"),
                    "message": metadata.get("message")
                }

                # Add resize-specific fields if present
                if metadata.get("resized"):
                    metadata_copy.update({
                        "new_size": metadata.get("new_size"),
                        "new_tokens_estimated": metadata.get("new_tokens_estimated"),
                        "scale_factor": metadata.get("scale_factor"),
                        "token_reduction": metadata.get("token_reduction"),
                        "output_path": metadata.get("output_path")
                    })

                # Add this attempt to history
                all_metadata.append(metadata_copy)

                # Add complete history to final metadata (no circular reference now)
                metadata["retry_history"] = all_metadata
                return resized_path, metadata

        except Exception as e:
            all_metadata.append({
                "attempt": attempt + 1,
                "target_tokens": target,
                "error": str(e)
            })

            # If this was last attempt, re-raise
            if attempt == max_attempts - 1:
                raise

    # Should not reach here, but return last attempt
    return resized_path, {"retry_history": all_metadata}


def get_image_info(image_path: str) -> dict:
    """Get image dimensions and estimated token count"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    tokens = estimate_image_tokens(width, height)

    return {
        "width": width,
        "height": height,
        "estimated_tokens": tokens,
        "within_budget": tokens <= MAX_TOKENS_TARGET,
        "recommended_resize": tokens > MAX_TOKENS_TARGET
    }
