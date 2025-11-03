#!/usr/bin/env python3
"""
Test image optimization functionality
"""
import sys
from pathlib import Path
from utils.image_optimizer import (
    get_image_info,
    resize_image_for_tokens,
    resize_with_retry,
    estimate_image_tokens
)

def test_estimation():
    """Test token estimation"""
    print("=" * 60)
    print("Token Estimation Tests")
    print("=" * 60)

    test_cases = [
        (512, 512, "Small image (512x512)"),
        (1024, 1024, "Medium image (1024x1024)"),
        (2048, 2048, "Large image (2048x2048)"),
        (4096, 4096, "Very large image (4096x4096)"),
        (1920, 1080, "HD image (1920x1080)"),
        (3840, 2160, "4K image (3840x2160)"),
    ]

    for width, height, description in test_cases:
        tokens = estimate_image_tokens(width, height)
        print(f"{description:30s} -> {tokens:5d} tokens")

    print()


def test_image_file(image_path: str):
    """Test image optimization on a real image file"""
    print("=" * 60)
    print(f"Testing Image: {image_path}")
    print("=" * 60)

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    # Get image info
    print("\n1. Getting image info...")
    info = get_image_info(image_path)
    print(f"   Dimensions: {info['width']}x{info['height']}")
    print(f"   Estimated tokens: {info['estimated_tokens']}")
    print(f"   Within budget: {info['within_budget']}")
    print(f"   Needs resize: {info['recommended_resize']}")

    if info['recommended_resize']:
        print("\n2. Resizing image with retry...")
        optimized_path, metadata = resize_with_retry(image_path)
        print(f"   Output: {optimized_path}")
        print(f"   Original size: {metadata['retry_history'][0]['original_size']}")
        print(f"   New size: {metadata['retry_history'][0]['new_size']}")
        print(f"   Token reduction: {metadata['retry_history'][0]['token_reduction']}")
        print(f"   Scale factor: {metadata['retry_history'][0]['scale_factor']:.3f}")
    else:
        print("\n2. Image is already within budget - no resize needed")

    print()


if __name__ == "__main__":
    # Run estimation tests
    test_estimation()

    # Test with a real image if provided
    if len(sys.argv) > 1:
        test_image_file(sys.argv[1])
    else:
        # Look for a test image in scipy package
        test_img = Path("/home/mark/vision-tool-server/venv/lib/python3.9/site-packages/scipy/ndimage/tests/dots.png")
        if test_img.exists():
            test_image_file(str(test_img))
        else:
            print("No test image provided. Usage: python test_image_optimization.py <image_path>")
