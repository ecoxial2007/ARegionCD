### Custom Phi-3 Vision Processor

#### Overview

This custom processor for `microsoft/Phi-3.5-vision-instruct` replaces the original to support spatial inputs like bounding boxes (bbox) and segmentation masks (mask).

Its main function is to ensure these annotations are transformed along with the image, maintaining perfect alignment for any region-based VLM tasks.

#### Usage

Place `processing_phi3_v.py` inside your local model directory (e.g., checkpoints/Phi-3.5-vision-instruct/).

