# Depth Trainer
This project aims to be a suite for preparing datasets that include depth values and training object detection models to utilize this data.

## Included
RGBDcombinator.py:
- Handles batches of rgb and depth img pairs, overlays depth greyscale within a new generated png as Alpha layer.

WebcamTest.py:
- tests a given model using a specified onboard camera. currently under construction.

## Dependencies 
- `matplotlib`
- `PIL`
- `ultralytics`
- `torch` can utilize gpu if proper package installed
- `cv2`
