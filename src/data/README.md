# Transforms
All transforms except those mentioned [below](#transforms-with-different-input--output-signature) expect input in following format:
1. Images: RGB, 0-255 range, float numpy array
2. Bounding box classes (if present): float numpy array
3. Bounding box coordinates (if present): XYX2Y2, absolute pixel format (unnormalized), float numpy array
4. Label class (if present): float numpy array
5. Label value (if present): float numpy array

and return output in the same format. Additionally, the output image is has the same dimensions as the input image.

## Transforms with different input / output signature
### Base Transforms
1. BaseImageOnlyAlbumentationsAug
2. BaseImageBboxAlbumentationsAug

These transforms can't be used directly.

### Helper Transforms
1. RandomApply
2. Lambda
3. Compose

Behaviour of these transforms depends on how they are used.

### Converters
1. ConvertFromInts: Converts input to numpy float32 array.
2. ToTensor: Converts inputs to Tensor. The numpy HWC image is converted to pytorch CHW tensor.
3. ToNumpy: Converts inputs to Numpy. The torch CHW tensor is converted to numpy HWC array.
4. ToPercentCoords: Converts bounding box coordinates from absolute format to normalized format.
5. ToAbsoluteCoords: Converts bounding box coordinates from normalized format to absolute format.
6. XYWHToXYX2Y2: Converts bounding box coordinates from XYWH to XYX2Y2 format.
7. XYX2Y2ToXYWH: Converts bounding box coordinates from XYX2Y2 to XYWH format.
8. Resize: Changes image dimensions.
9. ConvertColor: Changes colorspace of the image.

### Others
1. BlurGray: Return output image with 1 channel.
2. ClassicSegMean: Return output image with 4 channels.
3. ClassicSegOtsu: Return output image with 4 channels.


## Template
Assuming that the bounding boxes are initially in XYWH absolute pixel format, following template can be used:
```
1. XYWHToXYX2Y2 (only if you have bbox)
2. ConvertFromInts
.
.
All other transforms that you want to use (you need to include Resize)
.
.
n-1. ToPercentCoords (only if you have bbox)
n. ToTensor
```
This template is not valid for `SSDAugmentation` transform. It already contains these transforms and should be used alone.
