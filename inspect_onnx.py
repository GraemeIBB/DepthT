import onnxruntime as ort

sess = ort.InferenceSession("datasets\segmentation\yolo11n-seg.onnx")
print("== ONNX outputs ==")
for o in sess.get_outputs():
    print(f" {o.name:<20} shape={o.shape}")