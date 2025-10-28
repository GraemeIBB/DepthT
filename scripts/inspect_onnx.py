import onnxruntime as ort

sess = ort.InferenceSession("../models/best.onnx")
print("== ONNX outputs ==")
for o in sess.get_outputs():
    print(f" {o.name:<20} shape={o.shape}")