const onnxjs = require("onnxjs");
const model_path = "./onnx/out.onnx"
const session = new onnxjs.InferenceSession();
session.loadModel(model_path).then(e => console.log(e));
