within SMArtIInt.Tester.ExamplePI.ONNX;
model DoubleRoom_RNN_onnx
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare ONNX.TF_PI_RNN_onnx controller);
end DoubleRoom_RNN_onnx;
