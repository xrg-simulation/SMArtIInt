within SMArtIInt.Tester.ExamplePI;
model DoubleRoom_RNN_onnx
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare TF_PI_RNN_onnx controller);
end DoubleRoom_RNN_onnx;
