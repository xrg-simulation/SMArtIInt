within SMArtIInt.Tester.ExamplePI;
model DoubleRoom_RNN_tflite
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare TF_PI_RNN_tflite controller);
end DoubleRoom_RNN_tflite;
