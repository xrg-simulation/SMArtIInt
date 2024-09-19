within SMArtIInt.Tester.ExamplePI.TFLite;
model DoubleRoom_RNN_tflite
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare TFLite.TF_PI_RNN_tflite controller);
end DoubleRoom_RNN_tflite;
