within SMArtIInt.Tester.ExamplePI;
model DoubleRoom_RNNState_tflite
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare TF_PI_Stateful_tflite controller);
  annotation (experiment(
      StopTime=300000,
      __Dymola_NumberOfIntervals=5000,
      __Dymola_Algorithm="Dassl"));
end DoubleRoom_RNNState_tflite;
