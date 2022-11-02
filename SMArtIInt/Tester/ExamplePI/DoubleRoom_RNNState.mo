within SMArtIInt.Tester.ExamplePI;
model DoubleRoom_RNNState
  extends ReferenceModels.DoubleRoom_ContinuousPI(redeclare TF_PI_Stateful controller);
  annotation (experiment(
      StopTime=300000,
      __Dymola_NumberOfIntervals=5000,
      __Dymola_Algorithm="Dassl"));
end DoubleRoom_RNNState;
