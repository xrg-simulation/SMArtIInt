within SMArtIInt.Tester.ExamplePI.ONNX;
model StepTest_RNN_onnx
  extends TFLite.StepTest_RNN_tflite(redeclare ONNX.TF_PI_RNN_onnx controller);
equation
  connect(step.y, controller.u) annotation (Line(points={{-61,0},{-10.6,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_RNN_onnx;
