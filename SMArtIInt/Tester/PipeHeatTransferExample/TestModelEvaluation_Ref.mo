within SMArtIInt.Tester.PipeHeatTransferExample;
model TestModelEvaluation_Ref
  extends TestModelEvaluation_tflite
                             (redeclare Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer heatTransfer);
end TestModelEvaluation_Ref;
