within SMArtIInt.Tester.PipeHeatTransferExample.ReferenceModels;
model TestModelEvaluation_Ref
  extends TFLite.TestModelEvaluation_tflite
                             (redeclare Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer heatTransfer);
end TestModelEvaluation_Ref;
