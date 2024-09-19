within SMArtIInt.Tester.PipeHeatTransferExample.ReferenceModels;
model TestPipeRef
  extends TFLite.TestPipe_tflite
                  (pipe(redeclare model HeatTransfer =
          Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer));
end TestPipeRef;
