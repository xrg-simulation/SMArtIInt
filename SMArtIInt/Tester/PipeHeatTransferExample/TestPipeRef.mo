within SMArtIInt.Tester.PipeHeatTransferExample;
model TestPipeRef
  extends TestPipe_tflite
                  (pipe(redeclare model HeatTransfer =
          Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer));
end TestPipeRef;
