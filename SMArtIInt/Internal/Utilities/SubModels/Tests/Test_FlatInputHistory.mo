within SMArtIInt.Internal.Utilities.SubModels.Tests;
model Test_FlatInputHistory

  extends Modelica.Icons.Example;

  Modelica.Blocks.Sources.RealExpression realExpression(y=sin(time))
    annotation (Placement(transformation(extent={{-68,-10},{-48,10}})));


  RNNFlattenInput flatInputHistory_switchDelay(
    useClaRaDelay=true,
    nInputs=2,
    continuous=true,
    flatteningMethod=SMArtIInt.Internal.Utilities.SubModels.RNNFlatteningMethod.OldFirstTimeSeq,
    u={realExpression.y,-realExpression.y}) annotation (Placement(transformation(extent={{-8,-40},{12,-20}})));
equation

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=10,
      Tolerance=1e-05,
      __Dymola_Algorithm="Dassl"));
end Test_FlatInputHistory;
