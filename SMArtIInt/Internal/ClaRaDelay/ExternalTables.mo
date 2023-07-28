within SMArtIInt.Internal.ClaRaDelay;
class ExternalTables
  extends ExternalObject;
  function constructor
    extends Modelica.Icons.Function;
    input Integer size;
    output ExternalTables tables;
    external "C" tables = clara_initDelayArray(size) annotation (Library={"Delay-V1"});
  end constructor;

  function destructor "Release storage of table"
    extends Modelica.Icons.Function;
    input ExternalTables tables;
    external "C" clara_deleteDelayArray(tables) annotation (Library={"Delay-V1"});
  end destructor;
end ExternalTables;
