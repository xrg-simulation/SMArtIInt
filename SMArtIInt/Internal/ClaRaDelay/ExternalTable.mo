within SMArtIInt.Internal.ClaRaDelay;
class ExternalTable
  extends ExternalObject;
  function constructor
    extends Modelica.Icons.Function;
    output ExternalTable table;
    external "C" table = initDelay() annotation (Library={"Delay-V1"});
  end constructor;

  function destructor "Release storage of table"
    extends Modelica.Icons.Function;
    input ExternalTable table;
    external "C" deleteDelay(table) annotation (Library={"Delay-V1"});
  end destructor;
end ExternalTable;
