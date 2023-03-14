within SMArtIInt.Internal.ClaRaDelay;
class ExternalTable
  extends ExternalObject;
  function constructor
    extends Modelica.Icons.Function;
    output ExternalTable table;
    external "C" table = clara_initDelay() annotation (Library={"Delay-V1"});
  end constructor;

  function destructor "Release storage of table"
    extends Modelica.Icons.Function;
    input ExternalTable table;
    external "C" clara_deleteDelay(table) annotation (Library={"Delay-V1"});
  end destructor;
end ExternalTable;
