within SMArtIInt.Internal;
class ModelicaUtilityHelper
  extends ExternalObject;
  extends Modelica.Icons.SourcesPackage;

  function constructor
    output ModelicaUtilityHelper modelicaUtiltityHelper;
    external"C" modelicaUtiltityHelper = createModelicaUtitlityHelper() annotation (
        Include="#include \"ModelicaUtilityInterface.cpp\"",
        IncludeDirectory="modelica://SMArtIInt/Resources/Include/");
  end constructor;

  function destructor
    input ModelicaUtilityHelper modelicaUtiltityHelper;
  external"C" deleteModelicaUtitlityHelper(modelicaUtiltityHelper) annotation (Include="#include \"ModelicaUtilityInterface.cpp\"",
        IncludeDirectory="modelica://SMArtIInt/Resources/Include/");
  end destructor;
end ModelicaUtilityHelper;
