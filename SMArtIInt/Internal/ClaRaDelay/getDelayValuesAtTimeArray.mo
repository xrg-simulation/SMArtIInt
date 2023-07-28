within SMArtIInt.Internal.ClaRaDelay;
function getDelayValuesAtTimeArray
//__________________________________________________________________________//
// Component of the ClaRa library, version: 1.8.0                           //
//                                                                          //
// Licensed by the ClaRa development team under the 3-clause BSD License.   //
// Copyright  2013-2022, ClaRa development team.                            //
//                                                                          //
// The ClaRa development team consists of the following partners:           //
// TLK-Thermo GmbH (Braunschweig, Germany),                                 //
// XRG Simulation GmbH (Hamburg, Germany).                                  //
//__________________________________________________________________________//
// Contents published in ClaRa have been contributed by different authors   //
// and institutions. Please see model documentation for detailed information//
// on original authorship and copyrights.                                   //
//__________________________________________________________________________//

  input ClaRaDelay.ExternalTables tables;
  input Real simulationTime;
  input Real value;
  input Real getTime;
  input Integer index;
  output Real result;

external"C" result = clara_getDelayValuesAtTimeArray(tables, simulationTime, value, getTime, index)
annotation (Library={"Delay-V1"});

end getDelayValuesAtTimeArray;
