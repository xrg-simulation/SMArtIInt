within SMArtIInt.Internal.Utilities.SubModels;
model RNNFlattenInput

  import gdv = SMArtIInt.Internal.ClaRaDelay.getDelayValuesAtTimeArray;
  import HFM = SMArtIInt.Internal.Utilities.SubModels.RNNFlatteningMethod;

  parameter Boolean useClaRaDelay=true;
  parameter Integer nInputs=1 "Number of inputs";

  parameter Modelica.Units.SI.Time samplePeriod=0.1 "sampling interval";

  parameter Integer nHistoricElements=10 "Number of elements from sampling steps for each input fed to the neural net";

  parameter Boolean continuous=false;

  final parameter Integer n_tensorElements=nInputs*nHistoricElements;

  parameter RNNFlatteningMethod flatteningMethod=RNNFlatteningMethod.NewFirstInputSeq annotation (Evaluate=true);

  final parameter Modelica.Units.SI.Time startTime( fixed = false);
  Boolean sampleTrigger;

  input Real[nInputs] u annotation (Dialog);

  output Real[nHistoricElements*nInputs] inputFlattenTensor annotation (Dialog);

  //Pointers for inputFlattenTensor2 and delayedInputs
  //Note: in contrast to Modelica delay we need only 1 delay-table pointer per input.
  SMArtIInt.Internal.ClaRaDelay.ExternalTables pointer_inputFlattenTensor=
      SMArtIInt.Internal.ClaRaDelay.ExternalTables(nInputs);

  //////////////////////////////////////////////////////////////////////////////////
  //DelayTimes for ClaRaDelay
  //Note: Modelica delay(u[i],delayTime) expects delayTime to the difference between current time and past instance of time
  //      for which the delayed signal should be obtaind: delayTime=time - pastTime
  //      In contrast to that ClaRaDelay takes a vector of pastTimes: pastTime = time - delayTime
  //////////////////////////////////////////////////////////////////////////////////

  Real[nHistoricElements] delayTimes={max(0, time - samplePeriod*(t - 1)) for t in 1:nHistoricElements};

initial equation
  startTime = time;

equation
  //////////////////////////////////////////////////////////////////////////////////
  // example for call of ClaRaDelay analogue to Modelica delay                                   //
  //////////////////////////////////////////////////////////////////////////////////
  if continuous then

    sampleTrigger = false;

    for t in 1:nHistoricElements loop
      for i in 1:nInputs loop
        if flatteningMethod == RNNFlatteningMethod.NewFirstInputSeq then
          if (t == 1) then
            inputFlattenTensor[(t - 1)*nInputs + i] = u[i];
          else
            if not useClaRaDelay then
              inputFlattenTensor[(t - 1)*nInputs + i] = delay(u[i], samplePeriod*(t - 1));
            else
              inputFlattenTensor[(t - 1)*nInputs + i] = gdv(
                pointer_inputFlattenTensor,
                time,
                u[i],
                delayTimes[t],
                i);
            end if;
          end if;
        elseif flatteningMethod == RNNFlatteningMethod.NewFirstTimeSeq then
          if (t == 1) then
            inputFlattenTensor[(i - 1)*nHistoricElements + t] = u[i];
          else
            if not useClaRaDelay then
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = delay(u[i], samplePeriod*(t - 1));
            else
              inputFlattenTensor[(t - 1)*nInputs + i] = gdv(
                pointer_inputFlattenTensor,
                time,
                u[i],
                delayTimes[t],
                i);
            end if;
          end if;
        elseif flatteningMethod == RNNFlatteningMethod.OldFIrstInputSeq then
          if (t == nHistoricElements) then
            inputFlattenTensor[(t - 1)*nInputs + i] = u[i];
          else
            if not useClaRaDelay then
              inputFlattenTensor[(t - 1)*nInputs + i] = delay(u[i], samplePeriod*(nHistoricElements - t));
            else
              inputFlattenTensor[(t - 1)*nInputs + i] = gdv(
                pointer_inputFlattenTensor,
                time,
                u[i],
                delayTimes[nHistoricElements - t + 1],
                i);
            end if;
          end if;
        elseif flatteningMethod == RNNFlatteningMethod.OldFirstTimeSeq then
          if (t == nHistoricElements) then
            inputFlattenTensor[(i - 1)*nHistoricElements + t] = u[i];
          else
            if not useClaRaDelay then
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = delay(u[i], samplePeriod*(nHistoricElements - t));
            else
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = gdv(
                pointer_inputFlattenTensor,
                time,
                u[i],
                delayTimes[nHistoricElements - t + 1],
                i);
            end if;
          end if;
        end if;
      end for;
    end for;

  else
    sampleTrigger = sample(startTime, samplePeriod);

    when {sampleTrigger, initial()} then

      for t in 1:nHistoricElements loop
        for i in 1:nInputs loop
          if flatteningMethod == RNNFlatteningMethod.NewFirstInputSeq then
            if (t == 1) then
              inputFlattenTensor[(t - 1)*nInputs + i] = pre(u[i]);
            else
              inputFlattenTensor[(t - 1)*nInputs + i] = pre(inputFlattenTensor[(t - 2)*nInputs + i]);
            end if;
          elseif flatteningMethod == RNNFlatteningMethod.NewFirstTimeSeq then
            if (t == 1) then
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = pre(u[i]);
            else
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = pre(inputFlattenTensor[(t - 1) + (i - 1)*
                nHistoricElements]);
            end if;
          elseif flatteningMethod == RNNFlatteningMethod.OldFIrstInputSeq then
            if (t == nHistoricElements) then
              inputFlattenTensor[(t - 1)*nInputs + i] = pre(u[i]);
            else
              inputFlattenTensor[(t - 1)*nInputs + i] = pre(inputFlattenTensor[t*nInputs + i]);
            end if;
          elseif flatteningMethod == RNNFlatteningMethod.OldFirstTimeSeq then
            if (t == nHistoricElements) then
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = pre(u[i]);
            else
              inputFlattenTensor[(i - 1)*nHistoricElements + t] = pre(inputFlattenTensor[(i - 1)*nHistoricElements + t +
                1]);
            end if;
          end if;
        end for;
      end for;

    end when;
  end if;

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end RNNFlattenInput;
