#ifndef ModelicaUtilityInterface_C
#define ModelicaUtilityInterface_C

#include "ModelicaUtilityHelper.h"
#include "ModelicaUtilities.h"

static ModelicaUtilityHelper s_ModelicaUtilityHelper;

void* createModelicaUtitlityHelper()
{
	s_ModelicaUtilityHelper.ModelicaError = ModelicaError;
	s_ModelicaUtilityHelper.ModelicaMessage = ModelicaMessage;
	return &s_ModelicaUtilityHelper;
}

void deleteModelicaUtitlityHelper(void* externalObject)
{

}

#endif