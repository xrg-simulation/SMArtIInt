#ifndef ModelicaUtilityHelper_h
#define ModelicaUtilityHelper_h

typedef struct
{
	void (*ModelicaError)(const char*);
	void (*ModelicaMessage)(const char*);
} ModelicaUtilityHelper;

#endif

