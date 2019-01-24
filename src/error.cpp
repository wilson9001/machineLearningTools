// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "error.h"
#include <exception>
#include <signal.h>
#include <sys/stat.h>
#include <string>
#include <iostream>
#include <sstream>
#include <unistd.h>

using std::exception;
using std::string;
using std::cerr;

string g_errorMessage;

class MLException : public exception
{
public:
	virtual const char* what() const throw()
	{
		return g_errorMessage.c_str();
	}
};

MLException g_exception;

void ThrowError(string s1)
{
	g_errorMessage = s1;

	// This is a really good place to put a breakpoint.
	throw g_exception;
}


void ThrowError(string s1, string s2)
{
	string s = s1;
	s += s2;
	ThrowError(s);
}

void ThrowError(string s1, string s2, string s3)
{
	string s = s1;
	s += s2;
	s += s3;
	ThrowError(s);
}

void ThrowError(string s1, string s2, string s3, string s4)
{
	string s = s1;
	s += s2;
	s += s3;
	s += s4;
	ThrowError(s);
}

void ThrowError(string s1, string s2, string s3, string s4, string s5)
{
	string s = s1;
	s += s2;
	s += s3;
	s += s4;
	s += s5;
	ThrowError(s);
}

void ThrowError(string s1, string s2, string s3, string s4, string s5, string s6)
{
	string s = s1;
	s += s2;
	s += s3;
	s += s4;
	s += s5;
	s += s6;
	ThrowError(s);
}

void ThrowError(string s1, string s2, string s3, string s4, string s5, string s6, string s7)
{
	string s = s1;
	s += s2;
	s += s3;
	s += s4;
	s += s5;
	s += s6;
	s += s7;
	ThrowError(s);
}


#ifdef WIN32
void AssertFailed()
{
	cerr << "Debug Assert Failed!\n";
	cerr.flush();
	// __asm int 3 // tell the debugger to break
}
#else
void AssertFailed()
{
	cerr << "Debug Assert Failed!\n";
	cerr.flush();
	kill(getpid(), SIGINT); // tell the debugger to break
}
#endif
