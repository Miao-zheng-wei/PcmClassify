// PythonUtil.h
#pragma once

#include <memory>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>     
#include <Windows.h>  

#include <cstring>  // for strlen
#include <cstdlib>  // for mbstowcs
#include <cwchar>   // for wchar_t and mbstowcs in C++

namespace PythonUtil {

	// �Զ���ɾ��������������ָ�����PyObject
	struct PyObjectDeleter {
		void operator()(PyObject* ptr) const;
	};

	// ʹ���Զ���ɾ����������ָ�����Ͷ���
	using PyObjPtr = std::unique_ptr<PyObject, PyObjectDeleter>;

	// ��ӡPython������Ϣ�ĺ���
	void printPythonError();

	// ��ʼ��Python�����ĺ���
	bool initializePythonEnvironment();

	// ����Pythonģ��ĺ���
	PyObjPtr importPythonModule(const char* moduleName);

	// ��ȡPythonģ���еĺ���
	PyObjPtr getPythonFunction(PyObject* pModule, const char* functionName);

	// ����NumPy����ĺ���
	
	//PyObjPtr createNumPyArray(T* data, npy_intp rows, npy_intp cols, int dataType);
	template<typename T>
	PyObjPtr createNumPyArray(T* data, int rows, int cols, int dataType);

	//��C++��std::stringת��ΪPython��PyObjPtr
	PyObjPtr StringToPy(const std::string& p_obj);

	// ��C++��std::stringת��ΪPython��PyObject*�ĺ���
	wchar_t* c2w(const char *c);

	// ��C++��std::stringת��ΪPython��PyObject*
	static PyObject* StringToPy(std::string p_obj);
}

