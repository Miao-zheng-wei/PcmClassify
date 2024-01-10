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

	// 自定义删除器，用于智能指针管理PyObject
	struct PyObjectDeleter {
		void operator()(PyObject* ptr) const;
	};

	// 使用自定义删除器的智能指针类型定义
	using PyObjPtr = std::unique_ptr<PyObject, PyObjectDeleter>;

	// 打印Python错误信息的函数
	void printPythonError();

	// 初始化Python环境的函数
	bool initializePythonEnvironment();

	// 导入Python模块的函数
	PyObjPtr importPythonModule(const char* moduleName);

	// 获取Python模块中的函数
	PyObjPtr getPythonFunction(PyObject* pModule, const char* functionName);

	// 创建NumPy数组的函数
	
	//PyObjPtr createNumPyArray(T* data, npy_intp rows, npy_intp cols, int dataType);
	template<typename T>
	PyObjPtr createNumPyArray(T* data, int rows, int cols, int dataType);

	//将C++的std::string转换为Python的PyObjPtr
	PyObjPtr StringToPy(const std::string& p_obj);

	// 将C++的std::string转换为Python的PyObject*的函数
	wchar_t* c2w(const char *c);

	// 将C++的std::string转换为Python的PyObject*
	static PyObject* StringToPy(std::string p_obj);
}

