// PythonUtil.cpp
#include "PythonUtil.h"
namespace PythonUtil {

	// 自定义删除器的实现
	void PyObjectDeleter::operator()(PyObject* ptr) const {
		Py_XDECREF(ptr); // 安全地减少引用计数并可能释放对象
	}

	// 打印Python错误信息
	void printPythonError() {
		PyErr_Print(); // 打印Python的错误信息
		PyErr_Clear(); // 清除错误标志
	}

	//将C++的std::string转换为Python的PyObjPtr
	PyObjPtr StringToPy(const std::string& p_obj) {
		// 计算所需的宽字符数，包括空字符
		int wlen = ::MultiByteToWideChar(CP_UTF8, 0, p_obj.c_str(), -1, NULL, 0);
		if (wlen == 0) {
			// 转换失败，打印错误并返回nullptr
			printPythonError();
			return nullptr;
		}

		// 分配宽字符数组
		std::vector<wchar_t> wszString(wlen);
		// 执行转换
		if (::MultiByteToWideChar(CP_UTF8, 0, p_obj.c_str(), -1, &wszString[0], wlen) == 0) {
			// 转换失败，打印错误并返回nullptr
			printPythonError();
			return nullptr;
		}

		// 创建Python Unicode对象，注意-1是因为不需要复制最后的空字符
		PyObject* pyString = PyUnicode_FromWideChar(&wszString[0], wlen - 1);
		if (!pyString) {
			// 创建失败，打印错误并返回nullptr
			printPythonError();
			return nullptr;
		}

		return PyObjPtr(pyString); // 返回智能指针管理的Python Unicode对象
	}

	// 初始化Python环境
	bool initializePythonEnvironment() {
		Py_Initialize(); // 初始化Python
		if (!Py_IsInitialized()) {
			std::cerr << "Python environment initialization failed." << std::endl;
			return false;
		}
		import_array(); // 对于使用NumPy C API的程序，这是必须调用的宏
		return true;
	}

	// 导入Python模块
	PyObjPtr importPythonModule(const char* moduleName) {
		PyObject* pName = PyUnicode_DecodeFSDefault(moduleName); // 将C字符串转换为Python字符串
		if (!pName) {
			printPythonError();
			return nullptr;
		}
		PyObjPtr module(PyImport_Import(pName)); // 导入模块
		Py_DECREF(pName); // 减少引用计数
		if (!module) {
			std::cerr << "Failed to import Python module. " << moduleName << std::endl;
			printPythonError();
		}
		return module;
	}

	// 获取Python模块中的函数
	PyObjPtr getPythonFunction(PyObject* pModule, const char* functionName) {
		if (!pModule) return nullptr;
		PyObject* pFunc = PyObject_GetAttrString(pModule, functionName); // 获取属性
		if (!pFunc || !PyCallable_Check(pFunc)) { // 检查是否为可调用对象
			std::cerr << "Failed to import Python function. " << functionName << std::endl;
			printPythonError();
			Py_XDECREF(pFunc); // 确保释放非可调用或获取失败的对象
			return nullptr;
		}
		return PyObjPtr(pFunc); // 自动使用PyObjectDeleter
	}

	// 创建NumPy数组
	template<typename T>
	PyObjPtr createNumPyArray(T* data, int rows, int cols, int dataType) {
		npy_intp dims[2] = { static_cast<npy_intp>(rows), static_cast<npy_intp>(cols) }; // 定义数组维度
		PyObject* pArray = PyArray_SimpleNewFromData(2, dims, dataType, static_cast<void*>(data)); // 从数据创建NumPy数组
		if (!pArray) return nullptr;
		return PyObjPtr(pArray); // 返回智能指针管理的NumPy数组
	}


	//实例化
	template PyObjPtr PythonUtil::createNumPyArray<float>(float*, int, int, int);
	template PyObjPtr PythonUtil::createNumPyArray<int>(int*, int, int, int);
	template PyObjPtr PythonUtil::createNumPyArray<bool>(bool*, int, int, int);


	// 将c 风格字符串转换为宽字符串
	 wchar_t* c2w(const char *c)
	{
		const size_t cSize = strlen(c) + 1;
		wchar_t* wc = new wchar_t[cSize];
		size_t outSize;
		mbstowcs_s(&outSize, wc, cSize, c, cSize - 1);
		return wc;
	}

}


