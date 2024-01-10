// PythonUtil.cpp
#include "PythonUtil.h"
namespace PythonUtil {

	// �Զ���ɾ������ʵ��
	void PyObjectDeleter::operator()(PyObject* ptr) const {
		Py_XDECREF(ptr); // ��ȫ�ؼ������ü����������ͷŶ���
	}

	// ��ӡPython������Ϣ
	void printPythonError() {
		PyErr_Print(); // ��ӡPython�Ĵ�����Ϣ
		PyErr_Clear(); // ��������־
	}

	//��C++��std::stringת��ΪPython��PyObjPtr
	PyObjPtr StringToPy(const std::string& p_obj) {
		// ��������Ŀ��ַ������������ַ�
		int wlen = ::MultiByteToWideChar(CP_UTF8, 0, p_obj.c_str(), -1, NULL, 0);
		if (wlen == 0) {
			// ת��ʧ�ܣ���ӡ���󲢷���nullptr
			printPythonError();
			return nullptr;
		}

		// ������ַ�����
		std::vector<wchar_t> wszString(wlen);
		// ִ��ת��
		if (::MultiByteToWideChar(CP_UTF8, 0, p_obj.c_str(), -1, &wszString[0], wlen) == 0) {
			// ת��ʧ�ܣ���ӡ���󲢷���nullptr
			printPythonError();
			return nullptr;
		}

		// ����Python Unicode����ע��-1����Ϊ����Ҫ�������Ŀ��ַ�
		PyObject* pyString = PyUnicode_FromWideChar(&wszString[0], wlen - 1);
		if (!pyString) {
			// ����ʧ�ܣ���ӡ���󲢷���nullptr
			printPythonError();
			return nullptr;
		}

		return PyObjPtr(pyString); // ��������ָ������Python Unicode����
	}

	// ��ʼ��Python����
	bool initializePythonEnvironment() {
		Py_Initialize(); // ��ʼ��Python
		if (!Py_IsInitialized()) {
			std::cerr << "Python environment initialization failed." << std::endl;
			return false;
		}
		import_array(); // ����ʹ��NumPy C API�ĳ������Ǳ�����õĺ�
		return true;
	}

	// ����Pythonģ��
	PyObjPtr importPythonModule(const char* moduleName) {
		PyObject* pName = PyUnicode_DecodeFSDefault(moduleName); // ��C�ַ���ת��ΪPython�ַ���
		if (!pName) {
			printPythonError();
			return nullptr;
		}
		PyObjPtr module(PyImport_Import(pName)); // ����ģ��
		Py_DECREF(pName); // �������ü���
		if (!module) {
			std::cerr << "Failed to import Python module. " << moduleName << std::endl;
			printPythonError();
		}
		return module;
	}

	// ��ȡPythonģ���еĺ���
	PyObjPtr getPythonFunction(PyObject* pModule, const char* functionName) {
		if (!pModule) return nullptr;
		PyObject* pFunc = PyObject_GetAttrString(pModule, functionName); // ��ȡ����
		if (!pFunc || !PyCallable_Check(pFunc)) { // ����Ƿ�Ϊ�ɵ��ö���
			std::cerr << "Failed to import Python function. " << functionName << std::endl;
			printPythonError();
			Py_XDECREF(pFunc); // ȷ���ͷŷǿɵ��û��ȡʧ�ܵĶ���
			return nullptr;
		}
		return PyObjPtr(pFunc); // �Զ�ʹ��PyObjectDeleter
	}

	// ����NumPy����
	template<typename T>
	PyObjPtr createNumPyArray(T* data, int rows, int cols, int dataType) {
		npy_intp dims[2] = { static_cast<npy_intp>(rows), static_cast<npy_intp>(cols) }; // ��������ά��
		PyObject* pArray = PyArray_SimpleNewFromData(2, dims, dataType, static_cast<void*>(data)); // �����ݴ���NumPy����
		if (!pArray) return nullptr;
		return PyObjPtr(pArray); // ��������ָ������NumPy����
	}


	//ʵ����
	template PyObjPtr PythonUtil::createNumPyArray<float>(float*, int, int, int);
	template PyObjPtr PythonUtil::createNumPyArray<int>(int*, int, int, int);
	template PyObjPtr PythonUtil::createNumPyArray<bool>(bool*, int, int, int);


	// ��c ����ַ���ת��Ϊ���ַ���
	 wchar_t* c2w(const char *c)
	{
		const size_t cSize = strlen(c) + 1;
		wchar_t* wc = new wchar_t[cSize];
		size_t outSize;
		mbstowcs_s(&outSize, wc, cSize, c, cSize - 1);
		return wc;
	}

}


