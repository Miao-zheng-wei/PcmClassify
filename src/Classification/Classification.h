#pragma once
#pragma warning(disable:4996)

//local
#include "scFeatureCombinerCDHVFast.h"
#include "pcMlClassifyApi.h"
#include "PythonUtil.h"
#include "MlCommon.h"

//外部接口

//python
#include <Python.h>

//system
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <tchar.h>

namespace  Classify
{
	using namespace std;
	struct Classification
	{
	public:
		Classification(std::string pyPath = "./python310");
		~Classification();

		//! 随机森林训练分类器
		template <class PointT>
		bool trainClassifier_RF(RFTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! 神经网络训练分类器
		template <class PointT>
		bool trainClassifier_NN(NNTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! LG训练分类器
		template <class PointT>
		bool trainClassifier_LG(LGTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! 计算特征
		template <class PointT>
		bool getFeature(sc::FeatureMatrix &featMat, std::vector<PointT> *data, GenericProgressCallback *pdlg = 0);

		//! 执行分类代码
		template <class PointT>
		bool runPythonClassification(TestParam &param, TestResult &result, char* errorMes, GenericProgressCallback *pdlg = 0);

		//! 加载已有训练模型
		template <class PointT>
		bool loadTrainingModel(std::string modelPath, PyObject **model, char* errorMes, GenericProgressCallback* _pdlg = 0);

		//! 输出特征
		template <class PointT>
		bool writeFeature(std::vector<PointT> *pc, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector,
			float *featureArr, std::string filename, bool isTrain, int *labelArr = nullptr, GenericProgressCallback *pdlg = 0);

		//! 将体素标签转换为每个点的标签
		template <class PointT>
		void voxelLabel2PointLabel(int *voxelLabelArr, int *pointLabelArr, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *pdlg = 0);

		//! 体素化
		template <class PointT>
		void getVoxelPoints(float voxelsize, std::vector<PointT> *pointVector, std::vector<PointT> &m_dilutedPoints, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *pdlg = 0);

	public:

		//! 测试参数
		TestParam m_testParam;

		//! 特征数目
		int m_featureNum = FEA_NUM;
	};

	static PyObject* StringToPy(std::string p_obj);
}

namespace Classify
{
	Classification::Classification(std::string pyPath)
	{
		std::cout << pyPath << std::endl;
		Py_SetPythonHome(PythonUtil::c2w(pyPath.c_str()));
		Py_Initialize();
	}

	Classification::~Classification()
	{
	}

	// 随机森林分类器
	template <class PointT>
	bool Classification::trainClassifier_RF(RFTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);

		//对于使用NumPy C API的程序，这是必须调用的宏
		import_array();

		// 初始化Python环境
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.！");
			return false;
		}

		// 导入Python模块
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.！");
			return false;
		}

		// 导入python 函数
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "RF");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.！");
			return false;
		}

		if (!np.oneStep())return false;

		// 初始化训练结果数组
		std::fill_n(param->_TraincheckedResults, 16, true);

		npy_intp Dims_x[1]    = { param->_featArraySize };       //给定维度信息
		npy_intp Dims_y[1]    = { param->_labelArraySize };      //给定维度信息
		npy_intp Dims_bool[1] = { param->_checkedArraySize }; //给定维度信息

		// 创建numpy 数组
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// 将参数转化为python 对象
		PyObject* pyeval           = StringToPy(OutPath + "_eval.txt");
		PyObject* pymodel          = StringToPy(OutPath + "_RFmodel.pkl");
		PyObject* pyFeatureNum     = Py_BuildValue("i", param->_featureNum);
		PyObject* pyIsEval         = Py_BuildValue("i", static_cast<int>(param->_isEval));
		PyObject* pyIsModel        = Py_BuildValue("i", static_cast<int>(param->_isModel));
		PyObject* pyRFtreeMaxDepth = Py_BuildValue("i", param->_RFtreeMaxDepth);
		PyObject* pyRFtreenum      = Py_BuildValue("i", param->_RFtreenum);
		PyObject* pyRFcriterion    = Py_BuildValue("s", param->_RFcriterionId == 1 ? "entropy" : "gini");

		// 设置参数
		PythonUtil::PyObjPtr args(PyTuple_New(11));
		PyTuple_SetItem(args.get(), 0, pyFeatureArray);
		PyTuple_SetItem(args.get(), 1, pyLabelArray);
		PyTuple_SetItem(args.get(), 2, pyeval);
		PyTuple_SetItem(args.get(), 3, pymodel);
		PyTuple_SetItem(args.get(), 4, pyRFcriterion);
		PyTuple_SetItem(args.get(), 5, pyRFtreenum);
		PyTuple_SetItem(args.get(), 6, pyRFtreeMaxDepth);
		PyTuple_SetItem(args.get(), 7, pyIsEval);
		PyTuple_SetItem(args.get(), 8, pyIsModel);
		PyTuple_SetItem(args.get(), 9, pyFeatureNum);
		PyTuple_SetItem(args.get(), 10, pyCheckedArray);

		if (!np.oneStep())return false;

		//调用函数
		PythonUtil::PyObjPtr resultModel(PyObject_CallObject(pFun.get(), args.get()));
		if (!resultModel) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}

		if (!np.oneStep())return false;
		return true;
	}

	//神经网络分类器
	template <class PointT>
	bool Classification::trainClassifier_NN(NNTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);
		//对于使用NumPy C API的程序，这是必须调用的宏
		import_array();

		// 初始化Python环境
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.！");
			return false;
		}

		// 导入Python模块
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.！");
			return false;
		}

		// 导入python 函数
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "NN");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.！");
			return false;
		}
		if (!np.oneStep())return false;

		// 初始化训练结果数组
		std::fill_n(param->_TraincheckedResults, 16, true);

		// 创建numpy 数组
		npy_intp Dims_x[1]       = { param->_featArraySize };       //给定维度信息
		npy_intp Dims_y[1]       = { param->_labelArraySize };      //给定维度信息
		npy_intp Dims_bool[1]    = { param->_checkedArraySize }; //给定维度信息
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// 将参数转化为python 对象
		PyObject* pyeval        = StringToPy(OutPath + "_eval.txt");
		PyObject* pymodel       = StringToPy(OutPath + "_NNmodel.pkl");
		PyObject* pyFeatureNum  = Py_BuildValue("i", param->_featureNum);
		PyObject* pyIsEval      = Py_BuildValue("i", static_cast<int>(param->_isEval));
		PyObject* pyIsModel     = Py_BuildValue("i", static_cast<int>(param->_isModel));
		PyObject* pyNNactiveFun = Py_BuildValue("s", param->_NNactiveFun);
		PyObject* pyNNoptiFun   = Py_BuildValue("s", param->_NNoptiFun);
		PyObject* pyNNlr        = Py_BuildValue("f", param->_NNlr);
		PyObject* pyNNiter      = Py_BuildValue("i", param->_NNiter);

		PythonUtil::PyObjPtr args(PyTuple_New(12));
		PyTuple_SetItem(args.get(), 0, pyFeatureArray);
		PyTuple_SetItem(args.get(), 1, pyLabelArray);
		PyTuple_SetItem(args.get(), 2, pyeval);
		PyTuple_SetItem(args.get(), 3, pymodel);
		PyTuple_SetItem(args.get(), 4, pyNNactiveFun);
		PyTuple_SetItem(args.get(), 5, pyNNoptiFun);
		PyTuple_SetItem(args.get(), 6, pyNNlr);
		PyTuple_SetItem(args.get(), 7, pyNNiter);
		PyTuple_SetItem(args.get(), 8, pyIsEval);
		PyTuple_SetItem(args.get(), 9, pyIsModel);
		PyTuple_SetItem(args.get(), 10, pyFeatureNum);
		PyTuple_SetItem(args.get(), 11, pyCheckedArray);

		if (!np.oneStep())return false;

		//调用函数
		PythonUtil::PyObjPtr resultModel(PyObject_CallObject(pFun.get(), args.get()));
		if (!resultModel) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}

		// 存储结果
		if (!np.oneStep())return false;
		return true;
	}

	// LG分类器
	template <class PointT>
	bool Classification::trainClassifier_LG(LGTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);
		//对于使用NumPy C API的程序，这是必须调用的宏
		import_array();

		// 初始化Python环境
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.！");
			return false;
		}

		// 导入Python模块
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.！");
			return false;
		}

		// 导入python 函数
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "LG");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.！");
			return false;
		}
		if (!np.oneStep())return false;

		// 初始化训练结果数组
		std::fill_n(param->_TraincheckedResults, 16, true);

		// 创建numpy 数组
		npy_intp Dims_x[1]       = { param->_featArraySize };       //给定维度信息
		npy_intp Dims_y[1]       = { param->_labelArraySize };      //给定维度信息
		npy_intp Dims_bool[1]    = { param->_checkedArraySize }; //给定维度信息
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// 将参数转化为python 对象
		PyObject* pyeval           = StringToPy(OutPath + "_eval.txt");
		PyObject* pymodel          = StringToPy(OutPath + "_LGmodel.pkl");
		PyObject* pyFeatureNum     = Py_BuildValue("i", param->_featureNum);
		PyObject* pyIsEval         = Py_BuildValue("i", static_cast<int>(param->_isEval));
		PyObject* pyIsModel        = Py_BuildValue("i", static_cast<int>(param->_isModel));
		PyObject* pyXGtreeMaxDepth = Py_BuildValue("i", param->_LGtreeMaxDepth);
		PyObject* pyXGtreenum      = Py_BuildValue("i", param->_LGtreenum);
		PyObject* pyXGlr           = Py_BuildValue("f", param->_LGlr);
		PyObject* pyXGnumClass     = Py_BuildValue("i", param->_LGnumClass);
		PyObject* pyLGleavesNum    = Py_BuildValue("i", param->_LGleavesNum);

		PythonUtil::PyObjPtr args(PyTuple_New(13));
		PyTuple_SetItem(args.get(), 0, pyFeatureArray);
		PyTuple_SetItem(args.get(), 1, pyLabelArray);
		PyTuple_SetItem(args.get(), 2, pyeval);
		PyTuple_SetItem(args.get(), 3, pymodel);
		PyTuple_SetItem(args.get(), 4, pyXGtreeMaxDepth);
		PyTuple_SetItem(args.get(), 5, pyXGtreenum);
		PyTuple_SetItem(args.get(), 6, pyXGlr);
		PyTuple_SetItem(args.get(), 7, pyLGleavesNum);
		PyTuple_SetItem(args.get(), 8, pyIsEval);
		PyTuple_SetItem(args.get(), 9, pyIsModel);
		PyTuple_SetItem(args.get(), 10, pyFeatureNum);
		PyTuple_SetItem(args.get(), 11, pyXGnumClass);
		PyTuple_SetItem(args.get(), 12, pyCheckedArray);

		if (!np.oneStep())return false;

		//调用函数
		PythonUtil::PyObjPtr resultModel(PyObject_CallObject(pFun.get(), args.get()));
		if (!resultModel) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}
		if (!np.oneStep())return false;
		return true;
	}

	//
	template <class PointT>
	bool Classification::runPythonClassification(TestParam &param, TestResult &result, char* errorMes, GenericProgressCallback *pdlg) {
		NormalizedProgress np(pdlg, 4);
		import_array(); // This is necessary for NumPy
		// 初始化Python环境
		if (!PythonUtil::initializePythonEnvironment()) {
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.！");
			return false;
		}

		// 导入Python模块
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.！");
			return false;
		}

		// 获取python函数
		PythonUtil::PyObjPtr pFunc = PythonUtil::getPythonFunction(pModule.get(), "testing");
		if (!pFunc) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.！");
			return false;
		}

		// 体素化
		std::vector<PointT> pc;
		std::vector<cloud_point_index_idx> m_index_vector;
		std::vector<std::pair<unsigned int, unsigned int>> m_first_and_last_indices_vector;
		getVoxelPoints(param._voxelSize, param._initLabelPointCloud, pc, m_index_vector, m_first_and_last_indices_vector);
		if (!np.oneStep()) return false;

		// 特征提取
		float *ptr_test = nullptr;
		try {
			ptr_test = new float[pc.size() * m_featureNum];
		}
		catch (const std::bad_alloc&) {
			return false;
		}

		sc::FeatureMatrix featMat(ptr_test, pc.size(), m_featureNum); // 存测试集特征
		bool isFeature = getFeature(featMat, &pc, 0);
		if (!isFeature) {
			if (errorMes)
				strcpy(errorMes, "特征计算失败！");
			delete[] ptr_test;
			return false;
		}

		// 设置参数
		npy_intp Dims_x[1]        = { featMat.rows*featMat.cols }; //给定维度信息
		PyObject * pyFeatureNum   = Py_BuildValue("i", param._featureNum);
		PyObject * model          = param._model;
		PyObject * pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, ptr_test);

		// Ensure that the numpy array does not own the data
		PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(pyFeatureArray), NPY_ARRAY_OWNDATA);

		PythonUtil::PyObjPtr args(PyTuple_New(3));
		PyTuple_SetItem(args.get(), 0, pyFeatureArray);
		PyTuple_SetItem(args.get(), 1, model);
		PyTuple_SetItem(args.get(), 2, pyFeatureNum);

		//调用函数
		PythonUtil::PyObjPtr py_y_pred(PyObject_CallObject(pFunc.get(), args.get()));
		if (!py_y_pred) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
			delete[] ptr_test;
			return false;
		}

		if (!np.oneStep()) {
			delete[] ptr_test;
			return false;
		}

		// 处理分类结果
		PyArrayObject *pyResultArr = reinterpret_cast<PyArrayObject *>(py_y_pred.get());
		std::unique_ptr<int[]> labelArr(new int[param._initLabelPointCloud->size()]);
		int *voxelArr              = static_cast<int*>(PyArray_DATA(pyResultArr));
		voxelLabel2PointLabel< pcc::LASPoint>(voxelArr, labelArr.get(), m_index_vector, m_first_and_last_indices_vector);
		result._labels             = reinterpret_cast<unsigned long*>(labelArr.release());

		if (!np.oneStep()) {
			delete[] ptr_test;
			delete[] reinterpret_cast<int*>(result._labels);
			return false;
		}

		// 写入特征文件
		if (param._isFeature) {
			std::string featureFilePath = param._path + "TestFeature.txt";
			bool isWritten              = writeFeature(param._initLabelPointCloud, m_index_vector,
				m_first_and_last_indices_vector,
				ptr_test, featureFilePath, false, labelArr.get());
			if (!isWritten) {
				delete[] ptr_test;
				delete[] reinterpret_cast<int*>(result._labels);
				return false;
			}
		}

		// 清理内存
		delete[] ptr_test;
		std::vector<PointT>().swap(pc);
		std::vector<std::pair<unsigned int, unsigned int>>().swap(m_first_and_last_indices_vector);
		std::vector<cloud_point_index_idx>().swap(m_index_vector);

		return true;
	}

	//计算点云特征
	template <class PointT>
	bool Classification::getFeature(sc::FeatureMatrix &featMat, std::vector<PointT> *data, GenericProgressCallback* _pdlg)
	{
		sc::FeatureCombinerCDHVFast <PointT> featureCombiner;
		featureCombiner.setInputCloud(data);

		featureCombiner.buildSpatialIndexImpl();
		bool isFeature = featureCombiner.combineFeaturesImpl(featMat, _pdlg);
		return isFeature;
	}

	// 获取训练模型
	template <class PointT>
	bool Classification::loadTrainingModel(std::string modelPath, PyObject **model, char *errorMes, GenericProgressCallback* _pdlg)
	{
		// 使用NormalizedProgress来跟踪进度
		NormalizedProgress np(_pdlg, 4);

		// 确保Python环境已初始化
		if (!PythonUtil::initializePythonEnvironment()) {
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.！");
			return false;
		}
		if (!np.oneStep())return false;

		// 导入Python模块
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.！");
			return false;
		}
		if (!np.oneStep())return false;

		// 获取python 函数
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "loadModel");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.！");
			return false;
		}
		if (!np.oneStep())return false;

		// 设置参数
		PythonUtil::PyObjPtr args(PyTuple_New(1));
		PythonUtil::PyObjPtr arg_model(StringToPy(modelPath));
		PyTuple_SetItem(args.get(), 0, arg_model.release()); // 传递所有权给元组

		// 调用Python函数
		PythonUtil::PyObjPtr result(PyObject_CallObject(pFun.get(), args.get()));
		if (!result) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}
		if (!np.oneStep())return false;

		// 设置返回的模型对象
		*model = result.release(); // 传递所有权给调用者

		return true;
	}

	//写出特征值
	template <class PointT>
	bool  Classification::writeFeature(std::vector<PointT> *pc, std::vector<cloud_point_index_idx> &index_vector, std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, float *featureArr, std::string filename, bool isTrain, int *labelArr, GenericProgressCallback *_pdlg) // 默认nullptr，表示没有传递Progress回调函数
	{
		unsigned int first_index, last_index;
		//创建一个文件输出流，以写入模式打开文件
		std::ofstream fin(filename.c_str(), std::ios::out | std::ios::trunc);

		if (!fin)
		{
			return false;
		}

		//设置输出时的数值精度，保留4位小数
		fin.precision(4);

		//设置浮点数的格式为固定，即不使用科学记数法的形式
		fin.setf(std::ios::fixed, std::ios::floatfield);

		//创建一个进度指示器对象
		NormalizedProgress np(_pdlg, first_and_last_indices_vector.size());

		//当isTrain为true，需要输出label
		if (isTrain)
		{
			//输出当前向量的大小
			fin << first_and_last_indices_vector.size() << "\n";

			//遍历first_and_last_indices_vector
			for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
			{
				//每遍历一步，更新一次进度条
				if (!np.oneStep())return false;

				//获取当前元素的第一个索引值
				first_index = first_and_last_indices_vector[cp].first;

				//根据索引值，获取对应的点的索引
				int idx = index_vector[first_index].cloud_point_index;

				//输出格式为： x y z 标签 特征值
				fin << (*pc)[idx].x << " " << (*pc)[idx].y << " " << (*pc)[idx].z << " " << labelArr[idx] << " ";

				//输出所有特征值
				for (int i = 0; i < m_featureNum; i++)
				{
					fin << featureArr[idx*m_featureNum + i] << " ";
				}
				fin << "\n";
			}
		}
		else //isTrain为false，不需要输出label
		{
			//输出点云的数量
			fin << pc->size() << "\n";

			for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
			{
				if (!np.oneStep())return false;

				first_index = first_and_last_indices_vector[cp].first;
				last_index  = first_and_last_indices_vector[cp].second;

				//遍历所有的点
				for (unsigned int li = first_index; li < last_index; ++li)
				{
					int idx = index_vector[li].cloud_point_index;

					//输出格式为： x y z 特征值
					fin << (*pc)[idx].x << " " << (*pc)[idx].y << " " << (*pc)[idx].z << " ";

					//输出所有特征值
					for (int i = 0; i < m_featureNum; i++)
					{
						fin << featureArr[cp*m_featureNum + i] << " ";
					}
					fin << "\n";
				}
			}
		}

		//关闭文件流
		fin.close();

		return true;
	}

	template <class PointT>
	void Classification::voxelLabel2PointLabel(int *voxelLabelArr, int *pointLabelArr, std::vector<cloud_point_index_idx> &index_vector, std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *_pdlg)
	{
		unsigned int first_index = 0;
		unsigned int last_index = 0;
		int idx = 0;

		NormalizedProgress np(_pdlg, first_and_last_indices_vector.size());
		for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
		{
			first_index = first_and_last_indices_vector[cp].first;
			last_index  = first_and_last_indices_vector[cp].second;

			for (unsigned int li = first_index; li < last_index; ++li)
			{
				idx = index_vector[li].cloud_point_index;
				pointLabelArr[idx] = voxelLabelArr[cp];
			}
			np.oneStep();
		}
	}

	//体素化
	template<typename PointT>
	void Classification::getVoxelPoints(float voxelsize, std::vector<PointT> *pointVector, std::vector<PointT> &m_dilutedPoints, std::vector<cloud_point_index_idx> &index_vector, std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *_pdlg)
	{
		VoxelGrid<PointT>vf;
		vf.setInputCloud(*pointVector);

		vf.setLeafSize(voxelsize, voxelsize, voxelsize);
		vf.VoxelGrid_ApplyFilter(m_dilutedPoints, index_vector, first_and_last_indices_vector, _pdlg);
	}

	static PyObject* StringToPy(std::string p_obj)
	{
		int wlen = ::MultiByteToWideChar(CP_ACP, NULL, p_obj.c_str(), int(p_obj.size()), NULL, 0);
		wchar_t* wszString = new wchar_t[wlen + 1];
		::MultiByteToWideChar(CP_ACP, NULL, p_obj.c_str(), int(p_obj.size()), wszString, wlen);
		wszString[wlen] = '\0';
		PyObject* pb = PyUnicode_FromUnicode((const Py_UNICODE*)wszString, wlen);
		delete wszString;
		return pb;
	}
}
