#pragma once
#pragma warning(disable:4996)

//local
#include "scFeatureCombinerCDHVFast.h"
#include "pcMlClassifyApi.h"
#include "PythonUtil.h"
#include "MlCommon.h"

//�ⲿ�ӿ�

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

		//! ���ɭ��ѵ��������
		template <class PointT>
		bool trainClassifier_RF(RFTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! ������ѵ��������
		template <class PointT>
		bool trainClassifier_NN(NNTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! LGѵ��������
		template <class PointT>
		bool trainClassifier_LG(LGTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg = 0, char* errorMes = 0);

		//! ��������
		template <class PointT>
		bool getFeature(sc::FeatureMatrix &featMat, std::vector<PointT> *data, GenericProgressCallback *pdlg = 0);

		//! ִ�з������
		template <class PointT>
		bool runPythonClassification(TestParam &param, TestResult &result, char* errorMes, GenericProgressCallback *pdlg = 0);

		//! ��������ѵ��ģ��
		template <class PointT>
		bool loadTrainingModel(std::string modelPath, PyObject **model, char* errorMes, GenericProgressCallback* _pdlg = 0);

		//! �������
		template <class PointT>
		bool writeFeature(std::vector<PointT> *pc, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector,
			float *featureArr, std::string filename, bool isTrain, int *labelArr = nullptr, GenericProgressCallback *pdlg = 0);

		//! �����ر�ǩת��Ϊÿ����ı�ǩ
		template <class PointT>
		void voxelLabel2PointLabel(int *voxelLabelArr, int *pointLabelArr, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *pdlg = 0);

		//! ���ػ�
		template <class PointT>
		void getVoxelPoints(float voxelsize, std::vector<PointT> *pointVector, std::vector<PointT> &m_dilutedPoints, std::vector<cloud_point_index_idx> &index_vector,
			std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, GenericProgressCallback *pdlg = 0);

	public:

		//! ���Բ���
		TestParam m_testParam;

		//! ������Ŀ
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

	// ���ɭ�ַ�����
	template <class PointT>
	bool Classification::trainClassifier_RF(RFTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);

		//����ʹ��NumPy C API�ĳ������Ǳ�����õĺ�
		import_array();

		// ��ʼ��Python����
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.��");
			return false;
		}

		// ����Pythonģ��
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.��");
			return false;
		}

		// ����python ����
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "RF");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.��");
			return false;
		}

		if (!np.oneStep())return false;

		// ��ʼ��ѵ���������
		std::fill_n(param->_TraincheckedResults, 16, true);

		npy_intp Dims_x[1]    = { param->_featArraySize };       //����ά����Ϣ
		npy_intp Dims_y[1]    = { param->_labelArraySize };      //����ά����Ϣ
		npy_intp Dims_bool[1] = { param->_checkedArraySize }; //����ά����Ϣ

		// ����numpy ����
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// ������ת��Ϊpython ����
		PyObject* pyeval           = StringToPy(OutPath + "_eval.txt");
		PyObject* pymodel          = StringToPy(OutPath + "_RFmodel.pkl");
		PyObject* pyFeatureNum     = Py_BuildValue("i", param->_featureNum);
		PyObject* pyIsEval         = Py_BuildValue("i", static_cast<int>(param->_isEval));
		PyObject* pyIsModel        = Py_BuildValue("i", static_cast<int>(param->_isModel));
		PyObject* pyRFtreeMaxDepth = Py_BuildValue("i", param->_RFtreeMaxDepth);
		PyObject* pyRFtreenum      = Py_BuildValue("i", param->_RFtreenum);
		PyObject* pyRFcriterion    = Py_BuildValue("s", param->_RFcriterionId == 1 ? "entropy" : "gini");

		// ���ò���
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

		//���ú���
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

	//�����������
	template <class PointT>
	bool Classification::trainClassifier_NN(NNTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);
		//����ʹ��NumPy C API�ĳ������Ǳ�����õĺ�
		import_array();

		// ��ʼ��Python����
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.��");
			return false;
		}

		// ����Pythonģ��
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.��");
			return false;
		}

		// ����python ����
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "NN");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.��");
			return false;
		}
		if (!np.oneStep())return false;

		// ��ʼ��ѵ���������
		std::fill_n(param->_TraincheckedResults, 16, true);

		// ����numpy ����
		npy_intp Dims_x[1]       = { param->_featArraySize };       //����ά����Ϣ
		npy_intp Dims_y[1]       = { param->_labelArraySize };      //����ά����Ϣ
		npy_intp Dims_bool[1]    = { param->_checkedArraySize }; //����ά����Ϣ
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// ������ת��Ϊpython ����
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

		//���ú���
		PythonUtil::PyObjPtr resultModel(PyObject_CallObject(pFun.get(), args.get()));
		if (!resultModel) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}

		// �洢���
		if (!np.oneStep())return false;
		return true;
	}

	// LG������
	template <class PointT>
	bool Classification::trainClassifier_LG(LGTrainParam *param, std::string &OutPath, GenericProgressCallback *pdlg, char* errorMes) {
		NormalizedProgress np(pdlg, 3);
		//����ʹ��NumPy C API�ĳ������Ǳ�����õĺ�
		import_array();

		// ��ʼ��Python����
		if (!PythonUtil::initializePythonEnvironment())
		{
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.��");
			return false;
		}

		// ����Pythonģ��
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.��");
			return false;
		}

		// ����python ����
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "LG");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.��");
			return false;
		}
		if (!np.oneStep())return false;

		// ��ʼ��ѵ���������
		std::fill_n(param->_TraincheckedResults, 16, true);

		// ����numpy ����
		npy_intp Dims_x[1]       = { param->_featArraySize };       //����ά����Ϣ
		npy_intp Dims_y[1]       = { param->_labelArraySize };      //����ά����Ϣ
		npy_intp Dims_bool[1]    = { param->_checkedArraySize }; //����ά����Ϣ
		PyObject* pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, param->_featureArr);
		PyObject* pyLabelArray   = PyArray_SimpleNewFromData(1, Dims_y, NPY_INT, param->_labelArr);
		PyObject* pyCheckedArray = PyArray_SimpleNewFromData(1, Dims_bool, NPY_BOOL, param->_TraincheckedResults);

		// ������ת��Ϊpython ����
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

		//���ú���
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
		// ��ʼ��Python����
		if (!PythonUtil::initializePythonEnvironment()) {
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.��");
			return false;
		}

		// ����Pythonģ��
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.��");
			return false;
		}

		// ��ȡpython����
		PythonUtil::PyObjPtr pFunc = PythonUtil::getPythonFunction(pModule.get(), "testing");
		if (!pFunc) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.��");
			return false;
		}

		// ���ػ�
		std::vector<PointT> pc;
		std::vector<cloud_point_index_idx> m_index_vector;
		std::vector<std::pair<unsigned int, unsigned int>> m_first_and_last_indices_vector;
		getVoxelPoints(param._voxelSize, param._initLabelPointCloud, pc, m_index_vector, m_first_and_last_indices_vector);
		if (!np.oneStep()) return false;

		// ������ȡ
		float *ptr_test = nullptr;
		try {
			ptr_test = new float[pc.size() * m_featureNum];
		}
		catch (const std::bad_alloc&) {
			return false;
		}

		sc::FeatureMatrix featMat(ptr_test, pc.size(), m_featureNum); // ����Լ�����
		bool isFeature = getFeature(featMat, &pc, 0);
		if (!isFeature) {
			if (errorMes)
				strcpy(errorMes, "��������ʧ�ܣ�");
			delete[] ptr_test;
			return false;
		}

		// ���ò���
		npy_intp Dims_x[1]        = { featMat.rows*featMat.cols }; //����ά����Ϣ
		PyObject * pyFeatureNum   = Py_BuildValue("i", param._featureNum);
		PyObject * model          = param._model;
		PyObject * pyFeatureArray = PyArray_SimpleNewFromData(1, Dims_x, NPY_FLOAT, ptr_test);

		// Ensure that the numpy array does not own the data
		PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(pyFeatureArray), NPY_ARRAY_OWNDATA);

		PythonUtil::PyObjPtr args(PyTuple_New(3));
		PyTuple_SetItem(args.get(), 0, pyFeatureArray);
		PyTuple_SetItem(args.get(), 1, model);
		PyTuple_SetItem(args.get(), 2, pyFeatureNum);

		//���ú���
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

		// ���������
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

		// д�������ļ�
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

		// �����ڴ�
		delete[] ptr_test;
		std::vector<PointT>().swap(pc);
		std::vector<std::pair<unsigned int, unsigned int>>().swap(m_first_and_last_indices_vector);
		std::vector<cloud_point_index_idx>().swap(m_index_vector);

		return true;
	}

	//�����������
	template <class PointT>
	bool Classification::getFeature(sc::FeatureMatrix &featMat, std::vector<PointT> *data, GenericProgressCallback* _pdlg)
	{
		sc::FeatureCombinerCDHVFast <PointT> featureCombiner;
		featureCombiner.setInputCloud(data);

		featureCombiner.buildSpatialIndexImpl();
		bool isFeature = featureCombiner.combineFeaturesImpl(featMat, _pdlg);
		return isFeature;
	}

	// ��ȡѵ��ģ��
	template <class PointT>
	bool Classification::loadTrainingModel(std::string modelPath, PyObject **model, char *errorMes, GenericProgressCallback* _pdlg)
	{
		// ʹ��NormalizedProgress�����ٽ���
		NormalizedProgress np(_pdlg, 4);

		// ȷ��Python�����ѳ�ʼ��
		if (!PythonUtil::initializePythonEnvironment()) {
			if (errorMes)
				strcpy(errorMes, "Python environment initialization failed.��");
			return false;
		}
		if (!np.oneStep())return false;

		// ����Pythonģ��
		PythonUtil::PyObjPtr pModule = PythonUtil::importPythonModule("ClassificationPy");
		if (!pModule) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python module.��");
			return false;
		}
		if (!np.oneStep())return false;

		// ��ȡpython ����
		PythonUtil::PyObjPtr pFun = PythonUtil::getPythonFunction(pModule.get(), "loadModel");
		if (!pFun) {
			if (errorMes)
				strcpy(errorMes, "Failed to import Python function.��");
			return false;
		}
		if (!np.oneStep())return false;

		// ���ò���
		PythonUtil::PyObjPtr args(PyTuple_New(1));
		PythonUtil::PyObjPtr arg_model(StringToPy(modelPath));
		PyTuple_SetItem(args.get(), 0, arg_model.release()); // ��������Ȩ��Ԫ��

		// ����Python����
		PythonUtil::PyObjPtr result(PyObject_CallObject(pFun.get(), args.get()));
		if (!result) {
			PythonUtil::printPythonError();
			if (errorMes)
				strcpy(errorMes, (char *)PyUnicode_AsUTF8(PyObject_Str(PyErr_Occurred())));
			return false;
		}
		if (!np.oneStep())return false;

		// ���÷��ص�ģ�Ͷ���
		*model = result.release(); // ��������Ȩ��������

		return true;
	}

	//д������ֵ
	template <class PointT>
	bool  Classification::writeFeature(std::vector<PointT> *pc, std::vector<cloud_point_index_idx> &index_vector, std::vector<std::pair<unsigned int, unsigned int> > &first_and_last_indices_vector, float *featureArr, std::string filename, bool isTrain, int *labelArr, GenericProgressCallback *_pdlg) // Ĭ��nullptr����ʾû�д���Progress�ص�����
	{
		unsigned int first_index, last_index;
		//����һ���ļ����������д��ģʽ���ļ�
		std::ofstream fin(filename.c_str(), std::ios::out | std::ios::trunc);

		if (!fin)
		{
			return false;
		}

		//�������ʱ����ֵ���ȣ�����4λС��
		fin.precision(4);

		//���ø������ĸ�ʽΪ�̶�������ʹ�ÿ�ѧ����������ʽ
		fin.setf(std::ios::fixed, std::ios::floatfield);

		//����һ������ָʾ������
		NormalizedProgress np(_pdlg, first_and_last_indices_vector.size());

		//��isTrainΪtrue����Ҫ���label
		if (isTrain)
		{
			//�����ǰ�����Ĵ�С
			fin << first_and_last_indices_vector.size() << "\n";

			//����first_and_last_indices_vector
			for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
			{
				//ÿ����һ��������һ�ν�����
				if (!np.oneStep())return false;

				//��ȡ��ǰԪ�صĵ�һ������ֵ
				first_index = first_and_last_indices_vector[cp].first;

				//��������ֵ����ȡ��Ӧ�ĵ������
				int idx = index_vector[first_index].cloud_point_index;

				//�����ʽΪ�� x y z ��ǩ ����ֵ
				fin << (*pc)[idx].x << " " << (*pc)[idx].y << " " << (*pc)[idx].z << " " << labelArr[idx] << " ";

				//�����������ֵ
				for (int i = 0; i < m_featureNum; i++)
				{
					fin << featureArr[idx*m_featureNum + i] << " ";
				}
				fin << "\n";
			}
		}
		else //isTrainΪfalse������Ҫ���label
		{
			//������Ƶ�����
			fin << pc->size() << "\n";

			for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
			{
				if (!np.oneStep())return false;

				first_index = first_and_last_indices_vector[cp].first;
				last_index  = first_and_last_indices_vector[cp].second;

				//�������еĵ�
				for (unsigned int li = first_index; li < last_index; ++li)
				{
					int idx = index_vector[li].cloud_point_index;

					//�����ʽΪ�� x y z ����ֵ
					fin << (*pc)[idx].x << " " << (*pc)[idx].y << " " << (*pc)[idx].z << " ";

					//�����������ֵ
					for (int i = 0; i < m_featureNum; i++)
					{
						fin << featureArr[cp*m_featureNum + i] << " ";
					}
					fin << "\n";
				}
			}
		}

		//�ر��ļ���
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

	//���ػ�
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
