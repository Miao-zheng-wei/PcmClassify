#include "pcMlClassifyApi.h"
#include "MlCommon.h"
#include "Classification.h"

//local
#include "pcCommon.h"

//
#include <vector>

namespace pcMlClassify
{
	using namespace std;

	// ����ļ��Ƿ���ڣ�ʹ�� ifstream
	bool isFileExists_ifstream(string& name) {
		ifstream f(name.c_str());
		return f.good();
	}

	// ����ļ��Ƿ���ڣ�ʹ�� fopen
	bool isFileExists_fopen(string& name) {
		if (FILE* file = fopen(name.c_str(), "r")) {
			fclose(file);
			return true;
		}
		else {
			return false;
		}
	}

	// ����ļ��Ƿ���ڣ�ʹ�� access ����
	bool isFileExists_access(string& name) {
		return (access(name.c_str(), 0) != -1);
	}

	// ����ļ��Ƿ���ڣ�ʹ�� stat ����
	bool isFileExists_stat(string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);

	}

	// ���ô�����Ϣ
	void setErrorMessage(char* errorMes, const char* message) {
		if (errorMes) {
			strcpy(errorMes, message);
		}
	}

	// ��֤�����������Ч��
	bool validateInput(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, char* errorMes) {
		if (inPts.empty()) {
			setErrorMessage(errorMes, "����㼯Ϊ��");
			return false;
		}

		if (inPara->outfileDir.empty()) {
			setErrorMessage(errorMes, "δ�������·����");
			return false;
		}

		if (inPara->outfileName.empty()) {
			setErrorMessage(errorMes, "δ��������ļ�����");
			return false;
		}

		if (inPara->python310Path.empty()) {
			setErrorMessage(errorMes, "δ����Python310·����");
			return false;
		}

		if (!isFileExists_access(inPara->python310Path)) {
			setErrorMessage(errorMes, "Python310·����Ч��");
			return false;
		}

		return true;
	}

	// ����������֤ģ���ļ�·����Python·���Ƿ���ȷ���úʹ���
	bool validatePaths(ClassificationParams& inPara, char* errorMes) {
		// ���ģ���ļ�·���Ƿ�����
		if (inPara.modelPath.empty()) {
			if (errorMes)
				strcpy(errorMes, "δ����ģ���ļ�·����");
			return false;
		}

		// ���Python310·���Ƿ�����
		if (inPara.python310Path.empty()) {
			if (errorMes)
				strcpy(errorMes, "δ����Python310·����");
			return false;
		}

		// ���Python310·���Ƿ���Ч
		if (!isFileExists_access(inPara.python310Path)) {
			if (errorMes)
				strcpy(errorMes, "Python310·����Ч��");
			return false;
		}

		// �������·������Ч���򷵻� true
		return true;
	}

	bool TrainSample(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, ClassifierType ct, char* errorMes)
	{

		// ��������߼�
		if (!validateInput(inPts, inPara, errorMes))
		{
			return false;
		}

		WorkerProC pro_;
		pro_.setFunc(inPara->pFunc);
		pro_.setTotalSteps(4);

		//step1 ��ȡ��������
		std::vector<int> realLabels;
		{
			std::string logName = "���ڶ�ȡ��������";

			//
			NormalizedProgress nDlg(&pro_, inPts.size());
			for (auto& pt : inPts) {

				if (!nDlg.oneStep())
					return false;

				realLabels.push_back(pt.classification);
			}
		}

		//step2 �����������
		//float* ptr_test = new float[inPts.size() * 16];
		std::unique_ptr<float[]> ptr_test(new float[inPts.size() * 16]);
		{
			std::string logName = "���ڼ����������";

			sc::FeatureMatrix featMat(ptr_test.get(), inPts.size(), 16);

			Classify::Classification calfeature;
			if (!calfeature.getFeature(featMat, &inPts, &pro_)) {
				setErrorMessage(errorMes, "��������ʧ�ܣ�");
				return false;
			}
		}

		//step3 ѵ��ģ��
		{
			std::string logName = "����ѵ��ģ��";
			Classify::Classification ml;
			bool trainSuccess = false;

			switch (ct) {
			case RF: {

				auto rfInParam = dynamic_cast<RFInputParams*>(inPara);
				if (!rfInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "ģ�Ͳ���Ϊ�գ�");
				}

				// ���ɭ�ַ���������
				RFTrainParam rfTrainParam;
				rfTrainParam._path = rfInParam->outfileDir + "\\" + rfInParam->outfileName;
				rfTrainParam._labelArr = &realLabels[0];
				rfTrainParam._featureArr = &ptr_test[0];
				rfTrainParam._featArraySize = inPts.size() * 16;
				rfTrainParam._labelArraySize = inPts.size();
				// �������ɭ���ض��Ĳ���
				rfTrainParam._RFcriterionId = rfInParam->RFcriterionId;
				rfTrainParam._RFtreeMaxDepth = rfInParam->RFtreeMaxDepth;
				rfTrainParam._RFtreenum = rfInParam->RFtreenum;
				trainSuccess = ml.trainClassifier_RF< pcc::LASPoint>(&rfTrainParam, rfInParam->outfileDir, &pro_);
				break;
			}
			case NN: {
				// ���������������
				auto nnInParam = dynamic_cast<NNInputParams*>(inPara);
				if (!nnInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "ģ�Ͳ���Ϊ�գ�");
				}
				NNTrainParam nnTrainParam;
				nnTrainParam._path = nnInParam->outfileDir + "\\" + nnInParam->outfileName;
				nnTrainParam._labelArr = &realLabels[0];
				nnTrainParam._featureArr = &ptr_test[0];
				nnTrainParam._featArraySize = inPts.size() * 16;
				nnTrainParam._labelArraySize = inPts.size();
				// �����������ض��Ĳ���
				nnTrainParam._NNactiveFun = nnInParam->activeFun;
				nnTrainParam._NNoptiFun = nnInParam->optiFun;
				nnTrainParam._NNlr = nnInParam->lr;
				nnTrainParam._NNiter = nnInParam->iter;
				trainSuccess = ml.trainClassifier_NN< pcc::LASPoint>(&nnTrainParam, nnInParam->outfileDir, &pro_);
				break;
			}
			case LG: {
				// LG����������
				auto lgInParam = dynamic_cast<LGInputParams*> (inPara);
				if (!lgInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "ģ�Ͳ���Ϊ�գ�");
				}
				LGTrainParam lgTrainParam;
				lgTrainParam._path = lgInParam->outfileDir + "\\" + lgInParam->outfileName;
				lgTrainParam._labelArr = &realLabels[0];
				lgTrainParam._featureArr = &ptr_test[0];
				lgTrainParam._featArraySize = inPts.size() * 16;
				lgTrainParam._labelArraySize = inPts.size();
				// ����LightGBM�ض��Ĳ���
				lgTrainParam._LGtreeMaxDepth = lgInParam->treeMaxDepth;
				lgTrainParam._LGtreenum = lgInParam->treenum;
				lgTrainParam._LGlr = lgInParam->lr;
				lgTrainParam._LGnumClass = lgInParam->numClass;//
				lgTrainParam._LGleavesNum = lgInParam->leavesNum;
				trainSuccess = ml.trainClassifier_LG< pcc::LASPoint>(&lgTrainParam, lgInParam->outfileDir, &pro_);
				break;
			}
			default:
				setErrorMessage(errorMes, "δ֪�ķ�����ID��");
				return false;
			}

			if (!trainSuccess) {
				setErrorMessage(errorMes, "ģ��ѵ��ʧ�ܣ�");
				return false;
			}
		}
		return true;
	}

	bool ClassifyData(std::vector<pcc::LASPoint>& inPts, ClassificationParams& inPara, std::vector<unsigned>& labels, char* errorMes)
	{
		// ��֤ģ���ļ�·����Python·���Ƿ���ȷ���úʹ�
		if (!validatePaths(inPara, errorMes)) {
			std::cerr << errorMes << std::endl;
			return false;
		}

		// ��ʼ��������
		WorkerProC pro_;
		pro_.setFunc(inPara.pFunc);
		pro_.setTotalSteps(2);

		std::string logName = "���ڼ��ط���ģ��";

		//step1 ����ѵ���õ�ģ�͡�
		PyObject* pModel = nullptr;
		Classify::Classification api;
		PyObject* raw_m_model = nullptr;
		if (!api.loadTrainingModel< pcc::LASPoint>(inPara.modelPath, &raw_m_model, errorMes, &pro_)) {
			setErrorMessage(errorMes, "����ģ�͵���ʧ�ܣ�");
			return false;
		}

		// ȷ��raw_m_model���ǿ�ָ��
		if (raw_m_model && PyTuple_Check(raw_m_model)) {
			pModel = PyTuple_GetItem(raw_m_model, 0); // ���������ü���
			Py_INCREF(pModel); // �������ü�������Ϊ������Ҫ���ⲿʹ��pModel
		}

		//Step 2: ʹ��ģ�ͽ������ݷ���
		TestParam testParam;
		testParam._initLabelPointCloud = &inPts;
		testParam._model = pModel;
		testParam._featureNum = 16;
		testParam._isData = true;
		testParam._isFeature = false;

		TestResult testResult;
		Classify::Classification test;
		if (!test.runPythonClassification< pcc::LASPoint>(testParam, testResult, errorMes, &pro_))
		{
			setErrorMessage(errorMes, "��ģ�ͷ���ʧ�ܣ�");
			return false;
		}

		labels.reserve(inPts.size());
		labels.assign(testResult._labels, testResult._labels + inPts.size());

		// �����ڴ�
		delete[] testResult._labels;
		testResult._labels = nullptr;
		if (pModel) {
			Py_DECREF(pModel);
		}

		if (raw_m_model) {
			Py_DECREF(raw_m_model);
		}
		// ����ɹ�
		return true;
	}
}