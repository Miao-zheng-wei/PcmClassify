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

	// 检查文件是否存在，使用 ifstream
	bool isFileExists_ifstream(string& name) {
		ifstream f(name.c_str());
		return f.good();
	}

	// 检查文件是否存在，使用 fopen
	bool isFileExists_fopen(string& name) {
		if (FILE* file = fopen(name.c_str(), "r")) {
			fclose(file);
			return true;
		}
		else {
			return false;
		}
	}

	// 检查文件是否存在，使用 access 函数
	bool isFileExists_access(string& name) {
		return (access(name.c_str(), 0) != -1);
	}

	// 检查文件是否存在，使用 stat 函数
	bool isFileExists_stat(string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);

	}

	// 设置错误信息
	void setErrorMessage(char* errorMes, const char* message) {
		if (errorMes) {
			strcpy(errorMes, message);
		}
	}

	// 验证输入参数的有效性
	bool validateInput(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, char* errorMes) {
		if (inPts.empty()) {
			setErrorMessage(errorMes, "输入点集为空");
			return false;
		}

		if (inPara->outfileDir.empty()) {
			setErrorMessage(errorMes, "未设置输出路径！");
			return false;
		}

		if (inPara->outfileName.empty()) {
			setErrorMessage(errorMes, "未设置输出文件名！");
			return false;
		}

		if (inPara->python310Path.empty()) {
			setErrorMessage(errorMes, "未设置Python310路径！");
			return false;
		}

		if (!isFileExists_access(inPara->python310Path)) {
			setErrorMessage(errorMes, "Python310路径无效！");
			return false;
		}

		return true;
	}

	// 函数用于验证模型文件路径和Python路径是否被正确设置和存在
	bool validatePaths(ClassificationParams& inPara, char* errorMes) {
		// 检查模型文件路径是否设置
		if (inPara.modelPath.empty()) {
			if (errorMes)
				strcpy(errorMes, "未设置模型文件路径！");
			return false;
		}

		// 检查Python310路径是否设置
		if (inPara.python310Path.empty()) {
			if (errorMes)
				strcpy(errorMes, "未设置Python310路径！");
			return false;
		}

		// 检查Python310路径是否有效
		if (!isFileExists_access(inPara.python310Path)) {
			if (errorMes)
				strcpy(errorMes, "Python310路径无效！");
			return false;
		}

		// 如果所有路径都有效，则返回 true
		return true;
	}

	bool TrainSample(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, ClassifierType ct, char* errorMes)
	{

		// 参数检查逻辑
		if (!validateInput(inPts, inPara, errorMes))
		{
			return false;
		}

		WorkerProC pro_;
		pro_.setFunc(inPara->pFunc);
		pro_.setTotalSteps(4);

		//step1 读取点云数据
		std::vector<int> realLabels;
		{
			std::string logName = "正在读取点云数据";

			//
			NormalizedProgress nDlg(&pro_, inPts.size());
			for (auto& pt : inPts) {

				if (!nDlg.oneStep())
					return false;

				realLabels.push_back(pt.classification);
			}
		}

		//step2 计算点云特征
		//float* ptr_test = new float[inPts.size() * 16];
		std::unique_ptr<float[]> ptr_test(new float[inPts.size() * 16]);
		{
			std::string logName = "正在计算点云特征";

			sc::FeatureMatrix featMat(ptr_test.get(), inPts.size(), 16);

			Classify::Classification calfeature;
			if (!calfeature.getFeature(featMat, &inPts, &pro_)) {
				setErrorMessage(errorMes, "特征计算失败！");
				return false;
			}
		}

		//step3 训练模型
		{
			std::string logName = "正在训练模型";
			Classify::Classification ml;
			bool trainSuccess = false;

			switch (ct) {
			case RF: {

				auto rfInParam = dynamic_cast<RFInputParams*>(inPara);
				if (!rfInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "模型参数为空！");
				}

				// 随机森林分类器参数
				RFTrainParam rfTrainParam;
				rfTrainParam._path = rfInParam->outfileDir + "\\" + rfInParam->outfileName;
				rfTrainParam._labelArr = &realLabels[0];
				rfTrainParam._featureArr = &ptr_test[0];
				rfTrainParam._featArraySize = inPts.size() * 16;
				rfTrainParam._labelArraySize = inPts.size();
				// 设置随机森林特定的参数
				rfTrainParam._RFcriterionId = rfInParam->RFcriterionId;
				rfTrainParam._RFtreeMaxDepth = rfInParam->RFtreeMaxDepth;
				rfTrainParam._RFtreenum = rfInParam->RFtreenum;
				trainSuccess = ml.trainClassifier_RF< pcc::LASPoint>(&rfTrainParam, rfInParam->outfileDir, &pro_);
				break;
			}
			case NN: {
				// 神经网络分类器参数
				auto nnInParam = dynamic_cast<NNInputParams*>(inPara);
				if (!nnInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "模型参数为空！");
				}
				NNTrainParam nnTrainParam;
				nnTrainParam._path = nnInParam->outfileDir + "\\" + nnInParam->outfileName;
				nnTrainParam._labelArr = &realLabels[0];
				nnTrainParam._featureArr = &ptr_test[0];
				nnTrainParam._featArraySize = inPts.size() * 16;
				nnTrainParam._labelArraySize = inPts.size();
				// 设置神经网络特定的参数
				nnTrainParam._NNactiveFun = nnInParam->activeFun;
				nnTrainParam._NNoptiFun = nnInParam->optiFun;
				nnTrainParam._NNlr = nnInParam->lr;
				nnTrainParam._NNiter = nnInParam->iter;
				trainSuccess = ml.trainClassifier_NN< pcc::LASPoint>(&nnTrainParam, nnInParam->outfileDir, &pro_);
				break;
			}
			case LG: {
				// LG分类器参数
				auto lgInParam = dynamic_cast<LGInputParams*> (inPara);
				if (!lgInParam)
				{
					if (errorMes)
						setErrorMessage(errorMes, "模型参数为空！");
				}
				LGTrainParam lgTrainParam;
				lgTrainParam._path = lgInParam->outfileDir + "\\" + lgInParam->outfileName;
				lgTrainParam._labelArr = &realLabels[0];
				lgTrainParam._featureArr = &ptr_test[0];
				lgTrainParam._featArraySize = inPts.size() * 16;
				lgTrainParam._labelArraySize = inPts.size();
				// 设置LightGBM特定的参数
				lgTrainParam._LGtreeMaxDepth = lgInParam->treeMaxDepth;
				lgTrainParam._LGtreenum = lgInParam->treenum;
				lgTrainParam._LGlr = lgInParam->lr;
				lgTrainParam._LGnumClass = lgInParam->numClass;//
				lgTrainParam._LGleavesNum = lgInParam->leavesNum;
				trainSuccess = ml.trainClassifier_LG< pcc::LASPoint>(&lgTrainParam, lgInParam->outfileDir, &pro_);
				break;
			}
			default:
				setErrorMessage(errorMes, "未知的分类器ID！");
				return false;
			}

			if (!trainSuccess) {
				setErrorMessage(errorMes, "模型训练失败！");
				return false;
			}
		}
		return true;
	}

	bool ClassifyData(std::vector<pcc::LASPoint>& inPts, ClassificationParams& inPara, std::vector<unsigned>& labels, char* errorMes)
	{
		// 验证模型文件路径和Python路径是否被正确设置和存
		if (!validatePaths(inPara, errorMes)) {
			std::cerr << errorMes << std::endl;
			return false;
		}

		// 初始化进度条
		WorkerProC pro_;
		pro_.setFunc(inPara.pFunc);
		pro_.setTotalSteps(2);

		std::string logName = "正在加载分类模型";

		//step1 加载训练好的模型。
		PyObject* pModel = nullptr;
		Classify::Classification api;
		PyObject* raw_m_model = nullptr;
		if (!api.loadTrainingModel< pcc::LASPoint>(inPara.modelPath, &raw_m_model, errorMes, &pro_)) {
			setErrorMessage(errorMes, "分类模型导入失败！");
			return false;
		}

		// 确保raw_m_model不是空指针
		if (raw_m_model && PyTuple_Check(raw_m_model)) {
			pModel = PyTuple_GetItem(raw_m_model, 0); // 不增加引用计数
			Py_INCREF(pModel); // 增加引用计数，因为我们需要在外部使用pModel
		}

		//Step 2: 使用模型进行数据分类
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
			setErrorMessage(errorMes, "按模型分类失败！");
			return false;
		}

		labels.reserve(inPts.size());
		labels.assign(testResult._labels, testResult._labels + inPts.size());

		// 清理内存
		delete[] testResult._labels;
		testResult._labels = nullptr;
		if (pModel) {
			Py_DECREF(pModel);
		}

		if (raw_m_model) {
			Py_DECREF(raw_m_model);
		}
		// 分类成功
		return true;
	}
}