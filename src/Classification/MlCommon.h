#pragma once


#include "GenericProgressCallback.h"
#include "scBasics.h"

//python
#include <Python.h>

//
#include <string>

//
#include "pcCommon.h"
#define  FEA_NUM 16

struct FeatureParam
{
	FeatureParam() :
		featMat_train_file(),
		_trainPointCloud(nullptr),
		_path(""),
		_isFeature(false)
	{}
	//! 特征矩阵
	sc::FeatureMatrix  featMat_train_file;

	//! 训练集点云
	std::vector<pcc::LASPoint>* _trainPointCloud;

	//! 路径
	std::string  _path;

	//! 判断输入是否为特征文件
	bool _isFeature;

};


// 基类
struct GenericParam {
	float* _featureArr;
	int* _labelArr;
	int _featArraySize;
	int _labelArraySize;
	int _checkedArraySize;
	std::string _path;
	int _classifierId;
	bool _isModel;
	bool _isEval;
	std::vector<std::pair<int, int>> _indexes;
	std::vector<std::string> _filenames;
	int _featureNum;
	bool* _TraincheckedResults;

	GenericParam() : _featureArr(nullptr), _labelArr(nullptr), _featArraySize(0), _labelArraySize(0), _checkedArraySize(16),_classifierId(0), _isModel(true), _isEval(true), _featureNum(FEA_NUM), _TraincheckedResults(new bool[16]) {}
	//virtual ~GenericParam() { delete[] _TraincheckedResults; }
};

// 随机森林的特定参数
struct RFTrainParam : GenericParam {
	int _RFcriterionId; // 0 为基尼系数 1为信息熵
	int _RFtreenum;
	int _RFtreeMaxDepth;

	RFTrainParam() : _RFcriterionId(0), _RFtreenum(20), _RFtreeMaxDepth(10) {
		_classifierId = 0; // 假设随机森林分类器的ID为1
	}
};

// 神经网络的特定参数
struct NNTrainParam : GenericParam {
	std::string _NNactiveFun;
	std::string _NNoptiFun;
	float _NNlr;
	int _NNiter;

	NNTrainParam() : _NNactiveFun("relu"), _NNoptiFun("adam"), _NNlr(0.001), _NNiter(200) {
		_classifierId = 1; // 假设神经网络分类器的ID为2
	}
};

// LightGBM 的特定参数
struct LGTrainParam : GenericParam {
	int _LGtreeMaxDepth;
	int _LGtreenum;
	float _LGlr;
	int _LGnumClass;
	int _LGleavesNum;

	LGTrainParam() : _LGtreeMaxDepth(6), _LGtreenum(200), _LGlr(0.3), _LGnumClass(0), _LGleavesNum(31) {
		_classifierId = 2; // 假设LightGBM分类器的ID为3
	}
};




//struct TrainParam
//{
//	TrainParam() :
//		_path(""),
//		_classifierId(0),
//		_labelArr(nullptr),
//		_featureArr(nullptr),
//		_featArraySize(0),
//		_labelArraySize(0),
//
//		_RFcriterionId(0),//0 为基尼系数 1为信息熵
//		_RFtreenum(20),
//		_RFtreeMaxDepth(10),
//
//		_NNactiveFun("relu"),
//		_NNoptiFun("adam"),
//		_NNlr(0.001),
//		_NNiter(200),
//
//		_XGtreeMaxDepth(6),
//		_XGtreenum(200),
//		_XGlr(0.3),
//
//		_isEval(true),
//		_isModel(true),
//
//
//		_indexes(),
//		_filenames(),
//		_featureNum(FEA_NUM)
//
//	{}
//
//	//! 训练数据的特征数组
//	float* _featureArr;
//
//	//! 训练数据的类别数组
//	int* _labelArr;
//
//	//! 训练数据的类别数组
//	int _featArraySize;
//	int _labelArraySize;
//	int _checkedArraySize = 16;
//	//! 输出文件夹
//	std::string  _path;
//
//	//! 分类器参数
//	int _classifierId;
//
//	//********随机森林参数********//
//	//! 0 为基尼系数 1为信息熵
//	int _RFcriterionId;
//	int _RFtreenum;
//	int _RFtreeMaxDepth;
//
//	//********神经网络********//
//	//! 0 ReLu,1 Logistic,2 Tanh, 3 Identity
//	std::string _NNactiveFun;
//	std::string _NNoptiFun;
//	float _NNlr;
//	int _NNiter;
//
//	//********LGB********
//	int _XGtreeMaxDepth;
//	int _XGtreenum;
//	float _XGlr;
//	int _XGnumClass;
//	int _LGleavesNum;
//
//	bool _isModel;
//	bool _isEval;
//
//
//	std::vector<std::pair<int, int>> _indexes;
//	std::vector<std::string> _filenames;
//	int _featureNum;
//
//	bool* _TraincheckedResults = new bool[16];
//};

struct TestParam
{
	TestParam() :
		_initLabelPointCloud(nullptr),
		_model(nullptr),
		_isData(false),
		_featureArr(nullptr),
		_isFeature(false),
		_path(""),
		_voxelSize(0.3),
		_featureNum(FEA_NUM)
	{}

	//! 测试点云指针
	std::vector< pcc::LASPoint>* _initLabelPointCloud;

	//! 输入分类模型
	PyObject* _model;

	//! 输入是否为特征
	bool _isFeature;

	//! 输入是否为数据
	bool _isData;

	//! 输出文件夹
	std::string  _path;

	//! 特征矩阵
	float* _featureArr;

	//! 体素大小
	float _voxelSize;

	//! 特征数目
	int _featureNum;

	bool* _TraincheckedResults = new bool[16];
};

struct TrainResult
{
	TrainResult() :
		_model(nullptr)
	{}

	//! 返回分类模型
	PyObject* _model;

	//
	std::string _modelFile;
};

struct TestResult
{
	TestResult() : _labels(nullptr) { }

	//! 返回测试点云类别数组
	unsigned long* _labels;
};


class WorkerProC : public GenericProgressCallback
{
public:
	WorkerProC() :
		currentValue(0),
		totalsteps(1)
	{ };

	virtual ~WorkerProC() {}

	void setFunc(pcc::ProgressFunc _func)
	{
		func_ = _func;
	}

	void setTotalSteps(int steps)
	{
		totalsteps = steps;
	}

	virtual void update(float percent)
	{
		if (!func_)
			return;

		int value = static_cast<int>(percent);

		int mod_ = currentValue % 100;

		if (mod_ > value)
			mod_ = 0;

		currentValue += value - mod_;
		func_(currentValue / totalsteps);
	};

	virtual void setMAXRange(int maxvalue) {};

	virtual void setMethodTitle(const char* methodTitle) {};

	virtual void setInfo(const char* infoStr) {
		//strcpy(errorMes,infoStr);
		errorMes = infoStr;
	};

	virtual void start() {};

	virtual void stop() {};

	virtual bool isCancelRequested() { return false; };

	pcc::ProgressFunc func_;

	std::string errorMes;
private:
	unsigned currentValue;

	int totalsteps;
};
