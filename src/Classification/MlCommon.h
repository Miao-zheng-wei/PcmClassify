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
	//! ��������
	sc::FeatureMatrix  featMat_train_file;

	//! ѵ��������
	std::vector<pcc::LASPoint>* _trainPointCloud;

	//! ·��
	std::string  _path;

	//! �ж������Ƿ�Ϊ�����ļ�
	bool _isFeature;

};


// ����
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

// ���ɭ�ֵ��ض�����
struct RFTrainParam : GenericParam {
	int _RFcriterionId; // 0 Ϊ����ϵ�� 1Ϊ��Ϣ��
	int _RFtreenum;
	int _RFtreeMaxDepth;

	RFTrainParam() : _RFcriterionId(0), _RFtreenum(20), _RFtreeMaxDepth(10) {
		_classifierId = 0; // �������ɭ�ַ�������IDΪ1
	}
};

// ��������ض�����
struct NNTrainParam : GenericParam {
	std::string _NNactiveFun;
	std::string _NNoptiFun;
	float _NNlr;
	int _NNiter;

	NNTrainParam() : _NNactiveFun("relu"), _NNoptiFun("adam"), _NNlr(0.001), _NNiter(200) {
		_classifierId = 1; // �����������������IDΪ2
	}
};

// LightGBM ���ض�����
struct LGTrainParam : GenericParam {
	int _LGtreeMaxDepth;
	int _LGtreenum;
	float _LGlr;
	int _LGnumClass;
	int _LGleavesNum;

	LGTrainParam() : _LGtreeMaxDepth(6), _LGtreenum(200), _LGlr(0.3), _LGnumClass(0), _LGleavesNum(31) {
		_classifierId = 2; // ����LightGBM��������IDΪ3
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
//		_RFcriterionId(0),//0 Ϊ����ϵ�� 1Ϊ��Ϣ��
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
//	//! ѵ�����ݵ���������
//	float* _featureArr;
//
//	//! ѵ�����ݵ��������
//	int* _labelArr;
//
//	//! ѵ�����ݵ��������
//	int _featArraySize;
//	int _labelArraySize;
//	int _checkedArraySize = 16;
//	//! ����ļ���
//	std::string  _path;
//
//	//! ����������
//	int _classifierId;
//
//	//********���ɭ�ֲ���********//
//	//! 0 Ϊ����ϵ�� 1Ϊ��Ϣ��
//	int _RFcriterionId;
//	int _RFtreenum;
//	int _RFtreeMaxDepth;
//
//	//********������********//
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

	//! ���Ե���ָ��
	std::vector< pcc::LASPoint>* _initLabelPointCloud;

	//! �������ģ��
	PyObject* _model;

	//! �����Ƿ�Ϊ����
	bool _isFeature;

	//! �����Ƿ�Ϊ����
	bool _isData;

	//! ����ļ���
	std::string  _path;

	//! ��������
	float* _featureArr;

	//! ���ش�С
	float _voxelSize;

	//! ������Ŀ
	int _featureNum;

	bool* _TraincheckedResults = new bool[16];
};

struct TrainResult
{
	TrainResult() :
		_model(nullptr)
	{}

	//! ���ط���ģ��
	PyObject* _model;

	//
	std::string _modelFile;
};

struct TestResult
{
	TestResult() : _labels(nullptr) { }

	//! ���ز��Ե����������
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
