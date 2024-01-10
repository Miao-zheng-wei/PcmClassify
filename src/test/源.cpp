// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
#include "pcLasFilterApi.h"
#include "pcMlClassifyApi.h"

#include <windows.h>
#include <iostream>
#include <string>
#include <set>

std::string pythonPath;

std::string workSpace;

std::string logName;

void progressFunc(int _process)
{
	printf((logName + "-progress: %d%%\n").c_str(), _process);
};

// 机器学习分类
bool RFClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\测试数据—分类\\训练数据_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//读取数据
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\测试数据—分类\\训练数据.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	auto inPara = new pcMlClassify::RFInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::RF;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\测试数据—分类";
	inPara->outfileName = "训练数据";
	inPara->RFtreenum = 20;      // 决策树数量
	inPara->RFcriterionId = 0;   // 分类判断标准 0 基尼系数 1 信息熵
	inPara->RFtreeMaxDepth = 10; // 最大深度
	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // 清理分配的内存
		return false;
	}
}

// 机器学习分类
bool NNClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\测试数据—分类\\训练数据_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//读取数据
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\测试数据—分类\\训练数据.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	auto inPara = new pcMlClassify::NNInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::NN;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\测试数据—分类";
	inPara->outfileName = "训练数据";
	inPara->activeFun = "relu";
	inPara->optiFun = "adam";
	inPara->lr = 0.001;
	inPara->iter = 200;

	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // 清理分配的内存
		return false;
	}
}

// 机器学习分类
bool LGClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\测试数据—分类\\训练数据_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//读取数据
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\测试数据—分类\\训练数据.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	std::set<int> uniqueCategories;

	// 遍历 trainCloud 中的每个点
	for (const auto& point : trainCloud) {
		// 插入点的类别到集合中，集合会自动处理重复的元素
		uniqueCategories.insert(point.classification);
	}

	// 类别数量 lg 分类必须必须设置
	int num_class = uniqueCategories.size();

	auto inPara = new pcMlClassify::LGInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::LG;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\测试数据—分类";
	inPara->outfileName = "训练数据";
	inPara->treeMaxDepth = 6;
	inPara->treenum = 70;
	inPara->lr = 0.05;
	inPara->leavesNum = 31;
	inPara->numClass = num_class;

	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // 清理分配的内存
		return false;
	}
}

// 使用训练好的模型对点云分类
bool Classify(char* errorMes = 0)
{
	std::string ModelFile = "D:\\2023清华碳汇\\src\\x64\\Release\\测试数据—分类_RFmodel.pkl";
	std::vector<unsigned> labels;
	std::vector<pcc::LASPoint> testCloud;

	{
		logName = "'mlClassify'";

		//读取数据
		pcLas::InParamT inpara1;
		inpara1.filePath = workSpace + "\\测试数据—分类\\待分类数据.las";
		char* errorMes = 0;
		if (!pcLas::ReadAndFilter(inpara1, testCloud, progressFunc, errorMes)) {
			return false;
		}

		using namespace pcMlClassify;
		ClassificationParams inpara2;
		inpara2.pFunc = progressFunc;
		inpara2.modelPath = ModelFile;
		inpara2.python310Path = pythonPath;

		if (!pcMlClassify::ClassifyData(testCloud, inpara2, labels, errorMes))
		{
			return false;
		}
	}

	{
		logName = "'outClassifyFile'";

		for (int i = 0; i < testCloud.size(); ++i)
		{
			testCloud[i].classification = labels[i];
		}

		std::string outGroundFile = workSpace + "\\测试数据—分类\\待分类数据_classify.las";
		if (!pcLas::WriteLasFile(testCloud, outGroundFile, progressFunc, errorMes)) {
			return false;
		}

		std::cout << "the result of 'mlClassify' : " + outGroundFile << std::endl;
		return true;
	}
}

void FuncIntroduction()
{
	std::cout << "-> Please read this help" << std::endl;
	std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
	std::cout << "   1 is to do 'RF Classify'" << std::endl;
	std::cout << "   2 is to do 'NN Classify'" << std::endl;
	std::cout << "   3 is to do 'LG Classify'" << std::endl;
	std::cout << "   4 is to do 'Classify'" << std::endl;
	std::cout << "   0 is to exit this test" << std::endl;
	std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
	std::cout << "-> Which function will be tested?" << std::endl;
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
	//当前测试程序路径
	workSpace = argv[0];
	//当前测试程序所在文件夹
	workSpace = workSpace.substr(0, workSpace.rfind("\\"));
	//当前测试程序的上一级文件夹
	//workSpace = workSpace.substr(0, workSpace.rfind("\\"));

	pythonPath = workSpace + "\\python310";

	char errorMes[1024];

	bool isContinue = true;
	while (isContinue) {
		FuncIntroduction();

		int functionId = 0;

		//#ifdef _DEBUG
		std::cout << "The function ID is ";
		std::cin >> functionId;
		std::cout << std::endl;
		//#endif //

		bool isSuccess = true;
		switch (functionId)
		{
		case 0:
			isContinue = false;
			break;
		case 1:
		{
			isSuccess = RFClassify(errorMes);
			break;
		}
		case 2:
		{
			isSuccess = NNClassify(errorMes);
			break;
		}
		case 3:
		{
			isSuccess = LGClassify(errorMes);
			break;
		}

		case 4:
		{
			isSuccess = Classify(errorMes);
			break;
		}
		default:
			std::cout << "Invalid Entry! Please input again!" << std::endl;
			std::cout << std::endl;
			break;
		}

		if (!isSuccess) {
			std::cout << logName << " failed,error: " << errorMes << std::endl;
		}
		else
			std::cout << logName << " finished." << std::endl;

		system("pause");
	}

	return 0;
}