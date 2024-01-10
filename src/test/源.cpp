// ConsoleApplication1.cpp : �������̨Ӧ�ó������ڵ㡣
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

// ����ѧϰ����
bool RFClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\�������ݡ�����\\ѵ������_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//��ȡ����
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\�������ݡ�����\\ѵ������.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	auto inPara = new pcMlClassify::RFInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::RF;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\�������ݡ�����";
	inPara->outfileName = "ѵ������";
	inPara->RFtreenum = 20;      // ����������
	inPara->RFcriterionId = 0;   // �����жϱ�׼ 0 ����ϵ�� 1 ��Ϣ��
	inPara->RFtreeMaxDepth = 10; // ������
	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // ���������ڴ�
		return false;
	}
}

// ����ѧϰ����
bool NNClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\�������ݡ�����\\ѵ������_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//��ȡ����
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\�������ݡ�����\\ѵ������.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	auto inPara = new pcMlClassify::NNInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::NN;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\�������ݡ�����";
	inPara->outfileName = "ѵ������";
	inPara->activeFun = "relu";
	inPara->optiFun = "adam";
	inPara->lr = 0.001;
	inPara->iter = 200;

	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // ���������ڴ�
		return false;
	}
}

// ����ѧϰ����
bool LGClassify(char* errorMes = 0)
{
	std::string outModelFile = workSpace + "\\�������ݡ�����\\ѵ������_model.pkl";
	logName = "trainingModel";

	std::vector<pcc::LASPoint> trainCloud;

	//��ȡ����
	pcLas::InParamT inpara1;
	inpara1.filePath = workSpace + "\\�������ݡ�����\\ѵ������.las";
	if (!pcLas::ReadAndFilter(inpara1, trainCloud, progressFunc, errorMes)) {
		return false;
	}

	std::set<int> uniqueCategories;

	// ���� trainCloud �е�ÿ����
	for (const auto& point : trainCloud) {
		// ��������𵽼����У����ϻ��Զ������ظ���Ԫ��
		uniqueCategories.insert(point.classification);
	}

	// ������� lg ��������������
	int num_class = uniqueCategories.size();

	auto inPara = new pcMlClassify::LGInputParams();
	pcMlClassify::ClassifierType ct = pcMlClassify::ClassifierType::LG;
	inPara->pFunc = progressFunc;
	inPara->python310Path = pythonPath;
	inPara->outfileDir = workSpace + "\\�������ݡ�����";
	inPara->outfileName = "ѵ������";
	inPara->treeMaxDepth = 6;
	inPara->treenum = 70;
	inPara->lr = 0.05;
	inPara->leavesNum = 31;
	inPara->numClass = num_class;

	if (!pcMlClassify::TrainSample(trainCloud, inPara, ct, errorMes))
	{
		delete inPara; // ���������ڴ�
		return false;
	}
}

// ʹ��ѵ���õ�ģ�ͶԵ��Ʒ���
bool Classify(char* errorMes = 0)
{
	std::string ModelFile = "D:\\2023�廪̼��\\src\\x64\\Release\\�������ݡ�����_RFmodel.pkl";
	std::vector<unsigned> labels;
	std::vector<pcc::LASPoint> testCloud;

	{
		logName = "'mlClassify'";

		//��ȡ����
		pcLas::InParamT inpara1;
		inpara1.filePath = workSpace + "\\�������ݡ�����\\����������.las";
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

		std::string outGroundFile = workSpace + "\\�������ݡ�����\\����������_classify.las";
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
	//��ǰ���Գ���·��
	workSpace = argv[0];
	//��ǰ���Գ��������ļ���
	workSpace = workSpace.substr(0, workSpace.rfind("\\"));
	//��ǰ���Գ������һ���ļ���
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