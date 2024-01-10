
#ifndef _H_PCMLCLASSIFYAPI_H_  
#define _H_PCMLCLASSIFYAPI_H_ 


#ifdef PC_MLCLASSIFY_EXPORTS
#define PC_MLCLASSIFY_API __declspec(dllexport)
#else
#define PC_MLCLASSIFY_API __declspec(dllimport)
#endif

//
#include "pcCommon.h"

//
#include <string>
#include <vector>

namespace pcMlClassify
{
	enum ClassifierType
	{
		RF,
		NN,
		LG,
	};

	// 基类结构体，包含所有模型共有的输入参数
	struct GenericInParams
	{
		virtual ~GenericInParams() {} // 虚析构函数确保多态
		pcc::ProgressFunc pFunc;      // 进度回调函数指针
		std::string python310Path;    // python310环境目录，默认根目录
		std::string outfileDir;       // 输出文件路径:不得为空
		std::string outfileName;      // 输出文件名:不得为空
		int classifierId;			  //
		GenericInParams()
			: pFunc(nullptr), python310Path("./python310"), classifierId(-1) {}
	};

	// 随机森林参数结构体
	struct RFInputParams : public GenericInParams
	{
		int RFtreenum;           // 决策树数量
		int RFcriterionId;       // 分类判断标准 0 基尼系数 1 信息熵
		int RFtreeMaxDepth;          // 最大深度

		RFInputParams()
			: RFtreenum(20), RFcriterionId(0), RFtreeMaxDepth(10) {}
	};

	// 神经网络参数结构体
	struct NNInputParams : public GenericInParams
	{
		std::string activeFun;  // 激活函数
		std::string optiFun;    // 优化函数
		float lr;               // 学习率
		int iter;               // 迭代次数

		NNInputParams()
			: activeFun("relu"), optiFun("adam"), lr(0.001), iter(200) {}
	};

	// LG模型参数结构体
	struct LGInputParams : public GenericInParams
	{
		int treeMaxDepth;      // 最大深度
		int treenum;           // 决策树数量
		float lr;              // 学习率
		int leavesNum;         // 类别数量
		int numClass;          // 叶子节点数量

		LGInputParams()
			: treeMaxDepth(6), treenum(70), lr(0.05), leavesNum(31), numClass(0) {}
	};

	// 输入参数结构体，用于分类时的参数
	struct ClassificationParams
	{
		//! 进度回调函数指针
		pcc::ProgressFunc pFunc;

		//! python310环境目录，默认根目录
		std::string python310Path;

		//! 模型文件路径（*.pkl）
		std::string modelPath;

		ClassificationParams() :pFunc(nullptr), python310Path("./python310") {};
	};


	// 模型训练
	extern "C"  PC_MLCLASSIFY_API bool TrainSample(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, ClassifierType ct = ClassifierType::RF, char* errorMes = 0);

	// 使用训练好的模型对数据进行分类
	extern "C"  PC_MLCLASSIFY_API bool ClassifyData(std::vector<pcc::LASPoint>& inPts, ClassificationParams& inPara, std::vector<unsigned>& labels, char* errorMes = 0);
}

#endif //PCMLCLASSIFYAPI_H_  