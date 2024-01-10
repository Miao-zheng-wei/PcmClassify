
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

	// ����ṹ�壬��������ģ�͹��е��������
	struct GenericInParams
	{
		virtual ~GenericInParams() {} // ����������ȷ����̬
		pcc::ProgressFunc pFunc;      // ���Ȼص�����ָ��
		std::string python310Path;    // python310����Ŀ¼��Ĭ�ϸ�Ŀ¼
		std::string outfileDir;       // ����ļ�·��:����Ϊ��
		std::string outfileName;      // ����ļ���:����Ϊ��
		int classifierId;			  //
		GenericInParams()
			: pFunc(nullptr), python310Path("./python310"), classifierId(-1) {}
	};

	// ���ɭ�ֲ����ṹ��
	struct RFInputParams : public GenericInParams
	{
		int RFtreenum;           // ����������
		int RFcriterionId;       // �����жϱ�׼ 0 ����ϵ�� 1 ��Ϣ��
		int RFtreeMaxDepth;          // ������

		RFInputParams()
			: RFtreenum(20), RFcriterionId(0), RFtreeMaxDepth(10) {}
	};

	// ����������ṹ��
	struct NNInputParams : public GenericInParams
	{
		std::string activeFun;  // �����
		std::string optiFun;    // �Ż�����
		float lr;               // ѧϰ��
		int iter;               // ��������

		NNInputParams()
			: activeFun("relu"), optiFun("adam"), lr(0.001), iter(200) {}
	};

	// LGģ�Ͳ����ṹ��
	struct LGInputParams : public GenericInParams
	{
		int treeMaxDepth;      // ������
		int treenum;           // ����������
		float lr;              // ѧϰ��
		int leavesNum;         // �������
		int numClass;          // Ҷ�ӽڵ�����

		LGInputParams()
			: treeMaxDepth(6), treenum(70), lr(0.05), leavesNum(31), numClass(0) {}
	};

	// ��������ṹ�壬���ڷ���ʱ�Ĳ���
	struct ClassificationParams
	{
		//! ���Ȼص�����ָ��
		pcc::ProgressFunc pFunc;

		//! python310����Ŀ¼��Ĭ�ϸ�Ŀ¼
		std::string python310Path;

		//! ģ���ļ�·����*.pkl��
		std::string modelPath;

		ClassificationParams() :pFunc(nullptr), python310Path("./python310") {};
	};


	// ģ��ѵ��
	extern "C"  PC_MLCLASSIFY_API bool TrainSample(std::vector<pcc::LASPoint>& inPts, GenericInParams* inPara, ClassifierType ct = ClassifierType::RF, char* errorMes = 0);

	// ʹ��ѵ���õ�ģ�Ͷ����ݽ��з���
	extern "C"  PC_MLCLASSIFY_API bool ClassifyData(std::vector<pcc::LASPoint>& inPts, ClassificationParams& inPara, std::vector<unsigned>& labels, char* errorMes = 0);
}

#endif //PCMLCLASSIFYAPI_H_  