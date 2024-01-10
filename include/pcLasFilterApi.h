
#ifndef _H_PCLASFILTERAPI_H_
#define _H_PCLASFILTERAPI_H_


#ifdef PC_LASFILTER_EXPORTS
#define PC_LASFILTER_API __declspec(dllexport)
#else
#define  PC_LASFILTER_API __declspec(dllimport)
#endif

//
#include <pcCommon.h>

//
#include <vector>
#include <string>

//
namespace pcLas
{
	struct InParamT
	{
		//! �����ļ�
		std::string filePath;

		//! ��Ҫ��ȡ��������Ϊ�գ��򲻽��������ȡ
		std::vector<int> validCs;

		//! ����α߽磬���Ϊ�գ��򲻽��б߽緶ΧԼ��
		std::vector<pcc::Point2D> validPolygon;
	};

	//
	extern "C" PC_LASFILTER_API bool ReadAndFilter(const InParamT& inPara, std::vector<pcc::LASPoint>& outPts, pcc::ProgressFunc progress, char* errorMes = 0);

	//outFilePathΪ�����ɵ�las�ļ�·��ȫ��
	extern "C" PC_LASFILTER_API bool WriteLasFile(const std::vector<pcc::LASPoint>& inPts, std::string outFilePath, pcc::ProgressFunc progress, char* errorMes = 0);

};

#endif //PCLASFILTERAPI_H_