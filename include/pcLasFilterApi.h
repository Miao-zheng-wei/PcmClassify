
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
		//! 输入文件
		std::string filePath;

		//! 需要提取的类别，如果为空，则不进行类别提取
		std::vector<int> validCs;

		//! 多边形边界，如果为空，则不进行边界范围约束
		std::vector<pcc::Point2D> validPolygon;
	};

	//
	extern "C" PC_LASFILTER_API bool ReadAndFilter(const InParamT& inPara, std::vector<pcc::LASPoint>& outPts, pcc::ProgressFunc progress, char* errorMes = 0);

	//outFilePath为待生成的las文件路径全名
	extern "C" PC_LASFILTER_API bool WriteLasFile(const std::vector<pcc::LASPoint>& inPts, std::string outFilePath, pcc::ProgressFunc progress, char* errorMes = 0);

};

#endif //PCLASFILTERAPI_H_