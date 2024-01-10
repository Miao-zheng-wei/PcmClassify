
#include "pcLasFilterApi.h"

//local

#include "GenericProgressCallback.h"
#include "pcCommon.h"

//LASlib
#include <las\laszip_decompress_selective_v3.hpp>
#include <las\lasreader.hpp>
#include <las\laswriter.hpp>

//for test
#include <filesystem>
#include <algorithm>
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <vector>

//
namespace pcLas
{
	using namespace pcc;

	LASreader* GetLasReaderPtr(const char* lasFile)
	{
		U32 decompress_selective = LASZIP_DECOMPRESS_SELECTIVE_CHANNEL_RETURNS_XY;
		decompress_selective |= LASZIP_DECOMPRESS_SELECTIVE_Z;
		decompress_selective |= LASZIP_DECOMPRESS_SELECTIVE_CLASSIFICATION;
		decompress_selective |= LASZIP_DECOMPRESS_SELECTIVE_INTENSITY;
		decompress_selective |= LASZIP_DECOMPRESS_SELECTIVE_RGB;

		LASreadOpener lasreadopener;
		lasreadopener.set_file_name(lasFile);
		lasreadopener.set_decompress_selective(decompress_selective);
		return lasreadopener.open();
	}

	template<typename PointT>
	bool IsPointInsidePoly(const PointT& P, const std::vector<Point2D>& contour)
	{
		//number of vertices
		unsigned vertCount = contour.size();
		if (vertCount < 2)
			return false;

		//
		bool inside = false;

		//
		Point2D A = contour[0];
		for (unsigned i = 1; i <= vertCount; ++i) {
			
			Point2D B = contour[i % vertCount];

			//Point Inclusion in Polygon Test (inspired from W. Randolph Franklin - WRF)
			//The polyline is considered as a 2D polyline here!
			if ((B.y <= P.y && P.y < A.y) || (A.y <= P.y && P.y < B.y))
			{
				double t = (P.x - B.x)*(A.y - B.y) - (A.x - B.x)*(P.y - B.y);
				if (A.y < B.y)
					t = -t;
				if (t < 0)
					inside = !inside;
			}

			A = B;
		}

		return inside;
	}

	class WorkerProC : public GenericProgressCallback
	{
	public:
		WorkerProC() :
			currentValue(0),
			totalsteps(1)
		{ };

		virtual ~WorkerProC() {}

		void setFunc(ProgressFunc _func)
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

		ProgressFunc func_;

		std::string errorMes;
	private:
		unsigned currentValue;

		int totalsteps;
	};

	bool ReadAndFilter(const InParamT& inPara, std::vector<pcc::LASPoint>& outPts, pcc::ProgressFunc progress, char* errorMes /*= 0*/)
	{
		//
		WorkerProC pro_;
		pro_.setFunc(progress);
		pro_.setTotalSteps(1);
		pro_.setInfo("正在执行重建");

		auto& filePath = inPara.filePath;
		auto& validCs = inPara.validCs;
		auto& polygon = inPara.validPolygon;

		LASreader* reader = GetLasReaderPtr(filePath.c_str());
		if (reader) {

			//
			LASheader& header = reader->header;

			//
			NormalizedProgress nprogress(&pro_, reader->npoints);

			//
			pcc::LASPoint ppt;
			while (reader->read_point()) {

				if (!nprogress.oneStep())
					break;

				LASpoint& lasPt = reader->point;

				ppt.x = lasPt.get_x();
				ppt.y = lasPt.get_y();
				ppt.z = lasPt.get_z();
				ppt.classification = lasPt.get_classification();

				if (!validCs.empty() && (validCs.cend() == std::find(validCs.cbegin(), validCs.cend(), (int)ppt.classification)))
					continue;

				if (!polygon.empty() && !IsPointInsidePoly<pcc::LASPoint>(ppt, polygon))
					continue;

				ppt.intensity = lasPt.get_intensity();
				ppt.return_number = lasPt.get_return_number();
				ppt.number_of_returns = lasPt.get_number_of_returns();
				ppt.scan_direction_flag = lasPt.get_scan_direction_flag();
				ppt.edge_of_flight_line = lasPt.get_edge_of_flight_line();
				ppt.user_data = lasPt.get_user_data();
				ppt.point_source_ID = lasPt.get_point_source_ID();
				ppt.scan_angle_rank = lasPt.get_scan_angle_rank();

				outPts.push_back(ppt);
			}

			//
			reader->close(true);
			return true;

		}
		else {

			if (errorMes)
				strcpy(errorMes, "打开文件失败！");
		}

		return false;
	}

	static bool endsWith(const std::string& str, const std::string suffix) {
		if (suffix.length() > str.length()) { return false; }

		return (str.rfind(suffix) == (str.length() - suffix.length()));
	}

	bool WriteLasFile(const std::vector<pcc::LASPoint>& inPts, std::string outFilePath, pcc::ProgressFunc progress, char* errorMes /*= 0*/)
	{
		//
		if (inPts.empty()) {

			if (errorMes)
				strcpy(errorMes, "输入点集为空");
			return false;
		}

		//
		if (outFilePath.empty()) {

			if (errorMes)
				strcpy(errorMes, "未设置输出las文件全路径");
			return false;
		}

		//
		if (!endsWith(outFilePath, ".las")) {

			if (errorMes)
				strcpy(errorMes, "文件后缀非法");
			return false;
		}

		//
		WorkerProC pro_;
		pro_.setFunc(progress);
		pro_.setTotalSteps(1);
		pro_.setInfo("正在执行重建");

		bool isSuccess = false;

		LASwriteOpener laswriteopener;
		laswriteopener.set_file_name(outFilePath.c_str());

		LASheader _pheader;
		_pheader.point_data_format = 3;
		_pheader.point_data_record_length = 34;

		auto& front = inPts.front();
		_pheader.x_offset = front.x;
		_pheader.y_offset = front.y;
		_pheader.z_offset = front.z;

		_pheader.x_scale_factor = 0.001;
		_pheader.y_scale_factor = 0.001;
		_pheader.z_scale_factor = 0.001;

		if (LASwriter* laswriter = laswriteopener.open(&_pheader))
		{
			LASpoint lpt;
			lpt.init(&_pheader, _pheader.point_data_format, _pheader.point_data_record_length);

			//
			unsigned ptSize = inPts.size();
			//
			NormalizedProgress nprogress(&pro_, ptSize);

			for (unsigned i = 0; i < ptSize; ++i) {

				if (!nprogress.oneStep())
					break;

				auto& pt = inPts[i];
				if (i == 0) {

					_pheader.min_x = _pheader.max_x = pt.x;
					_pheader.min_y = _pheader.max_y = pt.y;
					_pheader.min_z = _pheader.max_z = pt.z;
				}
				else {

					if (pt.x < _pheader.min_x) _pheader.min_x = pt.x;
					else if (pt.x > _pheader.max_x) _pheader.max_x = pt.x;
					if (pt.y < _pheader.min_y) _pheader.min_y = pt.y;
					else if (pt.y > _pheader.max_y) _pheader.max_y = pt.y;
					if (pt.z < _pheader.min_z) _pheader.min_z = pt.z;
					else if (pt.z > _pheader.max_z) _pheader.max_z = pt.z;
				}

				lpt.set_x(pt.x);
				lpt.set_y(pt.y);
				lpt.set_z(pt.z);
				lpt.set_intensity(pt.intensity);
				lpt.set_gps_time(0);
				lpt.set_return_number(pt.return_number);
				lpt.set_classification(pt.classification);
				lpt.set_point_source_ID(pt.point_source_ID);
				lpt.set_scan_angle_rank(pt.scan_angle_rank);
				lpt.set_number_of_returns(pt.number_of_returns);
				lpt.set_scan_direction_flag(pt.scan_direction_flag);

				lpt.have_rgb = TRUE;

				lpt.set_R(pt.rgb[0]);
				lpt.set_G(pt.rgb[1]);
				lpt.set_B(pt.rgb[2]);

				laswriter->write_point(&lpt);
			}
			laswriter->update_header(&_pheader);
			laswriter->close(true);
			isSuccess = true;
		}
		else
		{
			if (errorMes)
				strcpy(errorMes, "打开文件失败！");
		}

		return isSuccess;
	}
};