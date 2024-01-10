
#ifndef _H_PCCOMMON_H_
#define _H_PCCOMMON_H_

//

namespace pcc {

	// 进度回调函数指针
	typedef void(*ProgressFunc)(int _process);


	// 二维点结构
	struct Point2D
	{
		double x, y;

		//
		Point2D() : x(0), y(0) {};

		//
		Point2D(double x_, double y_) : x(x_), y(y_) {};
	};

	// 三维点结构
	struct Point3D
	{
		double x, y, z;

		//
		Point3D() : x(0), y(0), z(0) {};

		//
		Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {};
	};

	//
	struct LASPoint:Point3D
	{
		using Point3D::x;
		using Point3D::y; 
		using Point3D::z;

		unsigned short intensity;
		unsigned char return_number;
		unsigned char number_of_returns;
		unsigned char scan_direction_flag ;
		unsigned char edge_of_flight_line;
		unsigned char classification;
		unsigned char user_data;
		unsigned short point_source_ID;
		char scan_angle_rank;

		unsigned char rgb[3];

		//!
		LASPoint() :Point3D()
		{
			intensity = 0;
			edge_of_flight_line = 0;
			scan_direction_flag = 0;
			number_of_returns = 0;
			return_number = 0;
			classification = 0;
			scan_angle_rank = 0;
			user_data = 0;
			point_source_ID = 0;
			rgb[0] = rgb[1] = rgb[2] = 0;
		};

		//!
		LASPoint(double x_, double y_, double z_) :Point3D(x_, y_, z_)
		{
			intensity = 0;
			edge_of_flight_line = 0;
			scan_direction_flag = 0;
			number_of_returns = 0;
			return_number = 0;
			classification = 0;
			scan_angle_rank = 0;
			user_data = 0;
			point_source_ID = 0;
			rgb[0] = rgb[1] = rgb[2] = 0;
		};
	};

};

#endif //PCCOMMON_H_