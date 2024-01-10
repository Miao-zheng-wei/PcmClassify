#pragma once
#include <vector>

//存放计算点云的idx和点云编号的cloud_point_index的结构头
struct cloud_point_index_idx
{
	unsigned int idx;
	unsigned int cloud_point_index;

	cloud_point_index_idx(unsigned int idx_, unsigned int cloud_point_index_) : idx(idx_), cloud_point_index(cloud_point_index_) {}
	bool operator < (const cloud_point_index_idx &p) const { return (idx < p.idx); }
};

//中间参数的结构体
struct Array4f
{
	float x;
	float y;
	float z;
	float C;
};


template <class PointT>
class VoxelGrid
{
public:
	VoxelGrid();
	~VoxelGrid();

	void setInputCloud(std::vector<PointT> &cloud);
	void setLeafSize(float lx, float ly, float lz);
	void GetMaxMin(Array4f&min_p, Array4f&max_p);
	void VoxelGrid_ApplyFilter( std::vector<PointT> &OutPointCloud);

private:
	float X_Voxel;
	float Y_Voxel;
	float Z_Voxel;
	std::vector<PointT> InputCloudPoint;
};

template <class PointT>
VoxelGrid<PointT>::VoxelGrid()
{
}

template <class PointT>
VoxelGrid<PointT>::~VoxelGrid()
{
}

template <class PointT>
void VoxelGrid<PointT>::setInputCloud(std::vector<PointT> &cloud)
{
	InputCloudPoint = cloud;
}

template <class PointT>
void VoxelGrid<PointT>::setLeafSize(float lx, float ly, float lz)
{
	X_Voxel = lx;
	Y_Voxel = ly;
	Z_Voxel = lz;
}

/*----------------------------
*功能：找到输入点云中的包围盒两个点的值（右上和左下）
*-----------------------------
*输入：vector<Point3D>&InputCloudPoint（Piont3D的原始点云数据）
*输出：点云的min_p和max_p
*/
template <class PointT>
void VoxelGrid<PointT>::GetMaxMin( Array4f&min_p, Array4f&max_p)
{
	//主要思路是找到x，y,z的最小值,这样就能得到点云立体包围的次村
	//找x,y,z最小值
	if (InputCloudPoint.size() == 0)
	{
		//cout << "输入点云为空" << endl;
		return;
	}
	float x_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.x < b.x; })).x;
	float y_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.y < b.y; })).y;
	float z_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.z < b.z; })).z;
	//给min_p赋值
	min_p.x = x_min;
	min_p.y = y_min;
	min_p.z = z_min;
	min_p.C = 1;
	//找x,y,z的最大值
	float x_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.x < b.x; })).x;
	float y_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.y < b.y; })).y;
	float z_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.z < b.z; })).z;
	//给max_p赋值
	max_p.x = x_max;
	max_p.y = y_max;
	max_p.z = z_max;
	max_p.C = 1;
	return;
}

/*----------------------------
*功能：体素化网格方法实现下采样（PCL中的源码C++实现）
*-----------------------------
*输入：vector<Point3D>&InputCloudPoint(Piont3D的原始点云数据,下采样的体素大小x,y,z)
*输出：vector<Point3D>&OutPointCloud(采样之后的之后的Point3D结构的点云数据)
*/
template <class PointT>
void VoxelGrid<PointT>::VoxelGrid_ApplyFilter(std::vector<PointT> &OutPointCloud)
{
	//先判断输入的点云是否为空
	if (InputCloudPoint.size() == 0)
	{
		//cout << "输入点云为空！" << endl;
		return;
	}
	//存放输入点云的最大与最小坐标
	Array4f min_p, max_p;
	GetMaxMin( min_p, max_p);

	Array4f inverse_leaf_size_;
	inverse_leaf_size_.x = 1 / X_Voxel;
	inverse_leaf_size_.y = 1 / Y_Voxel;
	inverse_leaf_size_.z = 1 / Z_Voxel;
	inverse_leaf_size_.C = 1;

	//计算最小和最大边界框值
	Array4f min_b_, max_b_, div_b_, divb_mul_;
	min_b_.x = static_cast<int> (floor(min_p.x * inverse_leaf_size_.x));
	max_b_.x = static_cast<int> (floor(max_p.x * inverse_leaf_size_.x));
	min_b_.y = static_cast<int> (floor(min_p.y * inverse_leaf_size_.y));
	max_b_.y = static_cast<int> (floor(max_p.y * inverse_leaf_size_.y));
	min_b_.z = static_cast<int> (floor(min_p.z * inverse_leaf_size_.z));
	max_b_.z = static_cast<int> (floor(max_p.z * inverse_leaf_size_.z));

	//计算沿所有轴所需的分割数
	div_b_.x = max_b_.x - min_b_.x + 1;
	div_b_.y = max_b_.y - min_b_.y + 1;
	div_b_.z = max_b_.z - min_b_.z + 1;
	div_b_.C = 0;

	//设置除法乘数
	divb_mul_.x = 1;
	divb_mul_.y = div_b_.x;
	divb_mul_.z = div_b_.x * div_b_.y;
	divb_mul_.C = 0;

	//用于计算idx和pointcloud索引的存储
	std::vector<cloud_point_index_idx> index_vector;
	index_vector.reserve(InputCloudPoint.size());

	//第一步：遍历所有点并将它们插入到具有计算idx的index_vector向量中;具有相同idx值的点将有助于产生CloudPoint的相同点
	for (int i = 0; i < InputCloudPoint.size(); i++)
	{
		int ijk0 = static_cast<int> (floor(InputCloudPoint[i].x * inverse_leaf_size_.x) - static_cast<float> (min_b_.x));
		int ijk1 = static_cast<int> (floor(InputCloudPoint[i].y * inverse_leaf_size_.y) - static_cast<float> (min_b_.y));
		int ijk2 = static_cast<int> (floor(InputCloudPoint[i].z * inverse_leaf_size_.z) - static_cast<float> (min_b_.z));

		//计算质心叶索引
		int idx = ijk0 * divb_mul_.x + ijk1 * divb_mul_.y + ijk2 * divb_mul_.z;
		index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int> (idx), i));
	}
	//第二步：使用表示目标单元格的值作为索引对index_vector向量进行排序;实际上属于同一输出单元格的所有点都将彼此相邻
	std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());

	//第三步：计数输出单元格，我们需要跳过所有相同的，相邻的idx值
	unsigned int total = 0;
	unsigned int index = 0;
	unsigned int min_points_per_voxel_ = 0;
	//first_and_last_indices_vector [i]表示属于对应于第i个输出点的体素的index_vector中的第一个点的index_vector中的索引，以及不属于第一个点的索引
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	first_and_last_indices_vector.reserve(index_vector.size());                              //分配内存空间

	while (index < index_vector.size())
	{
		unsigned int i = index + 1;
		while (i < index_vector.size() && index_vector[i].idx == index_vector[index].idx)
			++i;
		if (i - index >= min_points_per_voxel_)
		{
			++total;
			first_and_last_indices_vector.push_back(std::pair<unsigned int, unsigned int>(index, i));
		}
		index = i;
	}

	//第四步：计算质心，将它们插入最终位置
	//OutPointCloud.resize(total);      //给输出点云分配内存空间
	float x_Sum, y_Sum, z_Sum;
	PointT PointCloud;
	unsigned int first_index, last_index;
	for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
	{
		// 计算质心 - 来自所有输入点的和值，这些值在index_vector数组中具有相同的idx值
		first_index = first_and_last_indices_vector[cp].first;
		last_index = first_and_last_indices_vector[cp].second;
		x_Sum = 0;
		y_Sum = 0;
		z_Sum = 0;
		for (unsigned int li = first_index; li < last_index; ++li)
		{
			x_Sum += InputCloudPoint[index_vector[li].cloud_point_index].x;
			y_Sum += InputCloudPoint[index_vector[li].cloud_point_index].y;
			z_Sum += InputCloudPoint[index_vector[li].cloud_point_index].z;
		}
		PointCloud.x = x_Sum / (last_index - first_index);
		PointCloud.y = y_Sum / (last_index - first_index);
		PointCloud.z = z_Sum / (last_index - first_index);
		OutPointCloud.emplace_back(PointCloud);
	}

	return;
}

//取体素内第一个点代表该体素，并传回体素点的索引
template <class PointT>
void VoxelGrid<PointT>::VoxelGrid_ApplyFilter(std::vector<PointT> &OutPointCloud)
{
	//先判断输入的点云是否为空
	if (InputCloudPoint.size() == 0)
	{
		//cout << "输入点云为空！" << endl;
		return;
	}
	//存放输入点云的最大与最小坐标
	Array4f min_p, max_p;
	GetMaxMin(min_p, max_p);

	Array4f inverse_leaf_size_;
	inverse_leaf_size_.x = 1 / X_Voxel;
	inverse_leaf_size_.y = 1 / Y_Voxel;
	inverse_leaf_size_.z = 1 / Z_Voxel;
	inverse_leaf_size_.C = 1;

	//计算最小和最大边界框值
	Array4f min_b_, max_b_, div_b_, divb_mul_;
	min_b_.x = static_cast<int> (floor(min_p.x * inverse_leaf_size_.x));
	max_b_.x = static_cast<int> (floor(max_p.x * inverse_leaf_size_.x));
	min_b_.y = static_cast<int> (floor(min_p.y * inverse_leaf_size_.y));
	max_b_.y = static_cast<int> (floor(max_p.y * inverse_leaf_size_.y));
	min_b_.z = static_cast<int> (floor(min_p.z * inverse_leaf_size_.z));
	max_b_.z = static_cast<int> (floor(max_p.z * inverse_leaf_size_.z));

	//计算沿所有轴所需的分割数
	div_b_.x = max_b_.x - min_b_.x + 1;
	div_b_.y = max_b_.y - min_b_.y + 1;
	div_b_.z = max_b_.z - min_b_.z + 1;
	div_b_.C = 0;

	//设置除法乘数
	divb_mul_.x = 1;
	divb_mul_.y = div_b_.x;
	divb_mul_.z = div_b_.x * div_b_.y;
	divb_mul_.C = 0;

	//用于计算idx和pointcloud索引的存储
	std::vector<cloud_point_index_idx> index_vector;
	index_vector.reserve(InputCloudPoint.size());

	//第一步：遍历所有点并将它们插入到具有计算idx的index_vector向量中;具有相同idx值的点将有助于产生CloudPoint的相同点
	for (int i = 0; i < InputCloudPoint.size(); i++)
	{
		int ijk0 = static_cast<int> (floor(InputCloudPoint[i].x * inverse_leaf_size_.x) - static_cast<float> (min_b_.x));
		int ijk1 = static_cast<int> (floor(InputCloudPoint[i].y * inverse_leaf_size_.y) - static_cast<float> (min_b_.y));
		int ijk2 = static_cast<int> (floor(InputCloudPoint[i].z * inverse_leaf_size_.z) - static_cast<float> (min_b_.z));

		//计算质心叶索引
		int idx = ijk0 * divb_mul_.x + ijk1 * divb_mul_.y + ijk2 * divb_mul_.z;
		index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int> (idx), i));
	}
	//第二步：使用表示目标单元格的值作为索引对index_vector向量进行排序;实际上属于同一输出单元格的所有点都将彼此相邻
	std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());

	//第三步：计数输出单元格，我们需要跳过所有相同的，相邻的idx值
	unsigned int total = 0;
	unsigned int index = 0;
	unsigned int min_points_per_voxel_ = 0;
	//first_and_last_indices_vector [i]表示属于对应于第i个输出点的体素的index_vector中的第一个点的index_vector中的索引，以及不属于第一个点的索引
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	first_and_last_indices_vector.reserve(index_vector.size());                              //分配内存空间

	while (index < index_vector.size())
	{
		unsigned int i = index + 1;
		while (i < index_vector.size() && index_vector[i].idx == index_vector[index].idx)
			++i;
		if (i - index >= min_points_per_voxel_)
		{
			++total;
			first_and_last_indices_vector.push_back(std::pair<unsigned int, unsigned int>(index, i));
		}
		index = i;
	}

	//第四步：计算质心，将它们插入最终位置
	//OutPointCloud.resize(total);      //给输出点云分配内存空间
	float x_Sum, y_Sum, z_Sum;
	PointT PointCloud;
	unsigned int first_index, last_index;
	for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
	{
		// 计算质心 - 来自所有输入点的和值，这些值在index_vector数组中具有相同的idx值
		first_index = first_and_last_indices_vector[cp].first;
		last_index = first_and_last_indices_vector[cp].second;
		x_Sum = 0;
		y_Sum = 0;
		z_Sum = 0;
		for (unsigned int li = first_index; li < last_index; ++li)
		{
			x_Sum += InputCloudPoint[index_vector[li].cloud_point_index].x;
			y_Sum += InputCloudPoint[index_vector[li].cloud_point_index].y;
			z_Sum += InputCloudPoint[index_vector[li].cloud_point_index].z;
		}
		PointCloud.x = x_Sum / (last_index - first_index);
		PointCloud.y = y_Sum / (last_index - first_index);
		PointCloud.z = z_Sum / (last_index - first_index);
		OutPointCloud.emplace_back(PointCloud);
	}

	return;
}
