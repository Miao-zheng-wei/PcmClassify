#pragma once
#include <vector>

//��ż�����Ƶ�idx�͵��Ʊ�ŵ�cloud_point_index�Ľṹͷ
struct cloud_point_index_idx
{
	unsigned int idx;
	unsigned int cloud_point_index;

	cloud_point_index_idx(unsigned int idx_, unsigned int cloud_point_index_) : idx(idx_), cloud_point_index(cloud_point_index_) {}
	bool operator < (const cloud_point_index_idx &p) const { return (idx < p.idx); }
};

//�м�����Ľṹ��
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
*���ܣ��ҵ���������еİ�Χ���������ֵ�����Ϻ����£�
*-----------------------------
*���룺vector<Point3D>&InputCloudPoint��Piont3D��ԭʼ�������ݣ�
*��������Ƶ�min_p��max_p
*/
template <class PointT>
void VoxelGrid<PointT>::GetMaxMin( Array4f&min_p, Array4f&max_p)
{
	//��Ҫ˼·���ҵ�x��y,z����Сֵ,�������ܵõ����������Χ�Ĵδ�
	//��x,y,z��Сֵ
	if (InputCloudPoint.size() == 0)
	{
		//cout << "�������Ϊ��" << endl;
		return;
	}
	float x_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.x < b.x; })).x;
	float y_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.y < b.y; })).y;
	float z_min = (*min_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.z < b.z; })).z;
	//��min_p��ֵ
	min_p.x = x_min;
	min_p.y = y_min;
	min_p.z = z_min;
	min_p.C = 1;
	//��x,y,z�����ֵ
	float x_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.x < b.x; })).x;
	float y_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.y < b.y; })).y;
	float z_max = (*max_element(InputCloudPoint.begin(), InputCloudPoint.end(), [](PointT& a, PointT& b){return a.z < b.z; })).z;
	//��max_p��ֵ
	max_p.x = x_max;
	max_p.y = y_max;
	max_p.z = z_max;
	max_p.C = 1;
	return;
}

/*----------------------------
*���ܣ����ػ����񷽷�ʵ���²�����PCL�е�Դ��C++ʵ�֣�
*-----------------------------
*���룺vector<Point3D>&InputCloudPoint(Piont3D��ԭʼ��������,�²��������ش�Сx,y,z)
*�����vector<Point3D>&OutPointCloud(����֮���֮���Point3D�ṹ�ĵ�������)
*/
template <class PointT>
void VoxelGrid<PointT>::VoxelGrid_ApplyFilter(std::vector<PointT> &OutPointCloud)
{
	//���ж�����ĵ����Ƿ�Ϊ��
	if (InputCloudPoint.size() == 0)
	{
		//cout << "�������Ϊ�գ�" << endl;
		return;
	}
	//���������Ƶ��������С����
	Array4f min_p, max_p;
	GetMaxMin( min_p, max_p);

	Array4f inverse_leaf_size_;
	inverse_leaf_size_.x = 1 / X_Voxel;
	inverse_leaf_size_.y = 1 / Y_Voxel;
	inverse_leaf_size_.z = 1 / Z_Voxel;
	inverse_leaf_size_.C = 1;

	//������С�����߽��ֵ
	Array4f min_b_, max_b_, div_b_, divb_mul_;
	min_b_.x = static_cast<int> (floor(min_p.x * inverse_leaf_size_.x));
	max_b_.x = static_cast<int> (floor(max_p.x * inverse_leaf_size_.x));
	min_b_.y = static_cast<int> (floor(min_p.y * inverse_leaf_size_.y));
	max_b_.y = static_cast<int> (floor(max_p.y * inverse_leaf_size_.y));
	min_b_.z = static_cast<int> (floor(min_p.z * inverse_leaf_size_.z));
	max_b_.z = static_cast<int> (floor(max_p.z * inverse_leaf_size_.z));

	//����������������ķָ���
	div_b_.x = max_b_.x - min_b_.x + 1;
	div_b_.y = max_b_.y - min_b_.y + 1;
	div_b_.z = max_b_.z - min_b_.z + 1;
	div_b_.C = 0;

	//���ó�������
	divb_mul_.x = 1;
	divb_mul_.y = div_b_.x;
	divb_mul_.z = div_b_.x * div_b_.y;
	divb_mul_.C = 0;

	//���ڼ���idx��pointcloud�����Ĵ洢
	std::vector<cloud_point_index_idx> index_vector;
	index_vector.reserve(InputCloudPoint.size());

	//��һ�����������е㲢�����ǲ��뵽���м���idx��index_vector������;������ͬidxֵ�ĵ㽫�����ڲ���CloudPoint����ͬ��
	for (int i = 0; i < InputCloudPoint.size(); i++)
	{
		int ijk0 = static_cast<int> (floor(InputCloudPoint[i].x * inverse_leaf_size_.x) - static_cast<float> (min_b_.x));
		int ijk1 = static_cast<int> (floor(InputCloudPoint[i].y * inverse_leaf_size_.y) - static_cast<float> (min_b_.y));
		int ijk2 = static_cast<int> (floor(InputCloudPoint[i].z * inverse_leaf_size_.z) - static_cast<float> (min_b_.z));

		//��������Ҷ����
		int idx = ijk0 * divb_mul_.x + ijk1 * divb_mul_.y + ijk2 * divb_mul_.z;
		index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int> (idx), i));
	}
	//�ڶ�����ʹ�ñ�ʾĿ�굥Ԫ���ֵ��Ϊ������index_vector������������;ʵ��������ͬһ�����Ԫ������е㶼���˴�����
	std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());

	//�����������������Ԫ��������Ҫ����������ͬ�ģ����ڵ�idxֵ
	unsigned int total = 0;
	unsigned int index = 0;
	unsigned int min_points_per_voxel_ = 0;
	//first_and_last_indices_vector [i]��ʾ���ڶ�Ӧ�ڵ�i�����������ص�index_vector�еĵ�һ�����index_vector�е��������Լ������ڵ�һ���������
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	first_and_last_indices_vector.reserve(index_vector.size());                              //�����ڴ�ռ�

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

	//���Ĳ����������ģ������ǲ�������λ��
	//OutPointCloud.resize(total);      //��������Ʒ����ڴ�ռ�
	float x_Sum, y_Sum, z_Sum;
	PointT PointCloud;
	unsigned int first_index, last_index;
	for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
	{
		// �������� - �������������ĺ�ֵ����Щֵ��index_vector�����о�����ͬ��idxֵ
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

//ȡ�����ڵ�һ�����������أ����������ص������
template <class PointT>
void VoxelGrid<PointT>::VoxelGrid_ApplyFilter(std::vector<PointT> &OutPointCloud)
{
	//���ж�����ĵ����Ƿ�Ϊ��
	if (InputCloudPoint.size() == 0)
	{
		//cout << "�������Ϊ�գ�" << endl;
		return;
	}
	//���������Ƶ��������С����
	Array4f min_p, max_p;
	GetMaxMin(min_p, max_p);

	Array4f inverse_leaf_size_;
	inverse_leaf_size_.x = 1 / X_Voxel;
	inverse_leaf_size_.y = 1 / Y_Voxel;
	inverse_leaf_size_.z = 1 / Z_Voxel;
	inverse_leaf_size_.C = 1;

	//������С�����߽��ֵ
	Array4f min_b_, max_b_, div_b_, divb_mul_;
	min_b_.x = static_cast<int> (floor(min_p.x * inverse_leaf_size_.x));
	max_b_.x = static_cast<int> (floor(max_p.x * inverse_leaf_size_.x));
	min_b_.y = static_cast<int> (floor(min_p.y * inverse_leaf_size_.y));
	max_b_.y = static_cast<int> (floor(max_p.y * inverse_leaf_size_.y));
	min_b_.z = static_cast<int> (floor(min_p.z * inverse_leaf_size_.z));
	max_b_.z = static_cast<int> (floor(max_p.z * inverse_leaf_size_.z));

	//����������������ķָ���
	div_b_.x = max_b_.x - min_b_.x + 1;
	div_b_.y = max_b_.y - min_b_.y + 1;
	div_b_.z = max_b_.z - min_b_.z + 1;
	div_b_.C = 0;

	//���ó�������
	divb_mul_.x = 1;
	divb_mul_.y = div_b_.x;
	divb_mul_.z = div_b_.x * div_b_.y;
	divb_mul_.C = 0;

	//���ڼ���idx��pointcloud�����Ĵ洢
	std::vector<cloud_point_index_idx> index_vector;
	index_vector.reserve(InputCloudPoint.size());

	//��һ�����������е㲢�����ǲ��뵽���м���idx��index_vector������;������ͬidxֵ�ĵ㽫�����ڲ���CloudPoint����ͬ��
	for (int i = 0; i < InputCloudPoint.size(); i++)
	{
		int ijk0 = static_cast<int> (floor(InputCloudPoint[i].x * inverse_leaf_size_.x) - static_cast<float> (min_b_.x));
		int ijk1 = static_cast<int> (floor(InputCloudPoint[i].y * inverse_leaf_size_.y) - static_cast<float> (min_b_.y));
		int ijk2 = static_cast<int> (floor(InputCloudPoint[i].z * inverse_leaf_size_.z) - static_cast<float> (min_b_.z));

		//��������Ҷ����
		int idx = ijk0 * divb_mul_.x + ijk1 * divb_mul_.y + ijk2 * divb_mul_.z;
		index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int> (idx), i));
	}
	//�ڶ�����ʹ�ñ�ʾĿ�굥Ԫ���ֵ��Ϊ������index_vector������������;ʵ��������ͬһ�����Ԫ������е㶼���˴�����
	std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());

	//�����������������Ԫ��������Ҫ����������ͬ�ģ����ڵ�idxֵ
	unsigned int total = 0;
	unsigned int index = 0;
	unsigned int min_points_per_voxel_ = 0;
	//first_and_last_indices_vector [i]��ʾ���ڶ�Ӧ�ڵ�i�����������ص�index_vector�еĵ�һ�����index_vector�е��������Լ������ڵ�һ���������
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	first_and_last_indices_vector.reserve(index_vector.size());                              //�����ڴ�ռ�

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

	//���Ĳ����������ģ������ǲ�������λ��
	//OutPointCloud.resize(total);      //��������Ʒ����ڴ�ռ�
	float x_Sum, y_Sum, z_Sum;
	PointT PointCloud;
	unsigned int first_index, last_index;
	for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
	{
		// �������� - �������������ĺ�ֵ����Щֵ��index_vector�����о�����ͬ��idxֵ
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
