#pragma once

#include <vector>

#include <flann/flann.hpp>

/********************************************************************
 *@  Info          :    FlannKDTree 类注释
 *@  ClassName     :    
 *@  BaseClassname :    NULL
 *@  UsingNameSpace:    PCM
 *@  Author        :    Liuyang
 *@  Date          :    2019/1/24 9:36
 *@  Brief         :    
 *********************************************************************/
template <typename PointT>
class FlannKDTree
{
  public:
	FlannKDTree();
	~FlannKDTree();

  public:
	//构建KD-Tree
	bool BuildKDTree(std::vector<PointT> &pointCloud);

	//设置参与构建kd树的维度
	void SetDimMark(bool xDimMark = true, bool yDimMark = true, bool zDimMark = true);

	//Set the search epsilon precision (error bound) for nearest neighbors searches.
	void setEpsilon(float eps);

	//是否对搜索结果进行排序，排序会降低速度
	void setSortedResults(bool sorted = false); //好像没有作用

	//搜索给定阈值范围内的所有邻域点，返回搜索到的邻域点的数量
	int radiusSearch(const PointT &point,			  //param[in] point a given \a valid (i.e., finite) query point
					 double radius,					  //param[in] radius the radius of the sphere bounding all of p_q's neighbors
					 std::vector<int> &k_indices,	 //param[out] k_indices the resultant indices of the neighboring points
					 std::vector<float> &k_distances, //param[out] k_distances the resultant squared distances to the neighboring points
					 unsigned int max_nn = 0) const;  //param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be *returned.

	//搜索邻域内K个邻近点，返回搜索到的邻域点的数量
	int nearestKSearch(const PointT &point,					   //param[in] point a given \a valid (i.e., finite) query point
					   int k,								   //param[in] k the number of neighbors to search for
					   std::vector<int> &k_indices,			   //param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
					   std::vector<float> &k_distances) const; //param[out] k_distances the resultant squared distances to the neighboring points (must be resized to \a k* a priori!)

  private:
	//点云数据
	std::vector<PointT> m_pointCloud;

	//向量格式的点云数据
	float *m_pointCloudMat;

	//falnn kdtree指针
	::flann::Index<flann::L2_Simple<float>> *m_kdTreePtr;

	//标记参与构建kd树的维度，可能只有一或者两个维度参与构建kd树
	bool m_dimMark[3];

	//标记参与构建kd树的维度的数量
	int m_dimCount;

	//eps precision (error bound) for nearest neighbors searches
	float m_epsilon;

	//brief Whether the searched neighbors by radius will be sorted.
	bool m_sorted;

	//
	::flann::SearchParams m_param_k;

	//
	::flann::SearchParams m_param_radius;
};

template <typename PointT>
FlannKDTree<PointT>::FlannKDTree()
{
	//初始化
	m_pointCloudMat = nullptr;
	m_kdTreePtr = nullptr;
	for (int i = 0; i < 3; ++i)
	{
		m_dimMark[i] = true;
	}
	m_dimCount = 0;
	m_epsilon = 0.0f;
	m_sorted = false;
	m_param_k = ::flann::SearchParams(-1, m_epsilon);
	m_param_radius = ::flann::SearchParams(-1, m_epsilon, false);
}

template <typename PointT>
FlannKDTree<PointT>::~FlannKDTree()
{
	if (m_pointCloudMat != nullptr)
	{
		delete m_pointCloudMat;
		m_pointCloudMat = nullptr;
	}
	if (m_kdTreePtr != nullptr)
	{
		delete[] m_kdTreePtr;
		m_kdTreePtr = nullptr;
	}
}

template <typename PointT>
bool FlannKDTree<PointT>::BuildKDTree(std::vector<PointT> &pointCloud)
{
	m_pointCloud = pointCloud;

	//计算参与构建kd树的维度数量
	for (int i = 0; i < 3; ++i)
	{
		if (m_dimMark[i])
		{
			m_dimCount += 1;
		}
	}

	//std::cout << "参与建树的维度数量：" << m_dimCount << std::endl;

	//为存储点云数据的向量开辟空间
	m_pointCloudMat = new float[pointCloud.size() * m_dimCount];

	//点云数据转存
	if (m_dimCount == 3)
	{
		for (int i = 0; i < pointCloud.size(); ++i)
		{
			m_pointCloudMat[i * m_dimCount] = pointCloud[i].x;
			m_pointCloudMat[i * m_dimCount + 1] = pointCloud[i].y;
			m_pointCloudMat[i * m_dimCount + 2] = pointCloud[i].z;
		}
	}
	else
	{
		for (int i = 0; i < pointCloud.size(); ++i)
		{
			if (m_dimMark[0])
				m_pointCloudMat[i * m_dimCount] = pointCloud[i].x;
			if (m_dimMark[1])
				m_pointCloudMat[i * m_dimCount + 1] = pointCloud[i].y;
			if (m_dimMark[2])
				m_pointCloudMat[i * m_dimCount + 2] = pointCloud[i].z;
		}
	}

	//构建kd树
	m_kdTreePtr = new ::flann::Index<::flann::L2_Simple<float>>(
		::flann::Matrix<float>(m_pointCloudMat, pointCloud.size(), m_dimCount),
		::flann::KDTreeSingleIndexParams(15));

	m_kdTreePtr->buildIndex();

	//std::cout << "构建KD树完成" << m_dimCount << std::endl;

	return true;
}

template <typename PointT>
int FlannKDTree<PointT>::nearestKSearch(const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_distances) const
{
	//assert(point_representation_->isValid(point) && "Invalid (NaN, Inf) point coordinates given to nearestKSearch!");

	if (k > m_pointCloud.size())
	{
		k = m_pointCloud.size();
	}

	k_indices.resize(k);
	k_distances.resize(k);

	std::vector<float> query(m_dimCount);
	if (m_dimMark[0])
		query[0] = point.x;
	if (m_dimMark[1])
		query[1] = point.y;
	if (m_dimMark[2])
		query[2] = point.z;

	::flann::Matrix<int> k_indices_mat(&k_indices[0], 1, k);
	::flann::Matrix<float> k_distances_mat(&k_distances[0], 1, k);

	// Wrap the k_indices and k_distances vectors (no data copy)
	m_kdTreePtr->knnSearch(::flann::Matrix<float>(&query[0], 1, m_dimCount),
						   k_indices_mat,
						   k_distances_mat,
						   k,
						   m_param_k);

	//k_indices = k_distances_mat[0];
	//k_distances = k_distances_mat[0];

	return k;
}

template <typename PointT>
int FlannKDTree<PointT>::radiusSearch(const PointT &point, double radius, std::vector<int> &k_indices, std::vector<float> &k_distances, unsigned int max_nn) const
{
	//assert(point_representation_->isValid(point) && "Invalid (NaN, Inf) point coordinates given to radiusSearch!");

	std::vector<float> query(m_dimCount);
	if (m_dimMark[0])
		query[0] = point.x;
	if (m_dimMark[1])
		query[1] = point.y;
	if (m_dimMark[2])
		query[2] = point.z;

	// Has max_nn been set properly?
	if (max_nn == 0 || max_nn >= m_pointCloud.size())
		max_nn = m_pointCloud.size() - 1; // return all neighbors in radius

	std::vector<std::vector<int>> indices(1);
	std::vector<std::vector<float>> dists(1);

	int neighborsCountInRadius = m_kdTreePtr->radiusSearch(
		::flann::Matrix<float>(&query[0], 1, m_dimCount),
		indices,
		dists,
		static_cast<float>(radius * radius),
		m_param_radius);

	k_indices = indices[0];
	k_distances = dists[0];

	return neighborsCountInRadius;
}

template <typename PointT>
void FlannKDTree<PointT>::SetDimMark(bool xDimMark, bool yDimMark, bool zDimMark)
{
	//确保至少还有一个维度的数据参与构建kd树
	assert(!(!xDimMark && !yDimMark && !zDimMark));

	//赋予标记
	m_dimMark[0] = xDimMark; //标记X是否参与建树
	m_dimMark[1] = yDimMark; //标记Y是否参与建树
	m_dimMark[2] = zDimMark; //标记Z是否参与建树
}

template <typename PointT>
void FlannKDTree<PointT>::setEpsilon(float eps)
{
	m_epsilon = eps;
	m_param_k = ::flann::SearchParams(-1, m_epsilon);
	m_param_radius = ::flann::SearchParams(-1, m_epsilon, m_sorted);
}

template <typename PointT>
void FlannKDTree<PointT>::setSortedResults(bool sorted)
{
	m_sorted = sorted;
	m_param_k = ::flann::SearchParams(-1, m_epsilon);
	m_param_radius = ::flann::SearchParams(-1, m_epsilon, m_sorted);
}