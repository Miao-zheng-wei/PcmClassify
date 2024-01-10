#ifndef _SUPERVISED_CLASSIFICATION_DENSITYFAST_FEATURE_CALCULATOR_FAST_H_
#define _SUPERVISED_CLASSIFICATION_DENSITYFAST_FEATURE_CALCULATOR_FAST_H_

#define PI 3.14159265358979323846

#include "scFeatureCalculatorBase.h"
#include "VoxelGrid.h"
#include "FlannKDTree.h"
#include <vector>
#include <cmath>

/*! namespace supervised classification */
namespace sc {
	/*---------------------------------------------------------------------------*\
        template<typename PointT> class FeatureCalculatorExample
\*---------------------------------------------------------------------------*/

/*!
 * @brief Example derived class with some todo lists for the calculation of features.
*/
	template<typename PointT>
	class FeatureCalculatorDensityFast : public FeatureCalculatorBase<FeatureCalculatorDensityFast<PointT>, PointT>
	{
	public:
		FeatureCalculatorDensityFast(float radius = 2.5, float voxelsize = 1)
		{
			//20180828
			//m_sphereNNS = new Kdtree<PointT>(false);
			//m_cylinderNNS = new Kdtree<PointT>(2, 0, false);

			m_dimFeatures = 2;
			m_radius = radius;
			m_voxelsize = voxelsize;
			m_volume = 4.0/3.0*PI*(float)pow(m_radius,3);
		}
		~FeatureCalculatorDensityFast()
		{
			//if (m_sphereNNS)
			//{
			//	delete m_sphereNNS; m_sphereNNS = NULL;
			//}
			//if (m_cylinderNNS)
			//{
			//	delete m_cylinderNNS; m_cylinderNNS = NULL;
			//}

			m_dilutedPoints.clear();
		}

		//³éÏ¡  ½¨Ë÷Òý Ô²Öù¡¢ÇòÁÚÓò
		void buildSpatialIndex()
		{
			// 1. dilute input cloud
			//Voxelfilter<PointT> vf;
			//vf.setInputCloud(m_pointVector);
			//vf.setleafSize(m_voxelsize, m_voxelsize, m_voxelsize);
			//vf.filter(&m_dilutedPoints);

			//// 2. build kd-tree
			//m_sphereNNS->BuildKDTree(m_dilutedPoints);
			//m_cylinderNNS->BuildKDTree(m_dilutedPoints);
		}

		//////20180828 pengshuwen
		////´«ÈëÊ÷Ö¸Õë
		void setSpatialIndex(FlannKDTree<PointT>* mm_sphereNNS, FlannKDTree<PointT>* mm_cylinderNNS)
		{
			m_sphereNNS = mm_sphereNNS;
			m_cylinderNNS = mm_cylinderNNS;
		}

		bool calculateFeaturesImpl(FeatureMatrix& featMat)
		{
	//#pragma omp parallel for  
			for(int i=0; i<m_numPoints;i++)
			{
				FeatureVector featVec = featMat.row(i);
				calculateFeaturesImpl(i, featVec);
			}
			return true; 
		}

		inline bool calculateFeaturesImpl(int index, FeatureVector& featVec)
		{

			PointT &searchPoint = m_pointVector[index];
			
			std::vector<int> pointSphereRadiusSearch;
			std::vector<float> pointSphereRadiusSquaredDistance;

			std::vector<int> pointCylinderRadiusSearch;
			std::vector<float> pointCylinderRadiusSquaredDistance;


			float l_N3D = 0; //the number of points in the sphere
			float l_N2D = 0; //the number of points in the cylinder
			float l_density = 0; //the density of points in the sphere
			float l_DR = 0; // Density ratio

			if(m_sphereNNS->radiusSearch(searchPoint, m_radius, pointSphereRadiusSearch, pointSphereRadiusSquaredDistance)>0)
			{
				l_N3D = pointSphereRadiusSearch.size();
				l_density = l_N3D/m_volume;
			}

			if (m_cylinderNNS->radiusSearch(searchPoint, m_radius, pointCylinderRadiusSearch, pointCylinderRadiusSquaredDistance)>0)
			{
				l_N2D = pointCylinderRadiusSearch.size();
				l_DR = l_N3D/l_N2D * 3.0/4.0*m_radius;
			}

			featVec(0) = l_density; 
			featVec(1) = l_DR;

			return true;
		}

		void setSearchRadius(float radius)
		{
			m_radius = radius; 
		}

	private:
		std::vector<PointT> m_dilutedPoints;

		FlannKDTree<PointT>* m_sphereNNS;
		FlannKDTree<PointT>* m_cylinderNNS;
		float m_radius; 
		float m_voxelsize; 
		float m_volume;
	};


}// namespace sc




#endif // _SUPERVISED_CLASSIFICATION_DENSITYFAST_FEATURE_CALCULATOR_FAST_H_