#ifndef _SUPERVISED_CLASSIFICATION_DH_FEATURE_CALCULATOR_FAST_H_
#define _SUPERVISED_CLASSIFICATION_DH_FEATURE_CALCULATOR_FAST_H_

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#define PI 3.14159265358979323846
#include "packed/scFeatureCalculatorBase.h"
#include "VoxelGrid.h"
#include "FlannKDTree.h"
#include <vector>
#include <cmath>

#include <algorithm>
/*! namespace supervised classification */
namespace sc {
	template<typename PointT>
	class FeatureCalculatorDH : public FeatureCalculatorBase<FeatureCalculatorDH<PointT>, PointT>
	{
	public:
		FeatureCalculatorDH(float radius = 2, float voxelsize = 0.3)
		{
			m_dimFeatures = 7;
			m_radius = radius;
			m_voxelsize = voxelsize;
			m_volume = 4.0 / 3.0*PI*(float)pow(m_radius, 3);
		}
		~FeatureCalculatorDH()
		{
			//m_dilutedPoints.clear();
		}

		//抽稀  建索引 圆柱、球邻域
		void buildSpatialIndex()
		{
		}

		void setSpatialIndex(FlannKDTree<PointT>* mm_sphereNNS, FlannKDTree<PointT>* mm_cylinderNNS/*, std::vector<PointT> &mm_dilutedPoints*/)
		{
			m_sphereNNS = mm_sphereNNS;
			m_cylinderNNS = mm_cylinderNNS;
			//m_dilutedPoints = mm_dilutedPoints;
		}

		bool calculateFeaturesImpl(FeatureMatrix& featMat,GenericProgressCallback *m_pdlg = 0)
		{
			NormalizedProgress nProgress(m_pdlg, m_numPoints);
			for (int i = 0; i < m_numPoints; i++)
			{
				nProgress.oneStep();
				//if (!nProgress.oneStep())
				//{
				//	m_pdlg->stop();
	

				//	QCoreApplication::processEvents();//避免界面冻结
				//	QEventLoop eventloop;
				//	QTimer::singleShot(100, &eventloop, SLOT(quit()));
				//	eventloop.exec();

				//	return false;
				//}

				FeatureVector featVec = featMat.row(i);
				calculateFeaturesImpl(i, featVec);
			}
			return true;
		}

		inline bool calculateFeaturesImpl(int index, FeatureVector& featVec)
		{

			PointT &searchPoint = m_pointVector[index];
			ST_Height l_stuHeight;

			std::vector<int> pointRadiusSearch;
			std::vector<float> pointRadiusSquaredDistance;

			//std::vector<int> pointCylinderRadiusSearch;
			//std::vector<float> pointCylinderRadiusSquaredDistance;

			float l_N3D = 0; //the number of points in the sphere
			float l_N2D = 0; //the number of points in the cylinder
			float l_density = 0; //the density of points in the sphere
			float l_DR = 0; // Density ratio

			float *SphereZArr = NULL;
			float ZSum = 0;
			if (m_sphereNNS->radiusSearch(searchPoint, m_radius, pointRadiusSearch, pointRadiusSquaredDistance) > 0)
			{
				l_N3D = pointRadiusSearch.size();
				l_density = l_N3D / m_volume;

				//height
				SphereZArr = new float[pointRadiusSearch.size()];
				//#pragma omp parallel for  
				for (int k = 0; k < pointRadiusSearch.size(); ++k)
				{
					SphereZArr[k] = m_pointVector[pointRadiusSearch[k]].z;
					ZSum = ZSum + SphereZArr[k];
				}
			}

			float SphereMidZ  = ZSum / pointRadiusSearch.size();

			float Accum = 0;

			for (size_t j = 0; j < pointRadiusSearch.size(); j++)
			{
				Accum += (SphereZArr[j] - SphereMidZ)*(SphereZArr[j] - SphereMidZ);
			}

			l_stuHeight.SphereVariance = sqrt(Accum / (pointRadiusSearch.size()));


			//////////////////////////////////////////////////////////////////////////
			float *CylinderZArr = NULL;
			ZSum = 0;
			pointRadiusSquaredDistance.resize(0);
			pointRadiusSearch.resize(0);

			if (m_cylinderNNS->radiusSearch(searchPoint, m_radius/2, pointRadiusSearch, pointRadiusSquaredDistance) > 0)
			{
				l_N2D = pointRadiusSearch.size();
				l_DR = l_N3D / l_N2D * 3.0 / 4.0*m_radius;

				//Height
				CylinderZArr = new float[pointRadiusSearch.size()];
				//#pragma omp parallel for  
				for (int k = 0; k < pointRadiusSearch.size(); ++k)
				{
					CylinderZArr[k] = m_pointVector[pointRadiusSearch[k]].z;
					ZSum = ZSum + CylinderZArr[k];
				}
			}

			//height
			float l_CylinderZMin = 0, l_CylinderZMax = 0;

			//CompMaxZAndMinZ(l_CylinderZMin, l_CylinderZMax, CylinderZArr);
			l_CylinderZMin = *std::min_element(CylinderZArr, CylinderZArr + pointRadiusSearch.size());
			l_CylinderZMax = *std::max_element(CylinderZArr, CylinderZArr + pointRadiusSearch.size());

			float CylinderMidZ = ZSum / pointRadiusSearch.size();

			l_stuHeight.HeightAbove = l_CylinderZMax - searchPoint.z;
			l_stuHeight.HeightBelow = searchPoint.z - l_CylinderZMin;
			l_stuHeight.VerticalRange = l_CylinderZMax - l_CylinderZMin;

			Accum = 0;
			for (size_t j = 0; j < pointRadiusSearch.size(); j++)
			{
				Accum += (CylinderZArr[j] - CylinderMidZ)*(CylinderZArr[j] - CylinderMidZ);
			}

			l_stuHeight.CylinderVariance = sqrt(Accum / (pointRadiusSearch.size()));


			featVec(0) = l_density;						 // 点云在球形体积中的密度
			featVec(1) = l_DR;							 // 球形和圆柱形中的点的密度比
			featVec(2) = l_stuHeight.VerticalRange;		 // 圆柱形区域中点的z坐标的范围（最大z坐标 - 最小z坐标）
			featVec(3) = l_stuHeight.HeightAbove;		 // 圆柱形区域中点的最大z坐标与搜索点z坐标的差值。
			featVec(4) = l_stuHeight.HeightBelow;		 // 搜索点z坐标与圆柱形区域中点的最小z坐标的差值。
			featVec(5) = l_stuHeight.SphereVariance;     // 球形区域中点z坐标的标准差。
			featVec(6) = l_stuHeight.CylinderVariance;   // 圆柱形区域中点z坐标的标准差。


			delete []SphereZArr;
			delete []CylinderZArr;

			return true;
		}
//		inline bool calculateFeaturesImpl(int index, FeatureVector& featVec)
//		{
//
//			PointT &searchPoint = m_pointVector[index];
//#pragma omp parallel sections
//			{
//#pragma omp section
//				{
//					bool iscylinder = calculateFeaturesCylinder(searchPoint, featVec);
//					if (!iscylinder)
//						return false
//				}
//#pragma omp section
//				{
//					bool isSphere = calculateFeaturesSphere(searchPoint, featVec);
//					if (!isSphere)
//						return false;
//				}
//
//			}
//			return true;
//		}

		inline bool calculateFeaturesCylinder(PointT &searchPoint, FeatureVector& featVec)
		{
			float l_CylinderZ = 0, l_SphereZ = 0;
			ST_Height l_stuHeight;

			std::vector<int> pointCylinderRadiusSearch;
			std::vector<float> pointCylinderRadiusSquaredDistance;

			//////////////////////////////////////////////////////////////////////////
			float *CylinderZArr = NULL;
			if (m_cylinderNNS->radiusSearch(searchPoint, m_radius, pointCylinderRadiusSearch, pointCylinderRadiusSquaredDistance) > )
			{
				//l_N2D = pointCylinderRadiusSearch.size();
				//l_DR = l_N3D / l_N2D * 3.0 / 4.0*m_radius;

				//Height
				CylinderZArr = new float[pointCylinderRadiusSearch.size()];
				//#pragma omp parallel for  
				for (int k = 0; k < pointCylinderRadiusSearch.size(); ++k)
				{
					l_CylinderZ = m_pointVector[pointCylinderRadiusSearch[k]].z;
					CylinderZArr[k] = l_CylinderZ;
				}
			}

			//height
			float l_CylinderZMin = 0, l_CylinderZMax = 0;

			CompMaxZAndMinZ(l_CylinderZMin, l_CylinderZMax, CylinderZArr);

			float CylinderZSum = 0;

			for (size_t j = 0; j < pointCylinderRadiusSearch.size(); j++)
			{
				CylinderZSum = CylinderZSum + CylinderZArr[j];
			}

			float CylinderMidZ = CylinderZSum / pointCylinderRadiusSearch.size();

			l_stuHeight.HeightAbove = l_CylinderZMax - searchPoint.z;
			l_stuHeight.HeightBelow = searchPoint.z - l_CylinderZMin;
			l_stuHeight.VerticalRange = l_CylinderZMax - l_CylinderZMin;

			float CylinderAccum = 0;

			for (size_t j = 0; j < pointCylinderRadiusSearch.size(); j++)
			{
				CylinderAccum += (CylinderZArr[j] - CylinderMidZ)*(CylinderZArr[j] - CylinderMidZ);
			}

			float Cylinderstdev;
			if (pointCylinderRadiusSearch.size() != 1)
			{
				Cylinderstdev = 0;
			}
			else
			{
				l_stuHeight.CylinderVariance = sqrt(CylinderAccum / (pointCylinderRadiusSearch.size() - 1));
			}

			l_stuHeight.CylinderVariance = Cylinderstdev;

			featVec(1) = l_stuHeight.VerticalRange;
			featVec(2) = l_stuHeight.HeightAbove;
			featVec(3) = l_stuHeight.HeightBelow;
			featVec(4) = l_stuHeight.SphereVariance;

			if (CylinderZArr != NULL)
			{
				delete[]CylinderZArr;
			}
			return true;
		}
		inline bool calculateFeaturesSphere(PointT &searchPoint, FeatureVector& featVec)
		{
			float l_CylinderZ = 0, l_SphereZ = 0;
			ST_Height l_stuHeight;

			std::vector<int> pointSphereRadiusSearch;
			std::vector<float> pointSphereRadiusSquaredDistance;


			float l_N3D = 0; //the number of points in the sphere
			float l_N2D = 0; //the number of points in the cylinder
			float l_density = 0; //the density of points in the sphere
			//float l_DR = 0; // Density ratio

			float *SphereZArr = NULL;
			if (m_sphereNNS->radiusSearch(searchPoint, m_radius, pointSphereRadiusSearch, pointSphereRadiusSquaredDistance) > 0)
			{
				l_N3D = pointSphereRadiusSearch.size();
				l_density = l_N3D / m_volume;

				//height
				SphereZArr = new float[pointSphereRadiusSearch.size()];
				//#pragma omp parallel for  
				for (int k = 0; k < pointSphereRadiusSearch.size(); ++k)
				{
					l_SphereZ = m_pointVector[pointSphereRadiusSearch[k]].z;
					SphereZArr[k] = l_SphereZ;
				}
			}

			//height
			float SphereZSum = 0;

			for (size_t j = 0; j < pointSphereRadiusSearch.size(); j++)
			{
				SphereZSum = SphereZSum + SphereZArr[j];
			}

			float SphereMidZ = 0;
			SphereMidZ = SphereZSum / pointSphereRadiusSearch.size();

			float SphereAccum = 0;

			for (size_t j = 0; j < pointSphereRadiusSearch.size(); j++)
			{
				SphereAccum += (SphereZArr[j] - SphereMidZ)*(SphereZArr[j] - SphereMidZ);
			}

			float Spherestdev;
			if (pointSphereRadiusSearch.size() == 1)
			{
				Spherestdev = 0;
			}
			else
			{
				Spherestdev = sqrt(SphereAccum / (pointSphereRadiusSearch.size() - 1));
			}
			l_stuHeight.SphereVariance = Spherestdev;

			featVec(0) = l_density;
			featVec(4) = l_stuHeight.SphereVariance;

			if (SphereZArr != NULL)
			{
				delete[]SphereZArr;
			}
			return true;
		}

		void setSearchRadius(float radius)
		{
			m_radius = radius;
		}

	private:
		//std::vector<PointT> m_dilutedPoints;
		FlannKDTree<PointT>* m_sphereNNS;
		FlannKDTree<PointT>* m_cylinderNNS;
		float m_radius;
		float m_voxelsize;
		float m_volume;

		private:
			struct ST_Height
			{
				float VerticalRange;
				float HeightBelow;
				float HeightAbove;
				float CylinderVariance;
				float SphereVariance;
				ST_Height()
				{
					memset(this, 0, sizeof(ST_Height));
				}

				ST_Height(const ST_Height & other)
				{
					*this = other;
				}
			};

			//find the Minimum Z coordinate and Maximum Z coordinate
			void CompMaxZAndMinZ(float &l_CylinderZMin, float &l_CylinderZMax, float *CylinderZArr);

	};

	template<typename PointT>
	inline void FeatureCalculatorDH<PointT>::CompMaxZAndMinZ(float &l_CylinderZMin, float &l_CylinderZMax, float * CylinderZArr)
	{
		float l_TempZ = 0;
		if (CylinderZArr != NULL)
		{
			l_CylinderZMin = CylinderZArr[0];
			//find the Minimum Z coordinate 
			int ArrSize = sizeof(CylinderZArr) / sizeof(CylinderZArr[0]); //有错
			for (size_t i = 0; i < ArrSize; i++)
			{
				l_TempZ = CylinderZArr[i];
				if (l_CylinderZMin > l_TempZ)
				{
					l_CylinderZMin = l_TempZ;
				}
			}

			//find the Maximum Z coordinate 
			for (size_t i = 0; i < ArrSize; i++)
			{
				l_TempZ = CylinderZArr[i];
				if (l_CylinderZMax < l_TempZ)
				{
					l_CylinderZMax = l_TempZ;
				}
			}
		}
		else
		{
			l_CylinderZMax = l_TempZ;
			l_CylinderZMin = l_TempZ;
		}
	}

}
#endif