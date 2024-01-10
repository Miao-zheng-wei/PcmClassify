#ifndef _SUPERVISED_CLASSIFICATION_HEIGHTFAST_FEATURE_CALCULATOR_FAST_H_
#define _SUPERVISED_CLASSIFICATION_HEIGHTFAST_FEATURE_CALCULATOR_FAST_H_

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
class FeatureCalculatorHeightFast : public FeatureCalculatorBase<FeatureCalculatorHeightFast<PointT>, PointT>
{

public:

	FeatureCalculatorHeightFast(double radius = 2.5,float voxelsize =1)
	{
		// todo: 1. initialize the dimension of features.
		//       2. initialize multiple different spatial indices as null pointers.
		m_dimFeatures = 5;
		//m_sphereNNS = new Kdtree<PointT>(false);
		//m_cylinderNNS = new Kdtree<PointT>(2, 0, false);
		m_radius = radius ;
		m_voxelsize = voxelsize;
	}
	~FeatureCalculatorHeightFast() {

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

	//½¨Ë÷Òý Ô²Öù¡¢ÇòÁÚÓò
	void buildSpatialIndex()
	{
		//// 1. dilute input cloud
		//Voxelfilter<PointT> vf;
		//vf.setInputCloud(m_pointVector);
		//vf.setleafSize(m_voxelsize, m_voxelsize, m_voxelsize);
		//vf.filter(&m_dilutedPoints);

		//// 2. build kd-tree
		//m_sphereNNS->BuildKDTree(m_dilutedPoints);
		//m_cylinderNNS->BuildKDTree(m_dilutedPoints);

	}

	void setSpatialIndex(FlannKDTree<PointT>* mm_sphereNNS, FlannKDTree<PointT>* mm_cylinderNNS, std::vector<PointT> mm_dilutedPoints)
	{
		m_sphereNNS = mm_sphereNNS;
		m_cylinderNNS = mm_cylinderNNS;
		m_dilutedPoints = mm_dilutedPoints;
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

		//PointT &searchPoint = (*m_pointVector)[index];
		PointT &searchPoint = m_pointVector[index];

		float l_CylinderZ = 0,l_SphereZ = 0;
		ST_Height l_stuHeight;
		
		std::vector<int> pointCylinderRadiusSearch;
		std::vector<float> pointCylinderRadiusSquaredDistance;

		float *CylinderZArr = NULL;
		if (m_cylinderNNS->radiusSearch(searchPoint, m_radius, pointCylinderRadiusSearch, pointCylinderRadiusSquaredDistance)>0)
		{	
			CylinderZArr=new float [pointCylinderRadiusSearch.size()];
//#pragma omp parallel for  
			for(int k = 0; k < pointCylinderRadiusSearch.size (); ++k)
			{
				l_CylinderZ = m_dilutedPoints[ pointCylinderRadiusSearch[k] ].z ;
				CylinderZArr[k]=l_CylinderZ;
			}
		}

		
		float l_CylinderZMin = 0, l_CylinderZMax = 0;
		
		CompMaxZAndMinZ(l_CylinderZMin,l_CylinderZMax, CylinderZArr);
		
		float CylinderZSum=0;

		for (size_t j = 0; j<pointCylinderRadiusSearch.size(); j++)
		{
			CylinderZSum = CylinderZSum + CylinderZArr[j];
		}
		
		float CylinderMidZ = CylinderZSum / pointCylinderRadiusSearch.size();

		l_stuHeight.HeightAbove = l_CylinderZMax - searchPoint.z; 
		l_stuHeight.HeightBelow = searchPoint.z - l_CylinderZMin;
		l_stuHeight.VerticalRange = l_CylinderZMax - l_CylinderZMin;

		float CylinderAccum = 0;

		for (size_t j = 0; j<pointCylinderRadiusSearch.size(); j++)
		{
			CylinderAccum+= (CylinderZArr[j]-CylinderMidZ)*(CylinderZArr[j]-CylinderMidZ);
		}

		float Cylinderstdev;
		if (pointCylinderRadiusSearch.size() == 1)
		{
			Cylinderstdev = 0;
		}
		else
		{
			Cylinderstdev = sqrt(CylinderAccum / (pointCylinderRadiusSearch.size() - 1));
		}

		l_stuHeight.CylinderVariance = Cylinderstdev;

		//compute point cloud features based on sphere 

		std::vector<int> pointSphereRadiusSearch;
		std::vector<float> pointSphereRadiusSquaredDistance;

		float *SphereZArr = NULL;
		if (m_sphereNNS->radiusSearch(searchPoint, m_radius, pointSphereRadiusSearch, pointSphereRadiusSquaredDistance,25)>0)
		{
			SphereZArr = new float [pointSphereRadiusSearch.size()];
//#pragma omp parallel for  
			for(int k = 0; k < pointSphereRadiusSearch.size (); ++k)
			{
				l_SphereZ = m_dilutedPoints[ pointSphereRadiusSearch[k] ].z ;
				SphereZArr[k]=l_SphereZ;
			}
		}


		float SphereZSum=0;

		for (size_t j = 0; j<pointSphereRadiusSearch.size(); j++)
		{
			SphereZSum = SphereZSum + SphereZArr[j];
		}

		float SphereMidZ = 0;
		SphereMidZ = SphereZSum / pointSphereRadiusSearch.size();

		float SphereAccum = 0;

		for (size_t j = 0; j<pointSphereRadiusSearch.size(); j++)
		{
			SphereAccum+= (SphereZArr[j]-SphereMidZ)*(SphereZArr[j]-SphereMidZ);
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

		featVec(0) = l_stuHeight.VerticalRange;
		featVec(1) = l_stuHeight.HeightAbove;
		featVec(2) = l_stuHeight.HeightBelow;
		featVec(3) = l_stuHeight.SphereVariance;
		featVec(4) = l_stuHeight.CylinderVariance;
		
		if (SphereZArr!=NULL)
		{
			delete[]SphereZArr;
		}

		if (CylinderZArr != NULL)
		{
			delete[]CylinderZArr;
		}

		return true; 
	}


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
			memset(this,0,sizeof(ST_Height));
		}

		ST_Height(const ST_Height & other)
		{
			*this = other;
		}
	};

	//find the Minimum Z coordinate and Maximum Z coordinate
	void CompMaxZAndMinZ(float &l_CylinderZMin, float &l_CylinderZMax ,float *CylinderZArr);
private:
	std::vector<PointT> m_dilutedPoints;
	FlannKDTree<PointT>* m_cylinderNNS;
	FlannKDTree<PointT>* m_sphereNNS;
	double m_radius;
	float m_voxelsize; 
};


template<typename PointT>
inline void FeatureCalculatorHeightFast<PointT>::CompMaxZAndMinZ(float &l_CylinderZMin, float &l_CylinderZMax, float * CylinderZArr)
{
	float l_TempZ = 0;
	if (CylinderZArr != NULL)
	{
		l_CylinderZMin = CylinderZArr[0];
		//find the Minimum Z coordinate 
		int ArrSize = sizeof(CylinderZArr) / sizeof(CylinderZArr[0]);
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
}// namespace sc

#endif // _SUPERVISED_CLASSIFICATION_HEIGHTFAST_FEATURE_CALCULATOR_FAST_H_

