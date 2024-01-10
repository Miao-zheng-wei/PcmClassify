
#ifndef _SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_CDHV_FAST_H_
#define _SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_CDHV_FAST_H_


#include "scFeatureCombinerBase.h"
#include "scFeatureCalculatorCovFast.h"     /*!< covariance-based */
#include "scFeatureCalculatorDensityFast.h" /*!< density-based */
#include "scFeatureCalculatorHeightFast.h"  /*!< height-based */
#include "VoxelGrid.h"
#include "FeatureCalculatorDH.h"

/*! namespace supervised classification */
namespace sc {

	/*---------------------------------------------------------------------------*\
				template<typename PointT> class FeatureCombinerCDHVFast
	\*---------------------------------------------------------------------------*/

	/*!
	* @brief class combining covariance-based, density-based and height-based features.
	*/
	template<typename PointT>
	class FeatureCombinerCDHVFast : public FeatureCombinerBase<FeatureCombinerCDHVFast<PointT>, PointT>
	{
	public:
		FeatureCombinerCDHVFast();
		~FeatureCombinerCDHVFast();

		void setInputCloudImpl();

		bool buildSpatialIndexImpl();

		bool combineFeaturesImpl(FeatureMatrix& featMat, GenericProgressCallback *pdlg=0);

	private:
		/*! concrete feature calculator that will be combined. */
		FeatureCalculatorCovFast<PointT>* m_calculatorCov; /*!< covariance-based */
		FeatureCalculatorDH<PointT>* m_calculatorDH;


		int m_startIndex[3]; /*!< start index of feature dimension for all 4 feature calculators above */
		int m_endIndex[3]; /*!< end index of feature dimension for all 4 feature calculators above  */

		/*! concrete spatial indices that will be built and passed to feature calculators. */
		FlannKDTree<PointT>* m_kdtreeNNS; /*!< kdtree nearest neighbor search */
		////20180828
		//std::vector<PointT> m_dilutedPoints;
		FlannKDTree<PointT>* m_cylinderNNS;
		//FlannKDTree<PointT>* m_sphereNNS;
	//	float m_voxelsize;

	};

	/*---------------------------------------------------------------------------*\
								Implementation of
				  template<typename PointT> class FeatureCombinerCDHVFast
	\*---------------------------------------------------------------------------*/

	template<typename PointT>
	inline FeatureCombinerCDHVFast<PointT>::FeatureCombinerCDHVFast()
	{
		m_calculatorCov = new FeatureCalculatorCovFast<PointT>;
		m_calculatorDH = new FeatureCalculatorDH<PointT>;
		//m_calculatorHeight = new FeatureCalculatorHeightFast<PointT>;
	//	m_calculatorVerticalProfile = new FeatureCalculatorVerticalProfile<PointT>;

		m_kdtreeNNS = new FlannKDTree<PointT>();
		//m_sphereNNS = new FlannKDTree<PointT>();
		m_cylinderNNS = new FlannKDTree<PointT>();
		m_cylinderNNS->SetDimMark(true, true, false);

		m_dimFeatures = m_calculatorCov->getFeatureDimension()
			+ m_calculatorDH->getFeatureDimension();

		m_startIndex[0] = 0;
		m_endIndex[0] = m_startIndex[0] + m_calculatorCov->getFeatureDimension();
		m_startIndex[1] = m_endIndex[0];
		m_endIndex[1] = m_startIndex[1] + m_calculatorDH->getFeatureDimension();
		//m_startIndex[2] = m_endIndex[1];
		//m_endIndex[2] = m_startIndex[2] + m_calculatorHeight->getFeatureDimension();
		//m_startIndex[3] = m_endIndex[2];
		//m_endIndex[3] = m_startIndex[3] + m_calculatorVerticalProfile->getFeatureDimension();
	}

	template<typename PointT>
	inline FeatureCombinerCDHVFast<PointT>::~FeatureCombinerCDHVFast()
	{
		if (m_calculatorCov)
		{
			delete m_calculatorCov; m_calculatorCov = NULL;
		}
		if (m_calculatorDH)
		{
			delete m_calculatorDH; m_calculatorDH = NULL;
		}
		//if (m_calculatorHeight)
		//{
		//	delete m_calculatorHeight; m_calculatorHeight = NULL;
		//}

		if (m_kdtreeNNS)
		{
			delete m_kdtreeNNS; m_kdtreeNNS = NULL;
		}
		//if (m_sphereNNS)
		//{
		//	delete m_sphereNNS; m_sphereNNS = NULL;
		//}
		if (m_cylinderNNS)
		{
			delete m_cylinderNNS; m_cylinderNNS = NULL;
		}
	}

	template<typename PointT>
	inline void FeatureCombinerCDHVFast<PointT>::setInputCloudImpl()
	{
		m_calculatorCov->setInputCloud(&m_pointVector);
		m_calculatorDH->setInputCloud(&m_pointVector);
		//m_calculatorHeight->setInputCloud(&m_pointVector);
		//m_calculatorVerticalProfile->setInputCloud(m_pointVector);
	}

	template<typename PointT>
	inline bool FeatureCombinerCDHVFast<PointT>::buildSpatialIndexImpl()
	{

		//VoxelGrid<PointT>vf;
		//vf.setInputCloud(m_pointVector);
		//m_voxelsize = 0.3;

		//vf.setLeafSize(m_voxelsize, m_voxelsize, m_voxelsize);
		//vf.VoxelGrid_ApplyFilter(m_dilutedPoints, index_vector, first_and_last_indices_vector);

		m_kdtreeNNS->BuildKDTree(m_pointVector);

		//m_sphereNNS->BuildKDTree(m_dilutedPoints);

		m_cylinderNNS->BuildKDTree(m_pointVector);

		m_calculatorCov->setSpatialIndex(m_kdtreeNNS);

		m_calculatorDH->setSpatialIndex(m_kdtreeNNS, m_cylinderNNS);

		return true;
	}



	template<typename PointT>
	inline bool FeatureCombinerCDHVFast<PointT>::combineFeaturesImpl(FeatureMatrix & featMat, GenericProgressCallback *pdlg =0)
	{
		//基于协方差计算出来九个特征 
		FeatureMatrix featMatCov = featMat.col_range(m_startIndex[0], m_endIndex[0]);
		m_calculatorCov->calculateFeatures(featMatCov, pdlg);

		//基于直方图计算出来特征七个特征
		FeatureMatrix featMatDH = featMat.col_range(m_startIndex[1], m_endIndex[1]);
		m_calculatorDH->calculateFeatures(featMatDH, pdlg);

		return true;
	}


} // namespace sc

#endif // !_SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_CDHV_FAST_H_
