

#ifndef _SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_BASE_H_
#define _SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_BASE_H_

#include "scBasics.h"

/*! namespace supervised classification */
namespace sc {

/*---------------------------------------------------------------------------*\
   template<typename DerivedT, typename PointT> class FeatureCombinerBase
\*---------------------------------------------------------------------------*/

/*!
 * @brief Base class for the combination of different features. A derived class
 *        should call multiple feature calculators and combine them together.
 *
 *        The derived class must implement the following functions:
 *        1. bool buildSpatialIndexImpl();
 *        2. bool combineFeaturesImpl(FeatureMatrix& featMat);
 *        3. inline bool combineFeaturesImpl(size_t index, FeatureVector& featVec).
*/
template<typename DerivedT, typename PointT>
class FeatureCombinerBase
{
public:
	typedef std::vector<PointT> PointTVector;

	FeatureCombinerBase() : m_pointVector(NULL), m_numPoints(0), m_dimFeatures(0) {}
	~FeatureCombinerBase() {}

	/** \brief Set pointer to input cloud data.
	  * \param[in] points the pointer to vector of input points
	*/
	void setInputCloud(PointTVector *pointVector);

	/** \brief Build spatial index for neighborhood search.
	  *        Different spatial indices may be built according to feature calculation.
	  * \return "true" if succeeded.
	*/
	bool buildSpatialIndex();
	
	/** \brief Combine features of all samples for the training process.
	  * \param[out] featMatData the row-major feature matrix data with size [numPoints*dimFeatures]
	  * \return "true" if succeeded.
	*/
	bool combineFeatures(std::vector<FeatElemType>& featMatData);
	
	/** \brief Combine features of one sample for the evaluation process.
	  * \param[in] index the point index of input cloud
	  * \param[out] featVecData the feature vector data with size [dimFeatures]
	  * \return "true" if succeeded.
	*/
	inline bool combineFeatures(size_t index, std::vector<FeatElemType>& featVecData);

	/** \brief Get dimension of features.
	  * \return demension of features.
	*/
	size_t getFeatureDimension() { return m_dimFeatures; }

protected:
	/*! pointer to input cloud data  */
	PointTVector m_pointVector;
	/*! number of points */
	size_t m_numPoints;
	/*! dimension of features. it must be determined in the constructor of derived class */
	size_t m_dimFeatures;
	std::vector<int> m_idx;
};

/*---------------------------------------------------------------------------*\
                              Implementation of
    template<typename DerivedT, typename PointT> class FeatureCombinerBase
\*---------------------------------------------------------------------------*/

template<typename DerivedT, typename PointT>
inline void FeatureCombinerBase<DerivedT, PointT>::setInputCloud(PointTVector *pointVector)
{
	assert(NULL != pointVector);
	assert(pointVector->size() > 0);

	m_pointVector = *pointVector;
	m_numPoints = pointVector->size();

	static_cast<DerivedT*>(this)->setInputCloudImpl();
}

template<typename DerivedT, typename PointT>
inline bool FeatureCombinerBase<DerivedT, PointT>::buildSpatialIndex()
{
	return static_cast<DerivedT*>(this)->buildSpatialIndexImpl();
}

template<typename DerivedT, typename PointT>
inline bool FeatureCombinerBase<DerivedT, PointT>::combineFeatures(std::vector<FeatElemType>& featMatData)
{
	assert(m_numPoints > 0 && m_dimFeatures > 0);

	featMatData.clear();
	featMatData.resize(m_numPoints * m_dimFeatures, 0.0);
	FeatureMatrix featMat(&featMatData[0], m_numPoints, m_dimFeatures);

	return static_cast<DerivedT*>(this)->combineFeaturesImpl(featMat);
}

template<typename DerivedT, typename PointT>
inline bool FeatureCombinerBase<DerivedT, PointT>::combineFeatures(size_t index, std::vector<FeatElemType>& featVecData)
{
	assert(m_numPoints > 0 && m_dimFeatures > 0);
	assert(index >= 0 && index < m_numPoints);

	// data size must be prepared beforehand
	//featVecData.clear();
	//featVecData.resize(m_dimFeatures, 0.0);
	assert(featVecData.size() == m_dimFeatures);
	FeatureVector featVec(&featVecData[0], m_dimFeatures);

	return static_cast<DerivedT*>(this)->combineFeaturesImpl(index, featVec);
}

/*---------------------------------------------------------------------------*\
            template<typename PointT> class FeatureCombinerExample
\*---------------------------------------------------------------------------*/

/*!
 * @brief Example derived class with some todo lists for the combination of different features.
*/
template<typename PointT>
class FeatureCombinerExample : public FeatureCombinerBase<FeatureCombinerExample<PointT>, PointT>
{
public:
	FeatureCombinerExample()
	{
		// todo: 1. determine the total dimension of features.
		//          it's the sum of dimensions of all included feature calculators.
		//       2. allocate multiple different feature calculators.
		//       3. initialize multiple different spatial indices if required during feature calculation.
	}
	~FeatureCombinerExample()
	{
		// todo: 1. deallocate feature calculators
		//       2. deallocate spatial indices
	}

	void setInputCloudImpl()
	{
		// todo: 1. pass input cloud to feature calculators
	}

	bool buildSpatialIndexImpl()
	{
		// todo: 1. build spatial indices if required during feature calculation.
		//          they may be kdtree, octree, etc.
		//       2. pass sptial indices to feature calculators
		return false;
	}

	bool combineFeaturesImpl(FeatureMatrix& featMat)
	{
		// todo: 1. call multiple feature calculators.
		return false;
	}

	bool combineFeaturesImpl(size_t index, FeatureVector& featVec)
	{
		// todo: 1. call multiple feature calculators.
		return false;
	}

private:
	/*! multiple concrete feature calculator that will be combined. */
	//FeatureCalculator1* m_featCalculator1;
	//FeatureCalculator2* m_featCalculator2;
	//...

	/*! multiple concrete spatial indices that will be built and passed to feature calculators.
	    they may be kdtree, octree, etc. */
	//SpatialIndex1* m_spatIndex1;
	//SpatialIndex2* m_spatIndex2;
	//...
};

} // namespace sc

#endif // !_SUPERVISED_CLASSIFICATION_FEATURE_COMBINER_BASE_H_
