/******************************************************************************
*
* Copyright (C) 2017 Beijing GreenValley Technology Co., Ltd.
* All Rights Reserved.
*
* Project		: Supervised Classification
* Purpose		: base class for the calculation of different features
* Author		: Xie Dufang Ð»¶À·Å, df_xie@126.com; df.xie@outlook.com
* Created		: 2017-05-16-09:47 Tuesday
* Modified by	:
******************************************************************************/

#ifndef _SUPERVISED_CLASSIFICATION_FEATURE_CALCULATOR_BASE_H_
#define _SUPERVISED_CLASSIFICATION_FEATURE_CALCULATOR_BASE_H_

#include "scBasics.h"

/*! namespace supervised classification */
namespace sc {

/*---------------------------------------------------------------------------*\
   template<typename DerivedT, typename PointT> class FeatureCalculatorBase
\*---------------------------------------------------------------------------*/

/*!
 * @brief Base class for the calculation of features.
 *
 *        The derived class must implement the following functions:
 *        1. void setSpatialIndex(...);
 *        2. bool calculateFeaturesImpl(FeatureMatrix& featMat);
 *        3. inline bool calculateFeaturesImpl(size_t index, FeatureVector& featVec).
*/
template<typename DerivedT, typename PointT>
class FeatureCalculatorBase
{
public:
	typedef std::vector<PointT> PointTVector;

	FeatureCalculatorBase() : m_pointVector(NULL), m_numPoints(0), m_dimFeatures(0) {}
	~FeatureCalculatorBase() {}

	/** \brief Set pointer to input cloud data.
	  * \param[in] points the pointer to vector of input points
	*/
	void setInputCloud(PointTVector *pointVector);

	/** \brief Set spatial index for neighborhood search.
	  *        This method is commented off. Since different spatial indices
	  *        may be used in the derived class, there is no unified API
	  *        in the base class.
	*/
	//void setSpatialIndex(...)

	/** \brief Calculate features of all samples for the training process.
	  * \param[out] featMat the row-major feature matrix with size [numPoints*dimFeatures]
	  * \return "true" if succeeded.
	*/
	bool calculateFeatures(FeatureMatrix& featMat, GenericProgressCallback *pdlg);

	/** \brief Calculate features of one sample for the evaluation process.
	  * \param[in] index the point index of input cloud
	  * \param[out] featVec the feature vector with size [dimFeatures]
	  * \return "true" if succeeded.
	*/
	inline bool calculateFeatures(size_t index, FeatureVector& featVec);

	/** \brief Get dimension of features.
	  * \return dimension of features.
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
   template<typename DerivedT, typename PointT> class FeatureCalculatorBase
\*---------------------------------------------------------------------------*/

template<typename DerivedT, typename PointT>
inline void FeatureCalculatorBase<DerivedT, PointT>::setInputCloud(PointTVector  *pointVector)
{
	assert(NULL != pointVector);
	assert(pointVector->size() > 0);

	m_pointVector = *pointVector;
	m_numPoints = pointVector->size();
}

template<typename DerivedT, typename PointT>
inline bool FeatureCalculatorBase<DerivedT, PointT>::calculateFeatures(FeatureMatrix & featMat, GenericProgressCallback *pdlg = 0)
{
	assert(m_numPoints > 0 && m_dimFeatures > 0);
	assert(featMat.rows == m_numPoints && featMat.cols == m_dimFeatures);

	return static_cast<DerivedT*>(this)->calculateFeaturesImpl(featMat, pdlg);
}

template<typename DerivedT, typename PointT>
inline bool FeatureCalculatorBase<DerivedT, PointT>::calculateFeatures(size_t index, FeatureVector & featVec)
{
	assert(m_numPoints > 0 && m_dimFeatures > 0);
	assert(featVec.num_elements == m_dimFeatures);

	return static_cast<DerivedT*>(this)->calculateFeaturesImpl(index, featVec);
}

/*---------------------------------------------------------------------------*\
        template<typename PointT> class FeatureCalculatorExample
\*---------------------------------------------------------------------------*/

/*!
 * @brief Example derived class with some todo lists for the calculation of features.
*/
template<typename PointT>
class FeatureCalculatorExample : public FeatureCalculatorBase<FeatureCalculatorExample<PointT>, PointT>
{
public:
	FeatureCalculatorExample()
	{
		// todo: 1. initialize the dimension of features.
		//       2. initialize multiple different spatial indices as null pointers.
	}
	~FeatureCalculatorExample() {}

	void setSpatialIndex(...)
	{
		// todo: set pointer to prebuilt spatial indices
	}

	bool calculateFeaturesImpl(FeatureMatrix& featMat)
	{
		// todo: calculate features and store them in feature matrix
		return false;
	}

	inline bool calculateFeaturesImpl(size_t index, FeatureVector& featVec)
	{
		// todo: calculate features and store them in feature vector
		return false;
	}

private:
	/*! multiple concrete spatial indices. they may be kdtree, octree, etc.
	    they are not allocated or deallocated in this class. */
	//SpatialIndex1* m_spatIndex1;
	//SpatialIndex2* m_spatIndex2;
	//...
};

} // namespace sc

#endif // !_SUPERVISED_CLASSIFICATION_FEATURE_CALCULATOR_BASE_H_
