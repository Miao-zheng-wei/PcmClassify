/******************************************************************************
*
* Copyright (C) 2017 Beijing GreenValley Technology Co., Ltd.
* All Rights Reserved.
*
* Project		: Supervised Classification
* Purpose		: basic types of input point clouds, output probalibities,
*                 output labels, and features for training and evaluation. 
* Author		: Xie Dufang л����, df_xie@126.com; df.xie@outlook.com
* Created		: 2017-05-16-09:47 Tuesday
* Modified by	:
******************************************************************************/

#ifndef _SUPERVISED_CLASSIFICATION_BASICS_H_
#define _SUPERVISED_CLASSIFICATION_BASICS_H_

#define USE_EIGEN_ALIGNED_ALLOCATOR 0

#include <iostream>
#include <string>
#include <vector>
#if USE_EIGEN_ALIGNED_ALLOCATOR
#include <Eigen/Core>
#endif // USE_EIGEN_ALIGNED_ALLOCATOR
#include <assert.h>
#include "omp.h"
#include "GenericProgressCallback.h"
//#include <QTimer>
//#include <QEventLoop>
//#include <QCoreApplication>
/*! namespace supervised classification */
namespace sc {

/*---------------------------------------------------------------------------*\
                template <typename PointT> struct Cloud
\*---------------------------------------------------------------------------*/

/*!
 * @brief Struct defining point cloud type.
 *        PointT must contain at least x, y, z.
*/
template <typename PointT>
struct Cloud
{
#if USE_EIGEN_ALIGNED_ALLOCATOR
	typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > type;
#else
	typedef std::vector<PointT> type;
#endif // USE_EIGEN_ALIGNED_ALLOCATOR
};

/*---------------------------------------------------------------------------*\
              template <typename ElementType> struct DataView1D
\*---------------------------------------------------------------------------*/

/*!
 * @brief Struct managing pointer to one-dimensional data. The data can be
 *        e.g. a row or column of a matrix or the elements of an array or
 *        a std::vector - anything with a fixed step size between pointers
*/
template <typename ElementType>
struct DataView1D {
	//! \brief Element access
	//
	// \param idx the index of the element
	ElementType& operator()(size_t idx) const
	{
		return *(data + step * idx);
	}
	//! \brief Construct using pointer, size and (optional) step size
	DataView1D(ElementType* ptr, size_t size, ptrdiff_t step_size = 1) :
		data(ptr), step(step_size), num_elements(size)
	{
	}
	//! \brief Construct view from std::vector
	//
	// \param vec the vector, by-ref since we don't take ownership - no
	//            temporaries allowed. maybe needs to change
	DataView1D(std::vector<ElementType>& vec) :
		data(&vec[0]), step(1), num_elements(vec.size())
	{
	}

	//! \brief Construct empty view
	DataView1D() : data(0), step(1), num_elements(0)
	{
	}
	ElementType* data; //!< Pointer to first element
	ptrdiff_t step;    //!< Step size between elements - for a vector this is 1, no need to multiply by sizeof(ElementType)
	size_t num_elements; //!< Number of elements in the view
};

/*---------------------------------------------------------------------------*\
              template <typename ElementType> struct DataView2D
\*---------------------------------------------------------------------------*/

/*!
 * @brief Struct managing pointer to two-dimensional data. Step sizes of column
 *        and row can be set. Column step size is 1 for row-continuous data;
 *        Row step size is 1 for col-continuous data.
*/
template <typename ElementType>
struct DataView2D {
	//! \brief Construct empty view
	DataView2D() : data(0), row_step(1), col_step(1), rows(0), cols(0)
	{}
	//! \brief Construct view from memory using given step sizes
	DataView2D(ElementType* ptr, size_t rows_, size_t cols_, ptrdiff_t row_step_, ptrdiff_t col_step_) :
		data(ptr), row_step(row_step_), col_step(col_step_), rows(rows_), cols(cols_)
	{
	}
	//! \brief Construct view from a continuous block of memory in row-major order
	DataView2D(ElementType* ptr, size_t rows_, size_t cols_) :
		data(ptr), row_step(cols_), col_step(1), rows(rows_), cols(cols_)
	{
	}
	//! \brief Element access
	ElementType& operator()(size_t row_idx, size_t col_idx) const
	{
		return *(data + row_step * row_idx + col_step * col_idx);
	}
	//! \brief Return a 1D view of a row
	DataView1D<ElementType> row(size_t row_idx)
	{
		return DataView1D<ElementType>(data + row_step * row_idx, cols, col_step);
	}
	//! \brief Return a 1D view of a column
	DataView1D<ElementType> col(size_t col_idx)
	{
		return DataView1D<ElementType>(data + col_step * col_idx, rows, row_step);
	}
	//! \brief Return a new view, using a subset of rows
	DataView2D row_range(size_t start_row, size_t end_row) const
	{
		DataView2D ret(*this);
		ret.data = data + row_step * start_row;
		ret.rows = end_row - start_row;
		return ret;
	}
	//! \brief Return a new view, using a subset of columns
	DataView2D col_range(size_t start_col, size_t end_col) const
	{
		DataView2D ret(*this);
		ret.data = data + col_step * start_col;
		ret.cols = end_col - start_col;
		return ret;
	}
	//! \brief Transpose the matrix (actually done by swapping the steps, no
	//                               copying)
	DataView2D transpose() const
	{
		DataView2D ret(*this);
		std::swap(ret.row_step, ret.col_step);
		std::swap(ret.rows, ret.cols);
		return ret;
	}
	//! \brief Create a 2D view from a 1D one, represent as a column
	static DataView2D column(DataView1D<ElementType> vec)
	{
		return DataView2D(vec.data, vec.num_elements, 1);
	}
	//! \brief Create a 2D view from a 1D one, represent as a row
	static DataView2D row(DataView1D<ElementType> vec)
	{
		return DataView2D(vec.data, 1, vec.num_elements);
	}
	//! \brief Return the number of elements in this view
	size_t num_elements() { return rows * cols; }
	//! \brief Return true if the view is empty (no elements)
	bool   empty() { return num_elements() == 0; }
	//! \brief Return true if rows in this view are continuous in memory
	bool row_continuous() {
		return col_step == 1;
	}
	//! \brief Return true if columns in this view are continuous in memory
	bool col_continuous() {
		return row_step == 1;
	}
	//! \brief Get pointer to row (only valid if row-continuous)
	ElementType* row_pointer(size_t row_idx) {
		return data + row_idx * row_step;
	}
	//! \brief Get pointer to column (only valid if column-continuous)
	ElementType* col_pointer(size_t col_idx) {
		return data + col_idx * col_step;
	}

	ElementType* data; //!< Pointer to first element
	ptrdiff_t row_step; //!< Pointer difference between an element and its right neighbor - 1 if row-continuous
	ptrdiff_t col_step; //!< Pointer difference between an element and its bottom neighbor - 1 if column-continuous
	size_t rows; //!< Number of rows in the view
	size_t cols; //!< Number of columns in the view
};

/*---------------------------------------------------------------------------*\
             FeatureElementType, FeatureMatrix, FeatureVector
\*---------------------------------------------------------------------------*/

/*! type definition of feature element */
typedef float FeatureElementType, FeatElemType;

/*! type definition of feature matrix.
    coventionally, rows correspond to number of samples;
	columns correspond to dimension of features.
	this type ONLY manages pointer to memory block,
	is NOT responsible for allocation or deallocation. */
typedef DataView2D<FeatElemType> FeatureMatrix, FeatMat;

/*! type definition of feature vector. it can be a row or a column.
    coventionally, a row represents kD-features of one sample;
	a column represents 1D-features of multiple samples.
	this type ONLY manages pointer to memory block,
	is NOT responsible for allocation or deallocation. */
typedef DataView1D<FeatElemType> FeatureVector, FeatVec;

/*---------------------------------------------------------------------------*\
                    ProbabilityType, LabelType
\*---------------------------------------------------------------------------*/

/*! type definition of label probability.
    coventionally, value range [0, 1] */
typedef float ProbabilityType, ProbType;

/*! type definition of probability vector
    for a specific label of multiple samples.
	this type ONLY manages pointer to memory block,
	is NOT responsible for allocation or deallocation. */
typedef DataView1D<ProbType> ProbabilityVector, ProbVec;

/*! type definition of label */
typedef int LabelType;

/*! enumeration of label values */
//enum LabelValue
//{
//	NotClassified = 0,
//	Ground = 1,
//	Building = 2,
//	Vegetation = 3,
//	PowerLine = 4,
//	Pylon = 5
//};

/*! type definition of label vector of multiple samples.
    this type ONLY manages pointer to memory block,
	is NOT responsible for allocation or deallocation. */
typedef DataView1D<LabelType> LabelVector, LabelVec;

} // namespace sc

#endif // !_SUPERVISED_CLASSIFICATION_BASICS_H_
