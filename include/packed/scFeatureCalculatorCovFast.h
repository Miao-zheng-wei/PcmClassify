#ifndef _SUPERVISED_CLASSIFICATION_COVARIANCE_FEATURE_CALCULATOR_FAST_H_
#define _SUPERVISED_CLASSIFICATION_COVARIANCE_FEATURE_CALCULATOR_FAST_H_

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include "scFeatureCalculatorBase.h"
#include "FlannKDTree.h"
#include "math.h"
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
/*! namespace supervised classification */
namespace sc {

/*---------------------------------------------------------------------------*\
        template<typename PointT> class FeatureCalculatorExample
\*---------------------------------------------------------------------------*/

/*!
 * @brief Example derived class with some todo lists for the calculation of features.
*/
template<typename PointT>
class FeatureCalculatorCovFast : public FeatureCalculatorBase<FeatureCalculatorCovFast<PointT>, PointT>
{
public:
	FeatureCalculatorCovFast(int numNearestPts = 5)
	{
		// todo: 1. initialize the dimension of features.
		//       2. initialize multiple different spatial indices as null pointers.
		m_dimFeatures = 9;
		m_kdtreeNNS = NULL;
		m_numNearestPts = numNearestPts ;
	}
	~FeatureCalculatorCovFast()
	{
		//m_dilutedPoints.clear();
	}

	void setSpatialIndex(FlannKDTree<PointT>* kdtreeNNS)
	{
		m_kdtreeNNS = kdtreeNNS;
		//m_dilutedPoints = mm_dilutedPoints;
	}
	
	bool calculateFeaturesImpl(FeatureMatrix& featMat,GenericProgressCallback *m_pdlg = 0)
	{
		// todo: calculate features and store them in feature matrix
//#pragma omp parallel for num_threads(8)
		NormalizedProgress nProgress(m_pdlg, m_numPoints);
		for(int i=0; i< m_numPoints; i++)
		{
			//if (!nProgress.oneStep())
			//{
			//	m_pdlg->stop();
			//	return false;
			//}
			nProgress.oneStep();
			FeatureVector featVec = featMat.row(i);
			calculateFeaturesImpl(i, featVec);
		}
		return true; 
	}

	// todo: calculate features and store them in feature vector
	inline bool calculateFeaturesImpl(int index, FeatureVector& featVec)
	{
		float l_fEiVal1 = 0,l_fEiVal2 = 0,l_fEiVal3 = 0;

		PointT& searchPoint = m_pointVector[index];

		std::vector<int> pointIdxNKNSearch(m_numNearestPts);
		std::vector<float> pointNKNSquaredDistance(m_numNearestPts);

		float l_fXsum = 0, l_fYsum=0,l_fZsum=0;

		if(m_kdtreeNNS->nearestKSearch (searchPoint, m_numNearestPts, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
		{ 
			for( int k = 0; k < pointIdxNKNSearch.size (); ++k) 
			{
				l_fXsum = l_fXsum +  m_pointVector[pointIdxNKNSearch[k] ].x;
				l_fYsum = l_fYsum +  m_pointVector[pointIdxNKNSearch[k] ].y;
				l_fZsum = l_fZsum +  m_pointVector[pointIdxNKNSearch[k] ].z;
			}
		}

		PointT Midpoint; 

		Midpoint.x = l_fXsum/m_numNearestPts; Midpoint.y = l_fYsum/m_numNearestPts; Midpoint.z = l_fZsum/m_numNearestPts;

		ST_CovMat l_stuCovMat; 

		float l_fa = 0,l_fb = 0,l_fc = 0;

		for(size_t k=0; k<m_numNearestPts;k++) 
		{
			l_fa = m_pointVector[pointIdxNKNSearch[k] ].x - Midpoint.x;
			l_fb = m_pointVector[pointIdxNKNSearch[k] ].y - Midpoint.y;
			l_fc = m_pointVector[pointIdxNKNSearch[k] ].z - Midpoint.z;

			l_stuCovMat.r1c1 = l_stuCovMat.r1c1 + l_fa*l_fa; l_stuCovMat.r2c1 = l_stuCovMat.r2c1 + l_fb*l_fa; l_stuCovMat.r3c1 = l_stuCovMat.r3c1 + l_fa*l_fc;
			l_stuCovMat.r1c2 = l_stuCovMat.r1c2 + l_fa*l_fb; l_stuCovMat.r2c2 = l_stuCovMat.r2c2 + l_fb*l_fb; l_stuCovMat.r3c2 = l_stuCovMat.r3c2 + l_fb*l_fc;
			l_stuCovMat.r1c3 = l_stuCovMat.r1c3 + l_fa*l_fc; l_stuCovMat.r2c3 = l_stuCovMat.r2c3 + l_fb*l_fc; l_stuCovMat.r3c3 = l_stuCovMat.r3c3 + l_fc*l_fc;
		}


		Eigen::Matrix3f A;
		A << l_stuCovMat.r1c1, l_stuCovMat.r1c2, l_stuCovMat.r1c3, l_stuCovMat.r2c1, l_stuCovMat.r2c2, l_stuCovMat.r2c3, l_stuCovMat.r3c1, l_stuCovMat.r3c2, l_stuCovMat.r3c3;

//#pragma omp parallel  num_threads(8)
		{
			//Eigen::EigenSolver<Eigen::Matrix3f> es(A);
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>  es(A);

		//l_fEiVal1 = es.eigenvalues()[0].real();
		//l_fEiVal2 = es.eigenvalues()[1].real();
		//l_fEiVal3 = es.eigenvalues()[2].real();

		l_fEiVal1 = es.eigenvalues()[0];
		l_fEiVal2 = es.eigenvalues()[1];
		l_fEiVal3 = es.eigenvalues()[2];

		//Normalize the eigenvalues to sum up to 1.
		if (l_fEiVal1 < 0)
		{
			l_fEiVal1 = 0;
		}
		if (l_fEiVal2 < 0)
		{
			l_fEiVal2 = 0;
		}
		if (l_fEiVal3 < 0)
		{
			l_fEiVal3 = 0;
		}

		ST_Covariance l_stuCov;
		l_stuCov.Sum = l_fEiVal1+l_fEiVal2+l_fEiVal3;

		//2018.4.13
		if (l_stuCov.Sum==0)
		{
			l_fEiVal1 = 0;
			l_fEiVal2 =0;
			l_fEiVal3 = 0;
		}
		else
		{
			l_fEiVal1 = l_fEiVal1 / l_stuCov.Sum;
			l_fEiVal2 = l_fEiVal2 / l_stuCov.Sum;
			l_fEiVal3 = l_fEiVal3 / l_stuCov.Sum;
		}

		int l_nS1=0,l_nS2=1,l_nS3=2;

		sortEigenValue(l_fEiVal1, l_fEiVal2, l_fEiVal3, l_nS1, l_nS2, l_nS3);

		std::vector<float> l_FeatVec(9);
		
		if (l_fEiVal3 == 0 || l_fEiVal2 == 0 )
		{
			l_stuCov.Omnivarance = 0;
			l_stuCov.Eigenentropy = 0;
		}
		else
		{
			l_stuCov.Omnivarance = pow(float(l_fEiVal1*l_fEiVal2*l_fEiVal3), float(1.0) / float(3.0));
			l_stuCov.Eigenentropy = -(l_fEiVal1*logf(l_fEiVal1) + l_fEiVal2*logf(l_fEiVal2) + l_fEiVal3*logf(l_fEiVal3));
		}

		if (l_fEiVal1 == 0 || l_stuCov.Sum ==0)
		{
			l_stuCov.Anisotropy =0; l_stuCov.SurfaceVariation = 0;
			l_stuCov.Planarity = 0;  l_stuCov.Sphericity = 0;
			l_stuCov.Linearity = 0;
			l_stuCov.Omnivarance = 0;
			l_stuCov.Eigenentropy = 0;
		}
		else
		{
			l_stuCov.Anisotropy = (l_fEiVal1 - l_fEiVal3) / l_fEiVal1; l_stuCov.SurfaceVariation = l_fEiVal3 / (l_fEiVal1 + l_fEiVal2 + l_fEiVal3);
			l_stuCov.Planarity = (l_fEiVal2 - l_fEiVal3) / l_fEiVal1;  l_stuCov.Sphericity = l_fEiVal3 / l_fEiVal1;
			l_stuCov.Linearity = (l_fEiVal1 - l_fEiVal2) / l_fEiVal1;
		}

		l_FeatVec[0]=l_stuCov.Sum;
		l_FeatVec[1]=l_stuCov.Omnivarance; 
		l_FeatVec[2]=l_stuCov.Eigenentropy; 
		l_FeatVec[3]=l_stuCov.Anisotropy;
		l_FeatVec[4]=l_stuCov.Planarity;
		l_FeatVec[5]=l_stuCov.Linearity; 
		l_FeatVec[6]=l_stuCov.SurfaceVariation;
		l_FeatVec[7]=l_stuCov.Sphericity;                           
		
		Eigen::Vector3d V,W;
		float l_fMound = 0;
		V<<0,0,1;
		//W<<es.eigenvectors().col(l_nS3)[0].real(),es.eigenvectors().col(l_nS3)[1].real(),es.eigenvectors().col(l_nS3)[2].real
		W << es.eigenvectors().col(l_nS3)[0], es.eigenvectors().col(l_nS3)[1], es.eigenvectors().col(l_nS3)[2];
		l_fMound = V.dot(W);
		l_stuCov.Verticality = 1 - abs(l_fMound);	
		l_FeatVec[8]=l_stuCov.Verticality;

		for(size_t k = 0; k<m_dimFeatures; k++)
		{
			featVec(k) = l_FeatVec[k];
		}
		}
		return true;
	}

private:
	

	void sortEigenValue(float &l_fEiVal1, float &l_fEiVal2, float &l_fEiVal3, int &l_nS1, int &l_nS2, int &l_nS3);
	//calcualte Features 
	//void calculateFeaturesForAPoint(size_t i);

private:
	
	//Convariance tensor: 3*3 matrix 
	struct ST_CovMat
	{
		float r1c1; //the value of 1st row & 1st col
		float r2c1; //the value of 2st row & 1st col
		float r3c1; //the value of 3st row & 1st col
		float r1c2; //the value of 1st row & 2st col
		float r2c2; //the value of 2st row & 2st col
		float r3c2; //the value of 3st row & 2st col
		float r1c3; //the value of 1st row & 3st col
		float r2c3; //the value of 2st row & 3st col
		float r3c3; //the value of 3st row & 3st col
		ST_CovMat()
		{
			memset(this, 0, sizeof(ST_CovMat)); // used for pointere 
		}

		ST_CovMat(const ST_CovMat& other)
		{
			*this = other;
		}
	};

	//the features computed by eigenvalues of the local covariance tensor
	struct ST_Covariance 
	{
		float Sum;					//三个主要特征值的和。
		float Omnivarance;          //三个主要特征值的几何平均值。
		float Eigenentropy;         //基于特征值的信息熵。
		float Anisotropy;		    //各项异性
		float Planarity;			//平面度
		float Linearity;			//线性度
		float SurfaceVariation;     //表示局部点云的表面不规则程度
		float Sphericity;           //表示点云球面性质
		float Verticality;			// 表示法线和垂直方向偏差程度
		//float FstOrder1stAxis;
		//float FstOrder2ndAxis;
		//float SndOrder1stAxis;
		//float Sndorder2ndAxis;
		ST_Covariance()
		{

			memset(this,0,sizeof(ST_Covariance));
		}

		ST_Covariance(const ST_Covariance & other)
		{
			*this = other;
		}

	}; 


private:
	/*! multiple concrete spatial indices. they may be kdtree, octree, etc.
	    they are not allocated or deallocated in this class. */
	//SpatialIndex1* m_spatIndex1;
	//SpatialIndex2* m_spatIndex2;
	//...
	FlannKDTree<PointT>* m_kdtreeNNS;
	std::vector<PointT> m_dilutedPoints;
	int m_numNearestPts; // K nearest point; 
};

/*---------------------------------------------------------------------------*\
                          Implementation of
	 template<typename PointT> class FeatureCalculatorExample
\*---------------------------------------------------------------------------*/

template<typename PointT>
inline void FeatureCalculatorCovFast<PointT>::sortEigenValue(float &l_fEiVal1, float &l_fEiVal2, float &l_fEiVal3, int &l_nS1, int &l_nS2, int &l_nS3)
{
	float Temp_fEiVal = 0;
	int Temp_nS = 0;

	if(l_fEiVal1<l_fEiVal2)
	{
		Temp_fEiVal = l_fEiVal1;
		l_fEiVal1 = l_fEiVal2;
		l_fEiVal2 = Temp_fEiVal; 

		Temp_nS = l_nS1;
		l_nS1 = l_nS2;
		l_nS2 = Temp_nS;

	}
	if(l_fEiVal1<l_fEiVal3)
	{
		Temp_fEiVal = l_fEiVal1;
		l_fEiVal1 = l_fEiVal3;
		l_fEiVal3 = Temp_fEiVal;

		Temp_nS = l_nS1;
		l_nS1 = l_nS3;
		l_nS3 = Temp_nS;
	}
	if(l_fEiVal2<l_fEiVal3)
	{
		Temp_fEiVal = l_fEiVal2;
		l_fEiVal2 = l_fEiVal3;
		l_fEiVal3 = Temp_fEiVal;

		Temp_nS = l_nS2;
		l_nS2 = l_nS3;
		l_nS3 = Temp_nS;
	}
}

} // namespace sc

#endif // _SUPERVISED_CLASSIFICATION_COVARIANCE_FEATURE_CALCULATOR_FAST_H_
