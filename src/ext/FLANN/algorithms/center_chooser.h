/*
 * center_chooser.h
 *
 *  Created on: 2012-11-04
 *      Author: marius
 */

#ifndef CENTER_CHOOSER_H_
#define CENTER_CHOOSER_H_

#include <ext/FLANN/util/matrix.h>

namespace flann
{

template <typename Distance, typename ElementType>
struct squareDistance
{
    typedef typename Distance::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist*dist; }
};


template <typename ElementType>
struct squareDistance<L2_Simple<ElementType>, ElementType>
{
    typedef typename L2_Simple<ElementType>::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist; }
};

template <typename ElementType>
struct squareDistance<L2_3D<ElementType>, ElementType>
{
    typedef typename L2_3D<ElementType>::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist; }
};

template <typename ElementType>
struct squareDistance<L2<ElementType>, ElementType>
{
    typedef typename L2<ElementType>::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist; }
};


template <typename ElementType>
struct squareDistance<HellingerDistance<ElementType>, ElementType>
{
    typedef typename HellingerDistance<ElementType>::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist; }
};


template <typename ElementType>
struct squareDistance<ChiSquareDistance<ElementType>, ElementType>
{
    typedef typename ChiSquareDistance<ElementType>::ResultType ResultType;
    ResultType operator()( ResultType dist ) { return dist; }
};


template <typename Distance>
typename Distance::ResultType ensureSquareDistance( typename Distance::ResultType dist )
{
    typedef typename Distance::ElementType ElementType;

    squareDistance<Distance, ElementType> dummy;
    return dummy( dist );
}



template <typename Distance>
class CenterChooser
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    CenterChooser(const Distance& distance, const std::vector<ElementType*>& points) : distance_(distance), points_(points) {};

    virtual ~CenterChooser() {};

    void setDataSize(size_t cols) { cols_ = cols; }

    /**
     * Chooses cluster centers
     *
     * @param k number of centers to choose
     * @param indices indices of points to choose the centers from
     * @param indices_length length of indices
     * @param centers indices of chosen centers
     * @param centers_length length of centers array
     */
	virtual void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length) = 0;

protected:
	const Distance distance_;
    const std::vector<ElementType*>& points_;
    size_t cols_;
};


template <typename Distance>
class RandomCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    using CenterChooser<Distance>::points_;
    using CenterChooser<Distance>::distance_;
    using CenterChooser<Distance>::cols_;

    RandomCenterChooser(const Distance& distance, const std::vector<ElementType*>& points) :
    	CenterChooser<Distance>(distance, points) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        UniqueRandom r(indices_length);

        int index;
        for (index=0; index<k; ++index) {
            bool duplicate = true;
            int rnd;
            while (duplicate) {
                duplicate = false;
                rnd = r.next();
                if (rnd<0) {
                    centers_length = index;
                    return;
                }

                centers[index] = indices[rnd];

                for (int j=0; j<index; ++j) {
                    DistanceType sq = distance_(points_[centers[index]], points_[centers[j]], cols_);
                    if (sq<1e-16) {
                        duplicate = true;
                    }
                }
            }
        }

        centers_length = index;
    }
};



/**
 * Chooses the initial centers using the Gonzales algorithm.
 */
template <typename Distance>
class GonzalesCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    using CenterChooser<Distance>::points_;
    using CenterChooser<Distance>::distance_;
    using CenterChooser<Distance>::cols_;

    GonzalesCenterChooser(const Distance& distance, const std::vector<ElementType*>& points) :
        CenterChooser<Distance>(distance, points) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        int rnd = rand_int(n);
        assert(rnd >=0 && rnd < n);

        centers[0] = indices[rnd];

        int index;
        for (index=1; index<k; ++index) {

            int best_index = -1;
            DistanceType best_val = 0;
            for (int j=0; j<n; ++j) {
            	DistanceType dist = distance_(points_[centers[0]],points_[indices[j]],cols_);
                for (int i=1; i<index; ++i) {
                    DistanceType tmp_dist = distance_(points_[centers[i]],points_[indices[j]],cols_);
                    if (tmp_dist<dist) {
                        dist = tmp_dist;
                    }
                }
                if (dist>best_val) {
                    best_val = dist;
                    best_index = j;
                }
            }
            if (best_index!=-1) {
                centers[index] = indices[best_index];
            }
            else {
                break;
            }
        }
        centers_length = index;
    }
};


/**
 * Chooses the initial centers using the algorithm proposed in the KMeans++ paper:
 * Arthur, David; Vassilvitskii, Sergei - k-means++: The Advantages of Careful Seeding
 */
template <typename Distance>
class KMeansppCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    using CenterChooser<Distance>::points_;
    using CenterChooser<Distance>::distance_;
    using CenterChooser<Distance>::cols_;

    KMeansppCenterChooser(const Distance& distance, const std::vector<ElementType*>& points) :
        CenterChooser<Distance>(distance, points) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        double currentPot = 0;
        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = indices[index];

        // Computing distance^2 will have the advantage of even higher probability further to pick new centers
        // far from previous centers (and this complies to "k-means++: the advantages of careful seeding" article)
        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance_(points_[indices[i]], points_[indices[index]], cols_);
            closestDistSq[i] = ensureSquareDistance<Distance>( closestDistSq[i] );
            currentPot += closestDistSq[i];
        }


        const int numLocalTries = 1;

        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            for (int localTrial = 0; localTrial < numLocalTries; localTrial++) {

                // Choose our center - have to be slightly careful to return a valid answer even accounting
                // for possible rounding errors
                double randVal = rand_double(currentPot);
                for (index = 0; index < n-1; index++) {
                    if (randVal <= closestDistSq[index]) break;
                    else randVal -= closestDistSq[index];
                }

                // Compute the new potential
                double newPot = 0;
                for (int i = 0; i < n; i++) {
                    DistanceType dist = distance_(points_[indices[i]], points_[indices[index]], cols_);
                    newPot += std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
                }

                // Store the best result
                if ((bestNewPot < 0)||(newPot < bestNewPot)) {
                    bestNewPot = newPot;
                    bestNewIndex = index;
                }
            }

            // Add the appropriate center
            centers[centerCount] = indices[bestNewIndex];
            currentPot = bestNewPot;
            for (int i = 0; i < n; i++) {
                DistanceType dist = distance_(points_[indices[i]], points_[indices[bestNewIndex]], cols_);
                closestDistSq[i] = std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
            }
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }
};



/**
 * Chooses the initial centers in a way inspired by Gonzales (by Pierre-Emmanuel Viel):
 * select the first point of the list as a candidate, then parse the points list. If another
 * point is further than current candidate from the other centers, test if it is a good center
 * of a local aggregation. If it is, replace current candidate by this point. And so on...
 *
 * Used with KMeansIndex that computes centers coordinates by averaging positions of clusters points,
 * this doesn't make a real difference with previous methods. But used with HierarchicalClusteringIndex
 * class that pick centers among existing points instead of computing the barycenters, there is a real
 * improvement.
 */
template <typename Distance>
class GroupWiseCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    using CenterChooser<Distance>::points_;
    using CenterChooser<Distance>::distance_;
    using CenterChooser<Distance>::cols_;

    GroupWiseCenterChooser(const Distance& distance, const std::vector<ElementType*>& points) :
        CenterChooser<Distance>(distance, points) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        const float kSpeedUpFactor = 1.3f;

        int n = indices_length;

        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = indices[index];

        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance_(points_[indices[i]], points_[indices[index]], cols_);
        }


        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            DistanceType furthest = 0;
            for (index = 0; index < n; index++) {

                // We will test only the potential of the points further than current candidate
                if( closestDistSq[index] > kSpeedUpFactor * (float)furthest ) {

                    // Compute the new potential
                    double newPot = 0;
                    for (int i = 0; i < n; i++) {
                        newPot += std::min( distance_(points_[indices[i]], points_[indices[index]], cols_)
                                            , closestDistSq[i] );
                    }

                    // Store the best result
                    if ((bestNewPot < 0)||(newPot <= bestNewPot)) {
                        bestNewPot = newPot;
                        bestNewIndex = index;
                        furthest = closestDistSq[index];
                    }
                }
            }

            // Add the appropriate center
            centers[centerCount] = indices[bestNewIndex];
            for (int i = 0; i < n; i++) {
                closestDistSq[i] = std::min( distance_(points_[indices[i]], points_[indices[bestNewIndex]], cols_)
                                             , closestDistSq[i] );
            }
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }
};


}


#endif /* CENTER_CHOOSER_H_ */
