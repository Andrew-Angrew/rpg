#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#include <map>
#include <cmath>
#include <cassert>
#include <set>
#include <string>
#include <algorithm>

#include <omp.h>

#include <ctime>


int distance_counter = 0;
std::vector <std::vector <float> > model_features_train;
std::vector <std::vector <float> > model_features_test;
std::vector <std::vector <float> > itemRanks;
std::vector<float> rankToDist;
int relevanceVectorLength = -1;
bool usePrecomputedRanks = false;
bool useLp = true;
bool useL2 = true;
bool useMinSum = true;
bool useDotProduct = false;
int sumOrd = -1;
int hybridD = -1;

#include "hnswlib.h"
namespace hnswlib {
    using namespace std;

    void InitializeBaseConstruction(std::string basefile_, int baseSize_, int trainSize_, int relevanceVectorLength_,
                                    std::string constructionMetric, int sumOrd_=-1,
                                    int hybridD_=-1, std::string itemRanksFileName="", std::string rankToDistFileName="")
    {
        model_features_train = std::vector <std::vector <float> >(baseSize_, std::vector <float>(trainSize_));
        
        std::ifstream train_features(basefile_, std::ios::binary);
        for (int i = 0; i < baseSize_; i++) {
            train_features.read((char*)model_features_train[i].data(), sizeof(model_features_train[0][0]) * trainSize_);
        }
        train_features.close();

        relevanceVectorLength = relevanceVectorLength_;
        if (constructionMetric == "precomputed_ranks") {
            usePrecomputedRanks = true;
        }
        if (constructionMetric == "min_sum" || constructionMetric == "top_sum") {
            useLp = false;
            if (constructionMetric == "top_sum") {
                useMinSum = false;
            }
        } else if (constructionMetric == "l1") {
            useL2 = false;
        } else if (constructionMetric == "dot_product") {
            useDotProduct = true;
        }
        sumOrd = sumOrd_;
        hybridD = hybridD_;
        if (hybridD > 0) {
            itemRanks = std::vector<std::vector<float>>(baseSize_, std::vector<float>(trainSize_));
            std::ifstream itemRanksFile(itemRanksFileName, std::ios::binary);
            for (int i = 0; i < baseSize_; i++) {
                itemRanksFile.read((char*)itemRanks[i].data(), sizeof(itemRanks[0][0]) * trainSize_);
            }
            itemRanksFile.close();
			std::cerr << "Ranks loaded " << itemRanks.size() << " " << itemRanks[0].size() << std::endl;

            rankToDist = std::vector<float>(baseSize_);
            std::ifstream rankToDistFile(rankToDistFileName, std::ios::binary);
            rankToDistFile.read((char*)rankToDist.data(), sizeof(rankToDist[0]) * baseSize_);
            rankToDistFile.close();
			std::cerr << "RankToDist loaded " << rankToDist.size() << ": "; 
			std::cerr << rankToDist[0] << " " << rankToDist[1] << " " << rankToDist[2] << " " << std::endl;
        }
    }

    void InitializeSearch(std::string queryfile_, int baseSize_, int querySize_)
    {
        model_features_test = std::vector <std::vector <float> >(baseSize_, std::vector <float>(querySize_));
        
        std::ifstream test_features(queryfile_, std::ios::binary);
        for (int i = 0; i < baseSize_; i++) {
            test_features.read((char*)model_features_test[i].data(), sizeof(model_features_test[0][0]) * querySize_);
        }
        test_features.close();
    }

    float fvec_L2sqr(const float *x, const float *y, size_t d) {
        float PORTABLE_ALIGN32 TmpRes[8];
        #ifdef USE_AVX
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return (res);
        #else
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return (res);
        #endif
    }

    static float calcL2(int idx1, int idx2) {
		int relevanceVectorLength16 = (relevanceVectorLength / 16) * 16;
        float val = fvec_L2sqr(model_features_train[idx1].data(), model_features_train[idx2].data(), relevanceVectorLength16);
        for (int i = relevanceVectorLength16; i < relevanceVectorLength; i++) {
            float tmp = model_features_train[idx1][i] - model_features_train[idx2][i];
            val += tmp * tmp;
        }
        return val;
    }

	static float constructionDistance(const void *pVect1, const void *pVect2)
	{
        float float_query = ((float*)pVect1)[0];
        int idx_query = float_query;
        
        float float_item = ((float*)pVect2)[0];
        int idx_item = float_item;

        if (usePrecomputedRanks) {
            return model_features_train[idx_item][idx_query];
        }

        if (hybridD > 0) {
            float l2Dist = calcL2(idx_item, idx_query);
            float minSumRank = std::numeric_limits<float>::max();
            for (int i = 0; i < relevanceVectorLength; i++) {
                float tmp = itemRanks[idx_item][i] + itemRanks[idx_query][i];
                minSumRank = std::min(minSumRank, tmp);
            }
            int64_t maxRank = static_cast<int64_t>(hybridD) * static_cast<int64_t>(minSumRank);
            if (maxRank < static_cast<int>(rankToDist.size())) {
                return std::min(l2Dist, rankToDist[maxRank]);
            }
            return l2Dist;
        }


        float val = 0;

        if (useDotProduct) {
            for (int i = 0; i < relevanceVectorLength; i++) {
                val += model_features_train[idx_item][i] * model_features_train[idx_query][i];
            }
            return -val;
        }

        if (useLp) {
            if (useL2) {
                return calcL2(idx_item, idx_query);
            } else {
                for (int i = 0; i < relevanceVectorLength; i++) {
                    float tmp = model_features_train[idx_item][i] - model_features_train[idx_query][i];
                    val += std::abs(tmp);
                }
            }
        } else {
            if (useMinSum) {
                val = std::numeric_limits<float>::max();
                for (int i = 0; i < relevanceVectorLength; i++) {
                    float tmp = model_features_train[idx_item][i] + model_features_train[idx_query][i];
                    val = std::min(val, tmp);
                }
            } else {
                std::vector<float> sumRanks(relevanceVectorLength);
                for (int i = 0; i < relevanceVectorLength; i++) {
                    sumRanks[i] = model_features_train[idx_item][i] + model_features_train[idx_query][i];
                }
                std::nth_element(sumRanks.begin(), sumRanks.begin() + sumOrd, sumRanks.end());
                val = sumRanks[sumOrd];
            }
        }

        return val;
	}
	
    static float
		searchDistance(const void *pVect1, const void *pVect2)
	{
        float float_query = ((float*)pVect1)[0];
        int idx_query = float_query;
        
        float float_item = ((float*)pVect2)[0];
        int idx_item = float_item;
        
        distance_counter++;
        return -model_features_test[idx_item][idx_query];
	}

    int geDistanceCounter()
    {
        return distance_counter;
    }
	
	class L2Space : public SpaceInterface<float> {
		
		DISTFUNC<float> fstdistfunc_;
		size_t data_size_;
		size_t dim_;
	public:
		L2Space(size_t dim) {
			fstdistfunc_ = searchDistance;
            dim_ = 1;
		}

		DISTFUNC<float> get_dist_func() {
			return fstdistfunc_;
		}
		void *get_dist_func_param() {
			return &dim_;
		}

    };

}
