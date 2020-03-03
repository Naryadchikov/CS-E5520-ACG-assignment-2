#include "Bvh.hpp"
#include "filesaves.hpp"

#include <algorithm>
#include <numeric>


#define MAX_TRIS_PER_LEAF 3
#define MAX_TRIS_PER_LEAF_SAH 10


namespace FW
{
    Bvh::Bvh()
    {
    }


    // reconstruct from a file
    Bvh::Bvh(std::istream& is)
    {
        // Load file header.
        fileload(is, mode_);
        Statusbar nodeInfo("Loading nodes", 0);
        Loader loader(is, nodeInfo);

        // Load elements.
        {
            size_t size;
            fileload(is, size);

            for (size_t i = 0; i < size; ++i)
            {
                uint32_t idx;
                loader(idx);
                indices_.push_back(idx);
            }
        }

        // Load the rest.
        rootNode_.reset(new BvhNode(loader));
    }

    Bvh::Bvh(std::vector<RTTriangle>& triangles, SplitMode splitMode) : triangles_ptr(&triangles), mode_(splitMode),
                                                                        indices_(triangles.size())
    {
        rootNode_.reset(new BvhNode(0, triangles.size() - 1));

        std::iota(indices_.begin(), indices_.end(), 0);

        switch (mode_)
        {
        case SplitMode_None:
            {
                break;
            }
        case SplitMode_Sah:
            {
                constructTree_Sah(rootNode_);
                break;
            }
        case SplitMode_ObjectMedian:
            {
                constructTree_ObjectMedian(rootNode_);
                break;
            }
        case SplitMode_SpatialMedian: default:
            {
                constructTree_SpatialMedian(rootNode_);
                break;
            }
        }
    }

    void Bvh::save(std::ostream& os)
    {
        // Save file header.
        filesave(os, mode_);
        Statusbar nodeInfo("Saving nodes", 0);
        Saver saver(os, nodeInfo);

        // Save elements.
        {
            filesave(os, (size_t)indices_.size());

            for (auto& i : indices_)
            {
                saver(i);
            }
        }

        // Save the rest.
        rootNode_->save(saver);
    }

    void Bvh::constructTree_ObjectMedian(std::unique_ptr<BvhNode>& node)
    {
        std::pair<Vec3f, Vec3f> bbPoints = getBBPoints(node->startPrim, node->endPrim);

        node->bb = AABB(bbPoints.first, bbPoints.second);

        if (node->endPrim - node->startPrim + 1 > MAX_TRIS_PER_LEAF)
        {
            int longestAxis = getLongestAxis(bbPoints);

            std::stable_sort(indices_.begin() + node->startPrim, indices_.begin() + node->endPrim + 1,
                             [&](uint32_t i1, uint32_t i2)
                             {
                                 return (*triangles_ptr)[i1].bbCentroid()[longestAxis] <
                                     (*triangles_ptr)[i2].bbCentroid()[longestAxis];
                             });

            size_t splitIndex = (node->endPrim + node->startPrim) / 2;

            node->left.reset(new BvhNode(node->startPrim, splitIndex - 1));
            constructTree_ObjectMedian(node->left);

            node->right.reset(new BvhNode(splitIndex, node->endPrim));
            constructTree_ObjectMedian(node->right);
        }
    }

    void Bvh::constructTree_SpatialMedian(std::unique_ptr<BvhNode>& node)
    {
        std::pair<Vec3f, Vec3f> bbPoints = getBBPoints(node->startPrim, node->endPrim);

        node->bb = AABB(bbPoints.first, bbPoints.second);

        if (node->endPrim - node->startPrim + 1 > MAX_TRIS_PER_LEAF)
        {
            int longestAxis = getLongestAxis(bbPoints);
            size_t splitIndex = std::stable_partition(indices_.begin() + node->startPrim,
                                                      indices_.begin() + node->endPrim + 1,
                                                      [&](uint32_t n)
                                                      {
                                                          return (*triangles_ptr)[n].bbCentroid()[longestAxis]
                                                              < (bbPoints.second[longestAxis] +
                                                                  bbPoints.first[longestAxis]) * 0.5f;
                                                      }) - indices_.begin();

            if (splitIndex - 1 == node->endPrim || splitIndex == node->startPrim)
            {
                splitIndex = (node->endPrim + node->startPrim) / 2;
            }

            node->left.reset(new BvhNode(node->startPrim, splitIndex - 1));
            constructTree_SpatialMedian(node->left);

            node->right.reset(new BvhNode(splitIndex, node->endPrim));
            constructTree_SpatialMedian(node->right);
        }
    }

    void Bvh::constructTree_Sah(std::unique_ptr<BvhNode>& node)
    {
        std::pair<Vec3f, Vec3f> bbPoints = getBBPoints(node->startPrim, node->endPrim);

        node->bb = AABB(bbPoints.first, bbPoints.second);

        if (node->endPrim - node->startPrim + 1 > MAX_TRIS_PER_LEAF_SAH)
        {
            size_t sahSplitIndex = node->startPrim;
            float lowestScore = std::numeric_limits<float>::max();
            float bestSplitPlaneCoord = 0.f;
            int bestAxis = 0;

            for (int axis = 0; axis < 3; ++axis)
            {
                float splitCoef = 0.1f;

                while (splitCoef < 1.f)
                {
                    float splitPlaneCoord = bbPoints.first[axis] +
                        (bbPoints.second[axis] - bbPoints.first[axis]) * splitCoef;

                    size_t splitIndex = std::stable_partition(indices_.begin() + node->startPrim,
                                                              indices_.begin() + node->endPrim + 1,
                                                              [&](uint32_t n)
                                                              {
                                                                  return (*triangles_ptr)[n].bbCentroid()[axis] <
                                                                      splitPlaneCoord;
                                                              }) - indices_.begin();

                    float currentScore = getSahScore(node->startPrim, node->endPrim, splitIndex);

                    if (currentScore < lowestScore)
                    {
                        lowestScore = currentScore;
                        sahSplitIndex = splitIndex;
                        bestAxis = axis;
                        bestSplitPlaneCoord = splitPlaneCoord;
                    }

                    splitCoef += 0.1f;
                }
            }

            std::stable_partition(indices_.begin() + node->startPrim,
                                  indices_.begin() + node->endPrim + 1,
                                  [&](uint32_t n)
                                  {
                                      return (*triangles_ptr)[n].bbCentroid()[bestAxis] <
                                          bestSplitPlaneCoord;
                                  });

            if (sahSplitIndex - 1 == node->endPrim || sahSplitIndex == node->startPrim)
            {
                sahSplitIndex = (node->endPrim + node->startPrim) / 2;
            }

            node->left.reset(new BvhNode(node->startPrim, sahSplitIndex - 1));
            constructTree_Sah(node->left);

            node->right.reset(new BvhNode(sahSplitIndex, node->endPrim));
            constructTree_Sah(node->right);
        }
    }

    std::pair<Vec3f, Vec3f> Bvh::getBBPoints(size_t startPrim, size_t endPrim)
    {
        Vec3f min(std::numeric_limits<float>::max());
        Vec3f max(-std::numeric_limits<float>::max());

        for (size_t i = startPrim; i <= endPrim; ++i)
        {
            for (const auto v : ((*triangles_ptr)[indices_[i]]).m_vertices)
            {
                Vec3f p = v.p;

                min = FW::min(min, p);
                max = FW::max(max, p);
            }
        }

        return std::make_pair(min, max);
    }

    int Bvh::getLongestAxis(const std::pair<Vec3f, Vec3f> bbPoints)
    {
        int longestAxis = 0;
        Vec3f bbDiagonal = bbPoints.second - bbPoints.first;
        float longestAxisDistance = bbDiagonal.max();

        if (bbDiagonal.y == longestAxisDistance)
        {
            longestAxis = 1;
        }
        if (bbDiagonal.z == longestAxisDistance)
        {
            longestAxis = 2;
        }

        return longestAxis;
    }

    float Bvh::getSahScore(size_t startPrim, size_t endPrim, size_t splitIndex)
    {
        size_t nLeft = splitIndex - startPrim;
        float aLeft = nLeft != 0
                          ? getBBArea(startPrim, splitIndex - 1)
                          : 0;

        size_t nRight = endPrim - splitIndex + 1;
        float aRight = nRight != 0
                           ? getBBArea(splitIndex, endPrim)
                           : 0;

        return aLeft * nLeft + aRight * nRight;
    }

    float Bvh::getBBArea(size_t startPrim, size_t endPrim)
    {
        if (startPrim == endPrim)
        {
            return 0.f;
        }

        std::pair<Vec3f, Vec3f> bbPoints = getBBPoints(startPrim, endPrim);
        Vec3f bbDiagonal = bbPoints.second - bbPoints.first;

        return 2.f * (
            FW::abs(bbDiagonal.x * bbDiagonal.y) +
            FW::abs(bbDiagonal.x * bbDiagonal.z) +
            FW::abs(bbDiagonal.y * bbDiagonal.z)
        );
    }
}
