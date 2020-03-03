#pragma once

#include "rtutil.hpp"
#include "util.hpp"
#include "filesaves.hpp"
#include "RTTriangle.hpp"

#include <memory>


namespace FW
{
    class Bvh;

    // Bounding volume hierarchy tree node; stores the bounding box, interval of leafs covered, and child nodes.
    // To keep things simple, either store child nodes in the left and right pointers, or a range of triangles in
    // startPrim...endPrim. It is possible to also store geometry ranges in inner nodes of the triangles, but it
    // gets somewhat complicated pretty quickly.

    // The only real functionality of BvhNode is the save/load feature, otherwise it's a simple data holder.
    struct BvhNode : noncopyable
    {
        AABB bb;
        size_t startPrim, endPrim; // [start, end)
        std::unique_ptr<BvhNode> left;
        std::unique_ptr<BvhNode> right;

        BvhNode() :
            bb(),
            startPrim(0), endPrim(0)
        {
        }

        BvhNode::BvhNode(size_t start, size_t end) :
            bb(),
            startPrim(start), endPrim(end)
        {
        }

        BvhNode(Loader& is, size_t depth = 0);

        ~BvhNode()
        {
        }

        inline bool hasChildren() const
        {
            return !!left;
        }

        void save(Saver& os);
    };
}
