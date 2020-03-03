#include "Bvh.hpp"
#include <iostream>


namespace FW
{
    BvhNode::BvhNode(Loader& stream, size_t depth)
    {
        bool children;

        stream(bb.min.x)(bb.min.y)(bb.min.z);
        stream(bb.max.x)(bb.max.y)(bb.max.z);
        stream(startPrim)(endPrim);
        stream(children);

        if (children)
        {
            left.reset(new BvhNode(stream, depth + 1));
            right.reset(new BvhNode(stream, depth + 1));
        }
    }

    void BvhNode::save(Saver& stream)
    {
        bool children = (left != nullptr);
        stream(bb.min.x)(bb.min.y)(bb.min.z);
        stream(bb.max.x)(bb.max.y)(bb.max.z);
        stream(startPrim)(endPrim);
        stream(children);

        if (left)
        {
            left->save(stream);
            right->save(stream);
        }
    }
}
