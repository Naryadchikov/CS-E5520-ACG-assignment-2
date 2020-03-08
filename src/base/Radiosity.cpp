#include "Radiosity.hpp"
#include "AreaLight.hpp"
#include "RayTracer.hpp"


namespace FW
{
    // --------------------------------------------------------------------------

    Radiosity::~Radiosity()
    {
        if (isRunning())
        {
            m_context.m_bForceExit = true;
            while (m_launcher.getNumTasks() > m_launcher.getNumFinished())
                Sleep(1);
            m_launcher.popAll();
        }
    }


    // --------------------------------------------------------------------------
    void Radiosity::vertexTaskFunc(MulticoreLauncher::Task& task)
    {
        RadiosityContext& ctx = *(RadiosityContext*)task.data;

        if (ctx.m_bForceExit)
            return;

        Random rnd;

        // which vertex are we to compute?
        int v = task.idx;

        // fetch vertex and its normal
        Vec3f n = ctx.m_scene->vertex(v).n.normalized();
        Vec3f o = ctx.m_scene->vertex(v).p + 0.01f * n;

        // YOUR CODE HERE (R3):
        // This starter code merely puts the color-coded normal into the result.
        // Remove the dummy solution to make your own implementation work.
        //
        // In the first bounce, your task is to compute the direct irradiance
        // falling on this vertex from the area light source.
        // In the subsequent passes, you should compute the irradiance by a
        // hemispherical gathering integral. The commented code below gives you
        // an idea of the loop structure. Note that you also have to account
        // for how diffuse textures modulate the irradiance.

        // direct lighting pass? => integrate direct illumination by shooting shadow rays to light source
        if (ctx.m_currentBounce == 0)
        {
            Vec3f E(0.f);

            for (int r = 0; r < ctx.m_numDirectRays; ++r)
            {
                // draw sample on light source
                float pdf;
                Vec3f Pl;

                ctx.m_light->sample(pdf, Pl, 0, rnd);

                // construct vector from current vertex (o) to light sample
                Vec3f d = Pl - o;

                // trace shadow ray to see if it's blocked
                if (!ctx.m_rt->raycast(o, d))
                {
                    // if not, add the appropriate emission, 1/r^2 and clamped cosine terms, accounting for the PDF as well.
                    // accumulate into E
                    Vec3f unionD = d.normalized();
                    float cosThetaL = FW::clamp(FW::dot(unionD, -ctx.m_light->getNormal()), 0.f, 1.f);
                    float cosTheta = FW::clamp(FW::dot(unionD, n), 0.f, 1.f);
                    float distance = d.length();

                    E += (ctx.m_light->getEmission() * cosThetaL * cosTheta) / (pdf * distance * distance);
                }
            }
            // Note we are NOT multiplying by PI here;
            // it's implicit in the hemisphere-to-light source area change of variables.
            // The result we are computing is _irradiance_, not radiosity.
            ctx.m_vecCurr[v] = E * (1.f / ctx.m_numDirectRays);
            ctx.m_vecResult[v] = ctx.m_vecCurr[v];
        }
        else
        {
            // OK, time for indirect!
            // Implement hemispherical gathering integral for bounces > 1.

            // Get local coordinate system the rays are shot from.
            Mat3f B = formBasis(n);

            Vec3f E(0.f);

            for (int r = 0; r < ctx.m_numHemisphereRays; ++r)
            {
                // Draw a cosine weighted direction and find out where it hits (if anywhere)
                // You need to transform it from the local frame to the vertex' hemisphere using B.
                Vec2f rv = rnd.getVec2f();
                float sqrtX = FW::sqrt(rv.x);
                float theta = 2.f * FW_PI * rv.y;
                Vec3f cwd = B * Vec3f(sqrtX * FW::cos(theta),
                                      sqrtX * FW::sin(theta),
                                      FW::sqrt(FW::max(0.f, 1.f - rv.x)));

                // Make the direction long but not too long to avoid numerical instability in the ray tracer.
                // For our scenes, 100 is a good length. (I know, this special casing sucks.)
                float dirScale = 100.f;

                // Shoot ray, see where we hit
                Vec3f d = dirScale * cwd;
                const RaycastResult result = ctx.m_rt->raycast(o, d);

                if (result.tri != nullptr)
                {
                    // interpolate lighting from previous pass
                    const Vec3i& indices = result.tri->m_data.vertex_indices;

                    // check for backfaces => don't accumulate if we hit a surface from below!
                    if (FW::dot(d, -result.tri->normal()) < 0.f)
                    {
                        continue;
                    }

                    // fetch barycentric coordinates
                    float alpha = result.u;
                    float beta = result.v;

                    // Ei = interpolated irradiance determined by ctx.m_vecPrevBounce from vertices using the barycentric coordinates
                    Vec3f Ei = alpha * ctx.m_vecPrevBounce[indices[0]] +
                        beta * ctx.m_vecPrevBounce[indices[1]] +
                        (1.f - alpha - beta) * ctx.m_vecPrevBounce[indices[2]];

                    // Divide incident irradiance by PI so that we can turn it into outgoing
                    // radiosity by multiplying by the reflectance factor below.
                    Ei *= (1.f / FW_PI);

                    // check for texture
                    const auto mat = result.tri->m_material;

                    if (mat->textures[MeshBase::TextureType_Diffuse].exists())
                    {
                        // read diffuse texture like in assignment1
                        const Texture& tex = mat->textures[MeshBase::TextureType_Diffuse];
                        const Image& texImg = *tex.getImage();

                        Vec2f uv = alpha * ctx.m_scene->vertex(indices[0]).t +
                            beta * ctx.m_scene->vertex(indices[1]).t +
                            (1.f - alpha - beta) * ctx.m_scene->vertex(indices[2]).t;

                        Ei *= texImg.getVec4f(FW::getTexelCoords(uv, texImg.getSize())).getXYZ();
                    }
                    else
                    {
                        // no texture, use constant albedo from material structure.
                        // (this is just one line)
                        Ei *= mat->diffuse.getXYZ();
                    }

                    E += Ei; // accumulate
                }
            }
            // Store result for this bounce
            // Note that since we are storing irradiance, we multiply by PI(
            // (Remember the slides about cosine weighted importance sampling!)
            ctx.m_vecCurr[v] = E * (FW_PI / ctx.m_numHemisphereRays);

            // Also add to the global accumulator.
            ctx.m_vecResult[v] = ctx.m_vecResult[v] + ctx.m_vecCurr[v];

            // uncomment this to visualize only the current bounce
            // ctx.m_vecResult[v] = ctx.m_vecCurr[v];
        }
    }

    // --------------------------------------------------------------------------

    void Radiosity::startRadiosityProcess(MeshWithColors* scene, AreaLight* light, RayTracer* rt, int numBounces,
                                          int numDirectRays, int numHemisphereRays)
    {
        // put stuff the asyncronous processor needs 
        m_context.m_scene = scene;
        m_context.m_rt = rt;
        m_context.m_light = light;
        m_context.m_currentBounce = 0;
        m_context.m_numBounces = numBounces;
        m_context.m_numDirectRays = numDirectRays;
        m_context.m_numHemisphereRays = numHemisphereRays;

        // resize all the buffers according to how many vertices we have in the scene
        m_context.m_vecResult.resize(scene->numVertices());
        m_context.m_vecCurr.resize(scene->numVertices());
        m_context.m_vecPrevBounce.resize(scene->numVertices());
        m_context.m_vecResult.assign(scene->numVertices(), Vec3f(0, 0, 0));

        m_context.m_vecSphericalC.resize(scene->numVertices());
        m_context.m_vecSphericalX.resize(scene->numVertices());
        m_context.m_vecSphericalY.resize(scene->numVertices());
        m_context.m_vecSphericalZ.resize(scene->numVertices());

        m_context.m_vecSphericalC.assign(scene->numVertices(), Vec3f(0, 0, 0));
        m_context.m_vecSphericalX.assign(scene->numVertices(), Vec3f(0, 0, 0));
        m_context.m_vecSphericalY.assign(scene->numVertices(), Vec3f(0, 0, 0));
        m_context.m_vecSphericalZ.assign(scene->numVertices(), Vec3f(0, 0, 0));

        // fire away!
        // the solution exe is multithreaded
        // but you have to make sure your code is thread safe before enabling this!
        m_launcher.setNumThreads(m_launcher.getNumCores());
        //m_launcher.setNumThreads(1);

        m_launcher.popAll();
        m_launcher.push(vertexTaskFunc, &m_context, 0, scene->numVertices());
    }

    // --------------------------------------------------------------------------

    bool Radiosity::updateMeshColors(std::vector<Vec4f>& spherical1, std::vector<Vec4f>& spherical2,
                                     std::vector<float>& spherical3, bool spherical)
    {
        if (!m_context.m_scene || m_context.m_vecResult.size() == 0) return false;
        // Print progress.
        printf("%.2f%% done     \r", 100.0f * m_launcher.getNumFinished() / m_context.m_scene->numVertices());

        // Copy irradiance over to the display mesh.
        // Because we want outgoing radiosity in the end, we divide by PI here
        // and let the shader multiply the final diffuse reflectance in. See App::setupShaders() for details.
        for (int i = 0; i < m_context.m_scene->numVertices(); ++i)
        {
            // Packing data for the spherical harmonic extra.
            // In order to manage with fewer vertex attributes in the shader, the third component is stored as the w components of other actually three-dimensional vectors.
            if (spherical)
            {
                m_context.m_scene->mutableVertex(i).c = m_context.m_vecSphericalC[i] * (1.0f / FW_PI);
                spherical3[i] = m_context.m_vecSphericalZ[i].x * (1.0f / FW_PI);
                spherical1[i] = Vec4f(m_context.m_vecSphericalX[i], m_context.m_vecSphericalZ[i].y) * (1.0f / FW_PI);
                spherical2[i] = Vec4f(m_context.m_vecSphericalY[i], m_context.m_vecSphericalZ[i].z) * (1.0f / FW_PI);
            }
            else
            {
                m_context.m_scene->mutableVertex(i).c = m_context.m_vecResult[i] * (1.0f / FW_PI);
            }
        }
        return true;
    }

    // --------------------------------------------------------------------------

    void Radiosity::checkFinish()
    {
        // have all the vertices from current bounce finished computing?
        if (m_launcher.getNumTasks() == m_launcher.getNumFinished())
        {
            // yes, remove from task list
            m_launcher.popAll();

            // more bounces desired?
            if (m_context.m_currentBounce < m_context.m_numBounces)
            {
                // move current bounce to prev
                m_context.m_vecPrevBounce = m_context.m_vecCurr;
                ++m_context.m_currentBounce;
                // start new tasks for all vertices
                m_launcher.push(vertexTaskFunc, &m_context, 0, m_context.m_scene->numVertices());
                printf("\nStarting bounce %d\n", m_context.m_currentBounce);
            }
            else printf("\n DONE!\n");
        }
    }

    // --------------------------------------------------------------------------
} // namespace FW
