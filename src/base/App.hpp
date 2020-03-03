#pragma once


#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "gui/Image.hpp"
#include "3d/CameraControls.hpp"
#include "gpu/Buffer.hpp"

#include <vector>
#include <memory>

#include "RayTracer.hpp"

#include "AreaLight.hpp"
#include "Radiosity.hpp"


namespace FW
{
    //------------------------------------------------------------------------

    class App : public Window::Listener, public CommonControls::StateObject
    {
    private:
        enum Action
        {
            Action_None,

            Action_LoadMesh,
            Action_ReloadMesh,
            Action_SaveMesh,
            Action_LoadBVH,

            Action_ResetCamera,
            Action_EncodeCameraSignature,
            Action_DecodeCameraSignature,

            Action_NormalizeScale,
            Action_FlipXY,
            Action_FlipYZ,
            Action_FlipZ,

            Action_NormalizeNormals,
            Action_FlipNormals,
            Action_RecomputeNormals,

            Action_FlipTriangles,

            Action_CleanMesh,
            Action_CollapseVertices,
            Action_DupVertsPerSubmesh,
            Action_FixMaterialColors,
            Action_DownscaleTextures,
            Action_ChopBehindNear,

            // Assignment actions
            Action_TracePrimaryRays,
            Action_PlaceLightSourceAtCamera,

            Action_ComputeRadiosity,
            Action_LoadRadiosity,
            Action_SaveRadiosity,
            Action_ToggleSphericalHarmonics
        };

        enum CullMode
        {
            CullMode_None = 0,
            CullMode_CW,
            CullMode_CCW,
        };

        struct RayVertex
        {
            Vec3f pos;
            U32 color;
        };

        enum bvh_build_method { None, SAH };

        enum SamplingType { AO_sampling, AA_sampling };

        // this structure holds the necessary arguments when rendering using command line parameters
        struct
        {
            bool batch_render;
            SplitMode splitMode; // the BVH builder to use
            int spp; // samples per pixel to use
            SamplingType sample_type; // AO or AA sampling; AO includes one extra sample for the primary ray
            bool output_images; // might be useful to compare images with the example
            bool use_textures; // whether or not textures are used
            bool use_arealights; // whether or not area light sampling is used
            bool enable_reflections; // whether to compute reflections in whitted integrator
            float ao_length;
        } m_settings;

        struct
        {
            std::string state_name; // filenames of the state and scene files
            std::string scene_name;
            int rayCount;
            int build_time, trace_time;
        } m_results;

    public:
        App(std::vector<std::string>& cmd_args);
        virtual ~App(void);

        virtual bool handleEvent(const Window::Event& ev);
        virtual void readState(StateDump& d);
        virtual void writeState(StateDump& d) const;

    private:
        void process_args(std::vector<std::string>& args);

        void waitKey(void);
        void renderFrame(GLContext* gl);
        void renderScene(GLContext* gl, const Mat4f& worldToCamera, const Mat4f& projection);
        void loadMesh(const String& fileName);
        void saveMesh(const String& fileName);
        void loadRayDump(const String& fileName);

        static void downscaleTextures(MeshBase* mesh);
        static void chopBehindPlane(MeshBase* mesh, const Vec4f& pleq);

        static bool fileExists(const String& fileName);

        // 
        void constructTracer(void);

        void blitRttToScreen(GLContext* gl);

        void setupShaders(GLContext* gl);
        void sendTangents(std::vector<float>* sphericalCoefficients = nullptr);
        void loadRadiosity(const String& fileName);
        void saveRadiosity(const String& fileName);

    private:
        App(const App&); // forbidden
        App& operator=(const App&); // forbidden

    private:
        Window m_window;
        CommonControls m_commonCtrl;
        CameraControls m_cameraCtrl;

        Action m_action;
        String m_meshFileName;
        CullMode m_cullMode;
        Timer m_timer;

        std::unique_ptr<RayTracer> m_rt;
        std::vector<Vec3f> m_rtVertexPositions; // kept only for MD5 checksums
        std::vector<RTTriangle> m_rtTriangles;

        std::unique_ptr<MeshWithColors> m_mesh;
        std::unique_ptr<AreaLight> m_areaLight;
        std::unique_ptr<Radiosity> m_radiosity;
        std::vector<Vec3f> m_tangents, m_bitangents;
        std::vector<float> m_sphericalResults3;
        std::vector<Vec4f> m_sphericalResults1, m_sphericalResults2;
        int m_numHemisphereRays;
        int m_numDirectRays;
        int m_numBounces;
        float m_lightSize;
        float m_toneMapWhite;
        float m_toneMapBoost;
        Timer m_updateClock;
        GLuint m_tangentBuffer, m_bitangentBuffer, m_sphericalBuffer1, m_sphericalBuffer2;
        bool m_enableSH, m_shChanged;
    };


    //------------------------------------------------------------------------
}
