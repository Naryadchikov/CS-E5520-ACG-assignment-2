#define _CRT_SECURE_NO_WARNINGS

#include "App.hpp"
#include "base/Main.hpp"
#include "gpu/GLContext.hpp"
#include "3d/Mesh.hpp"
#include "io/File.hpp"
#include "io/StateDump.hpp"
#include "base/Random.hpp"

#include "RayTracer.hpp"
#include "rtlib.hpp"

#include <stdio.h>
#include <conio.h>
#include <map>
#include <algorithm>
#include <chrono>

#include <iostream>
#include <fstream>
#include <string>

using namespace FW;


//------------------------------------------------------------------------

bool fileExists(std::string fileName)
{
    return std::ifstream(fileName).good();
}

App::App(std::vector<std::string>& cmd_args)
    : m_commonCtrl(CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
      m_cameraCtrl(&m_commonCtrl, CameraControls::Feature_Default | CameraControls::Feature_StereoControls),
      m_action(Action_None),
      m_cullMode(CullMode_None),
      m_numHemisphereRays(256),
      m_numDirectRays(16),
      m_numBounces(1),
      m_lightSize(0.25f),
      m_toneMapWhite(1.0f),
      m_toneMapBoost(1.0f),
      m_enableSH(false),
      m_shChanged(false)
{
    m_commonCtrl.showFPS(true);
    m_commonCtrl.addStateObject(this);
    m_cameraCtrl.setKeepAligned(true);

    m_commonCtrl.addButton((S32*)&m_action, Action_LoadMesh, FW_KEY_M, "Load mesh or state... (M)");
    m_commonCtrl.addButton((S32*)&m_action, Action_ReloadMesh, FW_KEY_F5, "Reload mesh (F5)");
    m_commonCtrl.addButton((S32*)&m_action, Action_SaveMesh, FW_KEY_O, "Save mesh... (O)");
    m_commonCtrl.addButton((S32*)&m_action, Action_LoadBVH, FW_KEY_NONE, "Load BVH from file...");
    m_commonCtrl.addSeparator();

    m_commonCtrl.addButton((S32*)&m_action, Action_ResetCamera, FW_KEY_NONE, "Reset camera");
    m_commonCtrl.addButton((S32*)&m_action, Action_EncodeCameraSignature, FW_KEY_NONE, "Encode camera signature");
    m_commonCtrl.addButton((S32*)&m_action, Action_DecodeCameraSignature, FW_KEY_NONE, "Decode camera signature...");
    m_window.addListener(&m_cameraCtrl);
    m_commonCtrl.addSeparator();

    m_commonCtrl.addButton((S32*)&m_action, Action_NormalizeScale, FW_KEY_NONE, "Normalize scale");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_FlipXY,                  FW_KEY_NONE,    "Flip X/Y");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_FlipYZ,                  FW_KEY_NONE,    "Flip Y/Z");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_FlipZ,                   FW_KEY_NONE,    "Flip Z");
    m_commonCtrl.addSeparator();

    m_commonCtrl.addButton((S32*)&m_action, Action_NormalizeNormals, FW_KEY_NONE, "Normalize normals");
    m_commonCtrl.addButton((S32*)&m_action, Action_FlipNormals, FW_KEY_NONE, "Flip normals");
    m_commonCtrl.addButton((S32*)&m_action, Action_RecomputeNormals, FW_KEY_NONE, "Recompute normals");
    m_commonCtrl.addSeparator();

    m_commonCtrl.addToggle((S32*)&m_cullMode, CullMode_None, FW_KEY_NONE, "Disable backface culling");
    m_commonCtrl.addToggle((S32*)&m_cullMode, CullMode_CW, FW_KEY_NONE, "Cull clockwise faces");
    m_commonCtrl.addToggle((S32*)&m_cullMode, CullMode_CCW, FW_KEY_NONE, "Cull counter-clockwise faces");
    m_commonCtrl.addButton((S32*)&m_action, Action_FlipTriangles, FW_KEY_NONE, "Flip triangles");
    m_commonCtrl.addSeparator();

    //    m_commonCtrl.addButton((S32*)&m_action, Action_CleanMesh,               FW_KEY_NONE,    "Remove unused materials, denegerate triangles, and unreferenced vertices");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_CollapseVertices,        FW_KEY_NONE,    "Collapse duplicate vertices");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_DupVertsPerSubmesh,      FW_KEY_NONE,    "Duplicate vertices shared between multiple materials");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_FixMaterialColors,       FW_KEY_NONE,    "Override material colors with average over texels");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_DownscaleTextures,       FW_KEY_NONE,    "Downscale textures by 2x");
    //    m_commonCtrl.addButton((S32*)&m_action, Action_ChopBehindNear,          FW_KEY_NONE,    "Chop triangles behind near plane");
    //    m_commonCtrl.addSeparator();

    m_commonCtrl.addToggle((S32*)&m_action, Action_ToggleSphericalHarmonics, FW_KEY_F1,
                           "Render with spherical harmonics (F1)");
    m_commonCtrl.addButton((S32*)&m_action, Action_ComputeRadiosity, FW_KEY_ENTER, "Compute Radiosity (ENTER)");
    m_commonCtrl.addButton((S32*)&m_action, Action_PlaceLightSourceAtCamera,FW_KEY_SPACE,
                           "Place light at camera (SPACE)");
    m_commonCtrl.addButton((S32*)&m_action, Action_LoadRadiosity, FW_KEY_NONE, "Load radiosity solution");
    m_commonCtrl.addButton((S32*)&m_action, Action_SaveRadiosity, FW_KEY_NONE, "Save radiosity solution");

    m_commonCtrl.beginSliderStack();
    m_commonCtrl.addSlider(&m_numDirectRays, 1, 1024, true, FW_KEY_NONE, FW_KEY_NONE, "Direct lighting rays= %d");
    m_commonCtrl.addSlider(&m_numHemisphereRays, 1, 4096, true, FW_KEY_NONE, FW_KEY_NONE,
                           "Diffuse hemisphere rays= %d");
    m_commonCtrl.addSlider(&m_numBounces, 0, 8, false, FW_KEY_NONE, FW_KEY_NONE, "Number of indirect bounces= %d");
    m_commonCtrl.addSlider(&m_lightSize, 0.01f, 2.0f, false, FW_KEY_NONE, FW_KEY_NONE, "Light source area= %f");
    m_commonCtrl.endSliderStack();
    m_commonCtrl.beginSliderStack();
    m_commonCtrl.addSlider(&m_toneMapWhite, 0.05f, 10.0f, true, FW_KEY_NONE, FW_KEY_NONE,
                           "Tone mapping parameter 1= %f");
    m_commonCtrl.addSlider(&m_toneMapBoost, 0.05f, 10.0f, true, FW_KEY_NONE, FW_KEY_NONE,
                           "Tone mapping parameter 2= %f");
    m_commonCtrl.endSliderStack();

    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

    m_window.setTitle("Assignment 2");
    m_commonCtrl.setStateFilePrefix("state_assignment2_");

    m_radiosity.reset(new Radiosity);
    m_areaLight.reset(new AreaLight);
    setupShaders(m_window.getGL());

    glGenBuffers(1, &m_tangentBuffer);
    glGenBuffers(1, &m_bitangentBuffer);
    glGenBuffers(1, &m_sphericalBuffer1);
    glGenBuffers(1, &m_sphericalBuffer2);


    process_args(cmd_args);


    m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1));
    m_timer.start();
}

// returns the index of the needle in the haystack or -1 if not found
int find_argument(std::string needle, std::vector<std::string> haystack)
{
    for (unsigned j = 0; j < haystack.size(); ++j)
        if (!haystack[j].compare(needle))
            return j;

    return -1;
}

void App::process_args(std::vector<std::string>& args)
{
    // all of the possible cmd arguments and the corresponding enums (enum value is the index of the string in the vector)
    const std::vector<std::string> argument_names = {
        "-builder", "-spp", "-output_images", "-use_textures", "-bat_render", "-aa", "-ao", "-ao_length"
    };
    enum argument
    {
        arg_not_found = -1,
        builder = 0,
        spp = 1,
        output_images = 2,
        use_textures = 3,
        bat_render = 4,
        AA = 5,
        AO = 6,
        AO_length = 7
    };

    // similarly a list of the implemented BVH builder types
    const std::vector<std::string> builder_names = {"none", "sah", "object_median", "spatial_median", "linear"};
    enum builder_type
    {
        builder_not_found = -1,
        builder_None = 0,
        builder_SAH = 1,
        builder_ObjectMedian = 2,
        builder_SpatialMedian = 3,
        builder_Linear = 4
    };

    m_settings.batch_render = false;
    m_settings.output_images = false;
    m_settings.use_textures = true;
    m_settings.sample_type = AA_sampling;
    m_settings.ao_length = 1.0f;
    m_settings.spp = 1;
    m_settings.splitMode = SplitMode_Sah;

    for (unsigned i = 0; i < args.size(); ++i)
    {
        // try to recognize the argument
        argument cmd = argument(find_argument(args[i], argument_names));

        switch (cmd)
        {
        case bat_render:
            m_settings.batch_render = true;
            break;

        case output_images:
            m_settings.output_images = true;
            break;

        case use_textures:
            m_settings.use_textures = true;
            break;

        case AO:
            m_settings.sample_type = AO_sampling;
            break;

        case AA:
            m_settings.sample_type = AA_sampling;
            break;

        case spp:
            ++i;
            m_settings.spp = std::stoi(args[i]);
            break;

        case AO_length:
            ++i;
            m_settings.ao_length = std::stof(args[i]);
            break;

        case builder:
            {
                ++i;
                builder_type type = builder_type(find_argument(args[i], builder_names));

                if (type == builder_not_found)
                {
                    type = builder_SAH;
                    std::cout << "BVH builder not recognized, using Surface Area Heuristic" << std::endl;
                    break;
                }

                switch (type)
                {
                case builder_None:
                    m_settings.splitMode = SplitMode_None;
                    break;

                case builder_SAH:
                    m_settings.splitMode = SplitMode_Sah;
                    break;

                case builder_ObjectMedian:
                    m_settings.splitMode = SplitMode_ObjectMedian;
                    break;

                case builder_SpatialMedian:
                    m_settings.splitMode = SplitMode_SpatialMedian;
                    break;

                case builder_Linear:
                    m_settings.splitMode = SplitMode_Linear;
                    break;
                }

                break;
            }


        default:
            if (args[i][0] == '-')std::cout << "argument \"" << args[i] << "\" not found!" << std::endl;
        }
    }
    if (m_settings.batch_render)
        m_settings.use_textures = false;
}

//------------------------------------------------------------------------

App::~App()
{
    glDeleteBuffers(1, &m_tangentBuffer);
    glDeleteBuffers(1, &m_bitangentBuffer);
    glDeleteBuffers(1, &m_sphericalBuffer1);
    glDeleteBuffers(1, &m_sphericalBuffer2);
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
    if (ev.type == Window::EventType_Close)
    {
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }


    Action action = m_action;
    m_action = Action_None;
    String name;
    Mat4f mat;

    switch (action)
    {
    case Action_None:
        break;


    case Action_LoadMesh:
        name = m_window.showFileLoadDialog("Load mesh or state", getMeshImportFilter() + ",dat:State file");
        if (name.getLength())
            if (name.endsWith(".dat"))
                m_commonCtrl.loadState(name);
            else
                loadMesh(name);
        break;

    case Action_ReloadMesh:
        if (m_meshFileName.getLength())
            loadMesh(m_meshFileName);
        break;

    case Action_SaveMesh:
        name = m_window.showFileSaveDialog("Save mesh", getMeshExportFilter());
        if (name.getLength())
            saveMesh(name);
        break;

    case Action_LoadBVH:
        name = m_window.showFileLoadDialog("Load bvh", "hierarchy:BVH");
        m_rt->loadHierarchy(name.getPtr(), m_rtTriangles);
        break;

    case Action_ToggleSphericalHarmonics:
        m_enableSH = !m_enableSH;
        m_shChanged = true;

        break;
    case Action_ResetCamera:
        if (m_mesh)
        {
            m_cameraCtrl.initForMesh(m_mesh.get());
            m_commonCtrl.message("Camera reset");
        }
        break;

    case Action_EncodeCameraSignature:
        m_window.setVisible(false);
        printf("\nCamera signature:\n");
        printf("%s\n", m_cameraCtrl.encodeSignature().getPtr());
        waitKey();
        break;

    case Action_DecodeCameraSignature:
        {
            m_window.setVisible(false);
            printf("\nEnter camera signature:\n");

            char buf[1024];
            if (scanf_s("%s", buf, FW_ARRAY_SIZE(buf)))
                m_cameraCtrl.decodeSignature(buf);
            else
                setError("Signature too long!");

            if (!hasError())
                printf("Done.\n\n");
            else
            {
                printf("Error: %s\n", getError().getPtr());
                clearError();
                waitKey();
            }
        }
        break;

    case Action_NormalizeScale:
        if (m_mesh)
        {
            Vec3f lo, hi;
            m_mesh->getBBox(lo, hi);
            m_mesh->xform(Mat4f::scale(Vec3f(2.0f / (hi - lo).max())) * Mat4f::translate((lo + hi) * -0.5f));
        }
        break;

    case Action_FlipXY:
        std::swap(mat.col(0), mat.col(1));
        if (m_mesh)
        {
            m_mesh->xform(mat);
            m_mesh->flipTriangles();
        }
        break;

    case Action_FlipYZ:
        std::swap(mat.col(1), mat.col(2));
        if (m_mesh)
        {
            m_mesh->xform(mat);
            m_mesh->flipTriangles();
        }
        break;

    case Action_FlipZ:
        mat.col(2) = -mat.col(2);
        if (m_mesh)
        {
            m_mesh->xform(mat);
            m_mesh->flipTriangles();
        }
        break;

    case Action_NormalizeNormals:
        if (m_mesh)
            m_mesh->xformNormals(mat.getXYZ(), true);
        break;

    case Action_FlipNormals:
        mat = -mat;
        if (m_mesh)
            m_mesh->xformNormals(mat.getXYZ(), false);
        break;

    case Action_RecomputeNormals:
        if (m_mesh)
            m_mesh->recomputeNormals();
        break;

    case Action_FlipTriangles:
        if (m_mesh)
            m_mesh->flipTriangles();
        break;

    case Action_CleanMesh:
        if (m_mesh)
            m_mesh->clean();
        break;

    case Action_CollapseVertices:
        if (m_mesh)
            m_mesh->collapseVertices();
        break;

    case Action_DupVertsPerSubmesh:
        if (m_mesh)
            m_mesh->dupVertsPerSubmesh();
        break;

    case Action_FixMaterialColors:
        if (m_mesh)
            m_mesh->fixMaterialColors();
        break;

    case Action_DownscaleTextures:
        if (m_mesh)
            downscaleTextures(m_mesh.get());
        break;

    case Action_ChopBehindNear:
        if (m_mesh)
        {
            Mat4f worldToClip = m_cameraCtrl.getCameraToClip() * m_cameraCtrl.getWorldToCamera();
            Vec4f pleq = worldToClip.getRow(2) + worldToClip.getRow(3);
            chopBehindPlane(m_mesh.get(), pleq);
        }
        break;
    case Action_PlaceLightSourceAtCamera:
        m_areaLight->setOrientation(m_cameraCtrl.getCameraToWorld().getXYZ());
        m_areaLight->setPosition(m_cameraCtrl.getPosition());
        m_commonCtrl.message("Placed light at camera");
        break;

    case Action_ComputeRadiosity:
        m_radiosity->startRadiosityProcess(m_mesh.get(), m_areaLight.get(), m_rt.get(), m_numBounces, m_numDirectRays,
                                           m_numHemisphereRays);
        m_updateClock.start();
        break;

    case Action_LoadRadiosity:
        name = m_window.showFileLoadDialog("Load radiosity solution", "rad:Radiosity Solution");
        if (name.getLength())
            loadRadiosity(name);
        break;

    case Action_SaveRadiosity:
        name = m_window.showFileSaveDialog("Save radiosity solution", "rad:Radiosity Solution");
        if (name.getLength())
            saveRadiosity(name);
        break;
    default:
        FW_ASSERT(false);
        break;
    }

    if (ev.type == Window::EventType_KeyUp)
    {
    }


    m_window.setVisible(true);

    if (ev.type == Window::EventType_Paint)
        renderFrame(m_window.getGL());
    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& d)
{
    String meshFileName;

    d.pushOwner("App");
    d.get(meshFileName, "m_meshFileName");
    d.get((S32&)m_cullMode, "m_cullMode");
    d.get((S32&)m_numHemisphereRays, "m_numHemisphereRays");
    d.get((S32&)m_numDirectRays, "m_numDirectRays");
    d.get((S32&)m_numBounces, "m_numBounces");
    d.get((S32&)m_toneMapWhite, "m_tonemapWhite");
    d.get((S32&)m_toneMapBoost, "m_tonemapBoost");
    d.popOwner();

    m_areaLight->readState(d);
    m_lightSize = m_areaLight->getSize().x; // dirty; doesn't allow for rectangular lights, only square. TODO	

    if (m_meshFileName != meshFileName && meshFileName.getLength())
    {
        loadMesh(meshFileName);
    }
}

//------------------------------------------------------------------------

void App::writeState(StateDump& d) const
{
    d.pushOwner("App");
    d.set(m_meshFileName, "m_meshFileName");
    d.set((S32)m_cullMode, "m_cullMode");
    d.set((S32&)m_numHemisphereRays, "m_numHemisphereRays");
    d.set((S32&)m_numDirectRays, "m_numDirectRays");
    d.set((S32&)m_numBounces, "m_numBounces");
    d.set((S32&)m_toneMapWhite, "m_tonemapWhite");
    d.set((S32&)m_toneMapBoost, "m_tonemapBoost");
    d.popOwner();

    m_areaLight->writeState(d);
}

//------------------------------------------------------------------------

void App::waitKey(void)
{
    printf("Press any key to continue . . . ");
    _getch();
    printf("\n\n");
}

//------------------------------------------------------------------------

void App::renderFrame(GLContext* gl)
{
    // Setup transformations.

    Mat4f worldToCamera = m_cameraCtrl.getWorldToCamera();
    Mat4f projection = gl->xformFitToView(Vec2f(-1.0f, -1.0f), Vec2f(2.0f, 2.0f)) * m_cameraCtrl.getCameraToClip();

    // Initialize GL state.

    glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    if (m_cullMode == CullMode_None)
        glDisable(GL_CULL_FACE);
    else
    {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace((m_cullMode == CullMode_CW) ? GL_CCW : GL_CW);
    }
    GLContext::checkErrors();

    // No mesh => skip.

    if (!m_mesh)
    {
        gl->drawModalMessage("No mesh loaded!");
        return;
    }

    // setup tone mapping
    GLContext::Program* prog = gl->getProgram("MeshBase::draw_generic");
    prog->use();
    GLContext::checkErrors();

    gl->setUniform(prog->getUniformLoc("reinhardLWhite"), m_toneMapWhite);
    gl->setUniform(prog->getUniformLoc("tonemapBoost"), m_toneMapBoost);
    gl->setUniform(prog->getUniformLoc("useSphericalHarmonics"), m_enableSH);
    gl->setUniform(prog->getUniformLoc("t"), m_timer.getElapsed());
    if (m_mesh)
    {
        gl->setUniform(prog->getUniformLoc("ripple"), (m_mesh->numVertices() == 31440) ? 1 : 0);
    }
    GLContext::checkErrors();

    // if we are computing radiosity, refresh mesh colors every 0.5 seconds
    if (m_shChanged || (m_radiosity->isRunning() && m_updateClock.getElapsed() > 0.5f))
    {
        m_shChanged = false;

        if (m_radiosity->updateMeshColors(m_sphericalResults1, m_sphericalResults2, m_sphericalResults3, m_enableSH))
        {
            if (m_radiosity->isRunning())
                m_radiosity->checkFinish();

            glBindBuffer(GL_ARRAY_BUFFER, m_sphericalBuffer1);

            sendTangents(&m_sphericalResults3);

            GLuint spherical1Location = glGetAttribLocation(prog->getHandle(), "sphericalHarmonic1"),
                   spherical2Location = glGetAttribLocation(prog->getHandle(), "sphericalHarmonic2");

            if (spherical1Location != -1)
            {
                glBindBuffer(GL_ARRAY_BUFFER, m_sphericalBuffer1);
                glBufferData(GL_ARRAY_BUFFER, m_sphericalResults1.size() * sizeof(Vec4f), m_sphericalResults1.data(),
                             GL_STATIC_DRAW);
                glEnableVertexAttribArray(spherical1Location);
                glVertexAttribPointer(spherical1Location, 4, GL_FLOAT, false, 0, 0);
            }
            if (spherical2Location != -1)
            {
                glBindBuffer(GL_ARRAY_BUFFER, m_sphericalBuffer2);
                glBufferData(GL_ARRAY_BUFFER, m_sphericalResults2.size() * sizeof(Vec4f), m_sphericalResults2.data(),
                             GL_STATIC_DRAW);
                glEnableVertexAttribArray(spherical2Location);
                glVertexAttribPointer(spherical2Location, 4, GL_FLOAT, false, 0, 0);
            }
            GLContext::checkErrors();
            // restart cycle
            m_updateClock.start();
        }
    }

    // Render.
    GLContext::checkErrors();
    if (!gl->getConfig().isStereo)
        renderScene(gl, worldToCamera, projection);
    else
    {
        glDrawBuffer(GL_BACK_LEFT);
        renderScene(gl, m_cameraCtrl.getCameraToLeftEye() * worldToCamera, projection);
        glDrawBuffer(GL_BACK_RIGHT);
        glClear(GL_DEPTH_BUFFER_BIT);
        renderScene(gl, m_cameraCtrl.getCameraToRightEye() * worldToCamera, projection);
        glDrawBuffer(GL_BACK);
    }

    m_areaLight->setSize(Vec2f(m_lightSize));
    m_areaLight->draw(worldToCamera, projection);


    // Display status line.

    m_commonCtrl.message(sprintf("Triangles = %d, vertices = %d, materials = %d",
                                 m_mesh->numTriangles(),
                                 m_mesh->numVertices(),
                                 m_mesh->numSubmeshes()),
                         "meshStats");
}

//------------------------------------------------------------------------

void App::renderScene(GLContext* gl, const Mat4f& worldToCamera, const Mat4f& projection)
{
    // Draw mesh.
    if (m_mesh)
        m_mesh->draw(gl, worldToCamera, projection);
}


//------------------------------------------------------------------------


const static F32 texAttrib[] =
{
    0, 1,
    1, 1,
    0, 0,
    1, 0
};


//------------------------------------------------------------------------


void App::loadMesh(const String& fileName)
{
    // find the scene name; the file path without preceding folders and file extension

    std::string path = std::string(fileName.getPtr());

    size_t begin = path.find_last_of("/\\") + 1,
           end = path.find_last_of(".");

    m_results.scene_name = path.substr(begin, end - begin);

    std::cout << "Scene name: " << m_results.scene_name << std::endl;

    m_window.showModalMessage(sprintf("Loading mesh from '%s'...", fileName.getPtr()));
    String oldError = clearError();
    std::unique_ptr<MeshBase> mesh((MeshBase*)importMesh(fileName));

    String newError = getError();

    if (restoreError(oldError))
    {
        m_commonCtrl.message(sprintf("Error while loading '%s': %s", fileName.getPtr(), newError.getPtr()));
        return;
    }

    m_meshFileName = fileName;

    // convert input mesh to colored format
    m_mesh.reset(new MeshWithColors(*mesh));


    // fix input colors to white so we see something
    for (S32 i = 0; i < m_mesh->numVertices(); ++i)
        m_mesh->mutableVertex(i).c = Vec3f(1, 1, 1);

    m_commonCtrl.message(sprintf("Loaded mesh from '%s'", fileName.getPtr()));

    // build the BVH!
    constructTracer();
}

//------------------------------------------------------------------------

void App::saveMesh(const String& fileName)
{
    if (!m_mesh)
    {
        m_commonCtrl.message("No mesh to save!");
        return;
    }

    m_window.showModalMessage(sprintf("Saving mesh to '%s'...", fileName.getPtr()));
    String oldError = clearError();
    exportMesh(fileName, m_mesh.get());
    String newError = getError();

    if (restoreError(oldError))
    {
        m_commonCtrl.message(sprintf("Error while saving '%s': %s", fileName.getPtr(), newError.getPtr()));
        return;
    }

    m_meshFileName = fileName;
    m_commonCtrl.message(sprintf("Saved mesh to '%s'", fileName.getPtr()));
}

//------------------------------------------------------------------------

// This function iterates over all the "sub-meshes" (parts of the object with different materials),
// heaps all the vertices and triangles together, and calls the BVH constructor.
// It is the responsibility of the tree to free the data when deleted.
// This functionality is _not_ part of the RayTracer class in order to keep it separate
// from the specifics of the Mesh class.
void App::constructTracer()
{
    // fetch vertex and triangle data ----->
    m_rtTriangles.clear();
    m_rtTriangles.reserve(m_mesh->numTriangles());

    m_sphericalResults1.resize(m_mesh->numVertices());
    m_sphericalResults2.resize(m_mesh->numVertices());
    m_sphericalResults3.resize(m_mesh->numVertices());

    m_tangents.resize(m_mesh->numVertices());
    m_bitangents.resize(m_mesh->numVertices());

    for (int i = 0; i < m_mesh->numSubmeshes(); ++i)
    {
        const Array<Vec3i>& idx = m_mesh->indices(i);
        for (int j = 0; j < idx.getSize(); ++j)
        {
            const VertexPNTC &v0 = m_mesh->vertex(idx[j][0]),
                             &v1 = m_mesh->vertex(idx[j][1]),
                             &v2 = m_mesh->vertex(idx[j][2]);

            RTTriangle t = RTTriangle(v0, v1, v2);

            t.m_data.vertex_indices = idx[j];
            t.m_material = &(m_mesh->material(i));

            m_rtTriangles.push_back(t);
            // EXTRA: generate tangent space for triangle t here (you can probably reuse code from assn1 if you implemented tangent space normal mapping)
            for (int k = 0; k < 3; ++k)
            {
                m_tangents[idx[j][k]] += Vec3f(.0f);
                m_bitangents[idx[j][k]] += Vec3f(.0f);
            }
        }
    }


    for (auto& tangent : m_tangents)
        tangent.normalize();
    for (auto& bitangent : m_bitangents)
        bitangent.normalize();

    GLContext::checkErrors();
    sendTangents();

    // compute checksum

    m_rtVertexPositions.clear();
    m_rtVertexPositions.reserve(m_mesh->numVertices());
    for (int i = 0; i < m_mesh->numVertices(); ++i)
        m_rtVertexPositions.push_back(m_mesh->vertex(i).p);

    String md5 = RayTracer::computeMD5(m_rtVertexPositions);
    FW::printf("Mesh MD5: %s\n", md5.getPtr());

    // construct a new ray tracer (deletes the old one if there was one)
    m_rt.reset(new RayTracer());

    // whether we want to try loading a saved hierarchy from disk
    bool tryLoadHierarchy = true;

    // always construct when measuring performance
    if (m_settings.batch_render)
        tryLoadHierarchy = false;

    if (tryLoadHierarchy)
    {
        // check if saved hierarchy exists

        std::string meshName = m_meshFileName.getPtr();
        std::string hierarchyName = meshName.substr(0, meshName.find_last_of("."));
#ifdef _WIN64
		hierarchyName += "_x64";
#endif

        hierarchyName += ".hierarchy";

        String hierarchyCacheFile = hierarchyName.c_str();

        if (fileExists(hierarchyCacheFile.getPtr()))
        {
            // yes, load!
            m_rt->loadHierarchy(hierarchyCacheFile.getPtr(), m_rtTriangles);
            ::printf("Loaded hierarchy from %s\n", hierarchyCacheFile.getPtr());
        }
        else
        {
            // no, construct...
            LARGE_INTEGER start, stop, frequency;
            QueryPerformanceFrequency(&frequency);
            QueryPerformanceCounter(&start); // Start time stamp		

            m_rt->constructHierarchy(m_rtTriangles, m_settings.splitMode);

            QueryPerformanceCounter(&stop); // Stop time stamp

            int build_time = (int)((stop.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart);
            // Get timer result in milliseconds
            std::cout << "Build time: " << build_time << " ms" << std::endl;
            // .. and save!
            m_rt->saveHierarchy(hierarchyCacheFile.getPtr(), m_rtTriangles);
            ::printf("Saved hierarchy to %s\n", hierarchyCacheFile.getPtr());
        }
    }
    else
    {
        // nope, bite the bullet and construct it

        LARGE_INTEGER start, stop, frequency;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start); // Start time stamp		

        m_rt->constructHierarchy(m_rtTriangles, m_settings.splitMode);

        QueryPerformanceCounter(&stop); // Stop time stamp

        m_results.build_time = (int)((stop.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart);
        // Get timer result in milliseconds
        std::cout << "Build time: " << m_results.build_time << " ms" << std::endl;
    }
}


//------------------------------------------------------------------------


// This method sends the tangent and bitangent arrays to the GPU. One component of the spherical harmonics is also added as the fourth component of the tangent.
void App::sendTangents(std::vector<float>* sphericalCoefficients)
{
    std::vector<Vec4f> tangents(m_tangents.size());
    if (sphericalCoefficients == nullptr)
        for (int i = 0; i < m_tangents.size(); ++i)
            tangents[i] = Vec4f(m_tangents[i], .0f);
    else
        for (int i = 0; i < m_tangents.size(); ++i)
            tangents[i] = Vec4f(m_tangents[i], sphericalCoefficients->at(i));

    GLContext::Program* prog = m_window.getGL()->getProgram("MeshBase::draw_generic");

    GLuint tangentLocation = glGetAttribLocation(prog->getHandle(), "tangent"),
           bitangentLocation = glGetAttribLocation(prog->getHandle(), "bitangent");

    GLContext::checkErrors();
    if (tangentLocation != -1)
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_tangentBuffer);
        glBufferData(GL_ARRAY_BUFFER, tangents.size() * sizeof(Vec4f), tangents.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(tangentLocation);
        glVertexAttribPointer(tangentLocation, 4, GL_FLOAT, false, 0, 0);
    }
    if (bitangentLocation != -1)
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_bitangentBuffer);
        glBufferData(GL_ARRAY_BUFFER, m_bitangents.size() * sizeof(Vec3f), m_bitangents.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(bitangentLocation);
        glVertexAttribPointer(bitangentLocation, 3, GL_FLOAT, false, 0, 0);
    }
    GLContext::checkErrors();
}

//------------------------------------------------------------------------

void App::downscaleTextures(MeshBase* mesh)
{
    FW_ASSERT(mesh);
    Hash<const Image*, Texture> hash;
    for (int submeshIdx = 0; submeshIdx < mesh->numSubmeshes(); submeshIdx++)
        for (int textureIdx = 0; textureIdx < MeshBase::TextureType_Max; textureIdx++)
        {
            Texture& tex = mesh->material(submeshIdx).textures[textureIdx];
            if (tex.exists())
            {
                const Image* orig = tex.getImage();
                if (!hash.contains(orig))
                {
                    Image* scaled = orig->downscale2x();
                    hash.add(orig, (scaled) ? Texture(scaled, tex.getID()) : tex);
                }
                tex = hash.get(orig);
            }
        }
}

//------------------------------------------------------------------------

void App::chopBehindPlane(MeshBase* mesh, const Vec4f& pleq)
{
    FW_ASSERT(mesh);
    int posAttrib = mesh->findAttrib(MeshBase::AttribType_Position);
    if (posAttrib == -1)
        return;

    for (int submeshIdx = 0; submeshIdx < mesh->numSubmeshes(); submeshIdx++)
    {
        Array<Vec3i>& inds = mesh->mutableIndices(submeshIdx);
        int triOut = 0;
        for (int triIn = 0; triIn < inds.getSize(); triIn++)
        {
            if (dot(mesh->getVertexAttrib(inds[triIn].x, posAttrib), pleq) >= 0.0f ||
                dot(mesh->getVertexAttrib(inds[triIn].y, posAttrib), pleq) >= 0.0f ||
                dot(mesh->getVertexAttrib(inds[triIn].z, posAttrib), pleq) >= 0.0f)
            {
                inds[triOut++] = inds[triIn];
            }
        }
        inds.resize(triOut);
    }

    mesh->clean();
}

//------------------------------------------------------------------------

void App::setupShaders(GLContext* gl)
{
    GLContext::Program* prog = new GLContext::Program(
        "#version 120\n"
        FW_GL_SHADER_SOURCE(
            uniform mat4 posToClip;
            uniform mat4 posToCamera;
            uniform mat3 normalToCamera;
            attribute vec3 positionAttrib;
            attribute vec3 normalAttrib;
            attribute vec4 vcolorAttrib; // Workaround. "colorAttrib" appears to confuse certain ATI drivers.
            attribute vec2 texCoordAttrib;
            attribute vec4 sphericalHarmonic1;
            attribute vec4 sphericalHarmonic2;
            attribute vec4 tangent;
            // the fourth component is spherical harmonics data, both the tangent and bitangent are three-dimensional.
            attribute vec3 bitangent;
            centroid varying vec3 positionVarying;
            centroid varying vec3 normalVarying;
            centroid varying vec4 colorVarying;
            varying vec2 texCoordVarying;
            varying vec3 sphericalConstant;
            varying vec3 sphericalX;
            varying vec3 sphericalY;
            varying vec3 sphericalZ;

            void main()
            {
            // the coefficients have to be packed to keep the amount of attributes down - ugly but necessary (and not vital to understand)
            sphericalConstant = vcolorAttrib.xyz;
            sphericalX = sphericalHarmonic1.xyz;
            sphericalY = sphericalHarmonic2.xyz;
            sphericalZ = vec3(tangent.w, sphericalHarmonic1.w, sphericalHarmonic2.w);

            // if you need to invert a matrix, inverse() exists but needs version 150 or above.

            vec4 pos = vec4(positionAttrib, 1.0);
            gl_Position = posToClip * pos;
            positionVarying = (posToCamera * pos).xyz;
            normalVarying = normalAttrib;
            colorVarying = vcolorAttrib;
            texCoordVarying = texCoordAttrib;
            }
        ),
        "#version 120\n"
        FW_GL_SHADER_SOURCE(
            uniform bool hasNormals;
            uniform bool hasDiffuseTexture;
            uniform bool hasAlphaTexture;
            uniform bool hasNormalTexture;
            uniform vec4 diffuseUniform;
            uniform vec3 specularUniform;
            uniform float glossiness;
            uniform float reinhardLWhite;
            uniform float tonemapBoost;
            uniform sampler2D diffuseSampler;
            uniform sampler2D alphaSampler;
            uniform sampler2D normalSampler;
            uniform bool useSphericalHarmonics;
            centroid varying vec3 positionVarying;
            centroid varying vec3 normalVarying;
            centroid varying vec4 colorVarying;
            varying vec2 texCoordVarying;
            varying vec3 sphericalConstant;
            varying vec3 sphericalX;
            varying vec3 sphericalY;
            varying vec3 sphericalZ;
            uniform int ripple;
            uniform float t;
            void main()
            {
            vec4 diffuseColor = diffuseUniform;
            vec3 specularColor = specularUniform;

            vec3 normal;

            if (hasDiffuseTexture)
            diffuseColor.rgb = texture2D(diffuseSampler, texCoordVarying).rgb;

            if (hasNormalTexture)
            normal = normalize(normalVarying); // EXTRA: read the normal map instead
            else
            normal = normalize(normalVarying);

            if (useSphericalHarmonics) {
            vec3 pos = positionVarying;
            // produce the ripple effect for the normals of the ceiling in the cornell box
            if (ripple == 1 && normal.y < -.5 && pos.y>.9)
            normal = normalize(normal + .5 * vec3(pos.xz / (.1 + length(pos.xz)) * sin(dot(pos.xz, pos.xz) * 50. - 4. *
                t), .0).xzy);
            // EXTRA: evaluate the basis functions here
            }

            diffuseColor.rgb *= colorVarying.rgb;

            if (hasAlphaTexture)
            diffuseColor.a = texture2D(alphaSampler, texCoordVarying).g;

            if (diffuseColor.a <= 0.5)
            discard;

            diffuseColor *= tonemapBoost;
            float I = diffuseColor.r + diffuseColor.g + diffuseColor.b;
            I = I * (1.0 / 3.0);
            diffuseColor *= (1.0 + I / (reinhardLWhite*reinhardLWhite)) / (1.0 + I);

            // Gamma correction.
            diffuseColor.rgb = pow(max(diffuseColor.rgb, 0.0), vec3(1.0 / 2.2));

            gl_FragColor = vec4(diffuseColor.rgb, 1.0);
            }
        )

    );

    gl->setProgram("MeshBase::draw_generic", prog);
}

//------------------------------------------------------------------------

void App::loadRadiosity(const String& fileName)
{
    FW_ASSERT(m_radiosity != 0);

    // check the scene matches
    String md5 = RayTracer::computeMD5(m_rtVertexPositions);
    File f(fileName, File::Read);
    String savedMD5;
    f >> savedMD5;
    if (savedMD5 != md5)
        printf("Radiosity was computed for another scene!");

    // load..
    std::vector<Vec3f> v;
    v.resize(m_mesh->numVertices());
    f.read(&v[0], int(sizeof(Vec3f) * v.size()));

    // and update colors for display.
    for (S32 i = 0; i < m_mesh->numVertices(); ++i)
        m_mesh->mutableVertex(i).c = v[i];
}

//------------------------------------------------------------------------

void App::saveRadiosity(const String& fileName)
{
    FW_ASSERT(m_radiosity != 0);

    // save the MD5 hash to make sure we won't load the wrong solution later
    String md5 = RayTracer::computeMD5(m_rtVertexPositions);
    File f(fileName, File::Create);
    f << md5;

    // save..
    std::vector<Vec3f> v;
    v.resize(m_mesh->numVertices());
    for (S32 i = 0; i < m_mesh->numVertices(); ++i)
        v[i] = m_mesh->vertex(i).c;

    f.write(&v[0], int(sizeof(Vec3f) * v.size()));
}


//------------------------------------------------------------------------

void FW::init(std::vector<std::string>& args)
{
    new App(args);
}

//------------------------------------------------------------------------

bool App::fileExists(const String& fn)
{
    FILE* pF = fopen(fn.getPtr(), "rb");
    if (pF != 0)
    {
        fclose(pF);
        return true;
    }
    else
    {
        return false;
    }
}

//------------------------------------------------------------------------
