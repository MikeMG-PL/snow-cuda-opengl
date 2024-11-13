#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <thrust/device_new.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"
#include "point_loader.h"
#include "parser.h"

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"

const unsigned int WIDTH = 1920;
const unsigned int HEIGHT = 1080;

#include "imgui.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void errorCallback(int error, const char* description) {
    std::cerr << "Errors: " << description << std::endl;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main(int argc, const char* argv[]) {

    const char* margv[] = {
            "cuda-from-scratch.exe",
            "--config",
            "config"
    };

    int margc = sizeof(margv) / sizeof(margv[0]); // Number of arguments
    auto vm = parser::parseArgs(margc, margv);

    std::vector<Particle> particles;

    if (vm.count("model")) {
        auto model_vec = parser::parseModel(vm["model"].as<std::string>());

        for (const auto& config : model_vec) {
            auto model = PointLoader(config.path, config.translate * PARTICLE_DIAM, config.scale);
            for (const auto& pos : model.positions) {
                particles.push_back(
                    Particle(
                        pos,
                        config.velocity,
                        config.mass,
                        config.hardening,
                        config.young,
                        config.poisson,
                        config.compression,
                        config.stretch
                    )
                );
            }
        }
    }

    /*
    {
        // two balls
        // TIMESTEP 1e-4
        // HARDENING 10.0f
        // CRIT_COMPRESS 1.9e-2
        // CRIT_STRETCH 7.5e-3
        // ALPHA 0.95f
        // PATICLE_DIAM 0.010
        // STICKY_WALL 0.9
        // FRICTION 1.0
        // DENSITY 400
        // YOUNG 1.4e5
        // POSSION 0.2
        const int height = 70;
        Eigen::Vector3i center(70, height, 80);
        createSphere(particles, center, 20, Eigen::Vector3f(0.0f, 0.0f, -3.0f), mass, lambda, mu, 4);
        center(2) = 30;
        createSphere(particles, center, 7, Eigen::Vector3f(0.0f, 0.0f, 15.0f), mass, lambda, mu, 50);
    }
    */

    // std::cout << "number of particles: " << particles.size() << ", number of bytes in particles: " << particles.size() * sizeof(Particle) << std::endl;

    MPMSolver mpm_solver(particles);

    auto ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "TSK Snow MPM", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // Setup Dear ImGui context
    const char* glsl_version = "#version 130";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // glfw setting callback
    glfwSetErrorCallback(errorCallback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    glEnable(GL_DEPTH_TEST);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    Renderer renderer(WIDTH, HEIGHT, particles.size());
    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    mpm_solver.bindGLBuffer(renderer.getSnowVBO());

    // render loop
    int step = 0;
    bool start_simulation = false;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
            renderer.setOrigin();
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            renderer.setUp();
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            renderer.setFront();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            renderer.setSide();

        //std::cout << "step: " << step << std::endl;

        if (vm["save"].as<bool>()) {
            char pnt_fname[128];
            sprintf(pnt_fname, "points_%05d.dat", step);
            mpm_solver.writeToFile(pnt_fname);
        }
        step++;

        glfwPollEvents();

        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (!start_simulation)
        {
            static float critical_compression = 0.0f;
            static float critical_stretch = 0.0f;
            static float hardening_coefficient = 5.0f;
            static float initial_density = 300.0f;
            static float initial_youngs_modulus = 4.8e4f;
            static float poisson_ratio = 0.2f;

            ImGui::Begin("Snow Simulation Parameters");

            ImGui::Text("Adjust the snow material properties");

            ImGui::SliderFloat("Critical Compression", &critical_compression, 2.5e-3f, 1.9e-2f);
            ImGui::SliderFloat("Critical Stretch", &critical_stretch, 5.0e-3f, 7.5e-3f);
            ImGui::SliderFloat("Hardening Coefficient", &hardening_coefficient, 5.0f, 10.0f);
            ImGui::SliderFloat("Initial Density", &initial_density, 300.0f, 500.0f);
            ImGui::SliderFloat("Initial Young's Modulus", &initial_youngs_modulus, 4.8e4f, 1.4e5f);
            ImGui::Text("Poisson's Ratio = 0.2");

            if (ImGui::Button("Start Simulation!"))
                start_simulation = true;

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (start_simulation)
        {
            mpm_solver.simulate();
            mpm_solver.writeGLBuffer();
            renderer.render();
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
