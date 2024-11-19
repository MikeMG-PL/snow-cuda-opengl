#include <iostream>
#include <cstdio>
#include <vector>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"
#include "point_loader.h"
#include "parser.h"
#include "camera.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"

constexpr unsigned int width = 1920;
constexpr unsigned int height = 1080;

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

void error_callback(int error, const char* description)
{
    std::cerr << "Errors: " << description << std::endl;
}

void process_input(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main(int argc, char const* argv[])
{
    char const* margv[] = {
            "cuda-from-scratch.exe",
            "--config",
            "config"
    };

    int margc = sizeof(margv) / sizeof(margv[0]); // Number of arguments
    auto vm = parser::parse_args(margc, margv);

    std::vector<Particle> particles;

    if (vm.count("model"))
    {
        auto model_vec = parser::parseModel(vm["model"].as<std::string>());

        for (auto const& config : model_vec)
        {
            auto model = PointLoader(config.path, config.translate * particle_diameter, config.scale);

            for (auto const& pos : model.positions)
            {
                particles.emplace_back(
                    pos,
                    config.velocity,
                    config.mass,
                    config.hardening,
                    config.young,
                    config.poisson,
                    config.compression,
                    config.stretch

                );
            }
        }
    }

    MPMSolver mpm_solver(particles);

    auto ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    // glfw: initialize and configure
    if (!glfwInit())
        return EXIT_FAILURE;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(width, height, "TSK Snow MPM", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);

    // Setup Dear ImGui context
    char const* glsl_version = "#version 130";
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
    glfwSetErrorCallback(error_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << '\n';
        return EXIT_FAILURE;
    }

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << '\n';
    glEnable(GL_DEPTH_TEST);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    Camera camera(glm::vec3(-0.7f, 0.3f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    float delta_time = 0.0f;
    float last_frame = 0.0f;

    Renderer renderer(width, height, particles.size());
    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    mpm_solver.bind_gl_buffer(renderer.get_snow_vbo());

    // render loop
    int step = 0;
    bool start_simulation = false;
    bool pause_simulation = false;
    bool pressed = false;
    while (!glfwWindowShouldClose(window))
    {
        float current_frame = glfwGetTime();
        delta_time = current_frame - last_frame;
        last_frame = current_frame;

        glfwPollEvents();
        process_input(window);

        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
            renderer.set_origin();
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            renderer.set_up();
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            renderer.set_front();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            renderer.set_side();

        if (!pressed && glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
        {
            pause_simulation = !pause_simulation;
            pressed = true;
        }

        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE)
        {
            pressed = false;
        }

        // Process input
        if (pause_simulation)
        {
            // Process input
            bool forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
            bool backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
            bool left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
            bool right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
            bool roll_left = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
            bool roll_right = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;

            camera.process_keyboard(forward, backward, left, right, roll_left, roll_right, delta_time);

            // Process mouse input
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            static bool first_mouse = true;
            static float last_x = xpos, last_y = ypos;

            if (first_mouse)
            {
                last_x = xpos;
                last_y = ypos;
                first_mouse = false;
            }

            float xoffset = xpos - last_x;
            float yoffset = last_y - ypos; // Reversed since y-coordinates go bottom to top
            last_x = xpos;
            last_y = ypos;

            camera.process_mouse_movement(xoffset, yoffset);

            // Update the renderer's view matrix
            renderer.view_ = camera.get_view_matrix();
        }

        //std::cout << "step: " << step << std::endl;

        if (vm["save"].as<bool>())
        {
            char pnt_fname[128];
            sprintf(pnt_fname, "points_%05d.dat", step);
            mpm_solver.write_to_file(pnt_fname);
        }

        step++;

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
            static float hardening_coefficient = 0.1f;
            //static float initial_density = 300.0f;
            static float initial_youngs_modulus = 4.8e4f;
            static float poisson_ratio = 0.2f;

            ImGui::Begin("Snow Simulation Parameters");

            ImGui::Text("Adjust the snow material properties");

            ImGui::SliderFloat("Critical Compression", &critical_compression, 2.5e-3f, 1.9e-2f);
            ImGui::SliderFloat("Critical Stretch", &critical_stretch, 5.0e-3f, 7.5e-3f);
            ImGui::SliderFloat("Hardening Coefficient", &hardening_coefficient, 0.1f, 15.0f);
            //ImGui::SliderFloat("Initial Density", &initial_density, 300.0f, 500.0f);
            ImGui::SliderFloat("Initial Young's Modulus", &initial_youngs_modulus, 4.8e4f, 1.4e5f);
            ImGui::Text("Poisson's Ratio = 0.2");

            if (ImGui::Button("Start Simulation!"))
            {
                start_simulation = true;

                for (int i = 0; i < particles.size(); i++)
                {
                    particles[i].hardening = hardening_coefficient;
                    particles[i].lambda = (poisson_ratio * initial_youngs_modulus) / ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));
                    particles[i].mu = initial_youngs_modulus / (2.0f * (1.0f + poisson_ratio));
                    particles[i].compression = critical_compression;
                    particles[i].stretch = critical_stretch;
                }
            }

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
            if (!pause_simulation)
            {
                mpm_solver.simulate();
                mpm_solver.write_gl_buffer();
            }

            if (pause_simulation)
            {
                renderer.render(false);
            }
            else
            {
                renderer.render(true);
            }
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
