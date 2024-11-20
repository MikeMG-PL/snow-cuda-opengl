#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

Renderer::Renderer(int width, int height, int number)
    : width_(width),
    height_(height),
    number_(number)
{
    aspect_ratio_ = static_cast<float>(width_) / height_;

    view_ = origin_camera_;
    projection_ = glm::perspective(glm::radians(fov_), static_cast<float>(width_) / height_, 0.1f, 100.0f);

    // bind textures on corresponding texture units
    texture1_ = load_texture("../images/container.jpg");
    texture2_ = load_texture("../images/container.jpg");
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2_);

    // plane
    // ------------------------------------------------------------------------
    plane_shader_ = load_shader("./plane.vert", "./plane.frag");

    glUseProgram(plane_shader_);
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "projection"), 1, GL_FALSE, glm::value_ptr(projection_));

    glGenVertexArrays(1, &plane_buffers_.vao);
    glBindVertexArray(plane_buffers_.vao);

    glGenBuffers(1, &plane_buffers_.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, plane_buffers_.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wall_vertices_), wall_vertices_, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &plane_buffers_.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_buffers_.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_), indices_, GL_STATIC_DRAW);
    // ------------------------------------------------------------------------

    // snow
    // ------------------------------------------------------------------------
    snow_shader_ = load_shader("./snow.vert", "./snow.frag");

    glUseProgram(snow_shader_);
    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "projection"), 1, GL_FALSE, glm::value_ptr(projection_));
    glUniform1f(glGetUniformLocation(snow_shader_, "radius"), radius_);
    glUniform1f(glGetUniformLocation(snow_shader_, "scale"), width_ / aspect_ratio_ * (1.0f / tanf(fov_ * 0.5f)));

    glGenVertexArrays(1, &snow_buffers_.vao);
    glBindVertexArray(snow_buffers_.vao);

    glGenBuffers(1, &snow_buffers_.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, snow_buffers_.vbo);
    glBufferData(GL_ARRAY_BUFFER, number_ * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    // ------------------------------------------------------------------------
}

void Renderer::render(bool const save_frame)
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render_floor();
    render_snow();

    if (save_frame)
    {
        save_frame_to_file();
    }
}

void Renderer::render_wall()
{
    glUseProgram(plane_shader_);
    glUniform1i(glGetUniformLocation(plane_shader_, "texture1"), 1);
    glBindVertexArray(plane_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 1.5f, -1.5f));
    model = glm::scale(model, glm::vec3(3.0f, 3.0f, 3.0f));
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::render_floor()
{
    glUseProgram(plane_shader_);
    glUniform1i(glGetUniformLocation(plane_shader_, "texture1"), 0);
    glBindVertexArray(plane_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    model = glm::scale(model, glm::vec3(5.0f, 5.0f, 5.0f));
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::render_snow()
{
    glUseProgram(snow_shader_);
    glBindVertexArray(snow_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_POINTS, 0, GLsizei(number_));
}

void Renderer::save_frame_to_file()
{
    std::vector<unsigned char> pixels(width_ * height_ * 3); // RGB format
    glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::string additional_zeros;
    if (frameNumber < 10)
        additional_zeros = "00";
    else if (frameNumber < 100)
        additional_zeros = "0";

    std::string const filename = "../frames/frame" + additional_zeros + std::to_string(frameNumber) + ".png";

    for (int y = 0; y < height_ / 2; ++y)
    {
        int const oppositeY = height_ - y - 1;
        for (int x = 0; x < width_ * 3; ++x)
        {
            std::swap(pixels[y * width_ * 3 + x], pixels[oppositeY * width_ * 3 + x]);
        }
    }

    stbi_write_png(filename.c_str(), width_, height_, 3, pixels.data(), width_ * 3);
    frameNumber++;
}

GLuint Renderer::load_texture(const std::string& texture_path)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load and generate the texture
    stbi_set_flip_vertically_on_load(true);
    int width, height, channel;
    unsigned char* data =
        stbi_load(texture_path.c_str(), &width, &height, &channel, 0);
    if (data)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if (channel == 3)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                GL_UNSIGNED_BYTE, data);
        else if (channel == 4)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cerr << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    return texture;
}

void check_errors(GLuint id, const std::string& type)
{
    GLint success;
    if (type == "SHADER")
    {
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            GLint logLength;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLength);
            std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
            glGetShaderInfoLog(id, logLength, nullptr, infoLog.data());
            std::cerr << infoLog.data() << std::endl;
        }
    }
    else
    {
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success)
        {
            GLint logLength;
            glGetProgramiv(id, GL_INFO_LOG_LENGTH, &logLength);
            std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
            glGetProgramInfoLog(id, logLength, nullptr, infoLog.data());
            std::cerr << infoLog.data() << std::endl;
        }
    }
}

GLuint Renderer::load_shader(const std::string& vertex_path,
    const std::string& fragment_path)
{
    // Read shader code
    std::string vertex_code;
    std::string fragment_code;
    std::ifstream vertex_shader_file;
    std::ifstream fragment_shader_file;

    vertex_shader_file.open(vertex_path);
    fragment_shader_file.open(fragment_path);
    std::stringstream vertex_shader_stream, fragment_shader_stream;
    vertex_shader_stream << vertex_shader_file.rdbuf();
    fragment_shader_stream << fragment_shader_file.rdbuf();
    vertex_shader_file.close();
    fragment_shader_file.close();
    vertex_code = vertex_shader_stream.str();
    fragment_code = fragment_shader_stream.str();

    // Compile shaders with this lambda
    auto compile_shader = [](GLuint& id, const std::string& code)
        {
            const char* shader_code = code.c_str();
            glShaderSource(id, 1, &shader_code, nullptr);
            glCompileShader(id);
            check_errors(id, "SHADER");
        };

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    compile_shader(vertex_shader, vertex_code);
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    compile_shader(fragment_shader, fragment_code);

    GLuint shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    check_errors(shader_program, "PROGRAM");

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return shader_program;
}

GLuint Renderer::get_snow_vbo()
{
    return snow_buffers_.vbo;
}