#pragma once

#include <string>
#include <glad/glad.h>

GLuint load_shader(const std::string& vertex_path, const std::string& fragment_path);