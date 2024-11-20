#ifndef RENDER_H
#define RENDER_H
#include <string>

struct GLBuffers
{
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
};

struct CUDABUffers
{
    GLuint vao;
    GLuint vbo;
};

class Renderer
{
public:
    Renderer(int width, int height, int number);
    void render(bool const save_frame = true);
    GLuint get_snow_vbo();
    GLuint load_shader(const std::string& vertex_path, const std::string& fragment_path);

    glm::mat4 view_;

private:
    GLfloat const wall_vertices_[20] =
    {
        // positions        // texture coords
         1.5f,  1.5f, 0.0f, 1.0f, 1.0f,  //
        -1.5f,  1.5f, 0.0f, 0.0f, 1.0f,  //
        -1.5f, -1.5f, 0.0f, 0.0f, 0.0f,  //
         1.5f, -1.5f, 0.0f, 1.0f, 0.0f,  //
    };

    GLuint const indices_[6] =
    {
        0, 1, 2, //
        2, 3, 0
    };

    glm::mat4 origin_camera_ = glm::lookAt(glm::vec3(-0.7f, 0.3f, 5.0f),  // camera position
        glm::vec3(-0.7f, 0.3f, 0.0f),  // target position
        glm::vec3(0.3f, 1.0f, 0.0f)   // up vector
    );

    // window size
    int width_;
    int height_;
    float aspect_ratio_;

    // particle number;
    int number_;

    GLuint plane_shader_;
    GLuint snow_shader_;

    GLuint texture1_;
    GLuint texture2_;

    GLBuffers plane_buffers_;
    CUDABUffers snow_buffers_;

    glm::mat4 projection_;
    float fov_ = 15.0f;
    int frameNumber = 0;

    // snow point size;
    GLfloat radius_ = 0.075f;

    void render_wall();
    void render_floor();
    void render_snow();
    void save_frame_to_file();
};

#endif
