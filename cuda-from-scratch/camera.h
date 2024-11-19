#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
        : position_(position), front_(0.0f, 0.0f, -1.0f), world_up_(up), yaw_(yaw), pitch_(pitch), speed_(2.5f), sensitivity_(0.1f), roll_speed_(50.0f)
    {
        update_camera_vectors();
    }

    [[nodiscard]] glm::mat4 get_view_matrix() const
    {
        // Combine rotation around Z-axis with lookAt transformation
        glm::mat4 view = glm::lookAt(position_, position_ + front_, up_);
        std::cout << position_.x << " " << position_.y << " " << position_.z << "\n";
        glm::mat4 rollRotation = glm::rotate(glm::mat4(1.0f), glm::radians(roll_), front_);
        return rollRotation * view;
    }

    void process_keyboard(bool const forward, bool const backward, bool const left, bool const right, bool const roll_left,
        bool const roll_right, float const delta_time)
    {
        float const velocity = speed_ * delta_time;

        if (forward)
            position_ += front_ * velocity;
        if (backward)
            position_ -= front_ * velocity;
        if (left)
            position_ -= right_ * velocity;
        if (right)
            position_ += right_ * velocity;

        float const roll_velocity = roll_speed_ * delta_time;

        if (roll_left)
            roll_ += roll_velocity;

        if (roll_right)
            roll_ -= roll_velocity;

        if (roll_ > 360.0f)
            roll_ -= 360.0f;
        else if (roll_ < -360.0f)
            roll_ += 360.0f;
    }

    void process_mouse_movement(float xoffset, float yoffset, bool const constrain_pitch = true)
    {
        xoffset *= sensitivity_;
        yoffset *= sensitivity_;

        yaw_ += xoffset;
        pitch_ += yoffset;

        if (constrain_pitch)
        {
            if (pitch_ > 89.0f)
                pitch_ = 89.0f;

            if (pitch_ < -89.0f)
                pitch_ = -89.0f;
        }

        update_camera_vectors();
    }

    void set_speed(float const speed)
    {
        speed_ = speed;
    }

    void set_sensitivity(float const sensitivity)
    {
        sensitivity_ = sensitivity;
    }

    void set_roll_speed(float const roll_speed)
    {
        roll_speed_ = roll_speed;
    }

private:
    void update_camera_vectors()
    {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front.y = sin(glm::radians(pitch_));
        front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(front);

        right_ = glm::normalize(glm::cross(front_, world_up_));
        up_ = glm::normalize(glm::cross(right_, front_));
    }

    glm::vec3 position_ = {};
    glm::vec3 front_ = {};
    glm::vec3 up_ = {};
    glm::vec3 right_ = {};
    glm::vec3 world_up_ = {};
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    float roll_ = 0.0f; // Rotation around Z-axis
    float speed_ = 0.0f;
    float sensitivity_ = 0.0f;
    float roll_speed_ = 0.0f; // Speed of roll rotation
};
