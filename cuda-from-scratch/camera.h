#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
        : position_(position), front_(0.0f, 0.0f, -1.0f), world_up_(up), yaw_(yaw), pitch_(pitch),
        roll_(0.0f), speed_(2.5f), sensitivity_(0.1f), roll_speed_(50.0f) {
        updateCameraVectors();
    }

    glm::mat4 get_view_matrix() const {
        // Combine rotation around Z-axis with lookAt transformation
        glm::mat4 view = glm::lookAt(position_, position_ + front_, up_);
        std::cout << position_.x << " " << position_.y << " " << position_.z << "\n";
        glm::mat4 rollRotation = glm::rotate(glm::mat4(1.0f), glm::radians(roll_), front_);
        return rollRotation * view;
    }

    void process_keyboard(bool forward, bool backward, bool left, bool right, bool rollLeft, bool rollRight, float deltaTime) {
        float velocity = speed_ * deltaTime;
        if (forward)
            position_ += front_ * velocity;
        if (backward)
            position_ -= front_ * velocity;
        if (left)
            position_ -= right_ * velocity;
        if (right)
            position_ += right_ * velocity;

        float rollVelocity = roll_speed_ * deltaTime;
        if (rollLeft)
            roll_ += rollVelocity;
        if (rollRight)
            roll_ -= rollVelocity;

        if (roll_ > 360.0f)
            roll_ -= 360.0f;
        else if (roll_ < -360.0f)
            roll_ += 360.0f;
    }

    void process_mouse_movement(float xoffset, float yoffset, bool constrainPitch = true) {
        xoffset *= sensitivity_;
        yoffset *= sensitivity_;

        yaw_ += xoffset;
        pitch_ += yoffset;

        if (constrainPitch) {
            if (pitch_ > 89.0f)
                pitch_ = 89.0f;
            if (pitch_ < -89.0f)
                pitch_ = -89.0f;
        }

        updateCameraVectors();
    }

    void setSpeed(float speed) { speed_ = speed; }
    void setSensitivity(float sensitivity) { sensitivity_ = sensitivity; }
    void setRollSpeed(float rollSpeed) { roll_speed_ = rollSpeed; }

private:
    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front.y = sin(glm::radians(pitch_));
        front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(front);

        right_ = glm::normalize(glm::cross(front_, world_up_));
        up_ = glm::normalize(glm::cross(right_, front_));
    }

    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;
    glm::vec3 world_up_;
    float yaw_;
    float pitch_;
    float roll_; // Rotation around Z-axis
    float speed_;
    float sensitivity_;
    float roll_speed_; // Speed of roll rotation
};

#endif
