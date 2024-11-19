#version 410 core

in vec2 texture_coordinate;

uniform sampler2D texture1;

void main()
{
    gl_FragColor = vec4(0.1);
}
