#version 150

uniform float u_point_size;
uniform mat4 u_pmv_matrix;

in vec3 a_position;
in vec4 a_color;
out vec4 v_color;

void main(void) {
  gl_Position = u_pmv_matrix * vec4(a_position, 1);
  gl_PointSize = u_point_size;
  v_color = a_color;
}
