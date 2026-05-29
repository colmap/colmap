#version 150

uniform mat4 u_pmv_matrix;
uniform mat4 u_model_view_matrix;
uniform mat3 u_normal_matrix;

in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;
in vec3 a_color;

out vec3 v_position;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_color;

void main(void) {
  gl_Position = u_pmv_matrix * vec4(a_position, 1);
  v_position = (u_model_view_matrix * vec4(a_position, 1)).xyz;
  v_normal = u_normal_matrix * a_normal;
  v_uv = a_uv;
  v_color = a_color;
}
