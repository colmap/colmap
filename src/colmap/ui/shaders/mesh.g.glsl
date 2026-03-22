#version 150
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 v_position[3];
in vec3 v_normal[3];
in vec2 v_uv[3];
in vec3 v_color[3];

out vec3 g_position;
out vec3 g_normal;
out vec2 g_uv;
out vec3 g_color;
out vec3 g_barycentric;

void main() {
  vec3 bary[3] = vec3[3](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
  for (int i = 0; i < 3; ++i) {
    gl_Position = gl_in[i].gl_Position;
    g_position = v_position[i];
    g_normal = v_normal[i];
    g_uv = v_uv[i];
    g_color = v_color[i];
    g_barycentric = bary[i];
    EmitVertex();
  }
  EndPrimitive();
}
