#version 150

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 u_inv_viewport;
uniform float u_line_width;

in vec4 v_pos[2];
in vec4 v_color[2];
out vec4 g_color;

void main() {
  vec2 dir = normalize(v_pos[1].xy / v_pos[1].w - v_pos[0].xy / v_pos[0].w);
  vec2 normal_dir = vec2(-dir.y, dir.x);
  vec2 offset = (vec2(u_line_width) * u_inv_viewport) * normal_dir;

  gl_Position = vec4(v_pos[0].xy + offset * v_pos[0].w, v_pos[0].z, v_pos[0].w);
  g_color = v_color[0];
  EmitVertex();

  gl_Position = vec4(v_pos[1].xy + offset * v_pos[1].w, v_pos[1].z, v_pos[1].w);
  g_color = v_color[1];
  EmitVertex();

  gl_Position = vec4(v_pos[0].xy - offset * v_pos[0].w, v_pos[0].z, v_pos[0].w);
  g_color = v_color[0];
  EmitVertex();

  gl_Position = vec4(v_pos[1].xy - offset * v_pos[1].w, v_pos[1].z, v_pos[1].w);
  g_color = v_color[1];
  EmitVertex();

  EndPrimitive();
}
