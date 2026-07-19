#version 150

uniform bool u_wireframe;
uniform bool u_has_texture;
uniform bool u_color;
uniform sampler2D u_texture;

in vec3 g_position;
in vec3 g_normal;
in vec2 g_uv;
in vec3 g_color;
in vec3 g_barycentric;

out vec4 f_color;

void main(void) {
  // Two-sided lighting: flip normal if facing away from camera.
  vec3 normal = normalize(g_normal);
  if (dot(normal, -normalize(g_position)) < 0.0) {
    normal = -normal;
  }

  // Headlamp directional light (from camera toward scene).
  vec3 light_dir = normalize(-g_position);
  float diffuse = max(dot(normal, light_dir), 0.0);

  // Lambert diffuse + ambient.
  float intensity = 0.2 + 0.8 * diffuse;

  // Choose base color from texture, vertex color, or default gray.
  vec3 base_color = u_has_texture ? texture(u_texture, g_uv).rgb
                  : u_color       ? g_color
                  :                 vec3(0.784, 0.784, 0.784);

  if (u_wireframe) {
    float edge = min(g_barycentric.x, min(g_barycentric.y, g_barycentric.z));
    float threshold = fwidth(edge) * 1.5;
    if (edge > threshold) {
      discard;
    }
  }

  f_color = vec4(base_color * intensity, 1.0);
}
