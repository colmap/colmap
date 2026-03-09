#version 150

uniform bool u_wireframe;

in vec3 g_position;
in vec3 g_normal;
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

  if (u_wireframe) {
    float edge = min(g_barycentric.x, min(g_barycentric.y, g_barycentric.z));
    float threshold = fwidth(edge) * 1.5;
    if (edge > threshold) {
      discard;
    }
  }

  f_color = vec4(g_color * intensity, 1.0);
}
