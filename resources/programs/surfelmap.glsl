#version 330

#if defined VERTEX_SHADER

void main(void) {}

#elif defined GEOMETRY_SHADER
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform int width;
uniform int height;
uniform float conf_threshold;
uniform float radius = 0.003;

uniform sampler2D pointmap;
uniform sampler2D confs;
uniform sampler2D img;
uniform sampler2D label_texture; // New texture for labels

out vec3 normal;
out vec3 position;
out vec2 fragTexCoord;
out vec3 color;
out int fragLabelID; // Pass label ID to fragment shader

void main(void) {

  int y = gl_PrimitiveIDIn / int(width);
  int x = gl_PrimitiveIDIn - y * int(width);
  if (x < 1 || x >= int(width) - 1 || y < 1 || y >= int(height) - 1) {
    return;
  }

  float conf = texelFetch(confs, ivec2(x, y), 0).x;
  if (conf < conf_threshold) {
    return;
  }
  
  mat4 mv = m_camera * m_model;
  mat4 mvp = m_proj * mv;
  vec3 xyz00 = texelFetch(pointmap, ivec2(x, y), 0).xyz;
  vec3 xyz01 = texelFetch(pointmap, ivec2(x, y+1), 0).xyz;
  vec3 xyz10 = texelFetch(pointmap, ivec2(x+1, y), 0).xyz;
  vec3 color00 = texelFetch(img, ivec2(x, y), 0).xyz;
  int label_id = texelFetch(label_texture, ivec2(x,y), 0).r; // Fetch label ID


  vec3 right = xyz10 - xyz00;
  vec3 down = xyz01 - xyz00;
  vec3 N = normalize(cross(down, right));
  vec3 tangent = normalize(cross(N, vec3(0, -1, 0))); // assume y up
  vec3 bitangent = cross(N, tangent);

  vec3 p1 = xyz00 + (-tangent - bitangent) * radius;
  vec3 p2 = xyz00 + (tangent - bitangent) * radius;
  vec3 p3 = xyz00 + (-tangent + bitangent) * radius;
  vec3 p4 = xyz00 + (tangent + bitangent) * radius;


  normal = mat3(mv) * N;
  position = (mv * vec4(p1, 1.0)).xyz;
  gl_Position = mvp * vec4(p1, 1.0);
  fragTexCoord = vec2(-1.0, -1.0);
  color = color00;
  fragLabelID = label_id;
  EmitVertex();

  normal = mat3(mv) * N;
  position = (mv * vec4(p2, 1.0)).xyz;
  gl_Position = mvp * vec4(p2, 1.0);
  fragTexCoord = vec2(1.0, -1.0);
  color = color00;
  fragLabelID = label_id;
  EmitVertex();

  normal = mat3(mv) * N;
  position = (mv * vec4(p3, 1.0)).xyz;
  gl_Position = mvp * vec4(p3, 1.0);
  fragTexCoord = vec2(-1.0, 1.0);
  color = color00;
  fragLabelID = label_id;
  EmitVertex();

  normal = mat3(mv) * N;
  position = (mv * vec4(p4, 1.0)).xyz;
  gl_Position = mvp * vec4(p4, 1.0);
  fragTexCoord = vec2(1.0, 1.0);
  color = color00;
  fragLabelID = label_id;
  EmitVertex();
  
  EndPrimitive();
}

#elif defined FRAGMENT_SHADER
in vec3 normal;
in vec3 position;
in vec2 fragTexCoord;
in vec3 color;
in int fragLabelID; // Received from Geometry shader

out vec4 out_color;
uniform vec3 lightpos = vec3(0.1, 0.1, 0);
uniform vec3 phong = vec3(0.3, 0.5, 0.4);
uniform float spec = 32;
uniform vec3 base_color = vec3(1, 1, 1);
uniform bool use_img_texture; // Renamed from use_img
uniform bool show_normal;
uniform bool u_show_labels; // To toggle label display

// Simple hardcoded color map for labels
vec3 get_label_color(int label_id) {
    if (label_id == 0) return vec3(0.5, 0.5, 0.5); // Grey for label 0 (e.g. background)
    if (label_id == 1) return vec3(1.0, 0.0, 0.0); // Red for label 1
    if (label_id == 2) return vec3(0.0, 1.0, 0.0); // Green for label 2
    if (label_id == 3) return vec3(0.0, 0.0, 1.0); // Blue for label 3
    if (label_id == 4) return vec3(1.0, 1.0, 0.0); // Yellow for label 4
    if (label_id == 5) return vec3(1.0, 0.0, 1.0); // Magenta for label 5
    if (label_id == 6) return vec3(0.0, 1.0, 1.0); // Cyan for label 6
    if (label_id == 7) return vec3(1.0, 0.5, 0.0); // Orange for label 7
    if (label_id == 8) return vec3(0.5, 0.0, 1.0); // Purple for label 8
    if (label_id == 9) return vec3(0.0, 0.5, 1.0); // Light Blue for label 9
    return vec3(0.2, 0.2, 0.2); // Default color for other/unknown labels
}

void main() {
  float dist = length(fragTexCoord);
  // Discard fragments outside the radius to create a round disk
  if (dist > 1.0) {
      discard;
  }

  vec3 final_frag_color;
  if (u_show_labels && fragLabelID >= 0) { // Assuming -1 or some value for no label
      final_frag_color = get_label_color(fragLabelID);
  } else {
      final_frag_color = use_img_texture ? color : base_color;
  }

  // Lighting calculations (can be simplified if labels override lighting)
  float kA = use_img_texture ? phong[0] : 0.1;
  float kD = use_img_texture ? phong[1] : 0.2;
  float kS = use_img_texture ? phong[2] : 1.3;
  float spec_power = use_img_texture ? spec : 2;
  vec3 specular_color = vec3(1.0, 1.0, 1.0);

  vec3 L = normalize(lightpos - position);
  vec3 N = normalize(normal);
  float lambertian = max(dot(N, L), 0.0);
  float specular = 0;
  if (lambertian > 0.0) {
    vec3 R = 2 * (L * N) * N - L; // Reflection vector
    vec3 V = normalize(-position); // View vector
    specular = pow(max(dot(R, V), 0.0), spec_power);
  }

  vec3 lit_color;
  if (u_show_labels && fragLabelID >= 0) {
    // Apply minimal lighting to label colors, or just use them directly
    // For simplicity, let's use label color with some ambient and diffuse, no specular
    lit_color = final_frag_color * (phong[0] + lambertian * phong[1] * 0.5); // Modulate label color by simple lighting
  } else {
    // Full phong shading for non-label or when labels are off
    lit_color = final_frag_color * (kA + lambertian * kD ) + specular_color * kS * specular;
  }

  out_color = vec4(lit_color, 1.0);

  if (show_normal) {
    N = vec3(N.x, -N.y, -N.z); // Adjust normal display convention if needed
    out_color = vec4(-N * 0.5 + 0.5, 1.0);
  }
}
#endif