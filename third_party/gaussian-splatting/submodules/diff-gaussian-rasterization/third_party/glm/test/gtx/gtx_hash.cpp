#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <unordered_map>

int test_compile()
{
	int Error = 0;

    // Vector types
    std::unordered_map<glm::vec1, int> map_vec1;
    Error += ++map_vec1[glm::vec1(0.0f)];
    std::unordered_map<glm::vec2, int> map_vec2;
    Error += ++map_vec2[glm::vec2(0.0f)];
    std::unordered_map<glm::vec3, int> map_vec3;
    Error += ++map_vec3[glm::vec3(0.0f)];
    std::unordered_map<glm::vec4, int> map_vec4;
    Error += ++map_vec4[glm::vec4(0.0f)];

    // Quaternion types
    std::unordered_map<glm::quat, int> map_quat;
    Error += ++map_quat[glm::quat(0.0f, glm::vec3(0.0f))];
    std::unordered_map<glm::dualquat, int> map_dualquat;
    Error += ++map_dualquat[glm::dualquat(glm::vec3(0.0f))];

    // Matrix types
    std::unordered_map<glm::mat2x2, int> map_mat2x2;
    Error += ++map_mat2x2[glm::mat2x2(0.0f)];
    std::unordered_map<glm::mat2x3, int> map_mat2x3;
    Error += ++map_mat2x3[glm::mat2x3(0.0f)];
    std::unordered_map<glm::mat2x4, int> map_mat2x4;
    Error += ++map_mat2x4[glm::mat2x4(0.0f)];
    std::unordered_map<glm::mat3x2, int> map_mat3x2;
    Error += ++map_mat3x2[glm::mat3x2(0.0f)];
    std::unordered_map<glm::mat3x3, int> map_mat3x3;
    Error += ++map_mat3x3[glm::mat3x3(0.0f)];
    std::unordered_map<glm::mat3x4, int> map_mat3x4;
    Error += ++map_mat3x4[glm::mat3x4(0.0f)];
    std::unordered_map<glm::mat4x2, int> map_mat4x2;
    Error += ++map_mat4x2[glm::mat4x2(0.0f)];
    std::unordered_map<glm::mat4x3, int> map_mat4x3;
    Error += ++map_mat4x3[glm::mat4x3(0.0f)];
    std::unordered_map<glm::mat4x4, int> map_mat4x4;
    Error += ++map_mat4x4[glm::mat4x4(0.0f)];

	return Error > 0 ? 0 : 1;
}

int main()
{
	int Error = 0;

	Error += test_compile();

	return Error;
}