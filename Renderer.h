#pragma once
#include "VulkanContext.h"
#include "GravSolver.h"
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <glm/glm.hpp>

// Push constant for the 3D camera
struct GraphicsPushConstants {
    glm::mat4 mvp;
};

class Renderer {
public:
    Renderer(VulkanContext& context, const GravSolver& solver);
    ~Renderer() = default;

    void recordDrawCommands(vk::raii::CommandBuffer& cmdBuffer, uint32_t currentFrame, const glm::mat4& cameraViewProj);

private:
    VulkanContext& vkContext;
    const GravSolver& gravSolver;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    void createGraphicsPipeline();

    static std::vector<char> readFile(const std::string& filename);
};