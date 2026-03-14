#include "Renderer.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>

Renderer::Renderer(VulkanContext& context, const GravSolver& solver)
	: vkContext(context), gravSolver(solver)
{
	createGraphicsPipeline();
}

void Renderer::createGraphicsPipeline() {
    std::vector <char> vert = readFile("shaders/vert.spv");
    std::vector <char> frag = readFile("shaders/frag.spv");

    vk::ShaderModuleCreateInfo vertInfo({}, vert.size(), reinterpret_cast<const uint32_t*>(vert.data()));
    vk::raii::ShaderModule vertModule(vkContext.getDevice(), vertInfo);

    vk::ShaderModuleCreateInfo fragInfo({}, frag.size(), reinterpret_cast<const uint32_t*>(frag.data()));
    vk::raii::ShaderModule fragModule(vkContext.getDevice(), fragInfo);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, *vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, *fragModule, "main" }
    };

    vk::VertexInputBindingDescription bindingDescription{
        0,                                  //binding
        sizeof(ParticleData),               //stride
        vk::VertexInputRate::eVertex        //inputRate
    };

    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
        vk::VertexInputAttributeDescription{
            0,                                          //location
            0,                                          //binding
            vk::Format::eR32G32B32A32Sfloat,            //format
            0                                           //offset
        },
        vk::VertexInputAttributeDescription{
            1,                                          //location
            0,                                          //binding
            vk::Format::eR32G32B32A32Sfloat,            //format
            offsetof(ParticleData, velocity)            //offset
        }
    };
    vk::PipelineVertexInputStateCreateInfo   vertexInputInfo{
        {},                                                     // flags
        1,                                                      // vertexBindingDescriptionCount
        & bindingDescription,                                    // pVertexBindingDescriptions
        static_cast<uint32_t>(attributeDescriptions.size()),    // vertexAttributeDescriptionCount
        attributeDescriptions.data()                            // pVertexAttributeDescriptions
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        {},                                 // flags
        vk::PrimitiveTopology::ePointList,  // topology
        vk::False                           // primitiveRestartEnable
    };

    vk::PipelineViewportStateCreateInfo viewportState{
        {},         // flags
        1,          // viewportCount
        nullptr,    // pViewports 
        1,          // scissorCount
        nullptr     // pScissors
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {},                             // flags
        vk::False,                      // depthClampEnable
        vk::False,                      // rasterizerDiscardEnable
        vk::PolygonMode::eFill,         // polygonMode
        vk::CullModeFlagBits::eNone,    // cullMode
        vk::FrontFace::eClockwise,      // frontFace
        vk::False,                      // depthBiasEnable
        0.0f,                           // depthBiasConstantFactor 
        0.0f,                           // depthBiasClamp 
        0.0f,                           // depthBiasSlopeFactor 
        1.0f                            // lineWidth
    );

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {},                                 //flags
        vk::SampleCountFlagBits::e1,        //rasterizationSamples
        vk::False                           //sampleShadingEnable
    );


    vk::PipelineColorBlendAttachmentState colorBlendAttachment(
        vk::True,                                   //blendEnable
        vk::BlendFactor::eOne,                      //srcColorBlendFactor
        vk::BlendFactor::eOne,                      //dstColorBlendFactor
        vk::BlendOp::eAdd,                          //colorBlendOp
        vk::BlendFactor::eOneMinusSrcAlpha,         //srcAlphaBlendFactor
        vk::BlendFactor::eZero,                     //dstAlphaBlendFactor
        vk::BlendOp::eAdd,                          //alphaBlendOp
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA   //colorWriteMask
    );

    vk::PipelineColorBlendStateCreateInfo colorBlending(
        {},                         //flags
        vk::False,                  //logicOpEnable
        vk::LogicOp::eCopy,         //logicOp
        1,                          //attachmentCount
        &colorBlendAttachment       //pAttachments
    );

    vk::DynamicState dynamicStates[] = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState(
        {},                     //flags
        2,                      //dynamicStateCount
        dynamicStates           //pDynamicStates
    );

    vk::PushConstantRange pushConstantRange(
        vk::ShaderStageFlagBits::eVertex,           // stageFlags
        0,                                          // offset
        sizeof(GraphicsPushConstants)               // size (8 bytes total)
    );

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
        {},                     // flags
        0,                      // setLayoutCount
        nullptr,                // pSetLayouts
        1,                      // pushConstantRangeCount
        & pushConstantRange      // pPushConstantRanges
    );
    pipelineLayout = vk::raii::PipelineLayout(vkContext.getDevice(), pipelineLayoutInfo);

    vk::Format colorFormat = vkContext.getSwapChainSurfaceFormat().format;
    vk::PipelineRenderingCreateInfo renderingCreateInfo(
        0,
        1,
        &colorFormat,
        vk::Format::eUndefined,
        vk::Format::eUndefined
    );

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {},
        2,
        shaderStages,
        &vertexInputInfo,
        &inputAssembly,
        nullptr,
        &viewportState,
        &rasterizer,
        &multisampling,
        nullptr,
        &colorBlending,
        &dynamicState,
        *pipelineLayout
    );

    pipelineInfo.pNext = &renderingCreateInfo;

    graphicsPipeline = vk::raii::Pipeline(vkContext.getDevice(), nullptr, pipelineInfo);
}

void Renderer::recordDrawCommands(vk::raii::CommandBuffer& cmdBuffer, uint32_t currentFrame, const glm::mat4& cameraViewProj) {
    vk::Extent2D swapChainExtent = vkContext.getSwapChainExtent();

    cmdBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    cmdBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

    GraphicsPushConstants pushData{ cameraViewProj };
    cmdBuffer.pushConstants<GraphicsPushConstants>(
        *pipelineLayout,
        vk::ShaderStageFlagBits::eVertex,
        0,
        pushData
    );

    // 4. Bind the Compute Shader's output as the Vertex Buffer
    vk::Buffer vertexBuffer = gravSolver.getOutputBuffer();
    vk::DeviceSize offset = 0;
    cmdBuffer.bindVertexBuffers(0, { vertexBuffer }, { offset });

    // 5. Draw all the particles!
    cmdBuffer.draw(gravSolver.getParticleCount(), 1, 0, 0);
}

std::vector<char> Renderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
}