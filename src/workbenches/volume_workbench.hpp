#pragma once
#include <structures/workbench_base.hpp>
#include <util/memory_view.hpp>
#include <imgui/imgui.h>
#include <pipelines/volume_renderer.hpp>
#include <structures/enum_names.hpp>

namespace workbenches {

    class volume_workbench : public structures::workbench {
    public:
        struct attribute_order_info {
            uint32_t    attribut_index{};
            bool        active{ true };

            bool operator==(const attribute_order_info& o) const { return attribut_index == o.attribut_index && active == o.active; }
        };
    private:

        // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
        std::vector<std::unique_ptr<structures::median_type>>    _storage_median_type;
       

        void _update_plot_image();
    public:
        float cameraXPos;
        float cameraYPos;
        float cameraZPos;
        float cameraXAngle;
        float cameraYAngle;
        float cameraZAngle;
        int frame;
        
        

        struct plot_data {
            uint32_t                width{ 2000 };
            uint32_t                height{ 480 };
            structures::image_info  image{};
            VkImageView             image_view{};
            VkSampleCountFlagBits   image_samples{ VK_SAMPLE_COUNT_1_BIT };
            VkFormat                image_format{ VK_FORMAT_R16G16B16A16_UNORM };
            ImTextureID             image_descriptor{}; // called descriptor as internally it is a descriptor
        };
        enum class render_strategy {
            all,
            batched,
            COUNT
        };
        const structures::enum_names<render_strategy> render_strategy_names{
            "all",
            "batched",
        };
        const std::array<VkFormat, 4>                           available_formats{ VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT };

        structures::alpha_mapping_type                          alpha_mapping_typ{};
        structures::change_tracker<structures::volume_renderer::render_type> render_type{};
        structures::change_tracker<plot_data>                   plot_data{};
        structures::change_tracker<std::vector<structures::attribute>> attributes{};
        structures::change_tracker<std::vector<attribute_order_info>> attributes_order_info{};
        render_strategy                                         render_strategy{};

        volume_workbench(const std::string_view id);

        void calculateTF(structures::image_info image_info, size_t width, size_t height);
        void render_plot();
        // overriden methods
        void show() override;
    };

}