#pragma once
#include <vulkanMemoryAllocator/include/vk_mem_alloc.h>
#include <structures/vk_context.hpp>
#include <util/memory_view.hpp>
#include <structures/buffer_info.hpp>

namespace util{
namespace vma{
inline void upload_data(memory_view<const uint8_t> data, structures::buffer_info dst){
    void* mapped;
    auto res = vmaMapMemory(globals::vk_context.allocator, dst.allocation, &mapped); util::check_vk_result(res);
    std::memcpy(mapped, data.data(), data.byte_size());
    vmaUnmapMemory(globals::vk_context.allocator, dst.allocation);
}
}
}