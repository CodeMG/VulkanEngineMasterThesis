#include "workbenches_util.hpp"

#include <workbenches/data_workbench.hpp>
#include <workbenches/parallel_coordinates_workbench.hpp>
#include <workbenches/volume_workbench.hpp>
#include <structures/laod_behaviour.hpp>

namespace util{
namespace workbench{
void setup_default_workbenches(){
    // register all available workbenches
    auto data_wb = std::make_unique<workbenches::data_workbench>("Data workbench");
    data_wb->active = true;
    globals::dataset_dependencies.push_back(data_wb.get());
    globals::primary_workbench = data_wb.get();
    globals::workbenches.emplace_back(std::move(data_wb));

    auto volume_wb = std::make_unique<workbenches::volume_workbench>("Volume workbench");
    volume_wb->active = true;
    globals::secondary_workbench =volume_wb.get();
    globals::workbenches.emplace_back(std::move(volume_wb));

    globals::load_behaviour.on_load.push_back({false, 1, {0, std::numeric_limits<size_t>::max()}, {"Parallel coordinates workbench"}});
}
}
}