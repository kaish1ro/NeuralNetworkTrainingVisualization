#pragma once

#include "core/model/SequentialModel.h"
#include "app/panels/ControlPanel.h"

#include <filesystem>

namespace NN {

    class ModelSerializer {
    public:
        static void Save(const Models::SequentialModel& model,
            const App::TrainingParams& params,
            const std::filesystem::path& path);

        static std::pair<std::unique_ptr<Models::SequentialModel>, App::TrainingParams>
            Load(const std::filesystem::path& path);
    };

} // namespace NN