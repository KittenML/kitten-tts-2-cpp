#include "engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
static py::array_t<float> array(const std::vector<float> & samples) {
    py::array_t<float> result(samples.size());
    std::copy(samples.begin(), samples.end(), result.mutable_data());
    return result;
}
PYBIND11_MODULE(_native, m) {
    py::class_<kitten::engine>(m, "Engine")
        .def(py::init([](const std::map<std::string, std::string> & options) {
            py::gil_scoped_release release;
            return std::make_unique<kitten::engine>(options);
        }))
        .def("metadata", [](kitten::engine & self) { return self.metadata().dump(); })
        .def("normalize", &kitten::engine::normalize, py::call_guard<py::gil_scoped_release>())
        .def("generate", [](kitten::engine & self, const std::map<std::string, std::string> & options, py::object callback) {
            kitten::generation_result result;
            {
                py::gil_scoped_release release;
                result = self.generate(options, callback.is_none() ? std::function<bool(const std::vector<float> &)>{} :
                    [&callback](const std::vector<float> & samples) {
                        py::gil_scoped_acquire acquire;
                        return callback(array(samples)).cast<bool>();
                    });
            }
            return py::make_tuple(array(result.audio), result.report.dump());
        }, py::arg("options"), py::arg("callback") = py::none());
}
