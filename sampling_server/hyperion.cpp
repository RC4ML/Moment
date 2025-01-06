#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "server.h"  // 包含 Server 类和工厂函数的定义

namespace py = pybind11;

PYBIND11_MODULE(hyperion, m) {
    m.doc() = "Hyperion Python bindings";

    // 绑定 Server 接口
    py::class_<Server>(m, "Server")
        .def("initialize", &Server::Initialize, "Initialize the server",
             py::arg("gpu_number") = 1, py::arg("fanout") = std::vector<int>{})
        .def("presc", &Server::PreSc, "Set cache aggregate mode", py::arg("mode") = 0)
        .def("run", &Server::Run, "Run the server")
        .def("finalize", &Server::Finalize, "Finalize the server");

    // 工厂函数绑定
    m.def("NewGPUServer", &NewGPUServer, "Create a new GPU Server instance");
}
