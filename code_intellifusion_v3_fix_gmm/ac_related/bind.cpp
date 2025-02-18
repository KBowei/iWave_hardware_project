#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "arithmetic_coding.hpp"

namespace py = pybind11;


namespace py = pybind11;

PYBIND11_MODULE(arithmetic_coding, m) {
    // 绑定 ArithmeticPixelDecoder 类
    py::class_<ArithmeticPixelDecoder>(m, "ArithmeticPixelDecoder")
        .def(py::init<int, const char*, size_t, int16_t, int16_t, int16_t>())
        .def("read", &ArithmeticPixelDecoder::read);

    // 绑定 coding 函数
    m.def("coding", &coding, "编码函数", py::arg("gmms"), py::arg("datas"), py::arg("gmm_scales"));
}
