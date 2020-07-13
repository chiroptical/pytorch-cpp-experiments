#include <pybind11/pybind11.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

namespace py = pybind11;

class Module {
  private:
    torch::jit::script::Module module;
  public:
    Module() {}

    Module(const char* filename) {
      this->module = torch::jit::load(filename);
    }

    at::Tensor predict(std::vector<torch::jit::IValue> &inputs) {
      return this->module.forward(inputs).toTensor();
    }
};

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: example <model.pt>\n";
    return -1;
  }

  Module module;
  try {
    module = Module(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  auto output = module.predict(inputs);
  std::cout << output.slice(1, 0, 1);

  return 0;
}
