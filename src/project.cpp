#include "TritonCTF/project.hpp"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <triton/archEnums.hpp>

#include <LIEF/Abstract/Parser.hpp>
#include <LIEF/Abstract/enums.hpp>
#include <LIEF/ELF/Binary.hpp>
#include <LIEF/Visitor.hpp>

#include "TritonCTF/executor.hpp"

namespace tritonctf {

Project::Project(const char *filename) {
  binary_ = LIEF::Parser::parse(filename);
  if (binary_->header().is_32()) {
    default_stack_base_ = 0x600000;
  } else {
    default_stack_base_ = 0x7fffffff0000;
  }
  entrypoint_ = binary_->entrypoint();
}

std::unique_ptr<Executor> Project::CreateExecutor() {
  // std::unique_ptr<Executor> executor =
  //     std::make_unique<Executor>(binary_, entrypoint_, default_stack_base_);
  Executor *exec = new Executor(binary_, entrypoint_, default_stack_base_);
  std::unique_ptr<Executor> executor;
  executor.reset(exec);

  if (!executor->IsValid())
    return nullptr;

  return executor;
}

} // namespace tritonctf
