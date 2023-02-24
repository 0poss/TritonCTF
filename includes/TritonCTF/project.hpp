#ifndef TRITONCTF_PROJECT_HPP
#define TRITONCTF_PROJECT_HPP

#include <cstdint>
#include <map>

#include <LIEF/LIEF.hpp>
#include <triton/context.hpp>
#include <triton/tritonTypes.hpp>

#include "TritonCTF/executor.hpp"

namespace tritonctf {

class Project {
private:
  uint64_t default_stack_base_;
  uint64_t entrypoint_;
  std::shared_ptr<LIEF::Binary> binary_;

private:
  void LoadELF(LIEF::ELF::Binary *elf_binary, triton::Context *ctx);
  void LoadPE(LIEF::PE::Binary *pe_binary, triton::Context *ctx);

public:
  Project(const char *filename);

  inline void SetStackBase(uint64_t address) { default_stack_base_ = address; }

  inline uint64_t GetStackBase() { return default_stack_base_; }

  inline void SetEntrypoint(uint64_t address) { entrypoint_ = address; }

  inline uint64_t GetEntrypoint() { return entrypoint_; }

  inline LIEF::Binary *GetBinary() { return binary_.get(); }

  std::unique_ptr<Executor> CreateExecutor();
};

} // namespace tritonctf

#endif
