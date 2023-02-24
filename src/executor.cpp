#include "TritonCTF/executor.hpp"

#include <iomanip>

#include <triton/archEnums.hpp>
#include <triton/cpuInterface.hpp>

using namespace triton;
using namespace triton::arch;

namespace tritonctf {

Executor::Executor(std::shared_ptr<LIEF::Binary> bin, uint64_t default_ip,
                   uint64_t default_sp)
    : binary_(bin), context_(std::make_unique<Context>()) {

  switch (binary_->header().architecture()) {
  case LIEF::ARCH_ARM:
    context_->setArchitecture(triton::arch::ARCH_ARM32);
    break;
  case LIEF::ARCH_ARM64:
    context_->setArchitecture(triton::arch::ARCH_AARCH64);
    break;
  case LIEF::ARCH_X86:
    if (binary_->header().is_32()) {
      context_->setArchitecture(triton::arch::ARCH_X86);
    } else {
      context_->setArchitecture(triton::arch::ARCH_X86_64);
    }
    break;
  default:
    is_valid_ = false;
    return;
  }

  switch (binary_->format()) {
  case LIEF::FORMAT_ELF:
    is_valid_ = LoadELF(dynamic_cast<LIEF::ELF::Binary *>(binary_.get()),
                        context_.get());
    break;
  case LIEF::FORMAT_PE:
    is_valid_ =
        LoadPE(dynamic_cast<LIEF::PE::Binary *>(binary_.get()), context_.get());
    break;
  case LIEF::FORMAT_MACHO:
  case LIEF::FORMAT_UNKNOWN:
    is_valid_ = false;
    return;
  }

  if (!is_valid_)
    return;

  CpuInterface *cpu = context_->getCpuInstance();
  context_->setConcreteRegisterValue(cpu->getStackPointer(), default_sp);
  context_->setConcreteRegisterValue(cpu->getProgramCounter(), default_ip);
}

bool Executor::LoadPE(LIEF::PE::Binary *pe, Context *context) {
  std::vector<uint8_t> section_data;
  uint64_t section_start;
  uint64_t section_size;

  if (!pe || !context)
    return false;

  section_data.reserve(0x2000);

  for (auto &segment : pe->sections()) {
    section_start = segment.virtual_address();
    section_size = segment.virtual_size();

    if (!section_size)
      continue;

    section_data.assign(segment.content().begin(), segment.content().end());

    context->setConcreteMemoryAreaValue(section_start, section_data);
  }

  return true;
}

bool Executor::LoadELF(LIEF::ELF::Binary *elf, Context *context) {
  std::vector<uint8_t> segment_data;
  uint64_t segment_start;
  uint64_t segment_size;

  if (!elf || !context)
    return false;

  segment_data.reserve(0x2000);

  for (auto &segment : elf->segments()) {
    segment_start = segment.virtual_address();
    segment_size = segment.virtual_size();

    if (!segment_size)
      continue;

    segment_data.assign(segment.content().begin(), segment.content().end());

    context->setConcreteMemoryAreaValue(segment_start, segment_data);
  }

  return true;
}

exception_e Executor::Emulate(std::vector<uint64_t> stop_addresses) {
  exception_e result = NO_FAULT;
  const Register &instruction_pointer_reg =
      context_->getCpuInstance()->getProgramCounter();
  uint64_t instruction_pointer = static_cast<uint64_t>(
      context_->getConcreteRegisterValue(instruction_pointer_reg));
  Instruction instruction;

  std::sort(stop_addresses.begin(), stop_addresses.end());

  while (is_running_) {
    TriggerAddressHooks(instruction_pointer);
    instruction_pointer = static_cast<uint64_t>(
        context_->getConcreteRegisterValue(instruction_pointer_reg));

    if (std::binary_search(stop_addresses.begin(), stop_addresses.begin(),
                           instruction_pointer))
      break;

    instruction = FetchInstruction(instruction_pointer);
    TriggerPreInstructionCb(instruction);

    result = context_->buildSemantics(instruction);
    if (NO_FAULT != result)
      return result;

    TriggerPostInstructionCb(instruction);

    instruction_pointer = static_cast<uint64_t>(
        context_->getConcreteRegisterValue(instruction_pointer_reg));
  }

  return result;
}

void Executor::DumpRegisters() {

  if (ARCH_X86_64 != context_->getArchitecture()) {
    std::cerr << "[-] Unsupported architecture dump." << std::endl;
    std::terminate();
  }

#define DUMP_X86REG(reg) DumpReg(context_->registers.x86_##reg, #reg)

  DUMP_X86REG(rax);
  DUMP_X86REG(rbx);
  DUMP_X86REG(rcx);
  DUMP_X86REG(rdx);
  DUMP_X86REG(rdi);
  DUMP_X86REG(rsi);
  DUMP_X86REG(r8);
  DUMP_X86REG(r9);
  DUMP_X86REG(r10);
  DUMP_X86REG(r11);
  DUMP_X86REG(r12);
  DUMP_X86REG(r13);
  DUMP_X86REG(r14);
  DUMP_X86REG(r15);
  DUMP_X86REG(rbp);
  DUMP_X86REG(rsp);
  DUMP_X86REG(rip);

#undef DUMP_X86REG
}

void Executor::DumpStack() {
  CpuInterface *cpu = context_->getCpuInstance();
  uint32_t gpr_size = cpu->gprSize();
  const Register &sp_reg = cpu->getStackPointer();
  uint64_t sp = static_cast<uint64_t>(cpu->getConcreteRegisterValue(sp_reg));

  std::string line_status;
  line_status.reserve(100);

  for (size_t i = 0; i < 8; i++) {
    uint64_t stack_address = sp + i * gpr_size;
    uint64_t stack_value = static_cast<uint64_t>(
        cpu->getConcreteMemoryValue(MemoryAccess(stack_address, gpr_size)));

    std::cout << std::hex << std::setw(2) << std::setfill('0') << i;
    std::cout << ":";
    std::cout << std::setw(4) << std::setfill('0') << i * gpr_size << "|";

    line_status = "";
    if (context_->isMemoryTainted(stack_address))
      line_status += "[T] ";
    if (context_->isMemorySymbolized(stack_address))
      line_status += "[S] ";

    std::cout << std::setfill(' ') << std::setw(8) << line_status
              << std::setw(2 * gpr_size) << std::right << std::hex
              << std::showbase << stack_address << " <- " << stack_value
              << std::noshowbase;

    if (i < 7)
      std::cout << "\n";
  }
}

void Executor::Dump() {
  DumpRegisters();
  DumpStack();
}

} // namespace tritonctf
