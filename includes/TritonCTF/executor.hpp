#ifndef TRITONCTF_EXECUTOR_HPP
#define TRITONCTF_EXECUTOR_HPP

#include <memory>

#include <LIEF/LIEF.hpp>
#include <triton/context.hpp>

using namespace triton;
using namespace triton::arch;

namespace tritonctf {

enum class CallbackDirective {
  /** Fallthrough the rest of the hooks. */
  kFallthrough = 0,
  /** Don't execute the rest of the hooks. */
  kBreak,
};

enum class CallbackClass {
  /** Trigger the user-defined callback between the instruction decoding
   and the emulation.
   If the user-defined callback changes the instruction pointer,
   it is responsible for decoding the new instruction to execute. */
  kPreInstruction = 0,

  /** Trigger the user-defined callback after the instruction emulation. */
  kPostInstruction,

  /** Trigger the user-defined callback whenever the instruction pointer
     reaches the user-provided address. */
  kAddress,
};

class Executor;
using CallbackFn = CallbackDirective (*)(Executor *);
using InstructionCallback = CallbackDirective (*)(Executor *,
                                                  Instruction &inst);

class Executor {
private:
  /** Initialize the concrete state with the sections of the provided PE */
  static bool LoadPE(LIEF::PE::Binary *pe, Context *context);
  /** Initialize the concrete state with the segments of the provided PE */
  static bool LoadELF(LIEF::ELF::Binary *elf, Context *context);

  /** Symbolic, concrete and tainted state */
  std::unique_ptr<Context> context_;

  /** Pre instruction callback */
  std::vector<InstructionCallback> pre_instruction_cb_;
  /** Post instruction callback */
  std::vector<InstructionCallback> post_instruction_cb_;
  /** Hooks by address */
  std::map<uint64_t, std::vector<CallbackFn>> address_hooks_;

  /** The main binary */
  std::shared_ptr<LIEF::Binary> binary_;

  /** Is the executor correctly initialized ? */
  bool is_valid_;

  /** Is the emulator running ? */
  bool is_running_;

  inline void TriggerPreInstructionCb(Instruction &inst) {
    for (InstructionCallback &callback : pre_instruction_cb_) {
      callback(this, inst);
    }
  }

  inline void TriggerPostInstructionCb(Instruction &inst) {
    for (InstructionCallback &callback : post_instruction_cb_) {
      callback(this, inst);
    }
  }

  /** Call hooks for the given address */
  inline void TriggerAddressHooks(uint64_t address) {
    std::vector<CallbackFn> &address_hooks = address_hooks_[address];

    for (CallbackFn &hook : address_hooks) {
      CallbackDirective result = hook(this);

      if (CallbackDirective::kBreak == result)
        break;
    }
  }

  inline Instruction FetchInstruction(uint64_t address) {
    Instruction instruction;
    std::vector<uint8_t> opcodes =
        context_->getConcreteMemoryAreaValue(address, 16);

    instruction.setOpcode(opcodes.data(), 16);
    context_->disassembly(instruction);

    return instruction;
  }

  inline void DumpReg(const Register &reg, const char *name) {
    auto value = context_->getConcreteRegisterValue(reg);
    auto fill_size = 2 * reg.getSize();
    std::cout << std::setw(5) << std::setfill(' ') << name << " " << std::hex
              << std::left << std::setfill('0') << std::setw(fill_size)
              << value;

    if (context_->isRegisterTainted(reg))
      std::cout << " [T]";

    if (context_->isRegisterSymbolized(reg))
      std::cout << " [S]";

    std::cout << "\n";
  }

public:
  exception_e Emulate(std::vector<uint64_t> stop_addresses = {0});
  void DumpRegisters();
  void DumpStack();
  void Dump();

  Executor(std::shared_ptr<LIEF::Binary> bin, uint64_t default_ip,
           uint64_t default_sp);

  inline void EmulateReturn(uint64_t offset = 0) {
    CpuInterface *cpu = context_->getCpuInstance();
    const Register &stack_pointer_reg = cpu->getStackPointer();
    const Register &instruction_pointer_reg = cpu->getProgramCounter();
    uint32_t gpr_size = context_->getGprSize();

    uint64_t stack_pointer = static_cast<uint64_t>(
        context_->getConcreteRegisterValue(stack_pointer_reg));
    uint64_t return_address =
        static_cast<uint64_t>(context_->getConcreteMemoryValue(
            MemoryAccess(stack_pointer + offset, gpr_size)));

    context_->setConcreteRegisterValue(stack_pointer_reg,
                                       stack_pointer - gpr_size);
    context_->setConcreteRegisterValue(instruction_pointer_reg, return_address);
  }

  inline Context *GetContext() { return context_.get(); }

  inline bool IsValid() { return is_valid_; }

  inline bool IsRunning() { return is_running_; }

  inline void Terminate() { is_running_ = false; }

  inline void PushCallback(CallbackClass type, void *callback,
                           uint64_t address = 0) {
    switch (type) {
    case CallbackClass::kPreInstruction:
      pre_instruction_cb_.push_back(
          reinterpret_cast<InstructionCallback>(callback));
      break;
    case CallbackClass::kPostInstruction:
      post_instruction_cb_.push_back(
          reinterpret_cast<InstructionCallback>(callback));
      break;
    case CallbackClass::kAddress:
      address_hooks_[address].push_back(reinterpret_cast<CallbackFn>(callback));
      break;
    }
  }

  inline void PopCallback(CallbackClass type, uint64_t address = 0) {
    switch (type) {
    case CallbackClass::kPreInstruction:
      pre_instruction_cb_.pop_back();
      break;
    case CallbackClass::kPostInstruction:
      post_instruction_cb_.pop_back();
    case CallbackClass::kAddress:
      address_hooks_[address].pop_back();
      break;
    }
  }
};

} // namespace tritonctf

#endif
