{ pkgs ? import /home/oposs/Documents/nixpkgs {} }:
with pkgs;

mkShell {
  buildInputs = [
    clang-tools
    ninja
    z3
    libtriton
    lief
    cmake
    capstone
    llvm
    boost
  ];
}
