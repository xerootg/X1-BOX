project('x87', 'cpp',
        default_options: ['warning_level=1', 'cpp_std=c++17'])

x87_inc = include_directories('.')

x87_cpp_args = [
  # Force-include <array>: upstream x87fp80trans.cpp uses std::array<>
  # without #include <array>; works on libstdc++ via transitive header
  # pulls but fails on libc++ (Android NDK).
  '-include', 'array',
  '-Wno-unused-function',
  '-Wno-unused-variable',
  '-Wno-unused-parameter',
  '-Wno-sign-compare',
  '-Wno-implicit-fallthrough',
  '-Wno-missing-field-initializers',
  '-Wno-error',
]

libx87 = static_library(
  'x87',
  files(
    'x87fp64trans.cpp',
    'x87fp80.cpp',
    'x87fp80trans.cpp',
  ),
  include_directories: x87_inc,
  cpp_args: x87_cpp_args,
)

x87_dep = declare_dependency(
  link_with: libx87,
  include_directories: x87_inc,
)
