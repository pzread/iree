
add_executable("fuzz" "")
target_sources("fuzz"
  PRIVATE
  fuzz.cc
)

target_include_directories("fuzz" PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_options("fuzz" PRIVATE "-fsanitize=fuzzer")

target_link_libraries("fuzz"
  iree_base_base
  iree_base_tracing
  iree_hal_hal
  iree_runtime_runtime
  iree_base_internal_flatcc_building
  iree_base_internal_flatcc_debugging
  iree_base_internal_flatcc_parsing
)
