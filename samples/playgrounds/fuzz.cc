#include <stdlib.h>

#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/base/internal/flatcc/debugging.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

static iree_vm_module_t* hal_module = NULL;

iree_vm_instance_t* DoInitialization() {
  IREE_CHECK_OK(iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default()));

  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance));
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance));

  iree_hal_driver_t* driver = NULL;
  IREE_CHECK_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_string_view_literal("local-task"), iree_allocator_system(),
      &driver));

  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &device));

  IREE_CHECK_OK(iree_hal_module_create(instance, device,
                                       IREE_HAL_MODULE_FLAG_NONE,
                                       iree_allocator_system(), &hal_module));
  iree_hal_driver_release(driver);

  return instance;
}

int run(const uint8_t* vmfb_buf, size_t vmfb_len, char* entry_function_name) {
#if 0
  FILE* f = fopen(argv[1], "r");
  fseek(f, 0, SEEK_END);
  ssize_t vmfb_len = ftell(f);
  assert(vmfb_len > 0);
  fseek(f, 0, SEEK_SET);
  uint8_t* vmfb_buf = (uint8_t*)malloc(vmfb_len);
  size_t retn = fread(vmfb_buf, vmfb_len, 1, f);
  assert(retn == 1);
  fclose(f);

  iree_vm_module_t* bytecode_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, iree_const_byte_span_t{vmfb_buf, (size_t)vmfb_len},
      /*flatbuffer_allocator=*/iree_allocator_null(),
      /*allocator=*/iree_allocator_system(), &bytecode_module));

  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[2] = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      iree_allocator_system(), &context));
  // References to the modules can be released now.
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  iree_vm_function_t entry_function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(entry_function_name), &entry_function));

  size_t arg0_len = 1 * 320 * 320 * 3 * sizeof(float);
  uint8_t* arg0_buf = (uint8_t*)malloc(arg0_len);
  iree_hal_dim_t arg0_shape[] = {1, 320, 320, 3};
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device), IREE_ARRAYSIZE(arg0_shape), arg0_shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(arg0_buf, arg0_len), &arg0_buffer_view));

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(
      /*element_type=*/NULL,
      /*capacity=*/1, iree_allocator_system(), &inputs));
  iree_vm_ref_t arg0_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg0_buffer_view);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &arg0_buffer_view_ref));

  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(
      /*element_type=*/NULL,
      /*capacity=*/1, iree_allocator_system(), &outputs));

  IREE_CHECK_OK(iree_vm_invoke(
      context, entry_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  free(arg0_buf);

#endif
  return 0;
}

class DataGen {
 public:
  DataGen(const uint8_t* _data, size_t _size) : data(_data), size(_size) {}

  template <typename T>
  std::unique_ptr<T[]> genData(size_t len) {
    auto ret = std::unique_ptr<T[]>(new T[len]);
    uint8_t* buf = (uint8_t*)&ret.get()[0];
    for (size_t i = 0; i < len * sizeof(T); ++i) {
      if (off < size) {
        buf[i] = data[off];
        off += 1;
      } else {
        buf[i] = 0;
      }
    }
    return std::move(ret);
  }

  size_t genLen(size_t start = 0, size_t end = 10) {
    size_t value = genData<uint16_t>(1)[0];
    return (value % (end - start)) + start;
  }

  std::unique_ptr<uint8_t[]> genBitData(size_t len) {
    return genData<uint8_t>((len + 7) / 8);
  }

 private:
  const uint8_t* data;
  size_t size;
  size_t off = 0;
};

iree_vm_AttrDef_ref_t buildAttrDef(flatcc_builder_t* B, DataGen& G) {
  size_t keyLen = G.genLen();
  auto keyBuf = G.genData<char>(keyLen);
  auto key = flatbuffers_string_create(B, keyBuf.get(), keyLen);
  size_t valueLen = G.genLen();
  auto valueBuf = G.genData<char>(valueLen);
  auto value = flatbuffers_string_create(B, valueBuf.get(), valueLen);
  return iree_vm_AttrDef_create(B, key, value);
}

iree_vm_TypeDef_ref_t buildTypeDef(flatcc_builder_t* B, DataGen& G) {
  size_t valueLen = G.genLen();
  auto valueBuf = G.genData<char>(valueLen);
  auto value = flatbuffers_string_create(B, valueBuf.get(), valueLen);
  return iree_vm_TypeDef_create(B, value);
}

iree_vm_TypeDef_ref_t buildFunctionSignatureDef(flatcc_builder_t* B,
                                                DataGen& G) {
  size_t convLen = G.genLen();
  auto convBuf = G.genData<char>(convLen);
  auto conv = flatbuffers_string_create(B, convBuf.get(), convLen);

  std::vector<iree_vm_AttrDef_ref_t> attrs;
  size_t attrsLen = G.genLen();
  for (size_t i = 0; i < attrsLen; ++i) {
    attrs.push_back(buildAttrDef(B, G));
  }
  auto attr_vec = iree_vm_AttrDef_vec_create(B, attrs.data(), attrs.size());
  return iree_vm_FunctionSignatureDef_create(B, conv, attr_vec);
}

iree_vm_ModuleDependencyFlagBits_enum_t buildModuleDependencyFlagBits(
    flatcc_builder_t* B, DataGen& G) {
  auto values = G.genBitData(2);
  iree_vm_ModuleDependencyFlagBits_enum_t flags = values.get()[0];
  return flags;
}

iree_vm_ModuleDependencyDef_ref_t buildModuleDependencyDef(flatcc_builder_t* B,
                                                           DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  uint32_t minimumVersion = G.genData<uint32_t>(1)[0];
  iree_vm_ModuleDependencyFlagBits_enum_t flags =
      buildModuleDependencyFlagBits(B, G);
  return iree_vm_ModuleDependencyDef_create(B, name, minimumVersion, flags);
}

iree_vm_ImportFlagBits_enum_t buildImportFlagBits(flatcc_builder_t* B,
                                                  DataGen& G) {
  auto values = G.genBitData(2);
  iree_vm_ImportFlagBits_enum_t flags = values.get()[0];
  return flags;
}

iree_vm_ImportFunctionDef_ref_t buildImportFunctionDef(flatcc_builder_t* B,
                                                       DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  auto sign = buildFunctionSignatureDef(B, G);
  auto flags = buildImportFlagBits(B, G);
  return iree_vm_ImportFunctionDef_create(B, name, sign, flags);
}

iree_vm_ExportFunctionDef_ref_t buildExportFunctionDef(flatcc_builder_t* B,
                                                       DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  auto sign = buildFunctionSignatureDef(B, G);
  auto ordinal = G.genData<int32_t>(1)[0];
  return iree_vm_ExportFunctionDef_create(B, name, sign, ordinal);
}

iree_vm_InternalFunctionDef_ref_t buildInternalFunctionDef(flatcc_builder_t* B,
                                                           DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  auto sign = buildFunctionSignatureDef(B, G);
  return iree_vm_InternalFunctionDef_create(B, name, sign);
}

iree_vm_RodataSegmentDef_ref_t buildRodataSegmentDef(flatcc_builder_t* B,
                                                     DataGen& G) {
  iree_vm_RodataSegmentDef_start(B);
  size_t dataLen = G.genLen();
  auto dataBuf = G.genData<uint8_t>(dataLen);
  auto data = flatbuffers_uint8_vec_create(B, dataBuf.get(), dataLen);
  iree_vm_RodataSegmentDef_embedded_data_add(B, data);
  uint64_t extOff = G.genData<uint64_t>(1)[0];
  iree_vm_RodataSegmentDef_external_data_offset_add(B, extOff);
  uint64_t extLen = G.genData<uint64_t>(1)[0];
  iree_vm_RodataSegmentDef_external_data_length_add(B, extLen);
  return iree_vm_RodataSegmentDef_end(B);
}

iree_vm_RwdataSegmentDef_ref_t buildRwdataSegmentDef(flatcc_builder_t* B,
                                                     DataGen& G) {
  int32_t value = G.genData<int32_t>(1)[0];
  return iree_vm_RwdataSegmentDef_create(B, value);
}

iree_vm_ModuleStateDef_ref_t buildModuleStateDef(flatcc_builder_t* B,
                                                 DataGen& G) {
  int32_t cap = G.genData<int32_t>(1)[0];
  int32_t count = G.genData<int32_t>(1)[0];
  return iree_vm_ModuleStateDef_create(B, cap, count);
}

iree_vm_FunctionDescriptor_t buildFunctionDescriptor(flatcc_builder_t* B,
                                                     DataGen& G) {
  int32_t bc_off = G.genData<int32_t>(1)[0];
  int32_t bc_len = G.genData<int32_t>(1)[0];
  int16_t i32_reg_count = G.genData<int16_t>(1)[0];
  int16_t ref_reg_count = G.genData<int16_t>(1)[0];
  return iree_vm_FunctionDescriptor{.bytecode_offset = bc_off,
                                    .bytecode_length = bc_len,
                                    .i32_register_count = i32_reg_count,
                                    .ref_register_count = ref_reg_count};
}

iree_vm_CallSiteLocDef_ref_t buildCallSiteLocDef(flatcc_builder_t* B,
                                                 DataGen& G) {
  int32_t caller = G.genData<int32_t>(1)[0];
  int32_t callee = G.genData<int32_t>(1)[0];
  return iree_vm_CallSiteLocDef_create(B, caller, callee);
}

iree_vm_FileLineColLocDef_ref_t buildFileLineColLocDef(flatcc_builder_t* B,
                                                       DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  int32_t line = G.genData<int32_t>(1)[0];
  int32_t column = G.genData<int32_t>(1)[0];
  return iree_vm_FileLineColLocDef_create(B, name, line, column);
}

iree_vm_FusedLocDef_ref_t buildFusedLocDef(flatcc_builder_t* B, DataGen& G) {
  size_t metadataLen = G.genLen();
  auto metadataBuf = G.genData<char>(metadataLen);
  auto metadata = flatbuffers_string_create(B, metadataBuf.get(), metadataLen);
  size_t num = G.genLen();
  auto locsData = G.genData<int32_t>(num);
  auto locs = flatbuffers_int32_vec_create(B, locsData.get(), num);
  return iree_vm_FusedLocDef_create(B, metadata, locs);
}

iree_vm_NameLocDef_ref_t buildNameLocDef(flatcc_builder_t* B, DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  int32_t loc = G.genData<int32_t>(1)[0];
  return iree_vm_NameLocDef_create(B, name, loc);
}

iree_vm_BytecodeLocationDef_t buildBytecodeLocationDef(flatcc_builder_t* B,
                                                       DataGen& G) {
  int32_t off = G.genData<int32_t>(1)[0];
  int32_t loc = G.genData<int32_t>(1)[0];
  return iree_vm_BytecodeLocationDef{.bytecode_offset = off, .location = loc};
}

iree_vm_FunctionSourceMapDef_ref_t buildFunctionSourceMapDef(
    flatcc_builder_t* B, DataGen& G) {
  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  std::vector<iree_vm_BytecodeLocationDef_t> locs;
  size_t num = G.genLen();
  for (size_t i = 0; i < num; ++i) {
    auto loc = buildBytecodeLocationDef(B, G);
    locs.push_back(loc);
  }
  auto locsRef =
      iree_vm_BytecodeLocationDef_vec_create(B, locs.data(), locs.size());
  return iree_vm_FunctionSourceMapDef_create(B, name, locsRef);
}

iree_vm_LocationTypeDef_union_ref_t buildLocationTypeDef(flatcc_builder_t* B,
                                                         DataGen& G) {
  int32_t type = (G.genData<int32_t>(1)[0]) % 5;
  switch (type) {
    case 0:
      return iree_vm_LocationTypeDef_as_CallSiteLocDef(
          buildCallSiteLocDef(B, G));
    case 1:
      return iree_vm_LocationTypeDef_as_FileLineColLocDef(
          buildFileLineColLocDef(B, G));
    case 2:
      return iree_vm_LocationTypeDef_as_FusedLocDef(buildFusedLocDef(B, G));
    case 3:
      return iree_vm_LocationTypeDef_as_NameLocDef(buildNameLocDef(B, G));
    case 4:
    default:
      return iree_vm_LocationTypeDef_as_NONE();
  }
}

iree_vm_DebugDatabaseDef_ref_t buildDebugDatabaseDef(flatcc_builder_t* B,
                                                     DataGen& G) {
  size_t locLen = G.genLen();
  std::vector<iree_vm_LocationTypeDef_union_ref_t> locs;
  for (size_t i = 0; i < locLen; ++i) {
    locs.push_back(buildLocationTypeDef(B, G));
  }
  auto locs_vec =
      iree_vm_LocationTypeDef_vec_create(B, locs.data(), locs.size());
  size_t funcLen = G.genLen();
  std::vector<iree_vm_FunctionSourceMapDef_ref_t> funcs;
  for (size_t i = 0; i < funcLen; ++i) {
    funcs.push_back(buildFunctionSourceMapDef(B, G));
  }
  auto funcs_vec =
      iree_vm_FunctionSourceMapDef_vec_create(B, funcs.data(), funcs.size());
  return iree_vm_DebugDatabaseDef_create(B, locs_vec, funcs_vec);
}

iree_vm_BytecodeModuleDef_ref_t buildBytecodeModuleDef(flatcc_builder_t* B,
                                                       DataGen& G) {
  iree_vm_BytecodeModuleDef_start_as_root(B);

  size_t nameLen = G.genLen();
  auto nameBuf = G.genData<char>(nameLen);
  auto name = flatbuffers_string_create(B, nameBuf.get(), nameLen);
  iree_vm_BytecodeModuleDef_name_add(B, name);

  uint32_t version = G.genData<uint32_t>(1)[0];
  iree_vm_BytecodeModuleDef_version_add(B, version);

  {
    std::vector<iree_vm_AttrDef_ref_t> attrs;
    size_t attrsLen = G.genLen();
    for (size_t i = 0; i < attrsLen; ++i) {
      attrs.push_back(buildAttrDef(B, G));
    }
    auto attr_vec = iree_vm_AttrDef_vec_create(B, attrs.data(), attrs.size());
    iree_vm_BytecodeModuleDef_attrs_add(B, attr_vec);
  }

  {
    std::vector<iree_vm_TypeDef_ref_t> types;
    size_t typesLen = G.genLen();
    for (size_t i = 0; i < typesLen; ++i) {
      types.push_back(buildTypeDef(B, G));
    }
    auto type_vec = iree_vm_TypeDef_vec_create(B, types.data(), types.size());
    iree_vm_BytecodeModuleDef_types_add(B, type_vec);
  }

  {
    std::vector<iree_vm_ModuleDependencyDef_ref_t> moduleDeps;
    size_t moduleDepsLen = G.genLen();
    for (size_t i = 0; i < moduleDepsLen; ++i) {
      moduleDeps.push_back(buildModuleDependencyDef(B, G));
    }
    auto moduleDep_vec = iree_vm_ModuleDependencyDef_vec_create(
        B, moduleDeps.data(), moduleDeps.size());
    iree_vm_BytecodeModuleDef_dependencies_add(B, moduleDep_vec);
  }

  {
    std::vector<iree_vm_ImportFunctionDef_ref_t> elements;
    size_t elementsLen = G.genLen();
    for (size_t i = 0; i < elementsLen; ++i) {
      elements.push_back(buildImportFunctionDef(B, G));
    }
    auto vec = iree_vm_ImportFunctionDef_vec_create(B, elements.data(),
                                                    elements.size());
    iree_vm_BytecodeModuleDef_imported_functions_add(B, vec);
  }

  {
    std::vector<iree_vm_ExportFunctionDef_ref_t> elements;
    size_t elementsLen = G.genLen();
    for (size_t i = 0; i < elementsLen; ++i) {
      elements.push_back(buildExportFunctionDef(B, G));
    }
    auto vec = iree_vm_ExportFunctionDef_vec_create(B, elements.data(),
                                                    elements.size());
    iree_vm_BytecodeModuleDef_exported_functions_add(B, vec);
  }

  {
    std::vector<iree_vm_RodataSegmentDef_ref_t> elements;
    size_t elementsLen = G.genLen();
    for (size_t i = 0; i < elementsLen; ++i) {
      elements.push_back(buildRodataSegmentDef(B, G));
    }
    auto vec = iree_vm_RodataSegmentDef_vec_create(B, elements.data(),
                                                   elements.size());
    iree_vm_BytecodeModuleDef_rodata_segments_add(B, vec);
  }

  {
    std::vector<iree_vm_RwdataSegmentDef_ref_t> elements;
    size_t elementsLen = G.genLen();
    for (size_t i = 0; i < elementsLen; ++i) {
      elements.push_back(buildRwdataSegmentDef(B, G));
    }
    auto vec = iree_vm_RwdataSegmentDef_vec_create(B, elements.data(),
                                                   elements.size());
    iree_vm_BytecodeModuleDef_rwdata_segments_add(B, vec);
  }

  iree_vm_BytecodeModuleDef_module_state_add(B, buildModuleStateDef(B, G));

  {
    std::vector<iree_vm_FunctionDescriptor_t> elements;
    size_t elementsLen = G.genLen();
    for (size_t i = 0; i < elementsLen; ++i) {
      elements.push_back(buildFunctionDescriptor(B, G));
    }
    auto vec = iree_vm_FunctionDescriptor_vec_create(B, elements.data(),
                                                     elements.size());
    iree_vm_BytecodeModuleDef_function_descriptors_add(B, vec);
  }

  uint32_t bc_version = G.genData<uint32_t>(1)[0];
  iree_vm_BytecodeModuleDef_bytecode_version_add(B, bc_version);

  {
    size_t dataLen = G.genLen();
    auto dataBuf = G.genData<uint8_t>(dataLen);
    auto data = flatbuffers_uint8_vec_create(B, dataBuf.get(), dataLen);
    iree_vm_BytecodeModuleDef_bytecode_data_add(B, data);
  }

  iree_vm_BytecodeModuleDef_debug_database_add(B, buildDebugDatabaseDef(B, G));

  return iree_vm_BytecodeModuleDef_end_as_root(B);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
  static iree_vm_instance_t* instance = DoInitialization();
  (void)instance;

  flatcc_builder_t builder, *B;
  B = &builder;

  flatcc_builder_init(B);

  DataGen G(Data, Size);

  buildBytecodeModuleDef(B, G);

  flatcc_builder_clear(B);

#if 0
  iree_vm_module_t* bytecode_module = NULL;
  iree_status_t ret_status = iree_vm_bytecode_module_create(
      instance, iree_const_byte_span_t{Data, Size},
      /*flatbuffer_allocator=*/iree_allocator_null(),
      /*allocator=*/iree_allocator_system(), &bytecode_module);
  if (iree_status_is_ok(ret_status)) {
    bytecode_module->destroy(bytecode_module);
  } else {
    iree_status_free(ret_status);
  }
#endif

  return 0;
}
