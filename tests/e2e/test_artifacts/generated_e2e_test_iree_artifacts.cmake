iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileNetV2_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_int8.0_224_quantized.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "BertLargeTFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Unet2dPT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargefp16PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2Sfp16PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "EfficientNetV2Sfp16PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_3456x1024x2048_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_3456x1024x2048_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2560x2560x2560_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2560x2560x2560_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
  FRIENDLY_NAME "matmul_128x256x8192_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
  FRIENDLY_NAME "matmul_128x256x8192_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50TFBatch8(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50TFBatch256(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTFBatch16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTFBatch24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTFBatch48(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5LargeTFBatch24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargePTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargePTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargePTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargePTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargePTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50PTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50PTBatch128(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50PTBatch256(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50fp16PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50fp16PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50fp16PTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50fp16PTBatch128(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "Resnet50fp16PTBatch256(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargefp16PTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargefp16PTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargefp16PTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargefp16PTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH48(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH8(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH64(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH128(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH256(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=ampere-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [nvidia-ampere-vulkan_linux-vulkan_spirv][experimental-flags,tensorcore]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=ampere-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
  FRIENDLY_NAME "Unet2dPT(linalg) [nvidia-ampere-vulkan_linux-vulkan_spirv][experimental-flags,tensorcore]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=pascal-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [nvidia-pascal-vulkan_linux-vulkan_spirv][experimental-flags,simt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=pascal-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
  FRIENDLY_NAME "Unet2dPT(linalg) [nvidia-pascal-vulkan_linux-vulkan_spirv][experimental-flags,simt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,data-tiling,ukernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Unet2dPT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetB7PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargefp16PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2Sfp16PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2Sfp16PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_3456x1024x2048_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_3456x1024x2048_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2560x2560x2560_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2560x2560x2560_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_128x256x8192_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_128x256x8192_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch8(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch256(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch48(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargePTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargePTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargePTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargePTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargePTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50PTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50PTBatch128(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50PTBatch256(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50fp16PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50fp16PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50fp16PTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50fp16PTBatch128(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50fp16PTBatch256(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargefp16PTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargefp16PTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargefp16PTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargefp16PTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH48.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BERT_LARGE_JAX_384XI32_BATCH48(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH8.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH8(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH64(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH128(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH256.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "RESNET50_FP32_JAX_3X224X224XF32_BATCH256(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH1(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH16(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH24.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH24(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5_LARGE_FP32_JAX_512XI32_BATCH32(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=ampere-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [nvidia-ampere-vulkan_linux-vulkan_spirv][experimental-flags,tensorcore,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=ampere-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Unet2dPT(linalg) [nvidia-ampere-vulkan_linux-vulkan_spirv][experimental-flags,tensorcore,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=pascal-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "ClipTextSeqLen64PT(linalg) [nvidia-pascal-vulkan_linux-vulkan_spirv][experimental-flags,simt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=none"
    "--iree-vulkan-target-triple=pascal-unknown-linux"
    "--iree-stream-resource-index-bits=64"
    "--iree-vm-target-index-bits=64"
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-preprocessing-convert-conv2d-to-img2col,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Unet2dPT(linalg) [nvidia-pascal-vulkan_linux-vulkan_spirv][experimental-flags,simt,compile-stats]"
  PUBLIC
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-MobileNetV2_int8_tflite_
  ${PACKAGE_NAME}_model-BertForMaskedLMTF
  ${PACKAGE_NAME}_model-BertLargeTF
  ${PACKAGE_NAME}_model-BertLargefp16PTBatch1
  ${PACKAGE_NAME}_model-ClipTextSeqLen64PT
  ${PACKAGE_NAME}_model-EfficientNetB7PT
  ${PACKAGE_NAME}_model-EfficientNetV2SPT
  ${PACKAGE_NAME}_model-EfficientNetV2STF
  ${PACKAGE_NAME}_model-EfficientNetV2Sfp16PT
  ${PACKAGE_NAME}_model-MiniLML12H384Uncased
  ${PACKAGE_NAME}_model-Unet2dPT
  ${PACKAGE_NAME}_model-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_128x256x8192_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_128x256x8192_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2560x2560x2560_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2560x2560x2560_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_3456x1024x2048_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_3456x1024x2048_f32t_tile_config_default
)

add_dependencies(iree-benchmark-import-models-large
  ${PACKAGE_NAME}_model-BERT_LARGE_JAX_384XI32_BATCH1
  ${PACKAGE_NAME}_model-BERT_LARGE_JAX_384XI32_BATCH16
  ${PACKAGE_NAME}_model-BERT_LARGE_JAX_384XI32_BATCH24
  ${PACKAGE_NAME}_model-BERT_LARGE_JAX_384XI32_BATCH32
  ${PACKAGE_NAME}_model-BERT_LARGE_JAX_384XI32_BATCH48
  ${PACKAGE_NAME}_model-BertLargePTBatch1
  ${PACKAGE_NAME}_model-BertLargePTBatch16
  ${PACKAGE_NAME}_model-BertLargePTBatch24
  ${PACKAGE_NAME}_model-BertLargePTBatch32
  ${PACKAGE_NAME}_model-BertLargePTBatch48
  ${PACKAGE_NAME}_model-BertLargeTFBatch1
  ${PACKAGE_NAME}_model-BertLargeTFBatch16
  ${PACKAGE_NAME}_model-BertLargeTFBatch24
  ${PACKAGE_NAME}_model-BertLargeTFBatch32
  ${PACKAGE_NAME}_model-BertLargeTFBatch48
  ${PACKAGE_NAME}_model-BertLargeTFBatch64
  ${PACKAGE_NAME}_model-BertLargefp16PTBatch16
  ${PACKAGE_NAME}_model-BertLargefp16PTBatch24
  ${PACKAGE_NAME}_model-BertLargefp16PTBatch32
  ${PACKAGE_NAME}_model-BertLargefp16PTBatch48
  ${PACKAGE_NAME}_model-RESNET50_FP32_JAX_3X224X224XF32_BATCH1
  ${PACKAGE_NAME}_model-RESNET50_FP32_JAX_3X224X224XF32_BATCH128
  ${PACKAGE_NAME}_model-RESNET50_FP32_JAX_3X224X224XF32_BATCH256
  ${PACKAGE_NAME}_model-RESNET50_FP32_JAX_3X224X224XF32_BATCH64
  ${PACKAGE_NAME}_model-RESNET50_FP32_JAX_3X224X224XF32_BATCH8
  ${PACKAGE_NAME}_model-Resnet50PTBatch1
  ${PACKAGE_NAME}_model-Resnet50PTBatch128
  ${PACKAGE_NAME}_model-Resnet50PTBatch256
  ${PACKAGE_NAME}_model-Resnet50PTBatch64
  ${PACKAGE_NAME}_model-Resnet50PTBatch8
  ${PACKAGE_NAME}_model-Resnet50TFBatch1
  ${PACKAGE_NAME}_model-Resnet50TFBatch128
  ${PACKAGE_NAME}_model-Resnet50TFBatch256
  ${PACKAGE_NAME}_model-Resnet50TFBatch64
  ${PACKAGE_NAME}_model-Resnet50TFBatch8
  ${PACKAGE_NAME}_model-Resnet50fp16PTBatch1
  ${PACKAGE_NAME}_model-Resnet50fp16PTBatch128
  ${PACKAGE_NAME}_model-Resnet50fp16PTBatch256
  ${PACKAGE_NAME}_model-Resnet50fp16PTBatch64
  ${PACKAGE_NAME}_model-Resnet50fp16PTBatch8
  ${PACKAGE_NAME}_model-T5LargeTFBatch1
  ${PACKAGE_NAME}_model-T5LargeTFBatch16
  ${PACKAGE_NAME}_model-T5LargeTFBatch24
  ${PACKAGE_NAME}_model-T5LargeTFBatch32
  ${PACKAGE_NAME}_model-T5_LARGE_FP32_JAX_512XI32_BATCH1
  ${PACKAGE_NAME}_model-T5_LARGE_FP32_JAX_512XI32_BATCH16
  ${PACKAGE_NAME}_model-T5_LARGE_FP32_JAX_512XI32_BATCH24
  ${PACKAGE_NAME}_model-T5_LARGE_FP32_JAX_512XI32_BATCH32
)

add_dependencies(iree-benchmark-suites-comp-stats
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_compile-stats_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
)

add_dependencies(iree-benchmark-suites-comp-stats-large
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
)

add_dependencies(iree-benchmark-suites-cuda
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
)

add_dependencies(iree-benchmark-suites-cuda-large
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
)

add_dependencies(iree-benchmark-suites-default
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2Sfp16PT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
)

add_dependencies(iree-benchmark-suites-large
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BERT_LARGE_JAX_384XI32_BATCH48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargePTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch48_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch16_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch24_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch32_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargefp16PTBatch48_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-RESNET50_FP32_JAX_3X224X224XF32_BATCH8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch256_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch8_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch128_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch1_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch256_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch64_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50fp16PTBatch8_linalg___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH16_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH1_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH24_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-T5_LARGE_FP32_JAX_512XI32_BATCH32_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
)

add_dependencies(iree-benchmark-suites-riscv
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
)

add_dependencies(iree-benchmark-suites-vulkan-nvidia
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_
  ${PACKAGE_NAME}_iree-module-ClipTextSeqLen64PT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-ampere-vulkan_linux-vulkan_spirv__experimental-flags_tensorcore_
  ${PACKAGE_NAME}_iree-module-Unet2dPT_linalg___nvidia-pascal-vulkan_linux-vulkan_spirv__experimental-flags_simt_
)

add_dependencies(iree-benchmark-suites-x86_64
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetB7PT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2SPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_data-tiling_ukernel_
)

add_dependencies(iree-benchmark-suites-x86_64-large
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_
)
