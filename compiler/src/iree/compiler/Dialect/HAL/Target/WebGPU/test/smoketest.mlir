// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module attributes {
  hal.device.targets = [
    #hal.device.target<"webgpu", {
      executable_targets = [
        #hal.executable.target<"webgpu-wgsl", "webgpu-wgsl-fb", {
          spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
        }>
      ]
    }>
  ]
} {

// A minimal executable for translation to WGSL.
//   * A 'compute' op like 'linalg.generic' is required for codegen
//   * The computation just reads from a single binding, without using intermediate memory
stream.executable public @absf_dispatch {
  stream.executable.export @absf_dispatch workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @absf_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:16xf32>
      %0 = tensor.empty() : tensor<16xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = math.absf %arg2 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %2, %arg1, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
      return
    }
  }
}

}

//      CHECK:   hal.executable.binary public @webgpu_wgsl_fb attributes
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "webgpu-wgsl-fb"
