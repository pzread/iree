// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):

  // Step 1. Find the fill, matmul and generic ops
  // ===========================================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %matmul = transform.structured.match ops{["linalg.matmul_transpose_b"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %reduce, %broadcast = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 2. Tile the matmul and fuse the fill
  // ===========================================================================
  %forall_grid, %grid_reduction =
  transform.structured.tile_to_forall_op %broadcast tile_sizes [16] ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
  transform.structured.fuse_into_containing_op %reduce into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %matmul into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 3. Vectorize
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op

  // Step 4. Bufferize
  // ===========================================================================
  transform.apply_patterns to %func_3 {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
  } : !transform.any_op
  transform.apply_patterns to %func_3 {
    transform.apply_patterns.tensor.reassociative_reshape_folding
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_cse %func_3 : !transform.any_op
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  transform.apply_patterns to %func_3 {
    transform.apply_patterns.linalg.erase_unnecessary_inputs
  } : !transform.any_op
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

  // Step 6. Post-bufferization vector distribution
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [4, 8, 1] : (!transform.any_op) -> ()

  // Step 7. Do layout analysis and lower to mma
  // ===========================================================================
  %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  %func_11 = transform.iree.layout_analysis_and_distribution %func_10 : (!transform.any_op) -> (!transform.any_op)
}
