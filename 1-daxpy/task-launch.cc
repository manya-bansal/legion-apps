#include <iostream>
#include <vector>

#include "debug.h"
#include "legion.h"

using namespace Legion;

static int num_elem = 1000;
static int num_partitions = 4;

enum TaskID { TOP_LEVEL_TASK, DAXPY_TASK };
enum FieldIS { X_ID, Y_ID, Z_ID };

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {
  DEBUG_PRINT("In the top level task");

  // Make an index space for our arrays. The same
  // index space can be used for both the input and output in this case.
  Rect<1> index_space_range(0, num_elem - 1);
  IndexSpace index_space = runtime->create_index_space(ctx, index_space_range);
  runtime->attach_name(index_space, "index_space");

  // Create a field space from this Index Space.
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  // In the input field space, create fields for both of our inputs.
  FieldAllocator input_allocator =
      runtime->create_field_allocator(ctx, input_fs);
  // Each field is essentially adding a "coloumn" in the index space
  input_allocator.allocate_field(sizeof(float), X_ID);
  runtime->attach_name(input_fs, X_ID, "X");
  input_allocator.allocate_field(sizeof(float), Y_ID);
  runtime->attach_name(input_fs, Y_ID, "Y");

  // Do the same for an output field space.
  FieldSpace output_fs = runtime->create_field_space(ctx);
  FieldAllocator output_allocator =
      runtime->create_field_allocator(ctx, output_fs);
  output_allocator.allocate_field(sizeof(float), Z_ID);
  runtime->attach_name(output_fs, Z_ID, "Z");

  // Now, based on these field spaces, create logical regions.
  LogicalRegion input_lr =
      runtime->create_logical_region(ctx, index_space, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr =
      runtime->create_logical_region(ctx, index_space, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  // Initialize the data!
  // Create a physical parition, and then initialize the physical partition
  // using a FieldAccessor
  RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);
  req.add_field(X_ID);
  req.add_field(Y_ID);
  InlineLauncher input_launcher(req);
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  input_region.wait_until_valid();

  const FieldAccessor<READ_WRITE, float, 1> acc_x(input_region, X_ID);
  const FieldAccessor<READ_WRITE, float, 1> acc_y(input_region, Y_ID);

  for (PointInRectIterator<1> pir(index_space_range); pir(); pir++) {
    acc_x[*pir] = 1.0f;
    acc_y[*pir] = 3.0f;
  }

  // Now, partition the index space (which both input_fs and output_lr)
  // are derived from into num_parition pieces.
  Rect<1> color(0, num_partitions - 1);
  // Create an index space for color
  IndexSpace color_is = runtime->create_index_space(ctx, color);
  // I guess this creates a tree of paritions??
  IndexPartition partition =
      runtime->create_equal_partition(ctx, index_space, color_is);
  runtime->attach_name(partition, "partition");

  // Get the input and output's logical partiton now that the original index
  // space has been partitioned.
  LogicalPartition input_partition =
      runtime->get_logical_partition(ctx, input_lr, partition);
  runtime->attach_name(input_partition, "input_partition");
  LogicalPartition output_partition =
      runtime->get_logical_partition(ctx, output_lr, partition);
  runtime->attach_name(output_partition, "output_partition");

  ArgumentMap arg_map;
  float alpha = 3.14f;
  IndexLauncher daxpy_launcher(DAXPY_TASK, color_is,
                               TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_partition, 0, READ_WRITE, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(X_ID);
  daxpy_launcher.region_requirements[0].add_field(Y_ID);
  daxpy_launcher.add_region_requirement(RegionRequirement(
      output_partition, 0, WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(X_ID);
  runtime->execute_index_space(ctx, daxpy_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, index_space);
  runtime->destroy_index_space(ctx, color_is);
}

// Modify the result array by ref.
void daxpy_unfused(const Task *task, const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(float));
  const float alpha = *((const float *)task->args);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY, float, 1> acc_x(regions[0], X_ID);
  const FieldAccessor<READ_ONLY, float, 1> acc_y(regions[0], Y_ID);
  const FieldAccessor<WRITE_DISCARD, float, 1> acc_z(regions[1], Z_ID);
  printf("Running daxpy computation with alpha %.8g for point %d...\n", alpha,
         point);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
    std::cout << acc_x[*pir] << std::endl;
  }
}

int main(int argc, char **argv) {
  // Let's call our top-level task
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK);
  TaskVariantRegistrar registrar(TOP_LEVEL_TASK, "top_level_task");
  Runtime::preregister_task_variant<top_level_task>(registrar,
                                                    "top_level_task");
  TaskVariantRegistrar daxpy_register(DAXPY_TASK, "daxpy_task");
  daxpy_register.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  daxpy_register.set_leaf();
  Runtime::preregister_task_variant<daxpy_unfused>(daxpy_register,
                                                   "daxpy_task");

  return Runtime::start(argc, argv);
}
