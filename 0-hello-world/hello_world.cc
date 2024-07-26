#include <iostream>
#include <vector>

#include "legion.h"

using namespace Legion;

#define TOP_LEVEL_TASK_ID 0

void hello_world(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
  std::cout << "Hello Legion" << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "hello_world");
  Runtime::preregister_task_variant<hello_world>(registrar, "hello_word");

  return Runtime::start(argc, argv);
}
