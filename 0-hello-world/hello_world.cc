#include <iostream>
#include <vector>

#include "legion.h"

using namespace Legion;

void hello_word(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  std::cout << "Hello Legion" << std::endl;
}

int main(int argc, char **argv) { return 0; }
