#ifndef MY_DEBUG_H
#define MY_DEBUG_H

#define DEBUG_ON 1

#if DEBUG_ON
#define DEBUG_PRINT(x)                                                         \
  do {                                                                         \
    std::cout << x << std::endl;                                               \
  } while (0)
#else
#define DEBUG_PRINT(x)                                                         \
  do {                                                                         \
  } while (0)
#endif

#endif
