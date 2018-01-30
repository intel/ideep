#include <ideep/lru_cache.hpp>
using namespace ideep::utils;

int main() {
  lru_cache<int, int> base(3);
  base.insert(std::make_pair(6, 33));
  base.insert(std::make_pair(6, 33));
  base.insert(std::make_pair(6, 33));
  base.insert(std::make_pair(128, 28));

  int test_list[] = {128, 32, 6, 3};

  for (unsigned i = 0; i < sizeof(test_list)/sizeof(int); ++i) {
    printf("find %d\n", test_list[i]);
    auto it = base.find(test_list[i]);
    if (it != base.end())
      printf("value: %d\n", it->second);
    else
      printf("Nothing\n");
  }

  base.insert(std::make_pair(20, 111));

  printf("After insert (%d, %d)\n", 20, 111);
  for (unsigned i = 0; i < sizeof(test_list)/sizeof(int); ++i) {
    printf("find %d\n", test_list[i]);
    auto it = base.find(test_list[i]);
    if (it != base.end())
      printf("value: %d\n", it->second);
    else
      printf("Nothing\n");
  }

  return 0;
}
