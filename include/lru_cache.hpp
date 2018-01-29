#ifndef _LRU_CACHE_CPP
#define _LRU_CACHE_CPP

#include <string>
#include <unordered_map>
#include <list>
#include <ideep/tensor.hpp>

namespace ideep {
namespace utils {
template <class key_t, class value_t>
class lru_cache {
public:
  class node_t;

  typedef typename std::pair<key_t, value_t> value_type;

  typedef typename std::list<node_t>::iterator iterator;
  typedef typename std::list<node_t>::const_iterator const_iterator;

  typedef typename std::unordered_multimap<key_t, iterator>::iterator map_it;
  typedef typename std::unordered_multimap<key_t, iterator>::const_iterator
    const_map_it;

  class node_t : public std::pair<map_it, value_t> {};

  typedef typename std::pair<key_t, iterator> map_type;

  typedef typename std::list<node_t>::size_type size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}

  size_type size() const { to_vlist_.size(); }
  size_type max_size() const { return capacity_; }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (to_vlist_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
      to_vlist_.erase(last->first);
      vlist_.pop_back();
    }
  }

  iterator begin() noexcept {
    auto it = to_vlist_.begin();
    if (it == to_vlist_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  const_iterator begin() const noexcept {
    const auto it = to_vlist_.begin();
    if (it == to_vlist_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  iterator end() noexcept {
    return vlist_.end();
  }
  const_iterator end() const noexcept {
    return vlist_.end();
  }

  iterator find(const key_t &key) {
    auto it = to_vlist_.find(key);
    if (it == to_vlist_.end()) {
      return end();
    } else {
      to_vlist_.splice(to_vlist_.begin(), to_vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t &key) const {
    const auto it = to_vlist_.find(key);
    if (it == to_vlist_.end()) {
      return end();
    } else {
      // to_vlist_.splice(to_vlist_.begin(), to_vlist_, it->second);
      return it->second;
    }
  }

  bool empty() const noexcept {
    return vlist_.empty();
  }

  void clear() noexcept {
    vlist_.clear();
    to_vlist_.clear();
  }

  // Can we?
  // template <class... Args>
  // std::pair<iterator, bool> emplace(Args&&... args) {
  // }


  std::pair<iterator, bool> insert(const value_type& value) {
    auto it = to_vlist_.find(value.first);

    if (it == to_vlist_.end()) {
      vlist_.push_front(std::make_pair(it, value.second));
      to_vlist_.insert(std::make_pair(value.first, vlist_.begin()));
    } else
      return std::make_pair(it->second, false);

    // Trim cache
    while (to_vlist_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
      to_vlist_.erase(last->first);
      vlist_.pop_back();
    }

    return std::make_pair(vlist_.begin(), true);
  }

  iterator erase(const_iterator pos) {
    auto it = to_vlist_.erase(pos->first);
    vlist_.erase(pos);
    return it->first;
  }

  // Warning: carefully check iterator validity
  void swap(lru_cache & other) {
    std::swap(vlist_, other.vlist_);
    std::swap(to_vlist_, other.to_vlist_);
    std::swap(capacity_, other.capacity_);
  }

private:
  std::list<node_t> vlist_;
  // std::unordered_map<key_t, iterator> to_vlist_;
  std::unordered_multimap<key_t, iterator> to_vlist_;
  size_type capacity_;
};

// TODO: mutex it
template <class computation_t, class key_t = std::string, size_t capacity = 1024>
class computation_cache {
public:
  template <typename ...Ts>
  static inline computation_t fetch_or_create(const key_t& key, Ts&&... args) {
    const auto it = g_store().find(key);

    if (it != g_store().end()) {
      computation_t comp = std::move(it->second);
      g_store().erase(it);
      return comp;
    }

    return computation_t(std::forward<Ts>(args)...);
  }

  static inline bool release(
      const key_t& key, const computation_t& computation) {
    g_store().insert(std::make_pair(key, computation));
  }

  static lru_cache<key_t, computation_t> &g_store() {
    static lru_cache<key_t, computation_t> g_store_(capacity);
    return g_store_;
  }
};

template <typename T>
inline std::string to_string(const T arg) {
  return to_string<T>(arg);
}

inline std::string to_string(const tensor::dims arg) {
  return std::accumulate(std::next(arg.begin()), arg.end(),
      std::to_string(arg[0]), [](std::string a, int b) {
        return a + std::to_string(b);
      });
}

template <typename T, typename ...Ts>
inline std::string to_string(T&& arg, Ts&&... args) {
  return to_string(std::forward<T>(arg)) + to_string(std::forward<Ts>(args)...);
}

}
}
#endif
