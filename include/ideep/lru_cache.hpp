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

  // Only need opaque node_t pointer, it'll compile
  typedef typename std::list<node_t>::iterator iterator;
  typedef typename std::list<node_t>::const_iterator const_iterator;

  typedef typename std::unordered_multimap<key_t, iterator>::iterator map_it;
  typedef typename std::unordered_multimap<key_t, iterator>::const_iterator
    const_map_it;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
  public:
    node_t (const std::pair<map_it, value_t> &l) :
      std::pair<map_it, value_t>(l) {}
    node_t (std::pair<map_it, value_t> &&l) :
      std::pair<map_it, value_t>(std::move(l)) {}
  };

  typedef typename std::list<node_t>::size_type size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}

  size_type size() const { map_.size(); }
  size_type max_size() const { return capacity_; }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
      map_.erase(last->first);
      vlist_.pop_back();
    }
  }

  iterator begin() noexcept {
    auto it = map_.begin();
    if (it == map_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  const_iterator begin() const noexcept {
    const auto it = map_.begin();
    if (it == map_.end()) {
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
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t &key) const {
    const auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  bool empty() const noexcept {
    return vlist_.empty();
  }

  void clear() noexcept {
    vlist_.clear();
    map_.clear();
  }

  // Can we?
  // template <class... Args>
  // std::pair<iterator, bool> emplace(Args&&... args) {
  // }

  std::pair<iterator, bool> insert(const value_type& value) {
    auto map_it = map_.find(value.first);

    if (map_it == map_.end()) {
      vlist_.push_front(std::make_pair(map_it, value.second));
      auto list_it = vlist_.begin();
      auto updated = map_.insert(std::make_pair(value.first, list_it));
      // Update node to pointer to new map position
      list_it->first = updated;
    } else
      return std::make_pair(map_it->second, false);

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
      map_.erase(last->first);
      vlist_.pop_back();
    }

    return std::make_pair(vlist_.begin(), true);
  }

  iterator erase(iterator pos) {
    auto map_pos = pos->first;
    map_.erase(map_pos);
    return vlist_.erase(pos);
  }

  // Warning: carefully check iterator validity
  void swap(lru_cache & other) {
    std::swap(vlist_, other.vlist_);
    std::swap(map_, other.map_);
    std::swap(capacity_, other.capacity_);
  }

private:
  std::list<node_t> vlist_;
  // std::unordered_map<key_t, iterator> map_;
  std::unordered_multimap<key_t, iterator> map_;
  size_type capacity_;
};

// TODO: mutex it
template <class value_t, class key_t = std::string, size_t capacity = 1024>
class computation_cache {
public:
  using iterator = typename lru_cache<key_t, value_t>::iterator;

protected:
  template <typename ...Ts>
  static inline value_t create(Ts&&... args) {
    return value_t(std::forward<Ts>(args)...);
  }

  static inline value_t fetch(iterator it) {
    auto comp = std::move(it->second);
    g_store().erase(it);
    return comp;
  }

  static inline iterator find(const key_t& key) {
    return g_store().find(key);
  }

  static inline iterator end() {
    return g_store().end();
  }

public:

// Possible better performance, but use inside class scope only (private)
#define fetch_or_create_m(key, ...) \
  find(key) == end() ? create(__VA_ARGS__) : fetch(find(key));

  template <typename ...Ts>
  static inline value_t fetch_or_create(const key_t& key, Ts&&... args) {
    const auto it = g_store().find(key);

    if (it != g_store().end()) {
      value_t comp = std::move(it->second);
      g_store().erase(it);
      return comp;
    }

    return value_t(std::forward<Ts>(args)...);
  }

  static inline void release(
      const key_t& key, const value_t& computation) {
    g_store().insert(std::make_pair(key, computation));
  }

  static inline void release(
      const key_t& key, value_t&& computation) {
    g_store().insert(std::make_pair(key, std::move(computation)));
  }

  static lru_cache<key_t, value_t> &g_store() {
    static lru_cache<key_t, value_t> g_store_(capacity);
    return g_store_;
  }
};

template <typename T>
inline std::string to_string(const T arg) {
  return std::to_string(arg);
}

inline std::string to_string(const tensor::dims arg) {
  return std::accumulate(std::next(arg.begin()), arg.end(),
      std::to_string(arg[0]), [](std::string a, int b) {
        return a + 'x' + std::to_string(b);
      });
}

template <typename T, typename ...Ts>
inline std::string to_string(T&& arg, Ts&&... args) {
  return to_string(std::forward<T>(arg)) +
    '*' + to_string(std::forward<Ts>(args)...);
}

// Fast alternative to heavy string method
using bytestring = std::string;

template <typename T>
inline bytestring to_bytes(const T arg) {
  return bytestring();
}

inline bytestring to_bytes(const int arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  auto len = sizeof(arg);
  len -= (__builtin_clz(arg) / 8);

  return bytestring(as_cstring, len);
}

inline bytestring to_bytes(const float arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  return bytestring(as_cstring, sizeof(float));
}

inline bytestring to_bytes(const tensor::dims arg) {
  return std::accumulate(std::next(arg.begin()), arg.end(),
      to_bytes(arg[0]), [](bytestring a, int b) {
        return a + 'x' + to_bytes(b);
      });
}

template <typename T, typename ...Ts>
inline bytestring to_bytes(T&& arg, Ts&&... args) {
  return to_bytes(std::forward<T>(arg)) +
    '*' + to_bytes(std::forward<Ts>(args)...);
}


using key_t = std::string;

template <typename ...Ts>
inline key_t create_key(Ts&&... args) {
  return to_string(std::forward<Ts>(args)...);
}

}
}
#endif
