#pragma once
#include <bits/stdc++.h>
using namespace std;

/*
Usage:
  - Compile with -DLOCAL to enable debug.
  - Optional: -DDEBUG_AUTO_FLUSH to make cout unbuffered.
  - debug(x, y, expr(a, b), vec, map, ...)

Key fix vs your version:
  - debug(...) prints to the SAME stream as normal output (cout), so ordering is stable.
*/

#ifdef DEBUG_AUTO_FLUSH
struct _AutoFlush {
    _AutoFlush() {
        cout.setf(std::ios::unitbuf); // auto-flush after each insertion
    }
} _autoFlush;
#endif

#ifdef LOCAL

// Choose output stream for debug.
// Printing to cout ensures it won't interleave with cout answers.
#ifndef DBG_STREAM
#define DBG_STREAM cout
#endif

// ------------------ Pretty printers ------------------

// __int128
inline ostream& operator<<(ostream& os, __int128 x) {
    if (x == 0) { os << '0'; return os; }
    if (x < 0) { os << '-'; x = -x; }
    char s[64]; int n = 0;
    while (x) { s[n++] = char('0' + int(x % 10)); x /= 10; }
    while (n--) os << s[n];
    return os;
}

// pair
template<class A, class B>
inline ostream& operator<<(ostream& os, const pair<A,B>& p) {
    return os << "{" << p.first << " , " << p.second << "}";
}

// tuple
template<size_t I, class... Ts>
inline void _dbg_print_tuple(ostream& os, const tuple<Ts...>& t) {
    if constexpr (I < sizeof...(Ts)) {
        if constexpr (I) os << " , ";
        os << get<I>(t);
        _dbg_print_tuple<I + 1>(os, t);
    }
}
template<class... Ts>
inline ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
    os << "{";
    _dbg_print_tuple<0>(os, t);
    os << "}";
    return os;
}

// detect iterables
template<class C>
struct _dbg_iterable {
    template<class T>
    static auto test(int) -> decltype(begin(declval<T&>()), end(declval<T&>()), true_type{});
    template<class> static auto test(...) -> false_type;
    static constexpr bool value = decltype(test<C>(0))::value;
};

// treat std::string as scalar (not a container)
template<class T>
inline constexpr bool _dbg_is_string = is_same_v<decay_t<T>, string>;

// exclude C-strings and char arrays from generic container printer
template<class T>
inline constexpr bool _dbg_is_cstr =
    is_same_v<decay_t<T>, const char*> || is_same_v<decay_t<T>, char*>;

template<class T>
inline constexpr bool _dbg_is_char_array =
    is_array_v<remove_reference_t<T>> &&
    is_same_v<remove_cv_t<remove_extent_t<remove_reference_t<T>>>, char>;

// generic container printer (NOT for string/char*/char[N])
template<class C, enable_if_t<
    _dbg_iterable<C>::value &&
    !_dbg_is_string<C> &&
    !_dbg_is_cstr<C> &&
    !_dbg_is_char_array<C>, int> = 0>
inline ostream& operator<<(ostream& os, const C& cont) {
    os << '{';
    bool first = true;
    for (auto const& x : cont) {
        os << (first ? "" : " , ") << x;
        first = false;
    }
    os << '}';
    return os;
}

// nicer for std::map
template<class K, class V>
inline ostream& operator<<(ostream& os, const map<K,V>& m) {
    os << '{';
    bool first = true;
    for (auto const& kv : m) {
        os << (first ? "" : " , ") << kv.first << " : " << kv.second;
        first = false;
    }
    os << '}';
    return os;
}

// ------------------ Name parsing helpers ------------------

inline string _dbg_trim(string s) {
    int l = 0, r = (int)s.size() - 1;
    while (l <= r && isspace((unsigned char)s[l])) l++;
    while (l <= r && isspace((unsigned char)s[r])) r--;
    return (l > r) ? string() : s.substr(l, r - l + 1);
}

// find comma that separates macro arguments (top-level only)
inline const char* _dbg_find_top_level_comma(const char* s) {
    int depth_par = 0, depth_ang = 0, depth_br = 0;
    for (; *s; ++s) {
        char c = *s;
        if (c == '(') depth_par++;
        else if (c == ')') depth_par--;
        else if (c == '<') depth_ang++;
        else if (c == '>') depth_ang--;
        else if (c == '[') depth_br++;
        else if (c == ']') depth_br--;
        else if (c == ',' && depth_par == 0 && depth_ang == 0 && depth_br == 0) return s;
    }
    return nullptr;
}

// ------------------ debug(...) macro ------------------

#define debug(...) debug_out(#__VA_ARGS__, __VA_ARGS__)

inline void debug_out(const char*) {
    DBG_STREAM << '\n' << flush;
}

template<class T, class... R>
inline void debug_out(const char* names, T&& v, R&&... r) {
    const char* comma = _dbg_find_top_level_comma(names);
    string name = comma ? string(names, comma) : string(names);
    DBG_STREAM << "[" << _dbg_trim(name) << " = " << v << "]";
    if constexpr (sizeof...(r)) {
        DBG_STREAM << ", ";
        debug_out(comma ? comma + 1 : names, std::forward<R>(r)...);
    } else {
        DBG_STREAM << '\n' << flush;
    }
}

// ------------------ timers & memory usage ------------------

#define startClock do { clock_t _dbg_tStart = clock();
#define endClock   DBG_STREAM << fixed << setprecision(10) \
                    << "\nTime Taken: " \
                    << double(clock() - _dbg_tStart) / CLOCKS_PER_SEC \
                    << " seconds\n" << flush; } while(0)

#if defined(__linux__)
  #include <sys/resource.h>
  #include <sys/time.h>
  inline void printMemoryUsage() {
      rusage u{}; getrusage(RUSAGE_SELF, &u);
      // ru_maxrss is in KB on Linux
      DBG_STREAM << "Memory: " << (u.ru_maxrss / 1024.0) << " MB\n" << flush;
  }
#else
  inline void printMemoryUsage() {}
#endif

#else  // !LOCAL

#define debug(...)
#define startClock do {} while(0)
#define endClock   do {} while(0)
inline void printMemoryUsage() {}

#endif

