#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

struct custom {
    static const uint64_t C = 0x9e3779b97f4a7c15; const uint32_t RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const { return __builtin_bswap64((x ^ RANDOM) * C); }
    size_t operator()(const std::string& s) const { size_t hash = std::hash<std::string>{}(s); return hash ^ RANDOM; } };
template <class K, class V> using umap = std::unordered_map<K, V, custom>; 
template <class K> using uset = std::unordered_set<K, custom>; template<class K, class V = int> using gpt = gp_hash_table<K, V, custom>;

#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
inline ll gcd(ll a, ll b) {
    int neg = 0;
    if(a < 0) {
        neg ^= 1;
        a = -a;
    }
    if(b < 0) {
        neg ^= 1;
        b = -b;
    }
    while(b) {
        a %= b;
        swap(a, b);
    }
    return a * (neg ? -1 : 1);
}

ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
ll ceil_to_ll(db x) { return ceil(x - 1e-12L); }
int pct(ll x) { return __builtin_popcountll(x); }
int have_bit(ll x, int b) { return (x >> b) & 1; }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const string dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) { ll m = (n + 1) / 2; return m * m; }
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
ll geometric_sum(ll A, ll X, ll M) { // A^0 + A^1 + A^2 + ... + A^(x - 1), notice only to x - 1, works for any mod
    // https://atcoder.jp/contests/abc293/tasks/abc293_e
    if(X == 0) return 0;
    if(A == 1) return X % M;
    if(X % 2 == 1) {
        return (geometric_sum(A, X - 1, M) + modExpo(A, X - 1, M)) % M;
    }
    ll half = geometric_sum(A, X / 2, M);
    ll powA = modExpo(A, X / 2, M);
    return half * (1 + powA) % M;
}
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^0 + p^1 + p^2 + p^3 + ... + p^k
template<typename T> T geometric_power_range(T base, ll startExp, ll endExp) { // return base^startExp + base^(startExp + 1) + ... + base^endExp
    if(startExp > endExp) return 0;
    T first = base.pow(startExp);
    ll len = endExp - startExp + 1;
    return first * (base.pow(len) - 1) / (base - 1);
}
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
//bool is_two_prime_sum(ll n) { return n >= 4 && (n % 2 == 0 || isPrime(n - 2)); }
//bool is_three_prime_sum(ll n) { return n >= 6 && (n % 2 || is_two_prime_sum(n - 2)); }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

template<typename T, typename Compare>
vi closest_left(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = 0; i < n; i++) {
        auto& j = closest[i];
        while(j && cmp(a[i], a[j - 1])) j = closest[j - 1];
    }
    return closest;
}

template<typename T, typename Compare>
vi closest_right(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = n - 1; i >= 0; i--) {
        auto& j = closest[i];
        while(j < n - 1 && cmp(a[i], a[j + 1])) j = closest[j + 1];
    }
    return closest;
}
template<typename T, typename V = string>
vt<pair<T, int>> encode(const V& s) {
    vt<pair<T, int>> seg;
    for(auto& ch : s) {
        if(seg.empty() || ch != seg.back().ff) seg.pb({ch, 1});
        else seg.back().ss++;
    }
    return seg;
}
vs decode(const string& s, char off = ' ') {
    vs a;
    string t;
    for(auto& ch : s) {
        if(ch == off) {
            if(!t.empty()) a.pb(t);
            t = "";
        } else {
            t += ch;
        }
    }
    if(!t.empty()) a.pb(t);
    return a;
}

//     const db st = clock();
//     while((clock() - st) / CLOCKS_PER_SEC < 0.44) {
