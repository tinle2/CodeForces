#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 8e6 + 5;

template <int MOD>
struct mod_int {
    int value;
    
    mod_int(ll v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((ll)value * other.value % MOD); return *this; }
    mod_int pow(ll p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
    mod_int inv() const { return pow(MOD - 2); }
    mod_int& operator/=(const mod_int &other) { return *this *= other.inv(); }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const { return mod_int((ll)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((ll)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((ll)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((ll)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((ll)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator ll() const { return (ll)value; }
    explicit operator int() const { return value; }
    explicit operator db() const { return (db)value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend ostream& operator<<(ostream &os, const mod_int &a) { os << a.value; return os; }
    friend istream& operator>>(istream &is, mod_int &a) { ll v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<998244353>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

mint p10[MX], i10[MX];
#define lc i * 2 + 1
#define rc i * 2 + 2
#define lp lc, left, middle
#define rp rc, middle + 1, right
#define entireTree 0, 0, n - 1
#define midPoint left + (right - left) / 2
#define pushDown push(i, left, right)
#define iter int i, int left, int right

struct info {
    int len;
    int del;
    mint prod;
    int empty;
    info(int x = -1) : empty(x == -1), len(x == -1 || x == 10 ? 0 : 1), prod(x == -1 || x == 10 ? 0 : x), del(x == 10) { }
};
class SGT { 
    public: 
    int n;  
    vector<info> root;
	SGT(int n) {    
        this->n = n;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, info());    
    }

    void build(const vi& a) {
        build(entireTree, a);
    }

    void build(iter, const vi &a) {
        if(left == right) {
            root[i] = info(a[left]);
            return;
        }
        int middle = (left + right) / 2;
        build(lp, a);
        build(rp, a);
        root[i] = merge(lc, root[lc], root[rc]);
    }
    
    void update_at(int id, info val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, info val) {  
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = merge(lc, root[lc], root[rc]);
    }

    info merge(int i, const info& a, const info& b) {
        if(a.empty) return b;
        if(b.empty) return a;
        info res;
        res.empty = 0;
        if(a.len <= b.del) {
            res.prod = b.prod;
            res.del = b.del - a.len + a.del;
            res.len = b.len;
        } else {
            auto v = eval(i, b.del);
            res.len = a.len - b.del + b.len;
            res.prod = (a.prod - v) * i10[b.del] * p10[b.len] + b.prod;
            res.del = a.del;
        }
        return res;
    }

    mint eval(int i, int len) {
        // root[i] holds pre->B->D(C is deleted so not used)
        if(len == 0) return 0;
        if(root[i].len <= len) return root[i].prod;
        if(root[rc].len >= len) return eval(rc, len);
        int left_len = len - root[rc].len;
        auto v = root[lc].prod - eval(lc, len - root[rc].len + root[rc].del);
        return root[i].prod - v * i10[root[rc].del] * p10[root[rc].len];
    }

	info queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	info queries_at(iter, int id) {
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    info ans;
    info queries_range(int start, int end) { 
        ans = info();
        queries_range(entireTree, start, end);
        return ans;
    }
    
    void queries_range(iter, int start, int end) {   
        if(left > end || start > right) return;
        if(left >= start && right <= end) {
            ans = merge(i, root[i], ans);
            return;
        }
        int middle = midPoint;  
        queries_range(rp, start, end);
        queries_range(lp, start, end);
    }
};

void solve() {
    int n, q; cin >> n >> q;
    string s; cin >> s;
    s = ' ' + s;
    SGT root(n + 1);
    vi type(n + 1, -1);
    auto f = [](char c) -> int {
        return c == 'B' ? 10 : c - '0';
    };
    for(int i = 1; i <= n; i++) {
        type[i] = f(s[i]);
    }
    root.build(type);
    while(q--) {
        int op; cin >> op;
        if(op == 1) {
            int x; cin >> x;
            char c; cin >> c;
            root.update_at(x, f(c));
        } else {
            int l, r; cin >> l >> r;
            auto it = root.queries_range(l, r);
            if(it.len == 0) cout << -1 << '\n';
            else cout << it.prod << '\n';
        }
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    p10[0] = i10[0] = 1;
    for(int i = 1; i < MX; i++) {
        p10[i] = p10[i - 1] * 10;
    }
    i10[MX - 1] = p10[MX - 1].inv();
    for(int i = MX - 2; i >= 1; i--) {
        i10[i] = i10[i + 1] * 10;
    }
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
