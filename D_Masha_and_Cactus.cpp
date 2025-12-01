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
const static int MX = 1e5 + 5;

template<class T, typename F = function<T(const T&, const T&)>>
class FW {  
    public: 
    int n, N;
    vt<T> root;    
    T DEFAULT;
    F func;
    FW() {}
    FW(int n, T DEFAULT, F func = [](const T& a, const T& b) {return a + b;}) : func(func) { 
        this->n = n;    
        this->DEFAULT = DEFAULT;
		N = n == 0 ? -1 : log2(n);
        root.rsz(n, DEFAULT);
    }
    
    inline void update_at(int id, T val) {  
        assert(id >= 0);
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    inline T get(int id) {   
        assert(id < n);
        T res = DEFAULT;
        while(id >= 0) { 
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    inline T queries_range(int left, int right) {  
        return get(right) - get(left - 1);
    }

    inline T queries_at(int i) {
        return queries_range(i, i);
    }

    inline void update_range(int l, int r, T val) {
		if(l > r) return;
        update_at(l, val), update_at(r + 1, -val);
    }
	
	inline void reset() {
		root.assign(n, DEFAULT);
	}

	ll select(ll k) {
        ll pos = -1;
        T acc = DEFAULT;
        for(ll bit = 1LL << N; bit > 0; bit >>= 1) {
            ll np = pos + bit;
            if(np < n) {
                T cand = acc + root[np];
                if(cand < k) {
                    acc = cand;
                    pos = np;
                }
            }
        }
        return pos + 1;
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vvi graph(n + 1);
    const int K = log2(n) + 1;
    vvi dp(n + 1, vi(K));
    vi depth(n + 1);
    for(int i = 2; i <= n; i++) {
        int p; cin >> p;
        graph[p].pb(i);
        depth[i] = depth[p] + 1;
        dp[i][0] = p;
    }
    vi tin(n + 1), tout(n + 1);
    {
        int timer = 0;
        auto dfs = [&](auto& dfs, int node = 1) -> void {
            tin[node] = timer++;
            for(auto& nei : graph[node]) {
                dfs(dfs, nei);
            }
            tout[node] = timer - 1;
        };
        dfs(dfs);
    }
    for(int j = 1; j < K; j++) {
        for(int i = 1; i <= n; i++) {
            dp[i][j] = dp[dp[i][j - 1]][j - 1];
        }
    }
    auto lca = [&](int u, int v) -> int {
        if(depth[u] < depth[v]) swap(u, v);
        int D = depth[u] - depth[v];
        for(int j = K - 1; j >= 0; j--) {
            if(D >> j & 1) {
                u = dp[u][j];
            }
        }
        if(u == v) return u;
        for(int j = K - 1; j >= 0; j--) {
            if(dp[u][j] != dp[v][j]) {
                u = dp[u][j];
                v = dp[v][j];
            }
        }
        return dp[u][0];
    };
    vvar(3) Q(n + 1);
    while(q--) {
        int u, v, w; cin >> u >> v >> w;
        Q[lca(u, v)].pb({u, v, w});
    }
    auto add = [&](FW<ll>& fw, int u, ll s) -> void {
        fw.update_range(tin[u], tout[u], s);     
    };
    auto dist = [&](FW<ll>& fw, int u, int v, int l) -> ll {
        ll res = fw.get(tin[u]) + fw.get(tin[v]) - fw.get(tin[l]);
        return res;
    };
    FW<ll> fp(n + 1, 0, [](ll a, ll b) {return a + b;});
    FW<ll> fc(n + 1, 0, [](ll a, ll b) {return a + b;});
    vvll A(n + 1);
    auto dfs = [&](auto& dfs, int node = 1) -> ll {
        ll res = 0;
        for(auto& nei : graph[node]) {
            res += dfs(dfs, nei);
        }
        for(auto& w : A[node]) {
            add(fp, node, w);
        }
        for(auto& [u, v, w] : Q[node]) {
            ll now = w + dist(fp, u, v, node) - dist(fc, u, v, node);
            res = max(res, now);
        }
        add(fc, node, res);
        A[dp[node][0]].pb(res);
        return res;
    };
    cout << dfs(dfs) << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        // cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
