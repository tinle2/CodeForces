#include<bits/stdc++.h>
using namespace std;
#define all(x) begin(x), end(x)
#define ub upper_bound
#define lb lower_bound
#define ll long long
#define vi vector<int>
#define vvi vector<vi>
#define vll vector<ll>
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vpii vector<pii>
#define vpll vector<pll>
#define pb push_back
#define ff first
#define ss second
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define rev(x) reverse(all(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
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

class DSU { 
public: 
    int n, comp;  
    vi root, rank, col;  
    bool is_bipartite;  
    DSU(int n) {    
        this->n = n;    
        comp = n;
        root.resize(n, -1), rank.resize(n, 1), col.resize(n, 0);
        is_bipartite = true;
    }
    
    int find(int x) {   
        if(root[x] == -1) return x; 
        int p = find(root[x]);
        col[x] ^= col[root[x]];
        return root[x] = p;
    }
    
    bool merge(int a, int b) {
        int u = find(a);
        int v = find(b);
        if (u == v) {
            if(col[a] == col[b]) {
                is_bipartite = false;
            }
            return 0;
        }
        if(rank[u] < rank[v]) {
            swap(u, v);
            swap(a, b);
        }
		comp--;
        root[v] = u;
        rank[u] += rank[v];
        if(col[a] == col[b])
            col[v] ^= 1;
        return 1;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
    }
    
    int get_rank(int x) {    
        return rank[find(x)];
    }
    
	vector<vector<int>> get_group() {
        vector<vector<int>> ans(n);
        for(int i = 0; i < n; i++) {
            ans[find(i)].pb(i);
        }
        sort(all(ans), [](const vi& a, const vi& b) {return a.size() > b.size();});
        while(!ans.empty() && ans.back().empty()) ans.pop_back();
        return ans;
    }
};

void solve() {
    int n, m; cin >> n >> m;
    vector<array<int, 3>> edges(m); cin >> edges;
    for(auto& [u, v, w] : edges) {
        u--, v--;
    }
    sort(all(edges), [](const auto& x, const auto& y) {return x[2] < y[2];});
    auto f = [&](int x) -> pll {
        vector<array<int, 4>> now; 
        for(const auto& [u, v, w] : edges) {
            now.pb({u, v, abs(w - x), w});
        }
        sort(all(now), [](const auto& x, const auto& y) {
                if(x[2] != y[2]) return x[2] < y[2];
                return x[3] > y[3];
                });
        DSU root(n);
        ll res = 0, lt = 0;
        for(auto& [u, v, w, ow] : now) {
            if(root.merge(u, v)) {
                if(ow <= x) lt++;
                res += w;
            }
        }
        return {res, lt};
    };
    int p, k, a, b, c; cin >> p >> k >> a >> b >> c;
    vi potential;
    auto clamp = [&](int w) -> int {
        return max(0, min(w, c - 1));
    };
    auto push = [&](ll x) -> void {
        potential.pb(clamp(x));
    };
    for(int i = 0; i < m; i++) {
        for(int j = i; j < m; j++) {
            push((edges[i][2] + edges[j][2] + 1) / 2);
        }
    }
    push(0);
    push(c - 1);
    srtU(potential);
    const int M = potential.size();
    vll base(M), lt(M);
    for(int i = 0; i < M; i++) {
        auto [x, y] = f(potential[i]);
        base[i] = x;
        lt[i] = y;
    }
    vi Q(k + 1); 
    ll res = 0;
    for(int j = 1; j <= k; j++) {
        if(j <= p) {
            cin >> Q[j];
        } else {
            Q[j] = ((ll)Q[j - 1] * a + b) % c;
        }
        int x = Q[j];
        int i = lb(all(potential), x + 1) - begin(potential) - 1;
        ll now = base[i] + ((ll)x - potential[i]) * (lt[i] - (n - 1 - lt[i]));
        res ^= now;
    }
    cout << res << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
