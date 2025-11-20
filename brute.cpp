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

template<typename T = int>
class GRAPH {
public:
	int n, m;
    vvi dp;
    vi parent, subtree;
    vi tin, tout, low, ord, depth;
    vll depth_by_weight;
    vvi weight;
    int timer = 0;
    vt<unsigned> in_label, ascendant;
    vi par_head;
    unsigned cur_lab = 1;
    const vt<vt<T>> adj;

    GRAPH() {}

    GRAPH(const vt<vt<T>>& graph, int root = 0) : adj(graph) {
        n = graph.size();
        m = log2(n) + 1;
//        depth_by_weight.rsz(n);
//        weight.rsz(n, vi(m));
        dp.rsz(n, vi(m, -1));
        depth.rsz(n);
        parent.rsz(n, -1);
        subtree.rsz(n, 1);
        tin.rsz(n);
        tout.rsz(n);
        ord.rsz(n);
        dfs(root);
        init();
        in_label.rsz(n);
        ascendant.rsz(n);
        par_head.rsz(n + 1);
        sv_dfs1(root);
        ascendant[root] = in_label[root];
        sv_dfs2(root);
    }

	void dfs(int node, int par = -1) {
        tin[node] = timer++;
        ord[tin[node]] = node;
        for (auto& nei : adj[node]) {
            if (nei == par) continue;
            depth[nei] = depth[node] + 1;
//            depth_by_weight[nei] = depth_by_weight[node] + w;
//            weight[nei][0] = w;
            dp[nei][0] = node;
            parent[nei] = node;
            dfs(nei, node);
            subtree[node] += subtree[nei];
			// timer++; // use when merging segtree nodes for different subtree, size then become 2 * n
        }
        tout[node] = timer - 1;
    }

    bool is_ancestor(int par, int child) { return tin[par] <= tin[child] && tin[child] <= tout[par]; }

	void init() {
        for (int j = 1; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                int p = dp[i][j - 1];
                if(p == -1) continue;
                //weight[i][j] = max(weight[i][j - 1], weight[p][j - 1]);
                dp[i][j] = dp[p][j - 1];
            }
        }
    }

    void sv_dfs1(int u, int p = -1) {
        in_label[u] = cur_lab++;
        for(auto& v : adj[u]) if (v != p) {
            sv_dfs1(v, u);
            if(std::__countr_zero(in_label[v]) > std::__countr_zero(in_label[u]))
                in_label[u] = in_label[v];
        }
    }

    void sv_dfs2(int u, int p = -1) {
        for(auto& v : adj[u]) if (v != p) {
            ascendant[v] = ascendant[u];
            if(in_label[v] != in_label[u]) {
                par_head[in_label[v]] = u;
                ascendant[v] += in_label[v] & -in_label[v];
            }
            sv_dfs2(v, u);
        }
    }

    int lift(int u, unsigned j) const {
        unsigned k = std::__bit_floor(ascendant[u] ^ j);
        return k == 0 ? u : par_head[(in_label[u] & -k) | k];
    }

    int lca(int a, int b) {
        if(is_ancestor(a, b)) return a;
        if(is_ancestor(b, a)) return b;
        auto [x, y] = std::minmax(in_label[a], in_label[b]);
        unsigned j = ascendant[a] & ascendant[b] & -std::__bit_floor((x - 1) ^ y);
        a = lift(a, j);
        b = lift(b, j);
        return depth[a] < depth[b] ? a : b;
    }

    int path_queries(int u, int v) { // lca in logn
        if(depth[u] < depth[v]) swap(u, v);
        int res = 0;
        int diff = depth[u] - depth[v];
        for(int i = 0; i < m; i++) {
            if(diff & (1 << i)) { 
                res = max(res, weight[u][i]);
                u = dp[u][i]; 
            }
        }
        if(u == v) return res;
        for(int i = m - 1; i >= 0; --i) {
            if(dp[u][i] != dp[v][i]) {
                res = max({res, weight[u][i], weight[v][i]});
                u = dp[u][i];
                v = dp[v][i];
            }
        }
        return max({res, weight[u][0], weight[v][0]});
    }

    int dist(int u, int v) {
        int a = lca(u, v);
        return depth[u] + depth[v] - 2 * depth[a];
    }
	
	ll dist_by_weight(int u, int v) {
        int a = lca(u, v);
        return depth_by_weight[u] + depth_by_weight[v] - 2 * depth_by_weight[a];
    }

	int kth_ancestor(int u, ll k) {
        if(u < 0 || k > depth[u]) return -1;
        for(int i = 0; i < m && u != -1; ++i) {
            if(k & (1LL << i)) {
                u = (u >= 0 ? dp[u][i] : -1);
            }
        }
        return u;
    }

    int kth_ancestor_on_path(int u, int v, ll k) {
        int d = dist(u, v);
        if(k >= d) return v;
        int w  = lca(u, v);
        int du = depth[u] - depth[w];
        if(k <= du) return kth_ancestor(u, k);
        int rem = k - du;
        int dv  = depth[v] - depth[w];
        return kth_ancestor(v, dv - rem);
    }

    int kth_downward(int v, ll k) {
        if(k < 1 || k > depth[v] + 1) return -1;
        ll steps_up = depth[v] - (k - 1);
        return kth_ancestor(v, steps_up);
    }

    int max_intersection(int a, int b, int c) { // # of common intersection between path(a, c) OR path(b, c)
        auto cal = [&](int u, int v, int goal){
            return (dist(u, goal) + dist(v, goal) - dist(u, v)) / 2 + 1;
        };
        int res = 0;
        res = max(res, cal(a, b, c));
        res = max(res, cal(a, c, b));
        res = max(res, cal(b, c, a));
        return res;
    }
	
	int intersection(int a, int b, int c, int d) { // common edges between path[a, b] OR path[c, d]
        vi arr = {a, b, c, d, lca(a, b), lca(a, c), lca(a, d), lca(b, c), lca(b, d), lca(c, d)};
        srtU(arr);
        vi s;
        int res = 0;
        for(auto& x : arr) {
            if(dist(x, a) + dist(x, b) == dist(a, b) && dist(c, x) + dist(x, d) == dist(c, d)) {
                s.pb(x);
                for(auto& y : s) {
                    res = max(res, dist(x, y)); // +1 if looking for maximum node
                }
            }
        }
        return res;
    }

    bool is_continuous_chain(int a, int b, int c, int d) { // determine if path[a, b][b, c][c, d] don't have any intersection
        return dist(a, b) <= dist(a, c) && dist(d, c) <= dist(d, b) && intersection(a, b, c, d) == 0;
    }

    int rooted_lca(int a, int b, int c) { return lca(a, c) ^ lca(a, b) ^ lca(b, c); } 

    int next_on_path(int u, int v) { // closest_next_node from u to v
        if(u == v) return -1;
        if(is_ancestor(u, v)) return kth_ancestor(v, depth[v] - depth[u] - 1);
        return parent[u];
    }

    void reroot(int root) {
        fill(all(parent), -1);
        timer = 0;
        dfs(root);
        init();
        cur_lab = 1;
        sv_dfs1(root);
        ascendant[root] = in_label[root];
        sv_dfs2(root);
    }

    int comp_size(int c,int v){
        if(parent[v] == c) return subtree[v];
        return n - subtree[c];
    }

    int rooted_lca_potential_node(int a, int b, int c) { // # of nodes where rooted at will make lca(a, b) = c
        if(rooted_lca(a, b, c) != c) return 0;
        int v1 = next_on_path(c, a);
        int v2 = next_on_path(c, b);
        return n - (v1 == -1 ? 0 : comp_size(c, v1)) - (v2 == -1 ? 0 : comp_size(c, v2));
    }
	
	vi get_path(int u, int v) { // get every node in path [u, v]
        vi path1, path2;
        int c = lca(u, v);
        while(u != c) {
            path1.pb(u);
            u = parent[u];
        }
        while(v != c) {
            path2.pb(v);
            v = parent[v];
        }
        rev(path2);
        path1.pb(c);
        path1.insert(end(path1), all(path2));
        return path1;
    }
};

void solve() {
    int n; cin >> n;
    vvi graph(n);
    for(int i = 1; i < n; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        graph[u].pb(v);
        graph[v].pb(u);
    }
    int Q; cin >> Q;
    while(Q--) {
        int u, v; cin >> u >> v;
        u--, v--;
        vi dp(n, -1);
        queue<int> q;
        auto process = [&](int node, int cost) -> void {
            if(dp[node] != -1) return;
            dp[node] = cost;
            q.push(node);
        };
        process(u, 0);
        process(v, 0);
        while(!q.empty()) {
            auto node = q.front(); q.pop();
            for(auto& nei : graph[node]) {
                process(nei, dp[node] + 1);
            }
        }
        cout << sum(dp) << '\n';
    }
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
