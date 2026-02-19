#include<bits/stdc++.h>
#if defined(LOCAL) && __has_include("debug.h")
#include "debug.h"
#else
#define debug(...)
#endif
using i64 = long long;
using u64 = unsigned long long;
using i128 = __int128;

template<class T, typename TT = int, typename F = std::function<T(const T&, const T&)>>
class HLD {
    private:
	std::vector<std::pair<int, int>> get_path_helper(int node, int par) {
        std::vector<std::pair<int, int>> res;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                res.push_back({id[tp[node]], id[node]});
                node = parent[tp[node]];
            } else {   
                res.push_back({id[par] + 1, id[node]});
                break;  
            } 
        }   
        res.push_back({id[par], id[par]});
        return res;
    }

	T path_queries_helper(int node, int par) { // only query up to parent, don't include parent info
        T res = DEFAULT;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                T t = seg.queries_range(id[tp[node]], id[node]);
                res = func(t, res);
                node = parent[tp[node]];
            } else {   
                T t = seg.queries_range(id[par] + 1, id[node]);
                res = func(t, res);
                break;  
            } 
        }   
        return res; 
    }

	void update_path_helper(int node, int par, T val) {
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                seg.update_range(id[tp[node]], id[node], val);
                node = parent[tp[node]];
            } else {   
                seg.update_range(id[par] + 1, id[node], val); 
                break;  
            } 
        }   
    }
    public:
    SGT<T> seg;
    std::vector<int> id, tp, sz, parent, chain_id, rid;
    int chain_cnt;
    int ct;
    std::vector<std::vector<TT>> graph;
    int n;
    GRAPH<TT> g;
    T DEFAULT;
    F func;
    HLD() {}

    HLD(std::vector<std::vector<TT>>& _graph, std::vector<int> a, F func, int root = 0, T DEFAULT = 0) : graph(_graph), seg(_graph.size(), DEFAULT, func), g(graph, root), n(graph.size()), func(func), DEFAULT(DEFAULT) {
        this->parent = move(g.parent);
        this->sz = move(g.subtree);
        chain_cnt = 0, ct = 0;
        id.resize(n), tp.resize(n), chain_id.resize(n), rid.resize(n);
        dfs(root, -1, root);
        for(int i = 0; i < n; i++) seg.update_at(id[i], a[i]);
    }
        
	void dfs(int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        rid[id[node]] = node;
        tp[node] = top;
        if(node == top) chain_id[node] = chain_cnt++;
        else chain_id[node] = chain_id[top];
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs(nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs(nei, node, nei);  
        }   
    }

    int get_chain(int u) {
        return chain_id[u];
    }

    void update_chain(int u) {

    }

    void update_at(int i, T v) {
        seg.update_at(id[i], v);
    }
	
	void update_subtree(int i, T v) {
        seg.update_range(id[i], id[i] + sz[i] - 1, v);
    }

	std::vector<std::pair<int, int>> get_path(int u, int v) {
        int p = g.lca(u, v);
        auto path = get_path_helper(u, p);
        auto other = get_path_helper(v, p);
        other.pop_back();
        rev(other);
        path.insert(end(path), all(other));
        return path;
    }

	T path_queries(int u, int v) { // remember to include the info of parents
        int c = g.lca(u, v);
        T res = func(seg.queries_at(id[c]), func(path_queries_helper(u, c), path_queries_helper(v, c)));
        return res;
    }

    void update_path(int u, int v, T val) {
        int c = g.lca(u, v);
        update_path_helper(u, c, val);
        update_path_helper(v, c, val);
        seg.update_at(id[c], val);
    }

    int dist(int a, int b) {
        return g.dist(a, b);
    }

    int lca(int a, int b) {
        return g.lca(a, b);
    }

    bool contain_all_node(int u, int v) {
        return path_queries(u, v) == dist(u, v);
    }

    int climb(int u, int k) {
        while(u != -1 && k > 0) {
            int h = tp[u];
            int d = g.depth[u] - g.depth[h];
            if (k <= d) return rid[id[u] - k];
            k -= d + 1;
            u = parent[h];
        }
        return u;
    }

    int kth_on_path(int u, int v, int k) {
        int c = g.lca(u, v);
        int du = g.depth[u] - g.depth[c];
        if (k <= du) return climb(u, k);
        int dv = g.depth[v] - g.depth[c];
        return climb(v, dv - (k - du));
    }

    int kth_ancestor(int u, int k) {
        return climb(u, k);
    }
};

void solve() {
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int t = 1;
    //std::cin >> t;
    while(t--) {
        solve();
    }
    
    return 0;
}
