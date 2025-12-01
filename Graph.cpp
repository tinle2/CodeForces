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

https://codeforces.com/contest/1749/problem/F?adcd1e=caf4f277g6k8yy&csrf_token=6beea33a44ff1d0047d81022f5bd54ff&__cf_chl_tk=NS60GPs8ohrOPuSQY5cYAm5P2tPa3R.GhNrP4ZSVgWc-1764193587-1.0.1.1-LIrQY9Yh5ok.CX3Gu70kyj8yufbK16TomCmbCxKEPj4
template<class T, typename TT = int, typename F = function<T(const T&, const T&)>>
class HLD {
    private:
	vpii get_path_helper(int node, int par) {
        vpii res;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                res.pb({id[tp[node]], id[node]});
                node = parent[tp[node]];
            } else {   
                res.pb({id[par] + 1, id[node]});
                break;  
            } 
        }   
        res.pb({id[par], id[par]});
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
    vi id, tp, sz, parent, chain_id, rid;
    int chain_cnt;
    int ct;
    vt<vt<TT>> graph;
    int n;
    GRAPH<TT> g;
    T DEFAULT;
    F func;
    HLD() {}

    HLD(vt<vt<TT>>& _graph, vi a, F func, int root = 0, T DEFAULT = 0) : graph(_graph), seg(_graph.size(), DEFAULT, func), g(graph, root), n(graph.size()), func(func), DEFAULT(DEFAULT) {
        this->parent = move(g.parent);
        this->sz = move(g.subtree);
        chain_cnt = 0, ct = 0;
        id.rsz(n), tp.rsz(n), chain_id.rsz(n), rid.rsz(n);
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

	vpii get_path(int u, int v) {
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

template<class H, class T = long long>
struct path_update_logn {
	https://codeforces.com/gym/105937/problem/K
    const H& h;
    int n;
    FW<T> S, SD;
    T TOT;

    path_update_logn(const H& hld) : h(hld), n(hld.n), S(hld.n, 0), SD(hld.n, 0), TOT(0) {}

    inline T queries_range(const FW<T>& B, int l, int r) const {
        if(l > r) return 0;
        return B.get(r) - (l ? B.get(l - 1) : 0);
    }

    void update_path(int u, int v, T w) {
        int l = h.lca(u, v);
        TOT += (T)(h.depth[u] + h.depth[v] - 2 * h.depth[l] + 1) * w;
        S.update_at(h.tin[u], w);
        S.update_at(h.tin[v], w);
        S.update_at(h.tin[l], -w);
        if(h.parent[l] != -1) S.update_at(h.tin[h.parent[l]], -w);
        SD.update_at(h.tin[u], w * (T)h.depth[u]);
        SD.update_at(h.tin[v], w * (T)h.depth[v]);
        SD.update_at(h.tin[l], -w * (T)h.depth[l]);
        if(h.parent[l] != -1) SD.update_at(h.tin[h.parent[l]], -w * (T)h.depth[h.parent[l]]);
    }

    T subtree_sum(int u) const {
        T W = queries_range(S, h.tin[u], h.tout[u]);
        T WD = queries_range(SD, h.tin[u], h.tout[u]);
        return WD - (T)(h.depth[u] - 1) * W;
    }

    T query_with_root(int x, int rt) const {
        if(x == rt) return TOT;
        if(h.g.is_ancestor(x, rt)) {
            int y = h.kth_ancestor(rt, h.depth[rt] - h.depth[x] - 1);
            return TOT - subtree_sum(y);
        }
        return subtree_sum(x);
    }
};

template<typename T>
struct path_queries { // update point, query path from rt to v mostly
    int n;
    GRAPH<T> g;
    FW<int> fw;
    vi curr, tin, tout;

    path_queries(const vt<vt<T>>& graph, const vi& a, int rt = 0)
      : n(graph.size()), g(graph, rt),
        fw(n * 2 + 1, 0),
        curr(a), tin(graph.size()), tout(graph.size())
    {
        int timer = 0; 
        auto dfs = [&](auto& dfs, int node, int par) -> void {
            tin[node] = timer++;
            for(auto& nei : graph[node]) {
                if(nei == par) continue;
                dfs(dfs, nei, node);
            }
            tout[node] = timer++;
        }; dfs(dfs, rt, -1);
        for(int i = 0; i < n; ++i)
            fw.update_range(tin[i], tout[i] - 1, curr[i]);
    }

    void update_at(int i, int v) {
        int delta = v - curr[i];
        curr[i] = v;
        fw.update_range(tin[i], tout[i] - 1, delta);
    }
    
    ll query(int u) {
        return fw.get(tin[u]);
    }

    ll queries(int u, int v) {
        int w = g.lca(u, v);
        return query(u) + query(v) - 2 * query(w) + curr[w];
    }

    int is_parent(int p, int c) {
        return g.is_ancestor(p, c);
    }
};

class DSU { 
public: 
    int n, comp;  
    vi root, rank, col;  
    bool is_bipartite;  
    DSU(int n) {    
        this->n = n;    
        comp = n;
        root.rsz(n, -1), rank.rsz(n, 1), col.rsz(n, 0);
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
    
	vvi get_group() {
        vvi ans(n);
        for(int i = 0; i < n; i++) {
            ans[find(i)].pb(i);
        }
        sort(all(ans), [](const vi& a, const vi& b) {return a.size() > b.size();});
        while(!ans.empty() && ans.back().empty()) ans.pop_back();
        return ans;
    }
};

struct Persistent_DSU {
	int n, version;
	vvpii parent, rank, col;
	vpii bip;
	Persistent_DSU(int n) {
		this->n = n; version = 0;
		parent.rsz(n); rank.rsz(n); col.rsz(n);
		for (int i = 0; i < n; i++) {
			parent[i].pb({version, i});
			rank[i].pb({version, 1});
			col[i].pb({version, 0});
		}
        bip.pb({version, 1});
	}
 
	int find(int u, int ver) {
		auto pr = *(ub(all(parent[u]), make_pair(ver + 1, -1)) - 1);
		return pr.ss != u ? find(pr.ss, ver) : u;
	}
 
	int get_color(int u, int ver) {
		auto cp = *(ub(all(col[u]), make_pair(ver + 1, -1)) - 1);
		int c = cp.ss;
		int pu = find(u, ver);
		return pu == u ? c : c ^ get_color(pu, ver);
	}
 
	int get_rank(int u, int ver) {
		u = find(u, ver);
		auto it = *(ub(all(rank[u]), make_pair(ver + 1, -1)) - 1);
		return it.ss;
	}

	int merge(int u, int v, int ver) {
		u = find(u, ver), v = find(v, ver);
		int cu = get_color(u, ver), cv = get_color(v, ver);
		if(u == v) {
			if((cu ^ cv) != 1) {
				version = ver;
                bip.pb({version, 0});
			}
			return 0;
		}
		version = ver;
		int szu = rank[u].back().ss;
		int szv = rank[v].back().ss;
		if(szu < szv) swap(u, v);
		parent[v].pb({version, u});
		int new_sz = szu + szv;
		rank[u].pb({version, new_sz});
		int new_col = get_color(u, ver) ^ get_color(v, ver) ^ 1;
		col[v].pb({version, new_col});
		bip.pb({version, bip.back().ss});
		return version;
	}
 
	bool same(int u, int v, int ver) {
		return find(u, ver) == find(v, ver);
	}
 
	int get_bip(int ver) {
		auto it = ub(all(bip), make_pair(ver + 1, -1)) - 1;
		return it->ss;
	}

    int earliest_time(int u, int v, int m) {
        int left = 0, right = m, res = -1;
        while(left <= right) {
            int middle = (left + right) >> 1;
            if(same(u, v, middle)) res = middle, right = middle - 1;
            else left = middle + 1;
        }
        return res;
    }
};

class AUG_DSU { 
    // maintains potentials a[x] so that we can enforce a[u] = a[v] + d
  public: 
    int n, comp;  
    vi root, rank;
    vll weight;

    AUG_DSU(int n) : n(n), comp(n), root(n, -1), rank(n, 1), weight(n, 0) {}
    
    int find(int x) {   
        if (root[x] == -1) return x;
        int p = root[x];
        int r = find(p);
        weight[x] += weight[p]; 
        return root[x] = r;
    }
    
    // enforce  a[u] ≡ a[v] + d 
    // returns false if that contradicts existing info
    bool merge(int u, int v, ll d) {  
        int ru = find(u), rv = find(v);
        if (ru != rv) {
            comp--;
            weight[ru] = weight[v] + d - weight[u];
            root[ru] = rv;
            return true;
        }
        return weight[u] - weight[v] == d;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
    }
    
    int get_rank(int x) {    
        return rank[find(x)];
    }
	
	int potential(int u) {
        find(u);
        return weight[u];
    }


    // returns (a[u] - a[v]), or -1 if unknown
    ll diff(int u, int v) {
        if (!same(u, v)) return -1;
        find(u);  
        find(v);
        return (ll)(weight[u] - weight[v]);
    }
};

struct Reachability_Tree { // 2 * n - 1 not 2 * n for total vertices
    struct DSU {
        vi p, r;
        DSU(int n): p(n), r(n, 0) { iota(all(p), 0); }
        int find(int x) {
            return p[x] == x ? x : p[x] = find(p[x]);
        }
        bool same(int a, int b) {
            return find(a) == find(b);
        }
        void merge(int a, int b) {
            a = find(a); b = find(b);
            if(a == b) return;
            p[b] = a;
            if(r[a] == r[b]) r[a]++;
        }
    };
    int n;
    vi weight;
    GRAPH<int> g;
    DSU root;
    int rt;
    vvi graph;
    Reachability_Tree(int n) : n(n), root(n * 2), rt(n - 1), graph(n * 2), weight(n * 2) {}

    void add_edge(int u, int v, int w = 0) {
        if(root.same(u, v)) return;
        rt++;
        weight[rt] = w;
        graph[rt].pb(root.find(u));
        graph[rt].pb(root.find(v));
        root.merge(rt, u);
        root.merge(rt, v);
    }

    bool built = false;
    void init(var(3)& edges) {
        built = true;
        sort(all(edges), [](const ar(3)& a, const ar(3)& b) {return a[2] < b[2];});
        for(auto& [u, v, w] : edges) {
            add_edge(u, v, w);
        }
        g = GRAPH<int>(graph, rt);
    }
 
    void build() {
        built = true;
        g = GRAPH<int>(graph, rt);
    }

    int lca_query(int u, int v) {
        if(!built) build();
        int c = g.lca(u, v);
        return weight[c];
    }
};

struct functional_graph {
    vi a;
    functional_graph(const vi& a) : a(a) {}

    ll run() {
        int n = a.size();
        vi ans(n, -1);
        vvi rev_graph(n);
        for(int i = 0; i < n; i++) {
            rev_graph[a[i]].pb(i);
        }
        auto floyd = [&](int src) -> void {
            int x = src, y = src;
            do {
                x = a[x];
                y = a[a[y]];
            } while(x != y);
            do {
                ans[x] = x;
                x = a[x];
            } while(x != y);
            auto fill = [&](auto& fill, int node) -> void {
                for(auto& nei : rev_graph[node]) {
                    if(ans[nei] == -1) {
                        ans[nei] = node;
                        fill(fill, nei);
                    }
                }
            };
            do {
                fill(fill, x);
                x = a[x];
            } while(x != y);
        };
        for(int i = 0; i < n; i++) {
            if(ans[i] == -1) floyd(i);
        }
        int res = 0;
        for(int i = 0; i < n; i++) {
            if(ans[i] == i) res++;
        }
        return res;
    }
};

class SCC {
    public:
    int n, curr_comp;
    vvi graph, revGraph;
    vi vis, comp, comp_cnt, in_degree, out_degree;
    stack<int> s;
 
    SCC(int n) {
        this->n = n;
        curr_comp = 0;
        graph.rsz(n), revGraph.rsz(n), vis.rsz(n), comp.rsz(n, -1), comp_cnt.rsz(n);
		// don't forget to build after adding edges
    }
 
    void add_directional_edge(int a, int b) {    
        graph[a].pb(b); 
        revGraph[b].pb(a);
    }
 
    void dfs(int node) {
        if(vis[node]) return;
        vis[node] = true;
        for(auto& nei : graph[node]) dfs(nei);
        s.push(node);
    }
 
    void dfs2(int node) {
        if(comp[node] != -1) return;
        comp[node] = curr_comp;
        comp_cnt[curr_comp]++;
        for(auto& nei : revGraph[node]) dfs2(nei);
    }
 
    void build() {
        for(int i = 0; i < n; i++) dfs(i);
        while(!s.empty()) {
            int node = s.top(); s.pop();
            if(comp[node] != -1) continue;
            dfs2(node);
            curr_comp++;
        }
    }
    
    vvi compress_graph() {    
        assert(in_degree.empty() && out_degree.empty());
        vvi g(curr_comp);   
        in_degree = out_degree = vi(curr_comp);
        for(int i = 0; i < n; i++) {    
            for(auto& j : graph[i]) {   
                if(comp[i] != comp[j]) {    
                    g[comp[i]].pb(comp[j]);
                }
            }
        }
        for(int i = 0; i < curr_comp; i++) {
            auto& it = g[i];
            srtU(it);
            for(auto& j : it) {
                in_degree[j]++;
                out_degree[i]++;
            }
        }
        return g;
    }
 
    vpii get_augment_edges() { // minimum edges added so start from any node, you can reach every other nodes
                               // call compress_graph() first
        assert(!in_degree.empty() && !out_degree.empty());
        vi sources, sinks;
        for (int c = 0; c < curr_comp; c++) {
            if (in_degree[c] == 0) sources.pb(c);
            if (out_degree[c] == 0) sinks.pb(c);
        }
        if (curr_comp == 1) return {};
        int S = sources.size(), T = sinks.size();
        int k = max(S, T);
        vi rep(curr_comp, -1);
        for (int i = 0; i < n; i++) {
            int c = comp[i];
            if (rep[c] == -1) rep[c] = i;
        }
        vpii ans;
        for (int i = 0; i < k; i++) {
            int fromC = sinks[i % T];
            int toC   = sources[(i + 1) % S];
            ans.pb({rep[fromC], rep[toC]});
        }
        return ans;
    }
 
    bool same(int u, int v) {
        return comp[u] == comp[v];
    }
 
    int get_size(int u) {
        return comp_cnt[comp[u]];
    }
};

template<typename T>
struct CD { // centroid_decomposition
    int n, root;
    vt<vt<T>> graph, G;
    vi size, parent, vis;
    ll ans;
    GRAPH<T> g;
    vi best;
    CD(const vt<vt<T>>& graph) : graph(graph), n(graph.size()), g(graph), best(graph.size(), inf), G(graph.size()) {
        ans = 0;
        size.rsz(n);
        parent.rsz(n, -1);
        vis.rsz(n);
        root = init();
    }
 
    void get_size(int node, int par) { 
        size[node] = 1;
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            get_size(nei, node);
            size[node] += size[nei];
        }
    }
 
    int get_center(int node, int par, int size_of_tree) { 
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            if(size[nei] * 2 > size_of_tree) return get_center(nei, node, size_of_tree);
        }
        return node;
    }

    int get_centroid(int src) { 
        get_size(src, -1);
        int centroid = get_center(src, -1, size[src]);
        vis[centroid] = true;
        return centroid;
    }

    int mx;
    void modify(int node, int par, int depth, int delta) {
        for(auto& nei : graph[node]) {
            if(vis[nei] || nei == par) continue;
            modify(nei, node, depth + 1, delta);
        }
    }

    void cal(int node, int par, int depth) {
        for(auto& nei : graph[node]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, node, depth + 1);
        }
    }
 
    int get_max_depth(int node, int par = -1, int depth = 0) {
        int max_depth = depth;
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            max_depth = max(max_depth, get_max_depth(nei, node, depth + 1));
        }
        return max_depth;
    }

    void run(int root, int par) {
        mx = get_max_depth(root, par);
        for(auto& nei : graph[root]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, root, 1);
            modify(nei, root, 1, 1);
        }
    }

    int init(int root = 0, int par = -1) {
        root = get_centroid(root);
        parent[root] = par;
        if(par != -1) G[par].pb(root);
        run(root, par);
        for(auto& nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            init(nei, root);
        }
        return root;
    }
	
	vi id;
    vvi big;
    vvll big_prefix;
    vvvi small;
    vt<vvll> small_prefix;
    bool by_weight = false;
    inline void build() {
        id.rsz(n, -1);
        big = vvi(n);
        big_prefix = vvll(n);
        small = vvvi(n);
        small_prefix = vt<vvll>(n);
        {
            auto dfs = [&](auto& dfs, int node, int par) -> void {
                int c = 0;
                for(auto& nei : G[node]) {
                    if(nei == par) continue;
                    assert(id[nei] == -1);
                    id[nei] = c++;
                    dfs(dfs, nei, node);
                }
                small[node].rsz(c);
                small_prefix[node].rsz(c);
            }; dfs(dfs, root, -1);
            vvi().swap(G);
        }
        for(int i = 0; i < n; i++) {
            int u = i;
            int prev = -1;
            while(u != -1) {
                ll d = by_weight ? g.dist_by_weight(u, i) : g.dist(u, i);
                big[u].pb(d);
                if(prev != -1) {
                    small[u][id[prev]].pb(d);
                }
                prev = u;
                u = parent[u];
            }
        }
        for(int i = 0; i < n; i++) {
            big[i].pb(-inf);
            srt(big[i]);
            big_prefix[i].rsz(big[i].size());
            const int N = small[i].size();
            for(int j = 0; j < N; j++) {
                small[i][j].pb(-inf);
                srt(small[i][j]);
                small_prefix[i][j].rsz(small[i][j].size());
            }
        }
        for(int i = 0; i < n; i++) {
            int u = i, prev = -1;
            while(u != -1) {
                ll d = by_weight ? g.dist_by_weight(u, i) : g.dist(u, i);
				int j = get_id(big[u], d);
				big_prefix[u][j] += d;
				if(prev != -1) {
					int jj = id[prev];
					int k = get_id(small[u][jj], d);
					small_prefix[u][jj][k] += d;
				}
                prev = u;
                u = parent[u];
            }
        }
        for(int i = 0; i < n; i++) {
            const int M = big_prefix[i].size();
            for(int j = 1; j < M; j++) {
                big_prefix[i][j] += big_prefix[i][j - 1];
            }
            for(auto& it : small_prefix[i]) {
                const int N = it.size();
                for(int j = 1; j < N; j++) {
                    it[j] += it[j - 1];
                }
            }
        }
    }

    ll count_less_or_equal_to(int node, ll w) {
        ll res = 0;
        int u = node, prev = -1;
        while(u != -1) {
            ll d = w - (by_weight ? g.dist_by_weight(u, node) : g.dist(u, node));
			int j = get_id(big[u], d + 1) - 1; 
			assert(j >= 0);
			pll now = {0, 0};
			now.ff += big_prefix[u][j];
			now.ss += j;
			if(prev != -1) {
				int jj = id[prev];
				int k = get_id(small[u][jj], d + 1) - 1;
				assert(k >= 0);
				now.ff -= small_prefix[u][jj][k];
				now.ss -= k;
			}
			res += d * now.ss - now.ff;
            prev = u;
            u = parent[u];
        }
        return res;
    }

    inline int get_id(const vi& a, int x) {
        return int(lb(all(a), x) - begin(a));
    }


	//    vi tin, tout, depth;
//    vvpii pos;
//    vvi coord;
//    inline void build() {
//        GRAPH<T> gg(G, root);
//        vvi a;
//        vt<vt<T>>().swap(G);
//        swap(tin, gg.tin);
//        swap(tout, gg.tout);
//        swap(depth, gg.depth);
//        coord = vvi(n);
//        for(int i = 0; i < n; i++) {
////            Tree[i].reset();
//            int u = i; 
//            while(u != -1) {
//                coord[u].pb(tin[i]);
//                u = parent[u];
//            }
//        }
//        pos = vvpii(n);
//        vvi val(n);
//        for(int i = 0; i < n; i++) {
//            srtU(coord[i]);
//            const int N = coord[i].size();
//            val[i].rsz(N);
//            pos[i] = vpii(depth[i] + 1);
//        }
//        for(int i = 0; i < n; i++) {
//            int u = i;
//            while(u != -1) {
//                pos[i][depth[u]] = {get_id(coord[u], tin[i]), get_id(coord[u], tout[i] + 1)};
//                val[u][pos[i][depth[u]].ff] = g.dist(u, i); // changing it back is ok as well
//                u = parent[u];
//            }
//        }
////        for(int i = 0; i < n; i++) {
////            Tree[i].init(val[i]);
////        }
//        vi().swap(tin);
//        vi().swap(tout);
//    }
//	
//    inline int get_id(const vi& a, int x) {
//        return int(lb(all(a), x) - begin(a));
//    }
//
//    void update(int node) {
//        int u = node;
//        while(u != -1) {
////            int id = pos[node][depth[u]].ff;
//            best[u] = min(best[u], g.dist(u, node));
//            u = parent[u];
//        }
//    }
//
//    int queries(int node, ll k) {
//        int res = inf, prev = -1, u = node;
//        while(u != -1){ 
////            if(prev == -1) { // can include whole subtree
////                ll cnt = Tree[u].count_less_or_equal_to(1, inf, k);
////                ll sm = Tree[u].sum_less_or_equal_to(1, inf, k);
////                res += k * cnt - sm;
////            } else {
////                auto [l, r] = pos[prev][depth[u]]; // don't include the previous children
////                int seg_size = coord[u].size();
////                l--;
////                ll off = g.dist_by_weight(u, node);
////                ll cnt = 0, sm = 0;
////                cnt += Tree[u].count_less_or_equal_to(1, l + 1, k - off);
////                sm  += Tree[u].sum_less_or_equal_to  (1, l + 1, k - off);
////                cnt += Tree[u].count_less_or_equal_to(r + 1, inf, k - off);
////                sm  += Tree[u].sum_less_or_equal_to  (r + 1, inf, k - off);
////                res += (k - off) * cnt - sm;
////            }
////            prev = u;
//            res = min(res, g.dist(u, node) + best[u]);
//            u = parent[u];
//        }
//        return res;
//    }
};

struct CYCLE {
    vvi graph;
    vi degree;
    int n;
    CYCLE(vvi &graph, vi& degree) : graph(graph), degree(degree) { n = graph.size(); }
 
    vi reconstruct_cycle(int u, int v, const vi &parent) {
        vi pathU, pathV;
        for (int cur = u; cur != -1; cur = parent[cur]) pathU.pb(cur);
        for (int cur = v; cur != -1; cur = parent[cur]) pathV.pb(cur);
        rev(pathU);
        rev(pathV);
        int idx = 0;
        while (idx < (int)pathU.size() && idx < (int)pathV.size() && pathU[idx] == pathV[idx]) idx++;
        idx--;
        vi cycle;
        for (int i = (int)pathU.size() - 1; i >= idx; i--) cycle.pb(pathU[i]);
        for (int i = idx + 1; i < (int)pathV.size(); i++) cycle.pb(pathV[i]);
        return cycle;
    }
 
    vi find_shortest_cycle(int s) {
        vi dis(n, inf), parent(n, -1);
        queue<int> q;
        dis[s] = 0;
        q.push(s);
        int bestShortest = inf;
        int candU_short = -1, candV_short = -1;
        while (!q.empty()){
            int u = q.front();
            q.pop();
            for (int v : graph[u]){
                if (dis[u] + 1 < dis[v]) {
                    dis[v] = dis[u] + 1;
                    parent[v] = u;
                    q.push(v);
                } else if (v != parent[u] && dis[u] != inf && dis[v] != inf) {
                    int currLength = dis[u] + dis[v] + 1;
                    if (currLength < bestShortest) {
                        bestShortest = currLength;
                        candU_short = u;
                        candV_short = v;
                    }
                }
            }
        }
        vi shortestCycle;
        if (candU_short != -1) shortestCycle = reconstruct_cycle(candU_short, candV_short, parent);
        return shortestCycle;
    }

	vi find_longest_cycle_bidirected_graph() {
        vi depth(n, 0), parent(n, -1);
        vi stk;
        int bestLen = 0, bestU = -1, bestV = -1;

        auto dfs2 = [&](auto& dfs2, int u, int p) -> void {
            depth[u] = (int)stk.size() + 1;
            stk.push_back(u);
            for (int v : graph[u]) {
                if (v == p) continue;
                if (!depth[v]) {
                    parent[v] = u;
                    dfs2(dfs2, v, u);
                } else {
                    int len = depth[u] - depth[v] + 1;
                    if (len > bestLen) {
                        bestLen = len;
                        bestU = u;
                        bestV = v;
                    }
                }
            }
            stk.pop_back();
        };

        for (int i = 0; i < n; ++i) {
            if (!depth[i]) dfs2(dfs2, i, -1);
        }
        if (bestLen == 0) return {};
        vi cycle;
        int u = bestU;
        while (u != bestV) {
            cycle.pb(u);
            u = parent[u];
        }
        cycle.pb(bestV);
        rev(cycle);
        return cycle;
    }

    vi get_max_independent_set() {
        vb marked(n, false), visited(n, false);
        auto dfs = [&](auto& dfs, int u) -> void {
            visited[u] = true;
            for (int v : graph[u]) if (!visited[v]) dfs(dfs, v);
            if (!marked[u]) {
                for (int v : graph[u]) marked[v] = true;
            }
        };
        dfs(dfs, 0);
        vi res;
        for (int u = 0; u < n; ++u) if (!marked[u]) res.pb(u);
        return res;
    }

    vi find_longest_cycle_directed_graph() {
        vi color(n, 0), parent(n, -1), depth(n, 0);
        int bestLen = 0, bestU = -1, bestV = -1;
        auto dfs = [&](auto& dfs, int u) ->void {
            color[u] = 1;
            for (int v : graph[u]) {
                if (color[v] == 0) {
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    dfs(dfs, v);
                } else if (color[v] == 1) {
                    int len = depth[u] - depth[v] + 1;
                    if (len > bestLen) {
                        bestLen = len;
                        bestU = u;
                        bestV = v;
                    }
                }
            }
            color[u] = 2;
        };
        for (int i = 0; i < n; ++i) {
            if (color[i] == 0) {
                depth[i] = 0;
                dfs(dfs, i);
            }
        }
        if (bestLen == 0) return {};
        vi cycle;
        int cur = bestU;
        while (true) {
            cycle.pb(cur);
            if (cur == bestV) break;
            cur = parent[cur];
        }
        rev(cycle);
        return cycle;
    }

    vi longest_simple_path() { // return longest path where each vertex is visited once in a DIRECTED GRAPH
        queue<int> q;
        vi vis(n);
        for(int i = 0; i < n; i++) {
            if(degree[i] == 0) {
                q.push(i);
                vis[i] = true;
            }
        }
        while(!q.empty()) {
            auto node = q.front(); q.pop();
            vis[node] = true;
            for(auto& nei : graph[node]) {
                if(--degree[nei] == 0) q.push(nei);
            }
        }
        vi cycle(n);
        for(int i = 0; i < n; i++) {
            if(!vis[i]) {
                cycle[i] = true;
            }
        }
        DSU root(n);
        for(int i = 0; i < n; i++) {
            if(cycle[i]) {
                for(auto& j : graph[i]) {
                    if(cycle[j]) root.merge(i, j);
                }
            }
        }    
        vi dp(n, -1);
        vi next(n, -1);
        auto dfs = [&](auto& dfs, int node) -> int {
            auto& res = dp[node];
            if(res != -1) return res;
            if(cycle[node]) {
                return res = root.get_rank(node);
            }
            res = 1;
            for(auto& nei : graph[node]) {
                int v = dfs(dfs, nei) + 1;
                if(v > res) {
                    res = v;
                    next[node] = nei;
                }
            }
            return res;
        };
        for(int i = 0; i < n; i++) dfs(dfs, i);
        int mx = max_element(all(dp)) - begin(dp);
        vi path;
        int node = mx;
        while(node != -1) {
            path.pb(node);
            if(cycle[node]) {
                int u = node;
                while(true) {
                    int nxt = -1;
                    for(auto& nei : graph[node]) {
                        if(root.same(nei, node)) {
                            nxt = nei;
                            break;
                        }
                    }  
                    if(nxt == u) break;
                    node = nxt;
                    path.pb(node);
                } 
                break;
            }
            node = next[node];
        }
        return path;
    }
};

struct Tarjan {
    int n, m;
    vi tin, low, belong, bridges, size;
    vvpii graph;
    stack<int> s;
    int timer, comp;
    Tarjan(const vvpii& graph, int edges_size) : m(edges_size), graph(graph), timer(0), n(graph.size()), comp(0) {
        tin.rsz(n);
        low.rsz(n);
        bridges.rsz(edges_size);
        belong.rsz(n);
        for(int i = 0; i < n; i++) {
            if(!tin[i]) {
                dfs(i);
            }
        }
    }

    void dfs(int node = 0, int par = -1) {
        s.push(node);
        low[node] = tin[node] = ++timer;
        for(auto& [nei, id] : graph[node]) {  
            if(id == par) continue;
            if(!tin[nei]) {   
                dfs(nei, id);
                low[node] = min(low[node], low[nei]);   
                if(low[nei] > tin[node]) {  
                    bridges[id] = true;
                }
            }
            else {  
                low[node] = min(low[node], tin[nei]);
            }
        }
        if(low[node] == tin[node]) {
            int now = comp++;
            int cnt = 0;
            while(true) {
                int u = s.top(); s.pop();
                belong[u] = now;
                cnt++;
                if(u == node) break;
            }
            size.pb(cnt);
        }
    };

    vvpii compress_graph(vpii edges) { // return a root of 1 degree and a bridge graph
        assert(m == edges.size());
        vi degree(comp);
        vvpii G(comp);
        for(int i = 0; i < m; i++) {
            if(!bridges[i]) continue;
            auto& [u, v] = edges[i];
            u = belong[u], v = belong[v];
            assert(u != v);
            G[u].pb({v, i});
            G[v].pb({u, i});
            degree[u]++;
            degree[v]++;
        }
        return G;
    }

    void orientInternalEdges(vpii &edges) { // directed edges in same component to create a complete cycle
        vb oriented(m);
        vi disc(n, 0);
        int t2 = 0;
        auto dfs2 = [&](auto& dfs2, int u) -> void {
            for(auto &[v, id] : graph[u]) {
                if(bridges[id] || oriented[id]) continue;
                if(!disc[v]) {
                    edges[id] = {u, v};
                    oriented[id] = true;
                    disc[v] = ++t2;
                    dfs2(dfs2, v);
                } else {
                    if(disc[u] > disc[v]) edges[id] = {u, v};
                    else edges[id] = {v, u};
                    oriented[id] = true;
                }
            }
        };
        for(int u = 0; u < n; u++) {
            if(!disc[u]) {
                disc[u] = ++t2;
                dfs2(dfs2, u);
            }
        }
    }

    void directed_edge(vpii& edges) {
        // https://codeforces.com/contest/2113/problem/F
        orientInternalEdges(edges);
        auto tree = compress_graph(edges);
        int M = tree.size();
        vi vis(M);
        auto dfs = [&](auto& dfs, int node, int par) -> void {
            vis[node] = true;
            for(auto& [nei, id] : tree[node]) {
                if(nei == par) continue;
                auto& [u, v] = edges[id];
                if(belong[u] != node) swap(u, v);
                dfs(dfs, nei, node);
            }
        };
        for(int i = 0; i < M; i++) {
            if(!vis[i]) dfs(dfs, i, -1);
        }
    }

    bool get_group_flag = false;
    vvi get_group() {
        assert(!get_group_flag);
        get_group_flag = true;
        vvi group(comp);
        for(int i = 0; i < n; i++) {
            group[belong[i]].pb(i);
        }
        return group;
    }
};

struct block_cut_tree {
    int n, m, timer; 
    vvpii graph;
    vi tin, low, id, is_art;
    vvi comps, comp_vertices;
    vpii edges;
    vvi tree;
    vi is_simple_cycle;
    stack<int> s;
    block_cut_tree(int n, const vpii& edges) : edges(edges), n(n), m(edges.size()) {
        tin.rsz(n), low.rsz(n), id.rsz(n, -1), is_art.rsz(n), graph.rsz(n);
        for(int i = 0; i < m; i++) {
            auto& [u, v] = edges[i];
            graph[u].pb({v, i});
            graph[v].pb({u, i});
        }
        for(int i = 0; i < n; i++) {
            if(!tin[i]) dfs(i, -1);
        }
        build();
        // buildBCT();
    }

    void dfs(int node, int prev_id) {
        tin[node] = low[node] = ++timer;
        int child = 0;
        for(auto& [nei, j] : graph[node]) {
            if(j == prev_id) continue;
            if(!tin[nei]) {
                child++;
                s.push(j);
                dfs(nei, j);
                low[node] = min(low[node], low[nei]);
                if((prev_id == -1 && child > 1) || (prev_id != -1 && low[nei] >= tin[node])) is_art[node] = true;
                if(low[nei] >= tin[node]) {
                    comps.pb({});
                    auto& curr = comps.back();
                    while(true) {
                        int e = s.top(); s.pop();
                        curr.pb(e);
                        if(e == j) break;
                    }
                }
            } else if(tin[nei] < tin[node]) {
                s.push(j);
                low[node] = min(low[node], tin[nei]);
            }
        }
    }

    void buildBCT() {
        // https://codeforces.com/contest/487/problem/E
        // build a bipartile graph city -> supernode -> city -> supernode ...
        int C = comps.size();
        int N = n + C;
        tree.assign(N, {});
        for(int i = 0; i < C; i++) {
            int compNode = n + i;
            auto& comp = comps[i];
            vi verts;
            for(int eid : comp) {
                auto &[u, v] = edges[eid];
                verts.pb(u);
                verts.pb(v);
            }
            srtU(verts);
            for(int u : verts) {
                tree[compNode].pb(u);
                tree[u].pb(compNode);
            }
        }
    }

    void build() {
		// bipartile of comp -> articulation point -> comp -> articulation point
        int B = comps.size(); 
        is_simple_cycle.rsz(m);
        comp_vertices = vvi(B);
        int c = 0;
        for(int i = 0; i < n; i++) {
            if(is_art[i]) {
                id[i] = c++;
            }
        }
        int nc = c;
        for(int i = 0; i < B; i++) {
            int bn = nc++;
            for(auto& j : comps[i]) {
                auto& [u, v] = edges[j];
                comp_vertices[i].pb(u);
                comp_vertices[i].pb(v);
            }
            srtU(comp_vertices[i]);
            if(comps[i].size() == comp_vertices[i].size()) {
                for(auto& j : comps[i]) {
                    is_simple_cycle[j] = true;
                }
            }
            for(auto& j : comp_vertices[i]) {
                if(id[j] == -1) id[j] = bn; 
            }
        }
        for(int i = 0; i < n; i++) {
            if(id[i] == -1) {
                comp_vertices.pb({i});
                id[i] = nc++;
            }
        }
        tree.rsz(nc);
        for(int i = 0; i < (int)comp_vertices.size(); i++) {
            int bn = c++;
            for(auto& u : comp_vertices[i]) {
                if(is_art[u]) {
                    int j = id[u];
                    tree[j].pb(bn);
                    tree[bn].pb(j);
                } 
            }
        }
    }
};

struct two_sat {
    int N = 0;
    vpii edges;

    two_sat() {}

    two_sat(int n) : N(n) {}

    int addVar() {
        return N++;
    }

    void either(int x, int y) {
        edges.emplace_back(x, y);
    }

    void implies(int x, int y) {
        either(x ^ 1, y);
    }

    void must(int x) {
        either(x, x);
    }

    void add_clause(int a, bool sA, int b, bool sB) { // at least one is true
        int A = 2 * a + (sA ? 1 : 0);
        int B = 2 * b + (sB ? 1 : 0);
        either(A, B);
    }

    void add_or(int a, int b, bool v) {
        if (v) {
            add_clause(a, true, b, true);
        } else {
            add_clause(a, false, a, false);
            add_clause(b, false, b, false);
        }
    }

    void add_xor(int a, int b, bool v) {
        if (!v) {
            add_clause(a, false, b, true);
            add_clause(a, true,  b, false);
        } else {
            add_clause(a, false, b, false);
            add_clause(a, true,  b, true);
        }
    }

    void add_and(int a, int b, bool v) {
        if (v) {
            add_clause(a, true,  a, true);
            add_clause(b, true,  b, true);
        } else {
            add_clause(a, false, b, false);
        }
    }

    void at_most_one(const vi& l) {
        if (l.size() <= 1) return;
        int cur = l[0] ^ 1;
        for (int i = 2; i < (int)l.size(); ++i) {
            int aux = addVar();
            int at  = 2 * aux + 1;
            either(cur, l[i] ^ 1);
            either(cur, at);
            either(l[i] ^ 1, at);
            cur = at ^ 1;
        }
        either(cur, l[1] ^ 1);
    }

    vt<bool> satisfy() {
        int V = 2 * N;
        vvi adj(V), radj(V);
        for (auto& e : edges) {
            int u = e.first, v = e.second;
            adj[u ^ 1].pb(v);
            adj[v ^ 1].pb(u);
            radj[v].pb(u ^ 1);
            radj[u].pb(v ^ 1);
        }
        vi order; order.reserve(V);
        vector<char> used(V, 0);
        auto dfs1 = [&](auto& dfs1, int u) -> void {
            used[u] = 1;
            for(int w : adj[u]) if(!used[w]) dfs1(dfs1, w);
            order.pb(u);
        };
        for(int i = 0; i < V; ++i) if(!used[i]) dfs1(dfs1, i);
        vi comp(V, -1);
        int cid = 0;
        auto dfs2 = [&](auto& dfs2, int u) -> void {
            comp[u] = cid;
            for(int w : radj[u]) if(comp[w] == -1) dfs2(dfs2, w);
        };
        for(int i = V - 1; i >= 0; --i) {
            int u = order[i];
            if(comp[u] == -1) {
                dfs2(dfs2, u);
                cid++;
            }
        }
        vt<bool> res(N);
        for(int i = 0; i < N; ++i) {
            if(comp[2 * i] == comp[2 * i + 1]) return {};
            res[i] = comp[2 * i] > comp[2 * i + 1];
        }
        return res;
    }
};

template<int M>
struct max_clique { // maximum independent set is max clique on the complement graph
    static const int LIMIT = 1000;
    using mask = bitset<M>;

    int n;
    mask adj[M];
    ll maximal_count;
    bool too_many;
    int best;
    bool stop;
    vi dp, cur, ans;

    max_clique(int _n)
        : n(_n), maximal_count(0), too_many(false), best(0), stop(false)
    {
        for (int i = 0; i < n; ++i)
            adj[i].reset();
    }

    void add_edge(int u, int v) {
        adj[u].set(v);
        adj[v].set(u);
    }

    int ctz(const mask &b) const {
        for(int i = 0; i < n; ++i)
            if(b.test(i))
                return i;
        return n;
    }

    int solve() {
        best = 0;
        stop = false;
        mask R; R.reset();
        mask P; P.reset(); for (int i = 0; i < n; ++i) P.set(i);
        mask X; X.reset();
        bronk(R, P, X);
        return best;
    }

    void bronk(mask R, mask P, mask X) {
        if(stop) return;
        if(P.none() && X.none()) {
            int sz = (int)R.count();
            if(sz > best) {
                best = sz;
                ans = cur;                // RECORD best clique
            }
            return;
        }
        mask PX = P | X;
        int pivot = ctz(PX), maxCnt = -1;
        for(int u = pivot; u < n; ++u) {
            if(!PX.test(u)) continue;
            int c = (P & adj[u]).count();
            if(c > maxCnt) {
                maxCnt = c;
                pivot = u;
            }
        }
        mask can = P & (~adj[pivot]);
        while(can.any()) {
            int v = ctz(can);
            can.reset(v);

            cur.pb(v);
            bronk(R | mask().set(v), P & adj[v], X & adj[v]);
            cur.pop_back();

            P.reset(v);
            X.set(v);
        }
    }

    vector<int> get_maximum_clique() {
        solve();
        return ans;
    }

    void bronk_count(mask R, mask P, mask X) {
        if(too_many) return;
        if(P.none() && X.none()) {
            if (++maximal_count > LIMIT) too_many = true;
            return;
        }
        mask PX = P | X;
        int pivot = ctz(PX);
        int maxCnt = -1;
        for(int u = pivot; u < n; ++u) {
            if(!PX.test(u)) continue;
            mask inter = P & adj[u];
            int c = inter.count();
            if (c > maxCnt) {
                maxCnt = c;
                pivot = u;
            }
        }
        mask can = P & (~adj[pivot]);
        while(can.any() && !too_many) {
            int v = ctz(can);
            mask bit; bit.reset(); bit.set(v);
            can.reset(v);
            bronk_count(R | bit, P & adj[v], X & adj[v]);
            P.reset(v);
            X.set(v);
        }
    }

    ll count_maximal_cliques() {
        maximal_count = 0;
        too_many = false;
        mask R; R.reset();
        mask P; P.reset(); for (int i = 0; i < n; ++i) P.set(i);
        mask X; X.reset();
        bronk_count(R, P, X);
        return too_many ? -1 : maximal_count;
    }
};

template<typename T>
struct rerooting {
    int n;
    vt<vt<T>> graph;
    vll ans, dp;
    rerooting(const vt<vt<T>>& _graph, int root = 0) : graph(_graph), n(_graph.size()) {
        dp.rsz(n);
        ans.rsz(n);
		dfs1(root, -1);
        dfs2(root, -1);
    }

    void dfs1(int node, int par) {
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            dfs1(nei, node);
            // merging
        }
    }

    void dfs2(int node, int par) {
        ans[node] = dp[node];
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            // subtract contribution
            dfs2(nei, node);
            // add back contribution
        }
    }
};

template<typename T = int>
struct virtual_tree {
    GRAPH<T> g;
    using info = pair<int, ll>;
    vt<vt<info>> graph; // [node, dist]
    bool dist_by_weight;
    vi subtree, importance;
    int total;
    ll ans = 0;
    virtual_tree(const vt<vt<T>>& _graph, bool _dist_by_weight = false) : g(_graph), graph(_graph.size()), dist_by_weight(_dist_by_weight), subtree(_graph.size()), importance(_graph.size()) {}

    int build(vi& vertices) {
        int n = vertices.size();
        auto cmp = [&](const int& a, const int& b) -> bool {
            return g.tin[a] < g.tin[b];
        };
        sort(all(vertices), cmp);
        auto a(vertices);
        for(int i = 0; i < n - 1; i++) {
            int u = vertices[i], v = vertices[i + 1];
            a.pb(g.lca(u, v));
        }
        sort(all(a), cmp);
        a.erase(unique(all(a)), end(a));
        total = vertices.size();
        for(auto& u : a) {
            vt<info>().swap(graph[u]);
            subtree[u] = 0; 
            importance[u] = false;
        }
        for(auto& u : vertices) {
            importance[u] = true;
        }
        vi s;
        s.pb(a[0]);
        for(int i = 1; i < (int)a.size(); i++) {
            int u = a[i];
            while(!s.empty() && !g.is_ancestor(s.back(), u)) s.pop_back();
            int p = s.back();
            ll d = dist_by_weight ? g.dist_by_weight(p, u) : g.dist(p, u);
            graph[p].pb({u, d});
            s.pb(u);
        }
        return s[0];
    }

    ll dfs(int node, int par) { // return all pair shortest dist total sum
        subtree[node] = importance[node]; 
        ll ans = 0;
        for(auto& [nei, w] : graph[node]) {
            if(nei == par) continue;
            ans += dfs(nei, node);
            subtree[node] += subtree[nei];
            ans += (ll)subtree[nei] * (total - subtree[nei]) * w;
        }
        return ans;
    }
};

struct EulerianPath {
    int nodes, edges;
    bool directed;
    vvpii graph;
    vi deg, indeg, outdeg;
    vt<bool> used;
    vi ans_edges, ans_nodes;

    EulerianPath(int _nodes, bool _directed = false)
      : nodes(_nodes), edges(0), directed(_directed), graph(_nodes) {
        if(directed) indeg.assign(nodes,0), outdeg.assign(nodes,0);
        else deg.assign(nodes,0);
    }

    void add_edge(int u, int v, int id) {
        graph[u].emplace_back(v, id);
        edges++;
        if(directed) {
            outdeg[u]++;
            indeg[v]++;
        } else {
            graph[v].emplace_back(u, id);
            deg[u]++;
            deg[v]++;
        }
    }

    int find_start() const {
        int start = -1;
        if(!directed) {
            int odd = 0;
            for(int i = 0; i < nodes; i++) {
                if(deg[i] & 1) {
                    odd++;
                    start = i;
                }
                if(start < 0 && deg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(odd != 0 && odd != 2) return -1;
        } else {
            int plus1 = 0, minus1 = 0;
            for(int i = 0; i < nodes; i++) {
                int d = outdeg[i] - indeg[i];
                if(d == 1) { plus1++; start = i; }
                else if(d == -1) minus1++;
                else if(d != 0) return -1;
                if(start < 0 && outdeg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(!((plus1 == 1 && minus1 == 1) || (plus1 == 0 && minus1 == 0))) return -1;
        }
        return start;
    }

    void dfs(int u) {
		if(used.empty()) {
			used.rsz(edges);
		}
        while(!graph[u].empty()) {
            auto [v, id] = graph[u].back();
            graph[u].pop_back();
			while((int)used.size() <= id) used.pb(0);
            if(used[id]) continue;
            used[id] = true;
            dfs(v);
            ans_edges.pb(id);
        }
        ans_nodes.pb(u);
    }

    pair<vi, vi> get_path() {
        int start = find_start();
        if(start < 0) return {};
        used.rsz(edges);
        dfs(start);
        if((int)ans_edges.size() != edges) return {};
        rev(ans_nodes);
        rev(ans_edges);
        return {ans_nodes, ans_edges};
    }

    vvi get_all_cycle() {
        // https://oj.uz/problem/view/BOI14_postmen
        dfs(0);
        vi vis(nodes), curr;
        vvi ans;
        for(auto& x : ans_nodes) {
            if(vis[x]) {
                vi now;
                now.pb(x);
                while(!curr.empty() && curr.back() != x) {
                    int v = curr.back();
                    vis[v] = false;
                    now.pb(v);
                    curr.pop_back();
                }
                curr.pop_back();
                ans.pb(now);
            }
            curr.pb(x);
            vis[x] = true;
        }
        return ans;
    }
};

// Warning: when choosing flow_t, make sure it can handle the sum of flows, not just individual flows.
template<typename flow_t>
struct dinic {
    struct edge {
        int node, _rev;
        flow_t capacity;
 
        edge() {}
 
        edge(int _node, int _rev, flow_t _capacity) : node(_node), _rev(_rev), capacity(_capacity) {}
    };
 
    int V = -1;
    vt<vt<edge>> adj;
    vi dist, edge_index;
    vt<vt<flow_t>> _cap_snapshot;
    flow_t        _flow_snapshot;
    bool flow_called;
 
    dinic(int vertices = -1) {
        if (vertices >= 0)
            init(vertices);
    }
 
    void init(int vertices) {
        V = vertices;
        adj.assign(V, {});
        dist.resize(V);
        edge_index.resize(V);
        flow_called = false;
    }
 
    int _add_edge(int u, int v, flow_t capacity1, flow_t capacity2) {
        assert(0 <= u && u < V && 0 <= v && v < V);
        assert(capacity1 >= 0 && capacity2 >= 0);
        edge uv_edge(v, int(adj[v].size()) + (u == v ? 1 : 0), capacity1);
        edge vu_edge(u, int(adj[u].size()), capacity2);
        adj[u].push_back(uv_edge);
        adj[v].push_back(vu_edge);
        return adj[u].size() - 1;
    }
 
    int add_directional_edge(int u, int v, flow_t capacity) {
        return _add_edge(u, v, capacity, 0);
    }
 
    int add_bidirectional_edge(int u, int v, flow_t capacity) {
        return _add_edge(u, v, capacity, capacity);
    }
 
    edge &reverse_edge(const edge &e) {
        return adj[e.node][e._rev];
    }
 
    void bfs_check(queue<int> &q, int node, int new_dist) {
        if (new_dist < dist[node]) {
            dist[node] = new_dist;
            q.push(node);
        }
    }
 
    bool bfs(int source, int sink) {
        dist.assign(V, inf);
        queue<int> q;
        bfs_check(q, source, 0);
        while (!q.empty()) {
            int top = q.front(); q.pop();
            for (edge &e : adj[top])
                if (e.capacity > 0)
                    bfs_check(q, e.node, dist[top] + 1);
        }
 
        return dist[sink] < inf;
    }
 
    flow_t dfs(int node, flow_t path_cap, int sink) {
        if (node == sink)
            return path_cap;
 
        if (dist[node] >= dist[sink])
            return 0;
 
        flow_t total_flow = 0;
 
        // Because we are only performing DFS in increasing order of dist, we don't have to revisit fully searched edges
        // again later.
        while (edge_index[node] < int(adj[node].size())) {
            edge &e = adj[node][edge_index[node]];
 
            if (e.capacity > 0 && dist[node] + 1 == dist[e.node]) {
                flow_t path = dfs(e.node, min(path_cap, e.capacity), sink);
                path_cap -= path;
                e.capacity -= path;
                reverse_edge(e).capacity += path;
                total_flow += path;
            }
 
            // If path_cap is 0, we don't want to increment edge_index[node] as this edge may not be fully searched yet.
            if (path_cap == 0)
                break;
 
            edge_index[node]++;
        }
 
        return total_flow;
    }
 
    flow_t total_flow = 0;
    flow_t flow(int source, int sink) {
        assert(V >= 0);
 
        while (bfs(source, sink)) {
            edge_index.assign(V, 0);
            total_flow += dfs(source, inf, sink);
        }
 
        flow_called = true;
        return total_flow;
    }
 
    vector<bool> reachable;
 
    void reachable_dfs(int node) {
        reachable[node] = true;
 
        for (edge &e : adj[node])
            if (e.capacity > 0 && !reachable[e.node])
                reachable_dfs(e.node);
    }
 
    // Returns a list of {capacity, {from_node, to_node}} representing edges in the min cut.
    // TODO: for bidirectional edges, divide the resulting capacities by two.
    vector<pair<flow_t, pii>> min_cut(int source) {
        assert(flow_called);
        reachable.assign(V, false);
        reachable_dfs(source);
        vector<pair<flow_t, pii>> cut;
        for (int node = 0; node < V; node++)
            if (reachable[node])
                for (edge &e : adj[node])
                    if (!reachable[e.node])
                        cut.emplace_back(reverse_edge(e).capacity, make_pair(node, e.node));
 
        return cut;
    }
	
	vt<vt<flow_t>> assign_flow(int n) {
        vt<vt<flow_t>> assign(n, vt<flow_t>(n));   
        for(int i = 0; i < n; i++) {
            for(auto& it : adj[i]) {
                int j = it.node - n;
                auto e = reverse_edge(it);
                if(j >= 0 && j < n) {
                    assign[i][j] = e.capacity;
                }
            }
        }
        return assign;
    }
	
	vvi construct_path(int n, vi& a) {
        vi vis(n), A;
        vvi ans, G(n);

        auto dfs = [&](auto& dfs, int node) -> void {
            vis[node] = true;
            A.pb(node + 1); 
            for(auto& nei : G[node]) {
                if(!vis[nei]) {
                    dfs(dfs, nei);
                    return;
                }
            }
        };
        for(int i = 0; i < n; i++) {
            if(a[i] % 2 == 0) continue; // should only add node where going from source to this
            for(auto& it : adj[i]) {
                int j = it.node;
                if(j < n && it.capacity == 0) {
                    G[i].pb(j);
                    G[j].pb(i);
                }
            }
        }
        for(int i = 0; i < n; i++) {
            if(vis[i]) continue;
            A.clear();
            dfs(dfs, i);
            ans.pb(A);
        }
        return ans;
    }
	
	vpii construct_flow(int n, int m) { // max matching
        vpii matching;
        for (int u = 0; u < n; ++u) {
            for (auto &e : adj[u]) {
                int v = e.node;
                if (v >= n && v < n + m && e.capacity == 0) {
                    matching.emplace_back(u, v - n);
                }
            }
        }
        return matching;
    }

    vpii construct_min_vertex_cover(int n_left, int n_right, int src) {
        reachable.assign(V, false);
        reachable_dfs(src);
        vpii cover; // type 1 is picking left, type 2 is picking right
        for (int u = 0; u < n_left; ++u) {
            if (!reachable[u]) 
                cover.emplace_back(1, u);
        }
        for (int j = 0; j < n_right; ++j) {
            if (reachable[n_left + j])
                cover.emplace_back(2, j);
        }
        return cover;
    }

	// edges: vector of {from, edge_idx, element}
    vi construct_missing_flow(const var(3)& edges) const { // https://codeforces.com/contest/1783/problem/F
        vi ans;
        for (auto& [u, idx, element] : edges) {
            if (adj[u][idx].capacity > 0) ans.pb(element);
        }
        return ans;
    }

    void snapshot() {
        vvi().swap(_cap_snapshot);
        _cap_snapshot.rsz(V);
        for (int u = 0; u < V; ++u) {
            for (auto &e : adj[u]) {
                _cap_snapshot[u].pb(e.capacity);
            }
        }
        _flow_snapshot = total_flow;
    }

    void roll_back() { // https://codeforces.com/gym/101873/my
        for (int u = 0; u < V; ++u) {
            for (int i = 0; i < (int)adj[u].size(); ++i) {
                adj[u][i].capacity = _cap_snapshot[u][i];
            }
        }
        total_flow   = _flow_snapshot;
        flow_called  = false;
    }
};

template<typename T>
struct MCMF {
    private:
    int V;
    struct Edge {
        int to, _rev;
        T capacity, cost;
        Edge() {}

        Edge(int to, int _rev, T capacity, T cost) : to(to), _rev(_rev), capacity(capacity), cost(cost) {}
    };

    void add_edge(int u, int v, T capacity, T cost) {
        Edge a(v, int(graph[v].size()), capacity, cost);
        Edge b(u, int(graph[u].size()), 0, -cost);
        graph[u].pb(a);
        graph[v].pb(b);
    }
    public:

    vt<vt<Edge>> graph;
    MCMF(int V) : V(V), graph(V) {}

    void add_directional_edge(int u, int v, T capacity, T cost) {
        add_edge(u, v, capacity, cost);
    }

    void add_bidirectional_edge(int u, int v, T capacity, T cost) {
        add_edge(u, v, capacity, cost);
        add_edge(v, u, capacity, cost);
    }

    pair<T,T> min_cost_flow(int s, int t, T max_f = numeric_limits<T>::max()) {
        T flow = 0, flow_cost = 0;
        vi prev_v(V), prev_e(V);
        vt<T> dist(V);
        vb inq(V);
        const T INF_T = numeric_limits<T>::max();
        while (flow < max_f) {
            fill(dist.begin(), dist.end(), INF_T);
            fill(inq.begin(), inq.end(), false);
            queue<int> q;
            dist[s] = 0; inq[s] = true; q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop(); inq[u] = false;
                for (int i = 0; i < graph[u].size(); i++) {
                    auto &e = graph[u][i];
                    if (e.capacity > 0 && dist[e.to] > dist[u] + e.cost) {
                        dist[e.to] = dist[u] + e.cost;
                        prev_v[e.to] = u;
                        prev_e[e.to] = i;
                        if (!inq[e.to]) {
                            inq[e.to] = true;
                            q.push(e.to);
                        }
                    }
                }
            }
            if (dist[t] == INF_T) break;
            T df = max_f - flow;
            for (int v = t; v != s; v = prev_v[v]) {
                auto &e = graph[prev_v[v]][prev_e[v]];
                df = min(df, e.capacity);
            }
            flow += df;
            flow_cost += df * dist[t];
            for (int v = t; v != s; v = prev_v[v]) {
                auto &e = graph[prev_v[v]][prev_e[v]];
                e.capacity -= df;
                graph[v][e._rev].capacity += df;
            }
        }
        return {flow, flow_cost};
    }

    vpii construct_flow(int n, int m) const {
        vpii matching;
        for(int u = 0; u < n; ++u) {
            for(auto const &e : graph[u]) {
                if(e.to >= n && e.to < n + m && e.capacity == 0) {
                    matching.emplace_back(u, e.to - n);
                }
            }
        }
        return matching;
    }
};

struct Kuhn { // great for force matching, example mex-matching
    int n, L, tot;
    vvi adj;
    vi mate, vis;
    int stamp = 1;

    Kuhn(int N, int left)
        : n(N), L(left), tot(N + left),
          adj(tot), mate(tot, -1), vis(tot, 0) {}

    void add_edge(int u, int v) {
        v += n; // offset 
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    bool dfs(int v) {
        if (vis[v] == stamp) return false;
        vis[v] = stamp;
        for (int w : adj[v]) {
            if (vis[w] == stamp) continue;
            vis[w] = stamp;
            if (mate[w] == -1 || dfs(mate[w])) {
                mate[v] = w;
                mate[w] = v;
                return true;
            }
        }
        return false;
    }

    bool force(int u) { // force a match with u
        ++stamp;
        return dfs(u);
    }

    int max_match() {
        int matched = 0;
        bool progress = true;
        while (progress) {
            progress = false;
            for (int v = 0; v < L; ++v)
                if (mate[v] == -1 && force(v)) {
                    ++matched;
                    progress = true;
                }
        }
        return matched;
    }
};

struct Blossom {
    int n;
    vi match, Q, pre, base, hash, in_blossom, in_path;
    vvi adj;
    Blossom(int n) : n(n), match(n, -1), adj(n, vi(n)), hash(n), Q(n), pre(n), base(n), in_blossom(n), in_path(n) {}

    void insert(const int &u, const int &v) {
        adj[u][v] = adj[v][u] = 1;
    }

    int max_match() {
        fill(all(match), -1);
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (match[i] == -1) ans += bfs(i);
        }
        return ans;
    }

    int bfs(int p) {
        fill(all(pre), -1);
        fill(all(hash), 0);
        iota(all(base), 0);
        Q[0] = p;
        hash[p] = 1;
        for (int s = 0, t = 1; s < t; ++s) {
            int u = Q[s];
            for (int v = 0; v < n; ++v) {
                if (adj[u][v] && base[u] != base[v] && v != match[u]) {
                    if (v == p || (match[v] != -1 && pre[match[v]] != -1)) {
                        int b = contract(u, v);
                        for (int i = 0; i < n; ++i) {
                            if (in_blossom[base[i]]) {
                                base[i] = b;
                                if (hash[i] == 0) {
                                    hash[i] = 1;
                                    Q[t++] = i;
                                }
                            }
                        }
                    } else if (pre[v] == -1) {
                        pre[v] = u;
                        if (match[v] == -1) {
                            argument(v);
                            return 1;
                        } else {
                            Q[t++] = match[v];
                            hash[match[v]] = 1;
                        }
                    }
                }
            }
        }
        return 0;
    }

    void argument(int u) {
        while (u != -1) {
            int v = pre[u];
            int k = match[v];
            match[u] = v;
            match[v] = u;
            u = k;
        }
    }

    void change_blossom(int b, int u) {
        while (base[u] != b) {
            int v = match[u];
            in_blossom[base[v]] = in_blossom[base[u]] = true;
            u = pre[v];
            if (base[u] != b) {
                pre[u] = v;
            }
        }
    }

    int contract(int u, int v) {
        fill(all(in_blossom), 0);
        int b = find_base(base[u], base[v]);
        change_blossom(b, u);
        change_blossom(b, v);
        if (base[u] != b) pre[u] = v;
        if (base[v] != b) pre[v] = u;
        return b;
    }

    int find_base(int u, int v) {
        fill(all(in_path), 0);
        while (true) {
            in_path[u] = true;
            if (match[u] == -1) {
                break;
            }
            u = base[pre[match[u]]];
        }
        while (!in_path[v]) {
            v = base[pre[match[v]]];
        }
        return v;
    }
};

template <class T, T oo>
struct HopcroftKarp {
    int n, m; 
    vvi adj;
    vi pairU, pairV;
    vt<T> dist;

    HopcroftKarp(int n, int m) : n(n), m(m) {
        adj.resize(n);
        pairU.assign(n, -1);
        pairV.assign(m, -1);
        dist.assign(n, oo);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }

    bool bfs() {
        queue<int> q;
        for (int u = 0; u < n; u++) {
            if (pairU[u] == -1) {
                dist[u] = 0;
                q.push(u);
            } else {
                dist[u] = oo;
            }
        }
        T INF = oo;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (dist[u] < INF) {
                for (int v : adj[u]) {
                    if (pairV[v] == -1) {
                        INF = dist[u] + 1;
                    } else if (dist[pairV[v]] == oo) {
                        dist[pairV[v]] = dist[u] + 1;
                        q.push(pairV[v]);
                    }
                }
            }
        }
        return INF != oo;
    }

    bool dfs(int u) {
        if (u != -1) {
            for (int v : adj[u]) {
                int pu = pairV[v];
                if (pu == -1 || (dist[pu] == dist[u] + 1 && dfs(pu))) {
                    pairV[v] = u;
                    pairU[u] = v;
                    return true;
                }
            }
            dist[u] = oo;
            return false;
        }
        return true;
    }

    int max_match() {
        int matching = 0;
        while (bfs()) {
            for (int u = 0; u < n; u++) {
                if (pairU[u] == -1 && dfs(u)) {
                    matching++;
                }
            }
        }
        return matching;
    }
	
	vpii get_matching() const {
        vpii matchingPairs;
        for (int u = 0; u < n; u++) {
            if (pairU[u] != -1) {
                matchingPairs.push_back({u, pairU[u]});
            }
        }
        return matchingPairs;
    }

};

template<class T, T oo>
struct Hungarian {
    int n, m;
    vi maty, frm, used;
    vt<vt<T>> cst;
    vt<T> fx, fy, dst;

    Hungarian(int n, int m) {
        this->n = n;
        this->m = m;
        cst.resize(n + 1, vt<T>(m + 1, oo));
        fx.resize(n + 1);
        fy.resize(m + 1);
        dst.resize(m + 1);
        maty.resize(m + 1);
        frm.resize(m + 1);
        used.resize(m + 1);
    }

    void add_edge(int x, int y, T c) {
        cst[x][y] = c;
    }

    T min_cost() {
        random_device rd;
        mt19937 rng(rd());
        for (int x = 1; x <= n; x++) {
            int y0 = 0;
            maty[0] = x;
            for (int y = 0; y <= m; y++) {
                dst[y] = oo + 1;
                used[y] = 0;
            }
            int y1;
            do {
                used[y0] = 1;
                int x0 = maty[y0];
                T delta = oo + 1;
                vi perm(m);
                for (int i = 0; i < m; i++) {
                    perm[i] = i + 1;
                }
                shuffle(perm.begin(), perm.end(), rng);
                for (int idx = 0; idx < m; idx++) {
                    int y = perm[idx];
                    if (!used[y]) {
                        T curdst = cst[x0][y] - fx[x0] - fy[y];
                        if (dst[y] > curdst) {
                            dst[y] = curdst;
                            frm[y] = y0;
                        }
                        if (delta > dst[y]) {
                            delta = dst[y];
                            y1 = y;
                        }
                    }
                }
                for (int y = 0; y <= m; y++) {
                    if (used[y]) {
                        fx[maty[y]] += delta;
                        fy[y] -= delta;
                    } else {
                        dst[y] -= delta;
                    }
                }
                y0 = y1;
            } while (maty[y0] != 0);
            do {
                int y1 = frm[y0];
                maty[y0] = maty[y1];
                y0 = y1;
            } while (y0);
        }
        T res = 0;
        for (int y = 1; y <= m; y++) {
            T x = maty[y];
            if (cst[x][y] < oo)
                res += cst[x][y];
        }
        return res;
    }
};
