ll countPal(ll n) {
    // count palindrome <= n
    // https://lightoj.com/problem/palindromic-numbers
    if(n < 0) return 0;
    string s = to_string(n);
    int len = s.size();
    ll ans = 0;
    for(int L = 1; L < len; L++) {
        int half = (L + 1) / 2;
        ans += 9LL * (ll)pow(10, half - 1);
    }
    int half = (len + 1) / 2;
    ll prefix = stoll(s.substr(0, half));
    ll base = 1;
    for(int i = 0; i < half - 1; i++) base *= 10;
    ans += (prefix - base);
    string firstHalf = s.substr(0, half);
    string pal = firstHalf;
    int toMirror = (len % 2 == 0 ? half - 1 : half - 2);
    for(int i = toMirror; i >= 0; i--) {
        pal.pb(firstHalf[i]);
    }
    if(pal <= s) ans++;
    return ans + 1; // include "0"
}

ll min_lcm(ll l, ll r) {// find min_lcm of x, y such that x < y and L <= x < y <= R
    // https://toph.co/p/colded-lcm
    ll res = INF;
    for(ll g = 1; g * g <= r; g++) {
        ll x =  (l - 1) / g + 1;
        if(g * (x + 1) <= r) {
            res = min(res, g * x * (x + 1));
        }
    }
    for(ll x = 1; x * x <= r; x++) {
        ll g = (l - 1) / x + 1;
        if(g * (x + 1) <= r) {
            res = min(res, g * x * (x + 1));
        }
    }
    return res;
}

template<typename T> 
T knapsack_ways(const vpll& a, ll s) { // number of way to reach s given range [l, r] for each i you can pick from
    // https://lightoj.com/problem/cricket-ranking
    int n = a.size();
    vll diff;
    ll sum_left = 0;
    for(auto& [l, r] : a) {
        diff.pb(r - l);
        sum_left += l;
    }
    auto nCk = [&](ll n, ll r) -> T {
        if(n < r) return 0;
        T ans = 1;
        for(int i = 1 ; i <= r ; i++) {
            ans *= n - i + 1;
            ans /= i ;   
        }
        return ans ;
    };
    ll rem = s - sum_left;
    mint res = 0;
    if(rem >= 0) {
        ll total = 1LL << n;
        for(int mask = 0; mask < total; mask++) {
            ll sub = 0;
            int bits = pct(mask);
            for(int i = 0; i < n; i++) {
                if(have_bit(mask, i)) sub += diff[i] + 1;
            }
            ll R = rem - sub;
            mint ways = nCk(R + n - 1, n - 1);
            if(bits & 1) res -= ways;
            else res += ways;
        }
    }
    return res;
}

template<typename T>
T lcm_pairwise_sum(const vi& a) {
    // https://atcoder.jp/contests/agc038/tasks/agc038_c
    // calculate sum(lcm(a[i] * a[j])) for all pair i, j such that i < j
    // lcm(x, y) = (x * y) / gcd(x, y)
    // fix gcd(x, y) = d
    // lcm(x, y) = (x * y) / d
    // for each d, calculate pairwise_sum of a[i] * a[j] where gcd(a[i], a[j]) == d
    // SUM(a[i])^2 = SUM(a[i]^2) + 2 * SUMPAIRWISE(a[i] * a[j])
    // SUMPAIRWISE(a[i] * a[j]) = SUM(a[i])^2 - SUM(a[i]^2)
    const int M = MAX(a);
    vt<T> s(M + 1), s2(M + 1), f(M + 1);
    for(auto& x : a) {
        s[x] += x;
        s2[x] += T(x) * x;
    }
    for(int i = 1; i <= M; i++) {
        for(int j = i * 2; j <= M; j += i) {
            s[i] += s[j];
            s2[i] += s2[j];
        }
    }
    T ans = 0;
    for(int d = M; d >= 1; d--) {
        T now = (s[d] * s[d] - s2[d]) / 2;
        for(int j = d * 2; j <= M; j += d) {
            now -= f[j];
        }
        f[d] = now;
        ans += f[d] / d;
    }
    return ans;
}

template<typename T>
pair<T,T> fib_pair(ll n) { // return {f(n), f(n + 1)};
    if(n == 0) return {0, 1};
    auto [fk, fk1] = fib_pair<T>(n >> 1);
    T c = fk * ((fk1 << 1) - fk);
    T d = fk * fk + fk1 * fk1;
    if(n & 1) return {d, c + d};
    return {c, d};
}

template<typename T>
T fib_sum(ll n) { // return fib_sum of first nth fibonacci
    return n <= 0 ? 0 : fib_pair<T>(n + 2).ff - 1;
}

vi square_permutation(vi a) { // return a permutation where b[b[i]] = a[i], empty vector if not possible
    // https://codeforces.com/contest/612/problem/E
    int n = a.size();
    vi vis(n);
    vvi cycles;
    for(int i = 0; i < n; i++) {
        if(vis[i]) continue;
        vi cycle;
        int j = i;
        while(!vis[j]) {
            cycle.pb(j);
            vis[j] = true;
            j = a[j];
        }
        cycles.pb(cycle);
    }
    vi last(n + 1, -1);
    fill(all(a), -1);
    for(int i = 0; i < (int)cycles.size(); i++) {
        const int N = cycles[i].size();
        if(N & 1) {
            auto& curr = cycles[i];
            for(int j = 0; j < N; j++) {
                a[curr[j]] = curr[(j + (N + 1) / 2) % N];
            }
            continue;
        }
        if(last[N] == -1) last[N] = i;
        else {
            auto& A = cycles[i];
            auto& B = cycles[last[N]];
            last[N] = -1;
            for(int j = 0; j < N; j++) {
                a[A[j]] = B[j];
                a[B[j]] = A[(j + 1) % N];
            }
        }
    }
    if(count(all(a), -1)) return {};
    return a;
}

vpii spriral_matrix(const vvi& a, int x, int y) { // do a spriral_matrix surrounding x, y as a source
    int n = a.size();
    int m = a[0].size();
    auto in = [&](int r, int c) -> bool {
        return r >= 0 && c >= 0 && r < n && c < m;
    };
    vpii now;
    if(in(x, y)) now.pb(MP(x, y));
    int total = n * m;
    int step = 1;
    while(now.size() < total) {
        for(int i = 0; i < step && now.size() < total; i++) {
            y++;
            if(in(x, y)) now.pb(MP(x, y));
        }
        for(int i = 0; i < step && now.size() < total; i++) {
            x++;
            if(in(x, y)) now.pb(MP(x, y));
        }
        step++;
        for(int i = 0; i < step && now.size() < total; i++) {
            y--;
            if(in(x, y)) now.pb(MP(x, y));
        }
        for(int i = 0; i < step && now.size() < total; i++) {
            x--;
            if(in(x, y)) now.pb(MP(x, y));
        }
        step++;
    }
    return now;
}

template<typename T>
T lcm_mod(vi a) { // lcm of multiple number under a mod
    map<int, int> mp;
    for(auto& x : a) {
        for(auto& p : primes) {
            if(p * p > x) break;
            if(x % p) continue;
            int cnt = 0;
            while(x % p == 0) {
                cnt++;
                x /= p;
            }
            mp[p] = max(mp[p], cnt); // only max occcurences matter
        } 
        mp[x] = max(mp[x], 1);
    }
    T res = 1;
    for(auto& [x, v] : mp) {
        res *= T(x).pow(v);
    }
    return res;
}

string smallest_distinct(const string& s) { // return a lexicographically smallest string where each character appear only once
    int n = s.size();
    vi last(26, -1);
    for(int i = 0; i < n; i++) {
        last[s[i] - 'a'] = i;
    }
    string t;
    vi vis(26);
    for(int i = 0; i < n; i++) {
        if(vis[s[i] - 'a']) continue;
        while(!t.empty() && t.back() >= s[i]) {
            int j = t.back() - 'a';
            if(last[j] > i) t.pop_back(), vis[j] = 0;
            else break;
        }
        vis[s[i] - 'a'] = true;
        t.pb(s[i]);
    }
    return t;
}

int count_distinct_palindromic_subsequence(const string& S, int mod) { // https://leetcode.com/problems/count-different-palindromic-subsequences/description/
    int N = S.size();
    if (N == 0) return 0;
    map<char, int> mp;
    int K = 0;
    vi A(N);
    for (int i = 0; i < N; ++i) {
        char c = S[i];
        auto it = mp.find(c);
        if (it == mp.end()) {
            mp[c] = K++;
            A[i] = mp[c];
        } else {
            A[i] = it->second;
        }
    }
    vvi prv(N, vi(K, -1)), nxt(N, vi(K, -1));
    vi last(K, -1);
    for (int i = 0; i < N; ++i) {
        last[A[i]] = i;
        for (int x = 0; x < K; ++x)
            prv[i][x] = last[x];
    }
    fill(last.begin(), last.end(), -1);
    for (int i = N - 1; i >= 0; --i) {
        last[A[i]] = i;
        for (int x = 0; x < K; ++x)
            nxt[i][x] = last[x];
    }
    vvi memo(N, vi(N, -1));
    auto dfs = [&](auto& dfs, int i, int j) -> int {
        if (i > j) return 1;
        if (memo[i][j] != -1) return memo[i][j];
        ll ans = 1;
        for(int x = 0; x < K; ++x) {
            int i0 = nxt[i][x], j0 = prv[j][x];
            if(i0 != -1 && i0 <= j) ans = (ans + 1) % mod;
            if(i0 != -1 && j0 != -1 && i0 < j0) ans = (ans + dfs(dfs, i0 + 1, j0 - 1)) % mod;
        }
        return memo[i][j] = ans;
    };
    int result = dfs(dfs, 0, N - 1) - 1;
    if (result < 0) result += mod;
    return result;
}

int count_assignment(int n, const vpii& edges) { // count the number of way to assign edge to vertex without having node with degree >= 2
    // https://codeforces.com/contest/2098/problem/D
    // https://codeforces.com/problemset/problem/859/E
    struct DSU { 
        public: 
            int n, comp;  
            vi root, rank, col, self_loop, edges; 
            bool is_bipartite;  
            DSU(int n) {    
                this->n = n;    
                comp = n;
                root.rsz(n, -1), rank.rsz(n, 1), col.rsz(n, 0), self_loop.rsz(n), edges.rsz(n);
                is_bipartite = true;
            }

            int find(int x) {   
                if(root[x] == -1) return x; 
                int p = find(root[x]);
                col[x] ^= col[root[x]];
                return root[x] = p;
            }

            bool merge(int u, int v) {  
                u = find(u), v = find(v);   
                if(u == v) {    
                    if(col[u] == col[v]) 
                        is_bipartite = false;
                    return false;
                }
                if(rank[v] > rank[u]) swap(u, v); 
                comp--;
                self_loop[u] |= self_loop[v];
                edges[u] += edges[v];
                root[v] = u;
                col[v] = col[u] ^ col[v] ^ 1;
                rank[u] += rank[v];
                return true;
            }

            bool same(int u, int v) {    
                return find(u) == find(v);
            }

            void assign_self_loop(int x) {
                self_loop[find(x)] = true;
            }

            void increment_edges(int x) {
                edges[find(x)]++;
            }

            int get_rank(int x) {    
                return rank[find(x)];
            }

            vvi get_group() {
                vvi ans(n);
                for(int i = 0; i < n; i++) {
                    ans[find(i)].pb(i);
                }
                return ans;
            }
    };
    DSU root(n);
    for(auto& [u, v] : edges) {
        root.merge(u, v); 
        root.increment_edges(u); // add one more edge to the component
        if(u == v) root.assign_self_loop(u); // detect self loop
    }
    mint ans = 1;
    for(int i = 0; i < n; i++) {
        if(root.find(i) == i) {
            int v = root.get_rank(i);
            int e = root.edges[i];
            if(e > v) { // impossible
                return 0;
            }
            if(e == v - 1) {
                // it's a tree, so 1 node will have degree of 0
                // for that to happen, you have to directed all the edge outward from this edge
                // so the answer is just # of vertices
                ans *= v;
                continue;
            }
            if(v == e) {
                // exactly one cycle in this edge
                // you have to directed all the edges outside the cycle to go outward from this cycle
                // there's only one way to do that, but you can reverse the edge in the cycle to make another case
                // so a->b->c->a, anotherway is a<-b<-c<-
                // so answer is 2 if the cycle has len > 1
                // and 1 if the component has a self loop
                ans *= root.self_loop[i] ? 1 : 2;
            }
        }
    }
    return (int)ans;
}

ll kadane_2d(vvi& a) { // max subarray in 2d matrix
	// https://www.naukri.com/code360/problems/max-submatrix_1214973?leftPanelTabValue=SUBMISSION
    int n = a.size();
    int m = a[0].size();
    ll res = -INF;
    for(int top = 0; top < n; top++) {
        vll t(m);
        for(int bot = top; bot >= 0; bot--) {
            for(int j = 0; j < m; j++) {
                t[j] += a[bot][j];
            } 
            ll curr = 0;
            for(int j = 0; j < m; j++) {
                curr = max(t[j], curr + t[j]);
                res = max(res, curr);
            }
        }
    }
    return res;
}

pair<vi,vi> find_longest_cycle_bidirected_graph(const vvpii& graph){ // return a cycle and edges_id which it used
    int n = graph.size();
    vb visited(n, false), inStack(n, false);
    vi depth(n, 0), parent(n, -1), parentEdge(n, -1);
    int bestLen = 0, bestU = -1, bestV = -1, bestEdge = -1;
    auto dfs = [&](auto&& dfs, int u) -> void {
        visited[u] = inStack[u] = true;
        for (auto [v, eid] : graph[u]) {
            if (eid == parentEdge[u]) continue;
            if (!visited[v]) {
                parent[v] = u;
                parentEdge[v] = eid;
                depth[v] = depth[u] + 1;
                dfs(dfs, v);
            }
            else if (inStack[v]) {
                int len = depth[u] - depth[v] + 1;
                if (len > bestLen) {
                    bestLen = len;
                    bestU = u;
                    bestV = v;
                    bestEdge = eid;
                }
            }
        }
        inStack[u] = false;
    };

    for (int i = 0; i < n; i++)
        if (!visited[i])
            dfs(dfs, i);

    if (bestLen == 0)
        return {vi(), vi()};

    vi verts;
    int cur = bestU;
    while (cur != bestV) {
        verts.pb(cur);
        cur = parent[cur];
    }
    verts.pb(bestV);
    rev(verts);

    vi eids;
    for (int i = 1; i < (int)verts.size(); i++)
        eids.pb(parentEdge[verts[i]]);
    eids.pb(bestEdge);

    return {verts, eids};
}

bool is_symmetrical(const vvi& graph, int root = 0) {
    map<vi, int> hash_code;
    map<int, int> sym;
    int cnt = 0;
    auto dfs = [&](auto& dfs, int node = 0, int par = -1) -> int {
        vi child;
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            child.pb(dfs(dfs, nei, node));
        }
        srt(child);
        if(!hash_code.count(child)) {
            map<int, int> c;
            for(auto& it : child) c[it]++;
            bool bad = false;
            int odd = 0;
            for(auto& [x, v] : c) {
                if(v & 1) {
                    odd++;
                    bad |= !sym[x];
                }
            }
            hash_code[child] = ++cnt;
            sym[cnt] = odd < 2 && !bad;
        }
        return hash_code[child];
    };
    return sym[dfs(dfs, root)];
}

vi toposort(vvi& graph) {
    int n = graph.size();
    vi degree(n);
    for (int u = 0; u < n; u++)
        for (auto v : graph[u])
            degree[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (degree[i] == 0)
            q.push(i);
    vi ans;
    while (!q.empty()) {
        auto i = q.front(); q.pop(); ans.pb(i);
        for (auto& j : graph[i])
            if (--degree[j] == 0)
                q.push(j);
    }
    if (ans.size() != n) return {};
    return ans;
}

vi bidirectional_cycle_vector(int n, const vvi& graph) { // return a cycle_vector to mark which node is part of the cycle
    vi inCycle(n, true);
    vi degree(n);
    queue<int> q;
    for(int i = 0; i < n; i++) {
        for(auto& j : graph[i]) degree[j]++;
    }
    for (int i = 0; i < n; i++) {
        if(degree[i] <= 1) {
            q.push(i);
            inCycle[i] = false;
        }
    }
    while(!q.empty()) {
        auto u = q.front(); q.pop();
        for(int v : graph[u]) {
            if(inCycle[v]) {
                if(--degree[v] == 1) {
                    inCycle[v] = false;
                    q.push(v);
                }
            }
        }
    }
    return inCycle;
}

string construct_string(const string& s) { // construct lexicographically smallest string where no 2 adjacent character is the same
    // https://judge.eluminatis-of-lu.com/contest/686fe616d425270007014c27/1209
    const int n = s.size();
    const int K = 26;
    vi cnt(K);
    for(auto& ch : s) {
        cnt[ch - 'a']++;
    }
    int mx = MAX(cnt);
    if(mx * 2 - 1 > n) return "";
    string res;
    for(int i = 0; i < n; i++) {
        int mx = MAX(cnt);
        if(mx * 2 > (n - i)) {
            for(int j = 0; j < K; j++) {
                if(cnt[j] == mx) {
                    cnt[j]--;
                    res += char(j + 'a');
                    break;
                }
            }
        } else {
            for(int j = 0; j < K; j++) {
                if(cnt[j] && (res.empty() || res.back() - 'a' != j)) {
                    cnt[j]--;
                    res += char(j + 'a');
                    break;
                }
            }
        }
    }
    return res;
}

ll count_pop(ll n, int k) { // count how many number from [1, n] having pct as k
    ll res = 0;
    int ones = 0;
    for(int i = 63; i >= 0; --i) {
        if((n >> i) & 1) {
            if(k - ones >= 0 && k - ones <= i) {
                res += nCk_no_mod(i, k - ones);
            }
            ++ones;
            if(ones > k) break;
        }
    }
    if(ones == k) res++;
    return res;
}

template<typename T>
void minimal_rotation(T &s) {
    int n = int(s.size());
    assert(n > 0);
    int i = 0, ans = 0;
    while(i < n) {
        ans = i;
        int j = i + 1, k = i;
        while(j < 2 * n && !(s[j % n] < s[k % n])) {
            if(s[k % n] < s[j % n]) k = i;
            else k++;
            ++j;
        }
        while(i <= k) {
            i += j - k;
        }
    }
    rotate(s.begin(), s.begin() + ans, s.end());
}

vll spfa(int V, const vvpii& g) {
    // https://atcoder.jp/contests/abc404/tasks/abc404_g
    // https://cses.fi/problemset/task/3294/
    vll dist(V);
    vi inq(V, 1), cnt(V, 0);
    queue<int> q;  
    vi vis(V);
    for(int i = 0; i < V; i++) {
        if(vis[i]) continue; 
        q.push(i);
        while(!q.empty()) {
            int u = q.front(); q.pop();
            vis[u] = true;
            inq[u] = 0;
            for(auto [v, w] : g[u]) {
                if(dist[u] + w < dist[v]) { // careful to change sign if needed
                    dist[v] = dist[u] + w;
                    if(!inq[v]) {
                        if(++cnt[v] > V) {
                            return {};
                        }
                        q.push(v);
                        inq[v] = 1;
                    }
                }
            }
        }
    }
    for(int i = V - 1; i >= 1; i--) {
        dist[i] -= dist[i - 1];
    }
    return dist;
}

vvi rotate90(const vvi matrix) {
    int n = matrix.size(), m = matrix[0].size();
    vvi res(m, vi(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            res[j][n - 1 - i] = matrix[i][j];
    return res;
}

vvll rotate270(const vvll& A) {
    int n = (int)A.size(), m = (int)A[0].size();
    vvll B(m, vll(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[m - 1 - j][i] = A[i][j];
    return B;
}

ll allowed_kth(ll k, vi& allowed) { // find the kth number iff only digit in allowed vector show up
    // https://codeforces.com/contest/1811/problem/E
    ll res = 0, place = 1;
    int B = allowed.size();
    while(k > 0) {
        ll d = k % B;           
        ll e = allowed[d];     
        res += (ll)e * place;
        place *= 10;
        k /= B;
    }
    return res;
}

ll count_unique(const vi& a) { // sum of unique element over all subarray
    ll n = a.size();
    map<int, vi> mp;
    for(int i = 0; i < n; i++) mp[a[i]].pb(i);
    ll total = n * (n + 1) / 2;
    ll res = 0;
    for(auto& [_, it] : mp) {
        it.pb(n); 
        int last = -1;
        ll now = 0;
        for(auto& x : it) {
            ll d = x - last - 1;
            now += d * (d + 1) / 2;
            last = x;
        }
        res += total - now;
    }
    return res;
}

vi directional_cycle_vector(const vvi& out_graph) {
    int n = out_graph.size();
    vvi in_graph(n);
    vi in_deg(n, 0), out_deg(n, 0), in_cycle(n, true);
    for(int u = 0; u < n; u++) {
        out_deg[u] = out_graph[u].size();
        for(int v : out_graph[u]) {
            in_graph[v].pb(u);
            in_deg[v]++;
        }
    }
    queue<int> q;
    for(int i = 0; i < n; i++) {
        if(in_deg[i] == 0 || out_deg[i] == 0) {
            q.push(i);
            in_cycle[i] = false;
        }
    }
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        for(int v : out_graph[u]) {
            if(in_cycle[v]) {
                if(--in_deg[v] == 0) {
                    in_cycle[v] = false;
                    q.push(v);
                }
            }
        }
        for(int v : in_graph[u]) {
            if(in_cycle[v]) {
                if(--out_deg[v] == 0) {
                    in_cycle[v] = false;
                    q.push(v);
                }
            }
        }
    }
    return in_cycle;
}

template<typename T = int>
vi diameter_vector(const vt<vt<T>>& graph) { // return a vector where a[i] is the max diameter from the ith vertex
    int n = graph.size();
    auto bfs = [&](int src) -> vi {
        vi dp(n, -1);
        queue<int> q;
        auto process = [&](int u, int c) -> void {
            if(dp[u] == -1) {
                dp[u] = c;
                q.push(u);
            }
        };
        process(src, 0);
        while(!q.empty()) {
            int node = q.front(); q.pop();
            for(auto& nei : graph[node]) {
                process(nei, dp[node] + 1);
            }
        }
        return dp;
    };
    auto dummy = bfs(0);
    int a = max_element(all(dummy)) - begin(dummy);
    auto dp1 = bfs(a);
    int b = max_element(all(dp1)) - begin(dp1);
    auto dp2 = bfs(b);
    vi ans(n);
    for(int i = 0; i < n; i++) {
        ans[i] = max(dp1[i], dp2[i]);
    }
    return ans;
}

string validate_substring(int n, const string& t, vi a) { 
    // given a string s of len n full of '?'
    // and a vector a indicates where t is a substring of s
    // determine if it's a valid sequence of not
    string s = string(n, '?');
    srtU(a); rev(a);
    auto z = Z_Function(t);
    int m = t.size();
    for(auto& i : a) {
        i--;    
        int r = i + m;
        if(i + m > n) return "";
        for(int j = i; j < r; j++) {
            int id = j - i;
            if(s[j] == '?') {
                s[j] = t[id];
                continue;
            }
            if(z[id] < r - j) return "";
            break;
        }
    }
    return s;
}

vi derangement(const vi& a) { // return an array where nothing is in the original position
    vpii arr;
    map<int, int> mp;
    int mx = 0;
    int n = a.size();
    for(int i = 0; i < n; i++) {
        int x = a[i];
        arr.pb({a[i], i});
        mx = max(mx, ++mp[x]);
    }
    srt(arr);
    auto A(arr);
    auto b(a);
    ROTATE(arr, mx);
    for(int i = 0; i < n; i++) {
        b[A[i].ss] = arr[i].ff;
    }
    for(int i = 0; i < n; i++) {
        if(a[i] == b[i]) {
            return {};
        }
    }
    return b;
}

template<typename T>
vt<T> compute_lis(const vi& a, int K, const var(3)& queries) { // careful with k = 0 loop, it's k = 1 loop right now
    // https://usaco.org/index.php?page=viewproblem2&cpid=997
    int n = a.size();
    int q = queries.size();
    vt<T> ans(q);
    vt<vt<T>> pre(n), suff(n);
    vt<T> preT(n), suffT(n);
    auto dfs = [&](auto& dfs, int l, int r, const var(3)& Q) -> void {
        if(Q.empty()) return;
        if(l == r) {
            for(auto& [_, __, id] : Q) {
                ans[id] = 2;
            }
            return;
        }
        int m = (l + r) >> 1;
        {
            vt<vt<T>> dp(K + 1, vt<T>(K + 1));
            vt<T> C(K + 1);
            T S = 0;
            for(int i = m; i >= l; i--) {
                int L = a[i];
                auto old(dp);
                for(int R = L; R <= K; R++) {
                    T suff = 0;
                    for(int j = R; j >= L; j--) {
                        suff += old[j][R];
                    }
                    if(L == R) suff++;
                    dp[L][R] += suff;
                    S += suff;
                    C[R] += suff;
                }
                preT[i] = S;
                pre[i] = C;
            }
        }
        {
            vt<vt<T>> dp(K + 1, vt<T>(K + 1));
            vt<T> C(K + 1);
            T S = 0;
            for(int i = m + 1; i <= r; i++) {
                int R = a[i];
                auto old(dp);
                for(int L = R; L >= 1; L--) {
                    T pre = 0; 
                    for(int j = L; j <= R; j++) {
                        pre += old[L][j];
                    }
                    if(L == R) pre++;
                    dp[L][R] += pre;
                    S += pre;
                    C[L] += pre;
                }
                suffT[i] = S;
                suff[i] = C;
            }
        }
        var(3) left, right;
        for(auto& [L, R, id] : Q) {
            if(R <= m) left.pb({L, R, id});
            else if(L > m) right.pb({L, R, id});
            else {
                T res = 1 + preT[L] + suffT[R];
                T s = 0;
                for(int y = K; y >= 1; y--) {
                    s += suff[R][y];
                    res += pre[L][y] * s;
                }
                ans[id] = res;
            }
        }
        dfs(dfs, l, m, left);
        dfs(dfs, m + 1, r, right);
    };
    dfs(dfs, 0, n - 1, queries);
    return ans;
}

ll count_sqrt_divisor(ll n) {
    // count # of value such that x % sqrt(x) == 0
    if(n <= 0) return 0;
	ll sq = sqrt(n);
    // i^2 + 2*i + 1 - 1
    // i*(i + 2) / i
    ll res = 2 * (sq - 1) + (n / sq);
    return res;
}

vll shift_vector(const vi& a) { // maintain abs(a[i] - i) for each rotation
    // https://codeforces.com/contest/819/problem/B
    const int n = a.size();
    int lt = 0, gt = 0;
    vi cnt(n);
    ll orig = 0;
    for(int i = 0; i < n; i++) {
        orig += abs(a[i] - i);
        if(a[i] > i) {
            cnt[a[i] - i]++;
            lt++;
        } else {
            gt++;
        }
    }
    vll ans(n);
    ans[0] = orig;
    for(int k = 1; k < n; k++) {
        orig += gt-- - lt++;
        int id = (n - k) % n;
        orig += a[id] - abs(a[id] - (n - 1)) - 1; // -1 for greater counter
        if(a[id] + k < n) cnt[a[id] + k]++;
        lt -= cnt[k];
        gt += cnt[k];
        ans[k] = orig;
    }
    return ans;
}

template<typename T>
T get_val(const vll& a, ll k, int x) { // do m times a[i] += a[i - 1], what's the value of the index x
    // https://www.codechef.com/problems/STROPR?tab=statement
    if (k == 0) return T(a[x]);
    T B = 1;
    T res = 0;
    k = (ll)mint(k);
    for (int t = 0; t <= x; ++t) {
        int idx = x - t;
        res += T(a[idx]) * B;
        B *= k + t;
        B /= t + 1;
    }
    return res;
}

template<typename T>
T ways_to_assign_color(const vvi& graph, int C) { // assign color for path length <= 2
    // https://www.codechef.com/problems/TREECLR
    int n = graph.size();
    mint res = 1;
    auto dfs = [&](auto& dfs, int node = 0, int par = -1) -> void {
        int c = int(par != -1) + 1;
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            dfs(dfs, nei, node);
            res *= C - c++;
        }
        if(par == -1) res *= C;
    };
    dfs(dfs);
    return res;
}

db meeting_probability(int T1, int T2, int t1, int t2) {
    // https://www.codechef.com/problems/FRNDMTNG?tab=statement
    db SQ1, SQ2;
    if(T1 > T2) {
        swap(T1, T2);
        swap(t1, t2);
    }
    if(t1 > T2) t1 = T2;
    if(t2 > T1) t2 = T1;

    SQ1 = T1 * 0.5 * T1;
    if(T1 > t2) SQ1 -= (T1 - t2) * 0.5 * (T1 - t2);
    if(t1+T1 <= T2) {
        SQ2 = T1 * 1.0 * t1;
    }
    else {
        SQ2 = T1 * 0.5 * T1;
        SQ2 += (T1) * 1.0 * (T2-T1);
        SQ2 -= (T2-t1) * 0.5 * (T2-t1);
    }
    return (SQ1 + SQ2) / (T1 * 1ll * T2);
}

vll all_subarray_sum_minimum(const vi& a, const vpii& queries) {
    // https://codeforces.com/contest/2009/problem/G3
    // update is range set so lazy = -INF
    int n = a.size();
    auto right = closest_right(a, less_equal<int>());
    auto left = closest_left(a, less<int>());
    vll suffix(n + 1);
    for(int i = n - 1; i >= 0; i--) {    
        suffix[i] = suffix[right[i] + 1] + (ll)a[i] * (right[i] - i + 1);
    }
    lazy_seg suffix_tree(n), suffix_ans_tree(n), ans_i_tree(n);
    vll prefix(n + 1);
    for(int i = 1; i <= n; i++) {
        prefix[i] = prefix[i - 1] + suffix[i - 1];
    }
    vvpii G(n);
    int q = queries.size();
    for(int i = 0; i < q; i++) {
        auto& [l, r] = queries[i];
        G[r].pb({l, i});
    }
    vll A(q);
    for(int r = 0; r < n; r++) {
        int L = left[r]; 
        suffix_tree.update_range(L, r, suffix[r]);
        suffix_ans_tree.update_range(L, r, (ll)a[r] * (r - 1));
        ans_i_tree.update_range(L, r, a[r]);
        for(auto& [l, id] : G[r]) {
            ll now = prefix[r + 1] - prefix[l]; // suffix[left] + suffix[left + 1] + ... + suffix[right]
            now -= suffix_tree.queries_range(l, r); // - suffix[id]
            now -= suffix_ans_tree.queries_range(l, r);
            now += ans_i_tree.queries_range(l, r) * r;
            A[id] = now;
        }
    }
    return A;
}

ll sg(ll x) {
    while(x > 1) {
        int p = 1 << __lg(x);
        int h = p >> 1;
        if(x < p + h) return x - h;
        x -= p;
    }
    return 0;
}

int can_be_split(const string& s, int k) { // can s be split into subsequence of length k each
    // https://codeforces.com/group/o09Gu2FpOx/contest/541486/problem/V
    int n = s.size();
    if(n % k) return false;
    const int each = n / k;
    const int K = 26;
    set<int> pos[K];
    for(int i = 0; i < n; i++) {
        pos[s[i] - 'a'].insert(i);
    }
    vi prev(each, -1);
    vi vis(n);
    for(int i = 0; i < n; i++) {
        if(vis[i]) continue;
        vis[i] = true;
        int id = s[i] - 'a';
        for(int j = 0; j < each; j++) {
            auto it = pos[id].ub(prev[j]);
            if(it == end(pos[id])) return false;
            vis[*it] = true;
            prev[j] = *it;
            pos[id].erase(it);
        }
    }
    return true;
}

ll sum_of_second_max(vi a) {
    // https://codeforces.com/group/o09Gu2FpOx/contest/541486/problem/C
    int n = a.size();
    a.insert(begin(a), 0);
    auto left = closest_left(a, greater_equal<int>());
    auto right = closest_right(a, greater_equal<int>());
    vll pre(n + 2), suff(n + 2);
    for(int i = 1; i <= n; i++) {
        int L = left[i];
        pre[i] = pre[L - 1] + (ll)a[i] * (i - L + 1);
    }
    for(int i = n; i >= 1; i--) {
        int R = right[i];
        suff[i] = suff[R + 1] + (ll)a[i] * (R - i + 1);
    }
    vi id(n + 1);
    iota(all(id), 0);
    linear_rmq<int> T(id, [&](const int& i, const int& j) {return a[i] > a[j];});
    auto query_pre = [&](int l, int r) -> ll {
        if(l > r) return 0;
        int mx = T.query(l, r);
        return pre[r] - pre[mx] + (ll)a[mx] * (mx - l + 1);
    };
    auto query_suff = [&](int l, int r) -> ll {
        if(l > r) return 0;
        int mx = T.query(l, r);
        return suff[l] - suff[mx] + (ll)a[mx] * (r - mx + 1);
    };
    basic_segtree<int> max_tree(n + 1, -inf, [](const int& a, const int& b) {return max(a, b);});
    for(int i = 1; i <= n; i++) {
        max_tree.update_at(i, a[i]);
    }
    ll res = 0;
    auto dfs = [&](auto& dfs, int l, int r) -> void {
        if(l > r) return; 
        int m = T.query(l, r);
        res += query_pre(l, m - 1) + query_suff(m + 1, r); // where the border is m
        if(l < m && m < r) {
            if(m - l < r - m) {
                int mx = -inf;
                for(int i = m - 1; i >= l; i--) {
                    mx = max(mx, a[i]);
                    int R = min(r, max_tree.max_right(m + 1, [&](const int &x) {return x <= mx;}));
                    res += (R - m) * (ll)mx + query_suff(R + 1, r);
                }
            } else {
                int mx = -inf;
                for(int i = m + 1; i <= r; i++) {
                    mx = max(mx, a[i]);
                    int L = max(l, max_tree.min_left(m - 1, [&](const int& x) {return x <= mx;}));
                    res += (m - L) * (ll)mx + query_pre(l, L - 1);
                }
            }
        }
        dfs(dfs, l, m - 1);
        dfs(dfs, m + 1, r);
    };
    dfs(dfs, 1, n);
    return res;
}

string construct_median_string(string s, int k) { // reconstruct the binary string such that each window of size k would have ceil or floor k / 2 '1'
    int n = s.size();
    int m = 0;
    for(char c: s) m += (c == '1');
    int low = k / 2;
    int high = (k + 1) / 2;
    vi L(n + 1, 0), U(n + 1);
    for(int i = 0; i <= n; ++i) U[i] = min(i, m);
    L[0] = U[0] = 0;
    L[n] = U[n] = m;
    for(int i = 1; i <= n; ++i) {
        L[i] = max(L[i], L[i - 1]);
        if(i >= k) L[i] = max(L[i], L[i - k] + low);
    }
    for(int i = n - 1; i >= 0; --i) {
        L[i] = max(L[i], L[i + 1] - 1);
        if(i + k <= n) L[i] = max(L[i], L[i + k] - high);
    }
    for(int i = 0; i < n; ++i) {
        U[i + 1] = min(U[i + 1], U[i] + 1);
        if(i + k <= n) U[i + k] = min(U[i + k], U[i] + high);
    }
    for(int i = n; i >= 1; --i) {
        U[i - 1] = min(U[i - 1], U[i]);
        if(i >= k) U[i - k] = min(U[i - k], U[i] - low);
    }
    for(int i = 0; i <= n; ++i) {
        if(L[i] < 0) L[i] = 0;
        if(U[i] > min(i, m)) U[i] = min(i, m);
    }
    for(int i = 0; i <= n; ++i) {
        if(L[i] > U[i]) return "";
    }
    string ans;
    int curr = 0;
    for(int i = 0; i < n; ++i) {
        if(L[i + 1] <= curr && curr <= U[i + 1]) {
            ans.pb('0');
        } else {
            ++curr;
            if(curr < L[i + 1] || curr > U[i + 1]) return "";
            ans.pb('1');
        }
    }
    if(curr != m) return "";
    return ans;
}

ll sum_k_infinity(ll k) { // return sum of the first k digit of [123456789101112...]
    // https://codeforces.com/contest/2132/problem/D
    static ll p10[19];
    static bool init = false;
    if(!init) {
        p10[0] = 1;
        for(int i = 1; i < 19; i++) p10[i] = p10[i - 1] * 10;
        init = true;
    }
    auto S = [&](ll n) -> i128 {
        if(n <= 0) return 0;
        i128 res = 0;
        for(ll p = 1; p <= n; p *= 10) {
            ll left = n / (p * 10);
            ll cur  = (n / p) % 10;
            ll right = n % p;
            res += (i128)left * 45 * p;
            res += (i128)cur * (cur - 1) / 2 * p;
            res += (i128)cur * (right + 1);
        }
        return res;
    };
    auto range = [&](ll a, ll b) -> ll {
        if(b < a) return 0;
        return (ll)(S(b) - S(a - 1));
    };
    auto sumDigits = [&](ll x) -> ll {
        ll s = 0;
        while(x) { s += x % 10; x /= 10; }
        return s;
    };
    ll ans = 0;
    ll rem = k;
    for(int d = 1; rem > 0; d++) {
        i128 cnt = (i128)d * 9 * p10[d - 1];
        if(cnt <= rem) {
            ll L = p10[d - 1];
            ll R = p10[d] - 1;
            ans += range(L, R);
            rem -= (ll)cnt;
        } else {
            ll q = rem / d;
            ll r = rem % d;
            ll L = p10[d - 1];
            if(q > 0) ans += range(L, L + q - 1);
            if(r > 0) {
                ll num = L + q;
                ll trunc = num / p10[d - (int)r];
                ans += sumDigits(trunc);
            }
            rem = 0;
        }
    }
    return ans;
}

vvll count_subrectangle(const vs& S) { // return a vector [i][j] represents # of good subgrid with i as width and j as height
    // https://atcoder.jp/contests/abc420/tasks/abc420_f
    int n = S.size();
    int m = S[0].size();
    vvi R(n + 5, vi(m + 5)), U(n + 5, vi(m + 5)), D(n + 5, vi(m + 5));
    vvll ans(n + 5, vll(m + 5));
    vs mat(n + 5);
    mat[0].assign(m + 5, '0');
    mat[n + 1].assign(m + 5, '0');
    mat[n + 2].assign(m + 5, '0');
    mat[n + 3].assign(m + 5, '0');
    for(int i = 1; i <= n; ++i) {
        mat[i] = "0" + S[i - 1] + "000000";
        for(int j = m; j >= 1; --j) {
            if(mat[i][j] == '0') R[i][j] = 0;
            else R[i][j] = R[i][j + 1] + 1;
        }
    }
    vi st;
    for(int j = 1; j <= m; ++j) {
        st.clear();
        for(int i = 1; i <= n; ++i) {
            while(!st.empty() && R[i][j] < R[st.back()][j]) st.pop_back();
            U[i][j] = st.empty() ? 1 : (st.back() + 1);
            st.pb(i);
        }
        st.clear();
        for(int i = n; i >= 1; --i) {
            while(!st.empty() && R[i][j] <= R[st.back()][j]) st.pop_back();
            D[i][j] = st.empty() ? n : (st.back() - 1);
            st.pb(i);
        }
    }
    for(int i = 1; i <= n; ++i) {
        for(int j = 1; j <= m; ++j) {
            int w = R[i][j];
            ++ans[1][w];
            --ans[i - U[i][j] + 2][w];
            --ans[D[i][j] - i + 2][w];
            ++ans[D[i][j] - U[i][j] + 3][w];
        }
    }
    for(int j = 1; j <= m; ++j) {
        for(int i = 1; i <= n; ++i) ans[i][j] += ans[i - 1][j];
        for(int i = 1; i <= n; ++i) ans[i][j] += ans[i - 1][j];
    }
    for(int i = n; i >= 1; --i) {
        for(int j = m; j >= 2; --j) ans[i][j - 1] += ans[i][j];
    }
    vvll res(n + 1, vll(m + 1));
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= m; j++) {
            res[i][j] = ans[i][j];
        }
    }
    return res;
}

template<typename T> T dnc_mnmx(const vi& a) {
    // https://marisaoj.com/problem/423
    int n = a.size();
    vi lmn(n), lmx(n), rmn(n), rmx(n);
    auto dfs = [&](auto& self, int l, int r) -> T {
        if(l == r) return (T)a[l] * a[l]; // careful
        int m = (l + r) >> 1;
        T res = self(self, l, m) + self(self, m + 1, r);
        for(int i = m, mn = inf, mx = -inf; i >= l; i--) {
            mn = min(mn, a[i]);
            mx = max(mx, a[i]);
            lmn[i] = mn;
            lmx[i] = mx;
        }
        for(int i = m + 1, mn = inf, mx = -inf; i <= r; i++) {
            mn = min(mn, a[i]);
            mx = max(mx, a[i]);
            rmn[i] = mn;
            rmx[i] = mx;
        }
        { // lmn, lmx
            int j = m + 1, k = m + 1;     
            for(int i = m; i >= l; i--) {
                while(j <= r && rmn[j] >= lmn[i]) j++;
                while(k <= r && rmx[k] <= lmx[i]) k++;
                int len = min(j, k) - (m + 1);
                if(len > 0) res += (T)len * (lmn[i] * lmx[i]);
            }
        }
        { // rmn, rmx
            int j = m, k = m;
            for(int i = m + 1; i <= r; i++) {
                while(j >= l && lmn[j] > rmn[i]) j--;
                while(k >= l && lmx[k] < rmx[i]) k--;
                int len = m - max(j, k);
                if(len > 0) res += (T)len * (rmn[i] * rmx[i]);
            }
        }
        { // lmn, rmx
            T s = 0;
            for(int i = m, j = m + 1, k = m + 1; i >= l; i--) {
                while(j <= r && rmn[j] >= lmn[i]) s += rmx[j++];
                while(k < j && rmx[k] <= lmx[i]) s -= rmx[k++];
                res += s * lmn[i];
            } 
        }
        { // lmx, rmn
            T s = 0;
            for(int i = m + 1, j = m, k = m; i <= r; i++) {
                while(j >= l && lmn[j] > rmn[i]) s += lmx[j--];
                while(k > j && lmx[k] < rmx[i]) s -= lmx[k--];
                res += s * rmn[i];
            }
        }
        return res;
    };
    return dfs(dfs, 0, n - 1);
}

ll min_flip(ll O, ll Z, ll K) { // given a string s of O one, Z zero, in one move you can flip k index, return minimum move to make the string become all 1 or -1 if not possible
    // https://leetcode.com/problems/minimum-operations-to-equalize-binary-string/description/
    ll N = O + Z;
    if (N == K) {
        if (Z == 0) return 0;
        if (Z == N) return 1;
        return -1;
    }

    auto ceilDiv = [](ll x, ll y) -> ll { return (x + y - 1) / y; };

    ll ans = INF;

    if ((Z & 1) == 0) {
        ll M = max(ceilDiv(Z, K), ceilDiv(Z, N - K));
        M += (M & 1);
        ans = min(ans, M);
    }
    if (((Z & 1) == (K & 1))) {
        ll M = max(ceilDiv(Z, K), ceilDiv(N - Z, N - K));
        M += ((M & 1) == 0);
        ans = min(ans, M);
    }
    return ans < INF ? ans : -1;
}

template<typename T>
T count_b_mod_a_equal_a_xor_b(ll L, ll R) { // return number of pair where l <= a <= b <= r and b % a == b ^ a
    // https://codeforces.com/gym/106015/problem/B
    auto f = [](ll l, ll r) -> T {
        auto g = [](ll n) {
            string s;
            while(n) {
                s += char((n & 1) + '0');
                n >>= 1;
            }
            rev(s);
            return s;
        };
        string s = g(l), t = g(r);
        s = string(t.size() - s.size(), '0') + s;
        const int n = s.size();
        const int K = 60;
        static T dp[K][2][2];
        static int cached[K][2][2];
        memset(cached, 0, sizeof(cached));
        auto dfs = [&](auto &self, int i = 0, int t1 = 1, int t2 = 1) -> T {
            if(i == n) return 1;
            auto& res = dp[i][t1][t2];
            auto& c = cached[i][t1][t2];
            if(c) return res;
            c = true;
            res = 0;
            int low = t1 ? s[i] - '0' : 0;
            int high = t2 ? t[i] - '0' : 1;
            for(int a = low; a <= 1; a++) {
                for(int b = 0; b <= high; b++) {
                    if(a > b) continue;
                    res += self(self, i + 1, t1 && a == low, t2 && b == high);
                }
            }
            return res;
        };
        return dfs(dfs);
    };
    int msb_l = max_bit(L);
    int msb_r = max_bit(R);
    T res = 0; 
    for(int i = msb_l + 1; i < msb_r; i++) {
        res += T(3).pow(i); // 3 possible happening each bit [0, 0], [0, 1], [1, 1]
    }
    L -= 1LL << msb_l;
    R -= 1LL << msb_r;
    if(msb_l == msb_r) {
        res += f(L, R); 
    } else {
        res += f(L, (1LL << msb_l) - 1) + f(0, R);
    }
    return res;
}

string construct_string(const string& s, ll K) { // return the string with s appear as a subsequence K times
    // https://atcoder.jp/contests/cf16-exhibition-final/tasks/cf16_exhibition_final_g
    const int n = s.size();
    vll curr(n + 1, 1);
    curr[n] = 0;
    string ans;
    for(int i = 0; i < n - 1; i++) ans += s[i];
    for(int i = 0; (1LL << i) <= K; i++) {
        assert(curr[n - 1] == (1LL << i));
        if(K >> i & 1) {
            ans += s.back();
            curr[n] += curr[n - 1];
        }
        for(int j = n - 2; j >= max(0, i / (n + 1)); j--) {
            ll need = 2 * curr[j + 1];
            while(curr[j + 1] != need) {
                ans += s[j];
                curr[j + 1] += curr[j];
            }
        }
    }
    assert(curr[n] == K);
    return ans;
}

string construct_string(const string& target, ll k) { // shortest string with len of target = 2
    // https://www.codechef.com/problems/XDS?tab=statement
    ll h = sqrt(k);
    string t = target.substr(0, target.size() - 1);
    string tt = string(1, target.back());
    string s;
    for(int i = 0; i < h; i++) {
        s += t;
    }
    ll rem = k - h * h, curr = h;
    while(curr) {
        if(rem >= curr) {
            s += t;
            rem -= curr;
        } else {
            curr--;
            s += tt;
        }
    }
    return s;
}

template<typename T>
T count_perm_leq(string a, string b) { // count # of different reorder of a <= b
    const int K = 26;
    vi cnt(K); 
    for(auto& ch : a) cnt[ch - 'a']++;
    const int n = b.size();
    mint res = 0;
    mint s = comb.fact[sum(cnt)];
    for(auto& v : cnt) {
        s /= comb.fact[v];
    }
    for(int i = 0; i < n; i++) {
        int x = b[i] - 'a';
        int total = 0;
        s /= (n - i);
        for(int j = 0; j < x; j++) {
            total += cnt[j];
        }
        res += s * total;
        if(cnt[x] == 0) break;
        s *= cnt[x]--;
    }
    return res + 1;
}

map<pll, ll> dp;
ll xor_mst(ll l, ll r) { // find minimum mst of a graph with vertices from l to r where the cost of [u, v] is u ^ v
    // https://www.codechef.com/problems/MINSET?tab=statement
	if(l >= r) return 0;
    if(l + 1 == r) return l ^ r;
    if(dp.count({l, r})) return dp[{l, r}];
    auto& res = dp[{l, r}];
    ll p = 1;
    while(p * 2 <= r) p *= 2;
    if(p == r) {
        return res = p + l + xor_mst(l, r - 1);
    }
    return res = p + xor_mst(l, p - 1) + xor_mst(0, r - p);
}

vll dikstra_avoid_edge(int n, const var(3)& edges) { // for each edges i, what's the shortest path from [1, n] if we ignore this edge
    // https://codeforces.com/contest/1163/problem/F
    int m = edges.size();
    vvar(3) graph(n);
    for(int i = 0; i < m; i++) {
        const auto& [u, v, w] = edges[i];
        graph[u].pb({v, w, i});
        graph[v].pb({u, w, i});
    }
    vpii par(n, {-1, -1});
    vll dp(n, INF);
    vi on_path(n), l(n, -1), r(n, m);
    auto dikstra = [&](int src, int type) -> void {
        fill(all(dp), INF);
        min_heap<pll> q;
        auto process = [&](int node, ll cost, pii p) -> void {
            if(dp[node] > cost) {
                dp[node] = cost;
                if(p.ff != -1) {
                    if(type == 1 && !on_path[node]) {
                        l[node] = l[p.ff];
                    } else if(type == 2 && !on_path[node]) {
                        r[node] = r[p.ff]; 
                    }
                }
                par[node] = p;
                q.push({cost, node});
            }
        };
        process(src, 0, {-1, -1});
        while(!q.empty()) {
            auto [cost, node] = q.top(); q.pop();
            for(auto& [nei, w, i] : graph[node]) {
                process(nei, cost + w, {node, i});
            }
        }
    };
    dikstra(n - 1, 0);
    int curr = 0;
    vi ind(m, -1);
    int N = 0;
    while(curr != -1) {
        on_path[curr] = true;
        auto& [p, id] = par[curr];
        l[curr] = r[curr] = N;
        if(id != -1) ind[id] = N;
        N++;
        curr = p;
    }
    dikstra(0, 1);
    vll dp1(dp);
    dikstra(n - 1, 2);
    vll dp2(dp);
    vvll in(N + 1), out(N + 1);
    auto update = [&](int L, int R, ll w) -> void {
        if(L > R) return;
        if(L <= N && L >= 0) {
            in[L].pb(w);
        }
        if(R <= N && R >= 0) {
            out[R].pb(w);
        }
    };
    for(int i = 0; i < m; i++) {
        if(ind[i] == -1) {
            const auto& [u, v, w] = edges[i];
            update(l[u] + 1, r[v], dp1[u] + w + dp2[v]);
            update(l[v] + 1, r[u], dp1[v] + w + dp2[u]);
        }    
    }
    multiset<ll> s;
    vll ans(m);
    vll val(N + 1, INF);
    for(int i = 1; i <= N; i++) {
        for(auto& x : in[i]) s.insert(x);
        val[i] = s.empty() ? INF : *s.begin();
        for(auto& x : out[i]) s.erase(s.find(x));
    }
    for(int i = 0; i < m; i++) {
        if(ind[i] == -1) {
            ans[i] = dp1[n - 1];
        } else {
            ans[i] = val[ind[i] + 1];
        }
    }
    return ans;
}

pair<ll, vi> max_xor_via_swap(const vll& a, const vll& b) { // maximizing xor(a) + xor(b) and returning the number of indices need to swap
    // https://codeforces.com/gym/103414, problem G
    int n = a.size();
    const int l = 61;
    ull a_xor = 0, b_xor = 0;
    for(auto x : a) a_xor ^= x;
    for(auto x : b) b_xor ^= x;
    ull all_bits = (1ULL << l) - 1;
    ull u = (~(a_xor ^ b_xor)) & all_bits;

    vt<ull> v(n);
    for (int i = 0; i < n; i++) v[i] = (a[i] ^ b[i]) & u;

    vt<ull> base_val_by_bit(l, 0);
    vi base_id_by_bit(l, -1);
    vt<ull> base_val_by_id, base_pre_mask_by_id;
    vi base_lead_bit_by_id, base_orig_index_by_id;
    int rank = 0;

    auto insert = [&](ull x, int idx) {
        ull cur = x, dep = 0;
        for(int bit = l - 1; bit >= 0; --bit) {
            if(!((cur >> bit) & 1ULL)) continue;
            if(base_val_by_bit[bit]) {
                cur ^= base_val_by_bit[bit];
                dep ^= (1ULL << base_id_by_bit[bit]);
            } else {
                int id = rank++;
                base_val_by_id.pb(cur);
                base_pre_mask_by_id.pb(dep);
                base_lead_bit_by_id.pb(bit);
                base_orig_index_by_id.pb(idx);
                base_val_by_bit[bit] = cur;
                base_id_by_bit[bit] = id;
                return;
            }
        }
    };

    for(int i = 0; i < n; i++) if (v[i]) insert(v[i], i);

    ull cur = a_xor & u, res_val = cur, used_mask = 0;
    for(int bit = l - 1; bit >= 0; --bit) {
        int id = base_id_by_bit[bit];
        if(id == -1) continue;
        ull cand = res_val ^ base_val_by_id[id];
        if(cand > res_val) {
            res_val = cand;
            used_mask ^= (1ULL << id);
        }
    }

    ull best_sum = (a_xor ^ b_xor) + (res_val << 1);
    vi idxs;
    for(int k = rank - 1; k >= 0; --k) {
        if((used_mask >> k) & 1ULL) {
            idxs.pb(base_orig_index_by_id[k] + 1);
            used_mask ^= (1ULL << k);
            used_mask ^= base_pre_mask_by_id[k];
        }
    }
    srt(idxs);
    return {best_sum, idxs};
}

int total_subarray(int n, int x, int k) { // https://basecamp.eolymp.com/en/problems/12215#editorial
    mint total = 0;
    for(int len = 1; len <= x; len++) {
        total += comb.nCk(n - 1, len - 1); 
    }
    if(k == 0) {
        return (int)total;
    }
    vvmint dp(x + 1, vmint(n + 1));
    dp[0][0] = 1;
    for(int len = 1; len <= x; len++) {
        mint pre = 0;
        for(int i = 1; i <= n; i++) {
            pre += dp[len - 1][i - 1];
            if(i >= k) pre -= dp[len - 1][i - k];
            dp[len][i] = pre;
        }
    }
    total--;
    mint bad = 0;
    for(int size = 2; size <= x; size++) {
        for(int j = 1; j <= size; j++) {
            mint e = 0;
            for(int mn = 1; mn <= n; mn++) {
                int rem = n - mn * size;
                if(rem < 0) break;
                e += dp[size - j][rem];
            }
            e *= comb.nCk(size, j);
            bad += e;
        }
    }
    return int(total - bad);
}

vi find_isomophic_center(const vvi& graph) {
    // https://codeforces.com/contest/1182/problem/D
    int n = graph.size();
    vector<ull> hash(n);
    vi ok(n, 1);
    map<ull, ull> mp;
    auto f = [&](ull t) -> void {
        if(!mp.count(t)) {
            mp[t] = (rng() << 32) | rng();
        }
    };
    {
        auto dfs = [&](auto& dfs, int node = 0, int par = -1) -> void {
            ull s = 0, prev = 0;
            for(auto& nei : graph[node]) {
                if(nei == par) continue;
                dfs(dfs, nei, node);
                s += hash[nei];
                if(!ok[nei] || (prev && prev != hash[nei])) {
                    ok[node] = false;
                }
                prev = hash[nei];
            }
            f(s);
            hash[node] = mp[s];
        };
        dfs(dfs);
    }
    vi ans;
    {
        auto dfs = [&](auto& dfs, int node = 0, int par = -1, ull h = 0, int ok_up = 1) -> void {
            map<ull, int> cnt; 
            int bad = ok_up == 0;
            if(par != -1) cnt[h]++;
            ull curr = par == -1 ? 0 : h;
            for(auto& nei : graph[node]) {
                if(nei == par) continue;
                bad += !ok[nei];
                cnt[hash[nei]]++;
                curr += hash[nei];
            }
            if(bad == 0 && cnt.size() <= 1) {
                ans.pb(node);
            }
            for(auto& nei : graph[node]) {
                if(nei == par) continue;
                bad -= !ok[nei];
                if(--cnt[hash[nei]] == 0) cnt.erase(hash[nei]);
                curr -= hash[nei];
                f(curr);
                dfs(dfs, nei, node, mp[curr], cnt.size() <= 1 && bad == 0);
                cnt[hash[nei]]++;
                curr += hash[nei];
                bad += !ok[nei];
            }
        };
        dfs(dfs);
    }
    return ans;
}

ll domino_solver(vll h, vll c) {
    // https://codeforces.com/contest/1131/problem/G?adcd1e=caf4fvg4ke61uy&csrf_token=55be3d76734d8ccd743640b909b92ea9
    h.insert(begin(h), 0);
    c.insert(begin(c), 0);
    const int N = h.size();
    vi L(N), R(N);
    {
        stack<int> s;
        for(ll i = N - 1; i >= 1; i--) {
            while(!s.empty() && s.top() - h[s.top()] >= i) {
                L[s.top()] = i + 1;
                s.pop();
            }
            s.push(i);
        }
        while(!s.empty()) {
            L[s.top()] = 1;
            s.pop();
        }
    }
    {
        stack<int> s;
        for(int i = 1; i < N; i++) {
            while(!s.empty() && s.top() + h[s.top()] <= i) {
                R[s.top()] = i - 1;
                s.pop();
            }
            s.push(i);
        }
        while(!s.empty()) {
            R[s.top()] = N - 1;
            s.pop();
        }
    }
    vll dp(N, INF);
    dp[0] = 0;
    stack<int> s;
    for(int i = 1; i < N; i++) {
        while(!s.empty() && R[s.top()] < i) s.pop();
        dp[i] = dp[L[i] - 1] + c[i];
        if(!s.empty()) dp[i] = min(dp[i], dp[s.top() - 1] + c[s.top()]);
        if(s.empty() || (dp[i - 1] + c[i] < dp[s.top() - 1] + c[s.top()])) s.push(i);
    }
    return dp.back();
}

vvi mask_hamming_distance(int n, vi cnt) {
	// https://codeforces.com/contest/662/problem/C
    int N = 1 << n;
    assert((int)cnt.size() == N);
    // dp[k][mask] = #columns at Hamming distance exactly k from 'mask'
    vvi dp(n + 1, vi(N, 0));
    dp[0] = cnt;
    for(int k = 1; k <= n; ++k) {
        for(int mask = 0; mask < N; ++mask) {
            ll sum = (k > 1 ? 1LL * (k - 2 - n) * dp[k - 2][mask] : 0LL);
            for(int p = 0; p < n; ++p) {
                sum += dp[k - 1][mask ^ (1 << p)];
            }
            dp[k][mask] = (sum / k);
        }
    }
    vvi res(N, vi(n + 1, 0));
    for(int mask = 0; mask < N; ++mask)
        for(int k = 0; k <= n; ++k)
            res[mask][k] = dp[k][mask];
    return res;
}

template<typename T>
T count_perm_abs_sum(vi a, int l) { // count number of permutation of a such that sum(a[i] - pi) <= l
    // https://atcoder.jp/contests/abc134/tasks/abc134_f
    if(l < 0) return 0;
    const int n = (int)a.size();
    srt(a);

    vector<vector<T>> cur(l + 1, vector<T>(n + 1, T(0)));
    vector<vector<T>> nxt(l + 1, vector<T>(n + 1, T(0)));
    cur[0][0] = T(1);

    for(int i = 0; i < n; ++i) {
        int gap_top = (i == 0 ? 0 : a[i] - a[i - 1]);
        int add_per_open = gap_top + 1;
        for(int s = 0; s <= l; ++s) fill(nxt[s].begin(), nxt[s].end(), T(0));
        for(int s = 0; s <= l; ++s) {
            for(int j = 0; j <= n; ++j) {
                T ways = cur[s][j];
                if(ways == T(0)) continue;
                ll s2 = s + 1LL * j * add_per_open;
                if(s2 > l) continue;
                nxt[(int)s2][j] += ways;
                if(j + 1 <= n) nxt[(int)s2][j + 1] += ways;
                if(j > 0) {
                    nxt[(int)s2][j] += ways * T(2 * j);
                    nxt[(int)s2][j - 1] += ways * T(1LL * j * j);
                }
            }
        }
        swap(cur, nxt);
    }
    T ans = T(0);
    for(int s = 0; s <= l; ++s) ans += cur[s][0];
    return ans;
}

ll min_steps(ll b, ll d) { // while(b) b -= gcd(b, d), steps++; return steps
    // https://atcoder.jp/contests/arc159/tasks/arc159_b
    if(b == 0) return 0;
    if(d == 0) return 1;
    vll primes;
    {
        ll x = d;
        for(ll p = 2; p * p <= x; ++p) {
            if(x % p == 0) {
                primes.pb(p);
                while(x % p == 0) x /= p;
            }
        }
        if(x > 1) primes.pb(x);
    }
    ll res = 0;
    while(b > 0) {
        ll g = std::gcd(b, d);
        ll B = b / g;
        ll D = d / g;
        ll t = B;
        for(const auto& p : primes) {
            if(D % p == 0) {
                ll rem = B % p;
                if(rem < t) t = rem;
            }
        }
        res += t;
        b -= t * g;
    }
    return res;
}

ll step(ll a, ll b) { // while(a) [a, b] = [b, abs(a, b)], steps++;
    // https://codeforces.com/contest/1848/problem/C
    if(a == 0) return 0;
    if(b == 0) return 1;
    if(a >= b) {
        ll r = a % b;
        ll k = a / b;
        // f[k] = f[k - 2] + 3;
        return (k & 1 ? step(b, r) : step(r, b)) + k + k / 2;
    }
    return 1 + step(b, abs(a - b));
}

ll advance(ll n, ll k) { // n += n % 10 k times
    if(k == 0) return n;
    int d = n % 10;
    if(d == 0) return n;
    if(d == 5) {
        if(k >= 1) n += 5;
        return n;
    }
    int first_steps = min(4LL, k);
    for(int i = 0; i < first_steps; i++) {
        n += n % 10;
    }
    k -= first_steps;
    n += 20 * (k / 4);
    k %= 4;
    while(k--) n += n % 10;
    return n;
}

vi lexicographically_smallest_derangement(const vi& a) {
    int n = a.size();
    if(n == 0) return {};
    vi suf_same(n + 2, 0);
    vi suf_val(n + 2, 0);

    suf_same[n] = 1;
    suf_val[n] = a[n - 1];
    for(int i = n - 1; i >= 1; --i) {
        if(suf_same[i + 1] && a[i - 1] == suf_val[i + 1]) {
            suf_same[i] = 1;
            suf_val[i] = a[i - 1];
        }
    }
    set<int> rem;
    for(int v = 1; v <= n; ++v) {
        rem.insert(v);
    }
    vi p(n + 1, 0);

    for(int i = 1; i <= n; ++i) {
        int need = false;
        int v = 0;

        if(i < n && suf_same[i + 1]) {
            v = suf_val[i + 1];
            if(1 <= v && v <= n && rem.count(v)) need = true;
        }

        int x = -1;
        if(need) {
            if (a[i - 1] == v) return {};
            x = v;
        } else {
            auto it = rem.begin();
            if(it == rem.end()) return {};
            x = *it;
            if(x == a[i - 1]) {
                ++it;
                if(it == rem.end()) return {};
                x = *it;
            }
        }

        p[i] = x;
        rem.erase(x);
    }
    return vi(p.begin() + 1, p.end());
}