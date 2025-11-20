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

struct Undo_DSU {
    int n;
    using Record = ar(4);
    vi par, rank;
    stack<Record> st;

    Undo_DSU(int n) : n(n) {
        par.rsz(n);
        rank.rsz(n, 1);
        iota(par.begin(), par.end(), 0);
    }
    
    int find(int v) {
        return par[v] == v ? v : find(par[v]);
    }
    
    bool merge(int u, int v, bool save = true) {
        int ru = find(u), rv = find(v);
        if(ru == rv) {
            return false;
        }
        if(rank[ru] < rank[rv]) swap(ru, rv);
        st.push({ru, rank[ru], rv, rank[rv]});
        par[rv] = ru;
        rank[ru] += rank[rv];
        return true;
    }
    
    int rollBack() {
        if(!st.empty()) {
            Record rec = st.top();
            st.pop();
            int ru = rec[0], oldRankU = rec[1], rv = rec[2], oldRankV = rec[3];
            par[rv] = rv;
            rank[ru] = oldRankU;
            rank[rv] = oldRankV;
            return true;
        }
        return false;
    }
    
    bool same(int u, int v) {
        return find(u) == find(v);
    }
    
    int get_rank(int u) {
        return rank[find(u)];
    }
};

void solve() {
    int n, m; cin >> n >> m;
    vi u(m + 1), v(m + 1), d(m + 1);
    for(int i = 1; i <= m; i++) {
        cin >> u[i] >> v[i] >> d[i];
    }
    int q; cin >> q;
    vi ans(q, -1);
    const int B = 700;
    var(3) upd, queries;
    vi in(m + 1), used_edges(m + 1);
    Undo_DSU root(n + 1);
    for(int i = 0; i < q; i++) {
        int t; cin >> t;
        if(t == 1) {
            int b, r; cin >> b >> r;
            upd.pb({b, r, i});
        } else {
            int s, w; cin >> s >> w;
            queries.pb({s, w, i});
        }
        if(!((i + 1) % B == 0 || i == q - 1)) continue;
        vi ord(m);
        iota(all(ord), 1);
        sort(all(ord), [&](int x, int y) {return d[x] > d[y];});
        sort(all(queries), [](const auto& x, const auto& y) {return x[1] > y[1];});
        for(const auto& [b, r, id] : upd) {
            in[b] = 1;
        }
        const int U = upd.size();
        int j = 0;
        for(const auto& [s, w, tq] : queries) {
            while(j < m && d[ord[j]] >= w) {
                if(!in[ord[j]]) {
                    root.merge(u[ord[j]], v[ord[j]]);
                }
                j++;
            }
            int edges = 0;
            for(int k = U - 1; k >= 0; k--) {
                const auto& [b, r, tu] = upd[k];
                if(tu > tq || used_edges[b]) continue;
                used_edges[b] = 1;
                if(r < w) continue;
                if(root.merge(u[b], v[b])) edges++;
            }
            for(const auto& [b, r, tu] : upd) {
                if(used_edges[b]) continue;
                used_edges[b] = 1;
                if(d[b] < w) continue;
                if(root.merge(u[b], v[b])) edges++;
            }
            ans[tq] = root.get_rank(s);
            while(edges--) {
                root.rollBack();
            }
            for(const auto& [b, r, tu] : upd) {
                used_edges[b] = 0;
            }
        }
        for(const auto& [b, r, _] : upd) {
            in[b] = 0;
            d[b] = r;
        }
        var(3)().swap(upd);
        var(3)().swap(queries);
        while(root.rollBack());
    }
    for(auto& w : ans) {
        if(w != -1) {
            cout << w << '\n';
        }
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
