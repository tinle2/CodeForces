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

const int BUF_SZ = 1 << 15; // do init_output() at the start of the main function

inline namespace Input {
    char buf[BUF_SZ];
    int pos;
    int len;
    char next_char() {
        if (pos == len) {
            pos = 0;
            len = (int)fread(buf, 1, BUF_SZ, stdin);
            if (!len) { return EOF; }
        }
        return buf[pos++];
    }

    int read_int() {
        int x;
        char ch;
        int sgn = 1;
        while (!isdigit(ch = next_char())) {
            if (ch == '-') { sgn *= -1; }
        }
        x = ch - '0';
        while (isdigit(ch = next_char())) { x = x * 10 + (ch - '0'); }
        return x * sgn;
    }
}
inline namespace Output {
    char buf[BUF_SZ];
    int pos;

    void flush_out() {
        fwrite(buf, 1, pos, stdout);
        pos = 0;
    }

    void write_char(char c) {
        if (pos == BUF_SZ) { flush_out(); }
        buf[pos++] = c;
    }

    void write_int(ll x) {
        static char num_buf[100];
        if (x < 0) {
            write_char('-');
            x *= -1;
        }
        int len = 0;
        for (; x >= 10; x /= 10) { num_buf[len++] = (char)('0' + (x % 10)); }
        write_char((char)('0' + x));
        while (len) { write_char(num_buf[--len]); }
        write_char('\n');
    }

    void init_output() { assert(atexit(flush_out) == 0); }
}

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
        return root[x] = find(root[x]);
    }
};

void solve() {
    int n = read_int();
    vi l(n), r(n);
    vi b;
    {
        vpii a(n);
        for(auto& it : a) {
            it.ff = read_int();
        }
        for(auto& it : a) {
            it.ss = read_int();
        }
        sort(all(a), [](const auto& x, const auto& y) {return x.ss < y.ss;});
        for(int i = 0; i < n; i++) {
            l[i] = a[i].ff;
            r[i] = a[i].ss;
            b.pb(l[i]);
            b.pb(r[i]);
        }
        srtU(b);

        auto id = [&](int x) -> int {
            return int(lb(all(b), x) - begin(b));
        };
        for(int i = 0; i < n; i++) {
            l[i] = id(l[i]) + 1;
            r[i] = id(r[i]) + 1;
        }
    }
    const int N = b.size();
    vi diff(N + 1);
    DSU root(N + 1);
    auto run = [&](int k) -> int {
        fill(all(diff), 0);
        for(int i = 1; i <= N; i++) {
            root.root[i] = i - 1;
        }
        int last = 1;
        int res = 0;
        for(int i = 0, cnt = 0; i < n; i++) {
            if(l[i] < last) continue;
            root.root[r[i]] = -1;
            diff[r[i]]++;
            int p = root.find(l[i] - 1);
            if(p < last) {
                cnt++;
            } else {
                if(--diff[p] == 0) {
                    root.root[p] = p - 1;
                }
            }
            if(cnt == k) {
                cnt = 0;
                last = r[i] + 1;
                res++;
            }
        }
        return res;
    };
    vi ans(n + 2);

    ans[1] = run(1);
    ans[n] = run(n);
    auto dfs = [&](auto& dfs, int l, int r) -> void {
        if(l > r) return;
        if(ans[l - 1] == ans[r + 1]) {
            for(int i = l; i <= r; i++) {
                ans[i] = ans[i - 1];
            }
            return;
        }
        int m = (l + r) >> 1;
        ans[m] = run(m);
        dfs(dfs, l, m - 1);
        dfs(dfs, m + 1, r);
    };
    dfs(dfs, 2, n - 1);
    int res = 0;
    for(int i = 1; i <= n; i++) {
        res = max(res, ans[i] * i);
    }
    write_int(res);
}

signed main() {
    IOS;
    init_output();
    startClock
    int t = read_int();
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
