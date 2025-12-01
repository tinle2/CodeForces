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

int dp[20][20];
int a[20];
int choose[20][20];
void solve() {
    int n; cin >> n;
    for(int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    memset(dp, -1, sizeof(dp));
    // 0 0 0 0 0
    // 1 0 0 0 0
    // 2 1 0 0 0
    // 3 3 3 0 0
    // 3 0 0 0 0
    // 3 1 0 0 0
    // 3 2 2 0 0
    // 3 2 1 0 0
    // 4 4 4 4 4
    // 4 
    {
        auto dfs = [&](auto& dfs, int l, int r) -> int {
            if(l > r) return 0; 
            if(l == r) {
                return a[l] == 0 ? 1 : a[l];
            }
            auto& res = dp[l][r];
            if(res != -1) return res;
            res = (r - l + 1) * (r - l + 1);
            for(int k = l; k <= r; k++) {
                auto v = dfs(dfs, l, k - 1) + dfs(dfs, k + 1, r) + (a[k] == 0 ? 1 : a[k]);
                if(v > res) {
                    choose[l][r] = k;
                    res = v;
                }
            }
            return res;
        };
        cout << dfs(dfs, 1, n) << ' ';
    }
    vpii ops;
    {
        auto op = [&](int l, int r) -> void {
            int s[30] = {};
            for(int i = l; i <= r; i++) {
                if(a[i] <= 25) {
                    s[a[i]] = 1;
                }
            }
            int mex = 0;
            while(s[mex]) mex++;
            for(int i = l; i <= r; i++) {
                a[i] = mex;
            }
            ops.pb({l, r});
        };
        auto run = [&](auto& run, int l, int r) -> void {
            if(l > r) return;
            if(l == r) {
                while(a[l] != 1) {
                    op(l, r);
                }
                return;
            }
            for(int j = r - 1; j >= l; j--) {
                run(run, l, j);
            }
            while(a[r] != 0) {
                op(r, r);
            }
            op(l, r);
            assert(a[r] == r - l + 1);
        };
        auto dfs = [&](auto& dfs, int l, int r) -> void {
            if(l > r) return;
            if(l == r) {
                if(a[l] == 0) {
                    ops.pb({l, l});
                }
                return;
            }
            int k = choose[l][r];
            if(k == 0) {
                run(run, l, r);
                return;
            }
            dfs(dfs, l, k - 1);
            dfs(dfs, k, k);
            dfs(dfs, k + 1, r);
        };
        dfs(dfs, 1, n);
    }
    cout << ops.size() << '\n';
    for(auto& [l, r] : ops) {
        cout << l << ' ' << r << '\n';
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
