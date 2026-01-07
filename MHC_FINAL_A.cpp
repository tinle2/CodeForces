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

void solve() {
    int n, m; cin >> n >> m;
    vi a(n + 1), b(n + 1);
    vi pa(n + 1), pb(n + 1);
    for(int i = 1; i <= n; i++) {
        cin >> a[i];
        pa[a[i]] = i;
    }
    for(int i = 1; i <= n; i++) {
        cin >> b[i];
        pb[b[i]] = i;
    }
    set<pii> s;
    for(int i = 0; i < m; i++) {
        int x, y; cin >> x >> y;
        s.insert({x, y});
    }
    auto bad = [&](int x, int y) -> int {
        return s.count({min(x, y), max(x, y)});
    };
    auto sw = [&](int i, int j) -> int {
        if(bad(i, j)) return 0;
        swap(a[i], a[j]);
        pa[a[i]] = i;
        pa[a[j]] = j;
        return 1;
    };
    int res = 0;
    for(int i = 1; i <= n; i++) {
        if(pa[i] == pb[i]) continue;
        int k = i;
        while(k <= n && pa[k] != pb[i]) {
            k++;
        }
        if(k > n) {
            cout << -1 << '\n';
            return;
        }
        for(int v = k - 1; v >= i; v--) {
            if(!sw(pa[v], pa[v + 1])) {
                cout << -1 << '\n';
                return;
            }
            res++;
        }
    }
    cout << res << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
