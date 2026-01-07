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

const int K = 5e5 + 5;
bitset<K> possible_subsets_knapsack(int n, const vi &sizes) {
    vi freq(n + 1); 
    for (int s : sizes) {
        if (1 <= s && s <= n) {
            freq[s]++;
        }
    }
    bitset<K> knapsack;
    knapsack.set(0);
    for (int s = 1; s <= n; s++) {
        if (freq[s] >= 3) {
            int move = (freq[s] - 1) / 2;
            if (2 * s <= n) freq[2 * s] += move;
            freq[s] -= 2 * move;
        }
        for (int r = 0; r < freq[s]; r++)
            knapsack |= knapsack << s;
    }
    return knapsack;
}
void solve() {
    int n; cin >> n;
    vi a(n * 2); cin >> a;
    for(auto& x : a) {
        x--;
    }
    const int inf = 1e9;
    int mx1 = -inf, mx2 = -inf;
    for(auto& x : a) {
        if(x > mx1) {
            mx2 = mx1;
            mx1 = x;
        } else {
            mx2 = max(mx2, x);
        }
        if(mx2 > x) {
            cout << "No" << '\n';
            return;
        }
    }
    vvi graph(2 * n);
    stack<int> s;
    for(auto& x : a) {
        int mx = x;
        while(!s.empty() && x < s.top()) {
            int v = s.top(); s.pop();
            graph[x].pb(v);
            graph[v].pb(x);
            mx = max(mx, v);
        }
        s.push(mx);
    }
    vi vis(n * 2);
    int C[2] = {};
    auto dfs = [&](auto& dfs, int i = 0, int c = 0) -> void {
        if(vis[i]) return;
        C[c]++;
        vis[i] = 1;
        for(auto& j : graph[i]) {
            dfs(dfs, j, c ^ 1);
        } 
    };
    int base = 0;
    vi sz;
    for(int i = 0; i < 2 * n; i++) {
        if(!vis[i]) {
            C[0] = C[1] = 0;
            dfs(dfs, i);
            base += min(C[0], C[1]);
            sz.pb(abs(C[0] - C[1]));
        }
    }
    int need = n - base;
    if(need < 0 || need > n) {
        cout << "No" << '\n';
        return;
    }
    auto it = possible_subsets_knapsack(need + 1, sz);
    cout << (it.test(need) ? "Yes" : "No") << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
