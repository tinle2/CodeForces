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

const int M = 1e8;
const int MX = 2e5 + 5;
int mp[M], pre[MX], now[MX];
void solve() {
    // (r - l + 1) = (pre[r] - pre[l - 1]) * t
    // r - l + 1 = pre[r] * t - pre[l - 1] * t
    // r - pre[r] * t = l - 1 - pre[l - 1] * t
    string s; cin >> s;
    int n = s.size();
    s = ' ' + s;
    const int B = sqrt(n) + 1;
    ll res = 0;
    for(int i = 1; i <= n; i++) {
        pre[i] = pre[i - 1] + (s[i] == '1');
    }
    for(int t = 1; t < B; t++) {
        int mn = 0;
        for(int i = 0; i <= n; i++) {
            now[i] = i - pre[i] * t;
            mn = min(mn, now[i]);
        }
        for(int i = 0; i <= n; i++) {
            res += mp[now[i] - mn]++;
        }
        for(int i = 0; i <= n; i++) {
            mp[now[i] - mn] = 0;
        }
    }
    vi ones;
    for(int i = 1; i <= n; i++) {
        if(s[i] == '1') ones.pb(i);
    }
    ones.pb(n + 1);
    const int N = ones.size();
    for(int i = 1, j = 0; i <= n; i++) {
        while(ones[j] < i) j++;
        for(int k = 1; j + k < N && k * B <= n; k++) {
            int l = max(k * B, ones[j + k - 1] - i + 1);
            int r = ones[j + k] - 1 - i + 1;
            if(l <= r) {
                res += r / k - (l - 1) / k;
            }
        }
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
