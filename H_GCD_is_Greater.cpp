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

const int MX = 4e5 + 5;
const int B = 20;
vi zero[B], pos[MX];
int cnt[B];
void solve() {
    int n, x; cin >> n >> x;
    vi a(n + 1);
    memset(cnt, 0, sizeof(cnt));
    for(auto& it : zero) {
        vi().swap(it);
    }
    int M = 0;
    for(int i = 1; i <= n; i++) {
        cin >> a[i];
        M = max(M, a[i]);
        for(int j = 0; j < B; j++) {
            if(a[i] >> j & 1) {
                cnt[j]++;
            } else {
                zero[j].pb(i);
            }
        }
    }
    auto get_and = [&](int x, int y) -> int {
        int res = 0;
        for(int b = 0; b < B; b++) {
            int now = cnt[b];
            if(x >> b & 1) now--;
            if(y >> b & 1) now--;
            if(now == n - 2) res |= 1 << b;
        }
        return res;
    };
    auto out = [&](int i, int j) -> void {
        cout << "YES" << '\n';
        cout << 2 << ' ' << a[i] << ' ' << a[j] << '\n';
        cout << n - 2 << ' ';
        for(int k = 1; k <= n; k++) {
            if(i == k || j == k) continue;
            cout << a[k] << ' ';
        }
        cout << '\n';
    };
    vi bad(n + 1);
    for(auto& it : zero) {
        if((int)it.size() > 2 || (int)it.size() == 0) continue;
        for(auto& i : it) {
            bad[i] = 1;
            for(int j = 1; j <= n; j++) {
                if(i == j) continue;
                int g = gcd(a[i], a[j]);
                if(g > get_and(a[i], a[j]) + x) {
                    out(i, j);
                    return;
                }
            }
        }
    }
    for(int i = 1; i <= M; i++) {
        vi().swap(pos[i]);
    }
    int gand = a[1];
    for(int i = 1; i <= n; i++) {
        if(!bad[i] && (int)pos[a[i]].size() < 2) {
            pos[a[i]].pb(i);
        }
        gand &= a[i];
    }
    for(int g = M; g > gand + x; g--) {
        vi now;
        for(int i = g; i <= M && (int)now.size() < 2; i += g) {
            for(auto& j : pos[i]) {
                now.pb(j);
            }
        }
        if((int)now.size() >= 2) {
            assert(get_and(a[now[0]], a[now[1]]) + x < g);
            out(now[0], now[1]);
            return;
        }
    }
    cout << "NO" << '\n';
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
