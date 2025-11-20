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

pair<vll, vll> factorize(ll n) {
    using u64  = uint64_t;
    using u128 = unsigned __int128;
    vll pf;

    auto mul_mod = [](u64 a, u64 b, u64 m) -> u64 {
        return (u64)((u128)a * b % m);
    };
    auto pow_mod = [&](u64 a, u64 e, u64 m) -> u64{
        u64 r = 1;
        while(e) { if (e & 1) r = mul_mod(r, a, m); a = mul_mod(a, a, m); e >>= 1; }
        return r;
    };
    auto isPrime = [&](u64 x)-> bool {
        if (x < 2) return false;
        for(u64 p:{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
            if(x % p == 0) return x == p;
        u64 d = x - 1, s = 0;
        while((d & 1) == 0) { d >>= 1; ++s; }
        for(u64 a:{2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL}) {
            u64 y = pow_mod(a, d, x);
            if(y == 1 || y == x - 1) continue;
            bool comp = true;
            for(u64 r = 1; r < s; ++r) {
                y = mul_mod(y, y, x);
                if(y == x - 1) { comp = false; break; }
            }
            if(comp) return false;
        }
        return true;
    };
    auto rho = [&](u64 n) -> u64{                
        if((n & 1) == 0) return 2;
        mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count()); 
        uniform_int_distribution<u64> dist(2, n - 2);
        while(true) {
            u64 y = dist(rng), c = dist(rng), m = 128, g = 1, r = 1, q = 1, ys, x;
            auto f = [&](u64 v){ return (mul_mod(v, v, n) + c) % n; };
            while(g == 1) {
                x = y;  for(u64 i=0; i < r; ++i) y = f(y);
                u64 k = 0;
                while(k < r && g == 1) {
                    ys = y;
                    u64 lim = min(m, r - k);
                    for(u64 i = 0; i < lim; ++i){ y = f(y); q = mul_mod(q, (x > y ? x - y : y - x), n); }
                    g = gcd(q, n);  k += m;
                }
                r <<= 1;
            }
            if(g == n) {
                do { ys = f(ys); g = gcd((x > ys ? x - ys : ys - x), n); } while (g == 1);
            }
            if(g != n) return g;
        }
    };

    auto fact = [&](auto& fact, u64 v) -> void {
        static const int small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
        for(int p : small){ if((u64)p * (u64)p > v) break;
            while(v % p == 0){ pf.pb(p); v /= p; }
        }
        if(v == 1) return;
        if(isPrime(v)){ pf.pb(v); return; }
        u64 d = rho(v);
        fact(fact, d); fact(fact, v / d);

    };

    if(n <= 0) return {};          
    fact(fact, (u64)n);
    srt(pf);
    vpll uniq;
    vll P;
    for(size_t i = 0; i < pf.size();) {
        size_t j = i; while(j < pf.size() && pf[j] == pf[i]) ++j;
        uniq.pb({pf[i], int(j - i)});
        P.pb(pf[i]);
        i = j;
    }
    vll divs = {1};
    for(auto [p, e] : uniq) {
        size_t sz = divs.size();
        ll pk = 1;
        for(int k = 1; k <= e;++k){
            pk *= p;
            for(size_t i = 0; i < sz; ++i) divs.pb(divs[i] * pk);
        }
    }
    srt(divs);
    return {divs, P};
}

void solve() {
    int n, q; cin >> n >> q;
    vi a(n), b(n), c(n); cin >> a >> b >> c;
    vi g(n);
    vpll ans;
    for(int iter = 0; iter < 2; iter++) {
        auto [d0, p0] = factorize(a[0]);
        auto [d1, p1] = factorize(b[0]);
        const int N = d0.size();
        const int M = d1.size();
        auto id = [](const vll& a, int x) -> int {
            return int(lb(all(a), x) - begin(a));
        };
        vvi cnt(N, vi(M));
        vvll cost(N, vll(M));
        for(int i = 0; i < n; i++) {
            if(!iter) {
                g[i] = gcd(a[i], b[i]);
            }
            {
                int A = id(d0, gcd(a[i], a[0]));
                int B = id(d1, gcd(b[i], b[0]));
                cnt[A][B]++;
            }
            {
                int A = id(d0, gcd(a[0], b[i]));
                int B = id(d1, gcd(b[0], a[i]));
                cnt[A][B]++;
                cost[A][B] += c[i];
            }
            {
                int A = id(d0, gcd(a[0], g[i]));
                int B = id(d1, gcd(b[0], g[i]));
                cnt[A][B]--;
                cost[A][B] -= c[i];
            }
        }
        for(const auto& p : p0) {
            for(int i = N - 1, j = N - 1; i >= 0; i--) {
                if(d0[i] % p == 0) {
                    j = min(i - 1, j);
                    int k = d0[i] / p;
                    while(d0[j] != k) j--;
                    assert(j < i);
                    for(int jj = 0; jj < M; jj++) {
                        cnt[j][jj] += cnt[i][jj];
                        cost[j][jj] += cost[i][jj];
                    }
                } 
            }
        }
        for(const auto& p : p1) {
            for(int i = M - 1, j = M - 1; i >= 0; i--) {
                if(d1[i] % p == 0) {
                    j = min(i - 1, j);
                    int k = d1[i] / p;
                    while(d1[j] != k) j--;
                    assert(j < i);
                    for(int jj = 0; jj < N; jj++) {
                        cnt[jj][j] += cnt[jj][i];
                        cost[jj][j] += cost[jj][i];
                    }
                } 
            }
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++) {
                if(cnt[i][j] == n) {
                    ans.pb({cost[i][j] + iter * c[0], d0[i] + d1[j]});
                }
            }
        }
        swap(a[0], b[0]);
    }
    srtU(ans);
    for(int i = 1; i < int(ans.size()); i++) {
        ans[i].ss = max(ans[i].ss, ans[i - 1].ss); 
    }
    while(q--) {
        ll d; cin >> d;
        pll key = {d + 1, -1};
        cout << prev(upper_bound(all(ans), key))->ss << (q == 0 ? '\n' : ' ');
    }
    
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        // cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
