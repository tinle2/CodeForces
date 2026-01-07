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

template<class T, typename F = function<T(const T&, const T&)>>
class FW {  
    public: 
    int n, N;
    vector<T> root;    
    T DEFAULT;
    F func;
    FW() {}
    FW(int n, T DEFAULT, F func = [](const T& a, const T& b) {return a + b;}) : func(func) { 
        this->n = n;    
        this->DEFAULT = DEFAULT;
		N = n == 0 ? -1 : log2(n);
        root.resize(n, DEFAULT);
    }
    
    inline void update_at(int id, T val) {  
        assert(id >= 0);
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    inline T get(int id) {   
        assert(id < n);
        T res = DEFAULT;
        while(id >= 0) { 
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    inline T queries_range(int left, int right) {  
        return get(right) - get(left - 1);
    }

    inline T queries_at(int i) {
        return queries_range(i, i);
    }

    inline void update_range(int l, int r, T val) {
		if(l > r) return;
        update_at(l, val), update_at(r + 1, -val);
    }
	
	inline void reset() {
		root.assign(n, DEFAULT);
	}

	ll select(ll k) {
        ll pos = -1;
        T acc = DEFAULT;
        for(ll bit = 1LL << N; bit > 0; bit >>= 1) {
            ll np = pos + bit;
            if(np < n) {
                T cand = acc + root[np];
                if(cand < k) {
                    acc = cand;
                    pos = np;
                }
            }
        }
        return pos + 1;
    }
};

const int MX = 2e5 + 5;
void solve() {
    vvi divs(MX * 2);
    for(int i = 1; i < MX; i++) {
        for(int j = i; j < MX * 2; j += i) {
            divs[j].pb(i);
        }
    }
    vector<vpii> segs(MX);
    for(int k = 1; k < MX; k++) {
        const auto& D = divs[k * 2];
        const int N = D.size();
        for(int i = 0; i < N && D[i] < k; i++) {
            int cnt = 0;
            for(int j = i + 1; j < N && D[j] < k; j++) {
                int u = D[i];
                int v = D[j];
                if((u + v) > k || (k % u == 0 && k % v == 0)) cnt++;     
            }
            if(cnt) segs[D[i]].pb({k, cnt});
        }
    }
    vector<vpii> Q(MX);
    int q; cin >> q;
    vll ans(q);
    auto nc3 = [](ll n) -> ll {
        return (n * (n - 1) * (n - 2)) / 6;
    };
    for(int i = 0; i < q; i++) {
        int l, r; cin >> l >> r;
        Q[l].pb({r, i});
        ans[i] = nc3(r - l + 1);
    }
    FW<ll> root(MX * 2 + 10, 0);
    for(int i = MX; i >= 1; i--) {
        for(auto& [v, c] : segs[i]) {
            root.update_at(v, c);
        }
        for(auto& [r, id] : Q[i]) {
            ans[id] -= root.get(r);
        }
    }
    for(auto& x : ans) {
        cout << x << '\n';
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
