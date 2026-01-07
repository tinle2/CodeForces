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

pll operator+(const pll& A, const pll& B) {
    if(A.ss == 0) return B;
    if(B.ss == 0) return A;
    if(A.ff < B.ff) return A;
    if(B.ff < A.ff) return B;
    return {A.ff, A.ss + B.ss};
}
const ll INF = 1e15;
const int MX = 120005;
const int B = 400;
const int NB = MX / B + 5;
ll a[MX], lazy[NB], x[MX];
pll tracker[NB], mn[NB];
ll cnt[NB];
int seen[MX];
pll tracker_single[MX];

void solve() {
    int n; cin >> n;
    for(int i = 0; i < n; i++) {
        cin >> a[i];
    }
    auto BL = [&](int p) -> int { return (p / B) * B; };
    auto BR = [&](int p) -> int { return min(BL(p) + B - 1, n - 1); };
    auto rebuild = [&](int b) -> void {
        mn[b] = {INF, 0};
        for(int i = b * B; i <= BR(b * B); i++) {
            mn[b] = mn[b] + pll(x[i], 1);
        }
    };
    auto push = [&](int b) -> void {
        if(tracker[b].ss == 0 && lazy[b] == 0) return;
        for(int i = b * B; i <= BR(b * B); i++) {
            tracker_single[i] = tracker_single[i] + pll(tracker[b].ff + x[i], tracker[b].ss);    
            x[i] += lazy[b];
        }
        lazy[b] = 0;
        tracker[b] = {0, 0};
    };
    int test = 1;
    vi touch;
    auto update = [&](int l, int r, int d) -> void {
        int bl = l / B;
        int br = r / B;
        if(bl == br) {
            push(bl);
            for(int i = l; i <= r; i++) {
                x[i] += d;
            }
            rebuild(bl);
            return;
        }
        push(bl);
        push(br);
        for(int i = l; i <= BR(l); i++) {
            x[i] += d;
        }
        for(int i = BL(r); i <= r; i++) {
            x[i] += d;
        }
        rebuild(bl);
        rebuild(br);
        for(int i = bl + 1; i < br; i++) {
            lazy[i] += d;
        }
    };
    auto accumulate = [&](int r) -> void {
        for(int b = 0; b <= r / B; b++) {
            if(lazy[b] + mn[b].ff == -1) cnt[b] += mn[b].ss;
            tracker[b] = tracker[b] + pll(lazy[b], 1);
        }
    };
    for(int i = 0; i < n; i++) {
        x[i] = i;
    }
    for(int b = 0; b * B < n; b++) {
        rebuild(b);
    }
    auto query_single = [&](int i) -> int {
        return tracker_single[i].ff == -1 ? tracker_single[i].ss : 0;
    };
    auto query = [&](int l, int r) -> ll {
        int bl = l / B;
        int br = r / B;
        if(bl == br) {
            push(bl);
            ll res = 0;
            for(int i = l; i <= r; i++) {
                res += query_single(i);
            }
            rebuild(bl);
            return res;
        }
        push(bl);
        push(br);
        ll res = 0;
        for(int i = l; i <= BR(l); i++) {
            res += query_single(i);
        }
        for(int i = BL(r); i <= r; i++) {
            res += query_single(i);
        }
        rebuild(bl);
        rebuild(br);
        for(int i = bl + 1; i < br; i++) {
            res += cnt[i];
        }
        return res;
    };
    vector<vector<pii>> Q(n);
    int q; cin >> q;
    for(int i = 0; i < q; i++) {
        int l, r; cin >> l >> r;
        l--, r--;
        Q[r].pb({l, i});
    }
    stack<int> min_st, max_st;
    min_st.push(-1);
    max_st.push(-1);
    // mx - mn - r + l = 0
    vll ans(q);
    for(int i = 0; i < n; i++) {
        update(0, i, -i - 1);    
        while(min_st.size() > 1 && a[min_st.top()] > a[i]) {
            int j = min_st.top(); min_st.pop();
            update(min_st.top() + 1, j, a[j] - a[i]);
        }
        while(max_st.size() > 1 && a[max_st.top()] < a[i]) {
            int j = max_st.top(); max_st.pop();
            update(max_st.top() + 1, j, a[i] - a[j]);
        }
        min_st.push(i);
        max_st.push(i);
        accumulate(i);
        for(auto& [l, id] : Q[i]) {
            ans[id] = query(l, i);
        }
        update(0, i, i + 1);
    }
    for(auto& v : ans) {
        cout << v << '\n';
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    // cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
