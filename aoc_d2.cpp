#include <bits/stdc++.h>
using namespace std;

using i128 = __int128_t;
using ll = long long;

i128 blocks[19];
i128 pows10[19];

i128 f(i128 x) {
    if(x <= 11) return 0;
    int n = 0;
    vector<int> digs;
    {
        i128 v = x;
        while(v) {
            n++;
            digs.push_back((int)(v % 10));
            v /= 10;
        }
    }

    i128 ans = blocks[n - 1];
    vector<i128> dp(n / 2 + 1);

    for(int d = 1; d <= n / 2; d++) {
        if(n % d != 0) continue;

        i128 factor = 1;
        for(int i = 0; i < n / d - 1; i++) {
            factor = factor * pows10[d] + 1;
        }

        i128 prefsum = 0;

        for(int hidigs = 0; hidigs < n; hidigs++) {
            int idx = n - hidigs - 1;
            int limit = digs[idx];

            for(int ndig = (hidigs ? 0 : 1); ndig < limit; ndig++) {
                if(hidigs < d || digs[idx + d] == ndig) {
                    int power_idx = max(0, d - hidigs - 1);
                    i128 leftpower = pows10[power_idx];
                    dp[d] += leftpower * prefsum;
                    if(hidigs + 1 <= d) {
                        dp[d] += (i128)ndig * factor * leftpower * leftpower;
                    }
                    if(hidigs + 1 < d) {
                        dp[d] += factor * leftpower * (leftpower - 1) / 2;
                    }
                }
            }

            if(d <= hidigs && digs[idx + d] != digs[idx]) {
                break;
            }
            if(hidigs < d) {
                int power_idx = d - hidigs - 1;
                prefsum += (i128)digs[idx] * factor * pows10[power_idx];
            }
        }
    }

    for(int d = 1; d <= n / 2; d++) {
        if(n % d != 0) continue;
        for(int m = d * 2; m <= n / 2; m += d) {
            if(n % m == 0) {
                dp[m] -= dp[d];
            }
        }
    }

    i128 sum_dp = 0;
    for(i128 v : dp) sum_dp += v;

    return ans + sum_dp;
}

void solve() {
    pows10[0] = 1;
    for(int i = 0; i < 18; i++) {
        pows10[i + 1] = pows10[i] * 10;
    }

    for(int n = 2; n <= 18; n++) {
        i128 limit = pows10[n] - 1;
        blocks[n] = limit + f(limit);
    }

    vector<pair<ll, ll>> ranges;
    string s;
    ll last = -1;

    while(getline(cin, s)) {
        int L = (int)s.size();
        for(int i = 0; i < L; ) {
            if(s[i] >= '0' && s[i] <= '9') {
                int j = i;
                while (j < L && s[j] >= '0' && s[j] <= '9') j++;
                ll v = stoll(s.substr(i, j - i));
                if(last == -1) {
                    last = v;
                } else {
                    ranges.push_back({last, v});
                    last = -1;
                }
                i = j;
            } else {
                i++;
            }
        }
    }

    vector<pair<ll, bool>> events;
    for(auto &pr : ranges) {
        ll l = pr.first;
        ll r = pr.second;
        events.push_back({l, false});
        events.push_back({r + 1, true});
    }
    sort(events.begin(), events.end());

    i128 ans = 0;
    int bal = 0;

    for(auto &e : events) {
        ll at = e.first;
        bool type = e.second;
        if(type) {
            bal--;
            if(bal == 0) {
                ans += f((i128)at);
            }
        } else {
            if(bal == 0) {
                ans -= f((i128)at);
            }
            bal++;
        }
    }

    ll out = (ll)ans;
    cout << out << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie();
    cout.tie();
    solve();
}

