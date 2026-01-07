#include<bits/stdc++.h>
#if defined(LOCAL) && __has_include("debug.h")
#include "debug.h"
#else
#define debug(...)
#endif
using i64 = long long;
using u64 = unsigned long long;
using i128 = __int128;

const int inf = 1e9 + 7;
struct wavelet_psgt {
    private:
    struct Node {
        int cnt;
        i64 sm;
        Node(int cnt = 0, i64 sm = 0) : cnt(cnt), sm(sm) {}
        friend Node operator+(const Node& x, const Node& y) { 
            return {x.cnt + y.cnt, x.sm + y.sm}; 
        };
        friend Node operator-(const Node& x, const Node& y) { 
            return {x.cnt - y.cnt, x.sm - y.sm}; 
        };
    };
    int n;
    std::vector<Node> root;
    std::vector<int> t;
    std::vector<std::pair<int, int>> child;
    std::vector<int> a;

    int new_node() { 
        root.push_back(Node(0, 0)); 
        child.push_back({0, 0}); 
        return root.size() - 1; 
    }

    int get_id(i64 x) { 
        return int(lower_bound(begin(a), end(a), x) - begin(a)); 
    }

    public:
    wavelet_psgt() {}

    wavelet_psgt(const std::vector<int>& arr) : a(arr) {
        t.resize(arr.size());
        new_node(); 
        sort(begin(a), end(a));
        a.erase(unique(begin(a), end(a)), end(a));
        n = a.size();
        for(int i = 0, prev = 0; i < (int)arr.size(); i++) {
            t[i] = new_node();
            update(t[i], prev, get_id(arr[i]), Node(1, arr[i]), 0, n - 1);
            prev = t[i];
        }
    }

    void update(int curr, int prev, int id, Node delta, int left, int right) {  
        root[curr] = root[prev];    
        child[curr] = child[prev];
        if(left == right) { 
            root[curr] = root[curr] + delta;
            return;
        }
        int middle = (left + right) >> 1;
        if(id <= middle) {
            child[curr].first = new_node(), update(child[curr].first, child[prev].first, id, delta, left, middle);
        } else {
            child[curr].second = new_node(), update(child[curr].second, child[prev].second, id, delta, middle + 1, right);
        }
        root[curr] = root[child[curr].first] + root[child[curr].second];
    }

    int kth(int l, int r, int k) {
        return kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    i64 sum_kth(int l, int r, int k) {
        return sum_kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    int kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) {
            return -inf;
        }
        if(left == right) {
            return a[left];
        }
        int middle = (left + right) >> 1;
        int left_cnt = root[child[r].first].cnt - root[child[l].first].cnt;
        if(left_cnt >= k) {
            return kth(child[l].first, child[r].first, k, left, middle);
        }
        return kth(child[l].second, child[r].second, k - left_cnt, middle + 1, right);
    }

    i64 sum_kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) {
            return -inf;
        }
        if(k <= 0) {
            return 0;
        }
        if(left == right) {
            return (i64)k * a[left];
        }
        int middle = (left + right) >> 1;
        int left_cnt = root[child[r].first].cnt - root[child[l].first].cnt;
        if(left_cnt >= k) {
            return sum_kth(child[l].first, child[r].first, k, left, middle); 
        }
        return root[child[r].first].sm - root[child[l].first].sm + sum_kth(child[l].second, child[r].second, k - left_cnt, middle + 1, right);
    }

    int median(int l, int r) {
        return kth(l, r, (r - l + 2) / 2);
    }

    Node query_leq(int l, int r, int x) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], 0, get_id(x + 1) - 1, 0, n - 1);
    }

    Node query_eq(int l, int r, int x) {
        return query_leq(l, r, x) - query_leq(l, r, x - 1);
    }

    Node queries_range(int l, int r, i64 low, i64 high) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], get_id(low), get_id(high + 1) - 1, 0, n - 1);
    }

    Node query(int l, int r, int start, int end, int left, int right) {
        if(left > end || right < start || left > right) {
            return Node();
        }
        if(start <= left && right <= end) {
            return root[r] - root[l];
        }
        int middle = (left + right) >> 1;
        return query(child[l].first, child[r].first, start, end, left, middle) + query(child[l].second, child[r].second, start, end, middle + 1, right);
    }

    std::pair<int, int> kth_in_range(int l, int r, int start, int end, int k, int left, int right) {
        int C = root[r].cnt - root[l].cnt;
        if(left > end || right < start || left > right || C == 0) return {-1, 0};
        if(start <= left && right <= end && C < k) {
            return {-1, C};
        }
        if(left == right) {
            return {a[left], C};
        }
        int middle = (left + right) >> 1;
        auto [lv, lc] = kth_in_range(child[l].first, child[r].first, start, end, k, left, middle);
        if(lv != -1) {
            return {lv, -1};
        }
        auto [rv, rc] = kth_in_range(child[l].second, child[r].second, start, end, k - lc, middle + 1, right);
        if(rv != -1) {
            return {rv, -1};
        }
        return {-1, lc + rc};
    }

    int kth_in_range(int l, int r, i64 left, i64 right, int k) {
		// https://atcoder.jp/contests/abc324/tasks/abc324_g
        return kth_in_range(l == 0 ? 0 : t[l - 1], t[r], get_id(left), get_id(right + 1) - 1, k, 0, n - 1).first; 
    }
	
	i64 first_missing_number(int l, int r) { // https://cses.fi/problemset/task/2184/
        i64 s = 1;
        return first_missing_number(l == 0 ? 0 : t[l - 1], t[r], 0, n - 1, s);
    }

    i64 first_missing_number(i64 l, i64 r, i64 left, i64 right, i64 &s) {
        Node seg = root[r] - root[l];
        if(s < a[left] || seg.cnt == 0) {
            return s;
        }
        if(a[right] <= s) {
            s += seg.sm;
            return s;
        }
        i64 middle = (left + right) >> 1;
        first_missing_number(child[l].first, child[r].first, left, middle, s);
        first_missing_number(child[l].second, child[r].second, middle + 1, right, s);
        return s;
    }
};

template<class T, typename F = std::function<T(const T&, const T&)>>
class basic_segtree {
public:
    int n;    
    int size;  
    std::vector<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree() {}

    basic_segtree(int _n, T _DEFAULT, F _func = [](const T& a, const T& b) {return a + b;}) : n(_n), func(_func), DEFAULT(_DEFAULT) {
        size = 1;
        while(size < _n) size <<= 1;
        root.assign(size << 1, _DEFAULT);
    }
	
	void build(const std::vector<T>& a) {
        for(int i = 0; i < n; i++) 
            root[size + i] = a[i];
        for(int i = size - 1; i > 0; i--) 
            root[i] = func(root[i << 1], root[i << 1 | 1]);
    }
    
    void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size, root[idx] = val;
        for(idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
	T queries_range(int l, int r) {
        l = std::max(0, l), r = std::min(r, n - 1);
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        bool has_left = false, has_right = false;
        while(l <= r) {
            if((l & 1) == 1) {
                if(!has_left) res_left = root[l++];
                else res_left = func(res_left, root[l++]); 
                has_left = true;
            }
            if((r & 1) == 0) {
                if(!has_right) res_right = root[r--];
                else res_right = func(root[r--], res_right);
                has_right = true;
            }
            l >>= 1; r >>= 1;
        }
        if(!has_left) return res_right;
        if(!has_right) return res_left;
        return func(res_left, res_right);
    }

	
	T queries_at(int idx) {
        if(idx < 0 || idx >= n) return DEFAULT;
        return root[idx + size];
    }

	void update_range(int l, int r, i64 v) {}

    T get() {
        return root[1];
    }

    template<typename Pred>
    int max_right(int start, Pred P) const {
        if(start < 0) start = 0;
        if(start >= n) return n - 1;
        T sm = DEFAULT;
        int idx = start + size;
        do {
            while((idx & 1) == 0) idx >>= 1;
            if(!P(func(sm, root[idx]))) {
                while(idx < size) {
                    idx <<= 1;
                    T cand = func(sm, root[idx]);
                    if(P(cand)) {
                        sm = cand;
                        idx++;
                    }
                }
                return idx - size - 1;
            }
            sm = func(sm, root[idx]);
            idx++;
        } while((idx & -idx) != idx);
        return n - 1;
    }

    template<typename Pred>
    int min_left(int ending, Pred P) const {
        if(ending < 0) return 0;
        if(ending >= n) ending = n - 1;
        T sm = DEFAULT;
        int idx = ending + size + 1;
        do {
            idx--;
            while(idx > 1 && (idx & 1)) idx >>= 1;
            if(!P(func(root[idx], sm))) {
                while(idx < size) {
                    idx = idx * 2 + 1;
                    T cand = func(root[idx], sm);
                    if(P(cand)) {
                        sm = cand;
                        idx--;
                    }
                }
                return idx + 1 - size;
            }
            sm = func(root[idx], sm);
        } while((idx & -idx) != idx);
        return 0;
    }
};

const int MOD = 998244353;
void solve() {
    int n;
    std::cin >> n;
    std::vector<int> a(n);
    for(auto& x : a) {
        std::cin >> x;
    }
    wavelet_psgt pst(a);
    basic_segtree<int> mx(n, -1e9 + 7, [](int x, int y) {return std::max(x, y);});
    basic_segtree<int> mn(n, -1e9 + 7, [](int x, int y) {return std::max(x, y);});
    mx.build(a);
    mn.build(a);
    i64 res = 0;
    for(int i = 0; i < n; i++) {
        int j = i;
        while(j < n) {
            i64 mex = pst.first_missing_number(i, j);
            int k = mx.max_right(j + 1, [&](int x) {return x <= mex;});
            res += mex * (k - j + 1);
            res %= MOD;
            j = k + 1;
        }
    }
    std::cout << res << '\n';
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int t = 1;
    std::cin >> t;
    while(t--) {
        solve();
    }
    
    return 0;
}
