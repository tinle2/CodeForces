#define lc i * 2 + 1
#define rc i * 2 + 2
#define lp lc, left, middle
#define rp rc, middle + 1, right
#define entireTree 0, 0, n - 1
#define midPoint left + (right - left) / 2
#define pushDown push(i, left, right)
#define iter int i, int left, int right

template<typename T, typename I = ll, typename II = ll, typename F = function<T(const T, const T)>>
class SGT { 
    public: 
    int n;  
    vt<T> root;
    T DEFAULT;
    F func;
	SGT(int n, T DEFAULT = T(), F func = [](const auto& a, const auto& b) {return a + b;}) : func(func) {    
        this->n = n;
        this->DEFAULT = DEFAULT;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, DEFAULT);    
    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
		pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, I val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, I val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, val);
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

	void apply(iter, I val) {
        root[i].apply(val, right - left + 1);
    }

    void push(iter) {   
        if(root[i].have_lazy() && left != right) {
			int middle = midPoint;
            apply(lp, root[i].lazy), apply(rp, root[i].lazy);
            root[i].reset_lazy();
        }
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }

    void update_window(int L, int R, int len, T x) { // update [l, l + k - 1], [l + 1, l + k], ... [r, r + k] each with x
        update_range(L, L + len - 1, x);
        update_range(R + 1, R + len, -x);
    }

	T get() {
		return root[0];
	}
	
	template<typename Pred> // seg.min_left(ending, [](const int& a) {return a > 0;});
        int min_left(int ending, Pred f) { // min index where f[l, ending] is true
            T a = DEFAULT;
            auto ans = find_left(entireTree, ending, f, a);
            return ans == -1 ? ending + 1 : ans;
        }

    template<typename Pred>
        int max_right(int starting, Pred f) {
            T a = DEFAULT;
            auto ans = find_right(entireTree, starting, f, a);
            return ans == -1 ? starting - 1 : ans;
        }

    template<typename Pred>
        int find_left(iter, int end, Pred f, T& now) {
            pushDown;
            if(left > end) return -2;
            if(right <= end && f(func(root[i], now))) {
                now = func(root[i], now);
                return left;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int r = find_left(rp, end, f, now);
            if(r == -2) return find_left(lp, end, f, now);
            if(r == middle + 1) {
                int l = find_left(lp, end, f, now);
                if(l != -1) return l;
            }
            return r;
        }

    template<typename Pred>
        int find_right(iter, int start, Pred f, T &now) {
            pushDown;
            if(right < start) return -2;
            if(left >= start && f(func(now, root[i]))) {
                now = func(now, root[i]);
                return right;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int l = find_right(lp, start, f, now);
            if(l == -2) return find_right(rp, start, f, now);
            if(l == middle) {
                int r = find_right(rp, start, f, now);
                if(r != -1) return r;
            }
            return l;
        }
};

template<class T, typename F = function<T(const T&, const T&)>>
class basic_segtree {
public:
    int n;    
    int size;  
    vt<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree() {}

    basic_segtree(int _n, T _DEFAULT, F _func = [](const T& a, const T& b) {return a + b;}) : n(_n), func(_func), DEFAULT(_DEFAULT) {
        size = 1;
        while(size < _n) size <<= 1;
        root.assign(size << 1, _DEFAULT);
    }
    
    void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size, root[idx] = val;
        for(idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
	T queries_range(int l, int r) {
        l = max(0, l), r = min(r, n - 1);
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

	void update_range(int l, int r, ll v) {}

    T get() {
        return root[1];
    }

    template<typename Pred>
    int max_right(int start, Pred P) const {
        if(start < 0) start = 0;
        if(start >= n) return n;
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

template<typename T, typename lazy_type = ll>
struct lazy_seg {
    int n, n0, h;
    vt<T> tree;
    vi seglen;

    lazy_seg(int n_) : n(n_) , n0(1) , h(0) {
        while(n0 < n) {
            n0 <<= 1;
            ++h;
        }
        tree.assign(2 * n0, T());
        seglen.assign(2 * n0, 0);
        for (int i = n0; i < 2 * n0; ++i) {
            seglen[i] = 1;
        }
        for (int i = n0 - 1; i > 0; --i) {
            seglen[i] = seglen[i * 2] + seglen[i * 2 + 1];
        }
    }

    void apply_node(int p, lazy_type v) {
        tree[p].apply(v, seglen[p]);
    }

    void pull(int p) {
        tree[p] = tree[2 * p] + tree[2 * p + 1];
    }

    void push(int p) {
        if(tree[p].have_lazy()) {
            apply_node(2 * p, tree[p].lazy);
            apply_node(2 * p + 1, tree[p].lazy);
            tree[p].reset_lazy();
        }
    }

    void push_to(int p) {
        for (int i = h; i > 0; --i) {
            push(p >> i);
        }
    }

    void update_range(int l, int r, lazy_type v) {
        if(l > r) return;
        l = max(0, l);
        r = min(r, n - 1);
        int L = l + n0;
        int R = r + n0;
        push_to(L);
        push_to(R);
        int l0 = L, r0 = R + 1;
        while(l0 < r0) {
            if(l0 & 1) apply_node(l0++, v);
            if(r0 & 1) apply_node(--r0, v);
            l0 >>= 1;
            r0 >>= 1;
        }
        for(int i = 1; i <= h; ++i) {
            if(((L >> i) << i) != L) {
                pull(L >> i);
            }
            if((((R + 1) >> i) << i) != (R + 1)) {
                pull(R >> i);
            }
        }
    }

    void update_at(int p, T v) {
        if(p < 0 || p >= n) return;
        int pos = p + n0;
        push_to(pos);
        tree[pos] = v;
        for(pos >>= 1; pos > 0; pos >>= 1) {
            pull(pos);
        }
    }

    T queries_at(int p) {
        if(p < 0 || p >= n) return T();
        int pos = p + n0;
        push_to(pos);
        return tree[pos];
    }

    T queries_range(int l, int r) {
        if(l > r) return T();
        l = max(0, l);
        r = min(r, n - 1);
        int L = l + n0;
        int R = r + n0;
        push_to(L);
        push_to(R);
        T resL;
        T resR;
        int l0 = L;
        int r0 = R + 1;
        while(l0 < r0) {
            if(l0 & 1) resL = resL + tree[l0++];
            if(r0 & 1) resR = tree[--r0] + resR;
            l0 >>= 1;
            r0 >>= 1;
        }
        return (resL + resR);
    }

	T get() {
        return queries_range(0, n - 1);
    }

    template<typename Pred>
        int max_right(int l, Pred P) {
            if(l < 0) l = 0;
            if(l >= n) return n;
            T sm;
            int idx = l + n0;
            push_to(idx);
            int tmp = idx;
            do {
                while((tmp & 1) == 0) tmp >>= 1;
                T cand = sm + tree[tmp];
                if(!P(cand)) {
                    while(tmp < n0) {
                        push(tmp);
                        tmp <<= 1;
                        T cand2 = sm + tree[tmp];
                        if(P(cand2)) {
                            sm = cand2;
                            tmp++;
                        }
                    }
                    return tmp - n0 - 1;
                }
                sm = sm + tree[tmp];
                tmp++;
            } while((tmp & -tmp) != tmp);
            return n - 1;
        }

    template<typename Pred>
        int min_left(int r, Pred P) {
            if(r < 0) return 0;
            if(r >= n) r = n - 1;
            T sm;
            int idx = r + n0 + 1;
            push_to(idx - 1);
            do {
                idx--;
                while(idx > 1 && (idx & 1)) idx >>= 1;
                T cand = tree[idx] + sm;
                if(!P(cand)) {
                    while(idx < n0) {
                        push(idx);
                        idx = idx * 2 + 1;
                        T cand2 = tree[idx] + sm;
                        if(P(cand2)) {
                            sm = cand2;
                            idx--;
                        }
                    }
                    return idx + 1 - n0;
                }
                sm = tree[idx] + sm;
            } while((idx & -idx) != idx);
            return 0;
        }
};

struct info {
    const static ll lazy_value = 0;
    ll s;
    ll lazy;
    info(ll v = INF) : s(v), lazy(lazy_value) { }

    int have_lazy() {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(ll v, int len) {
        s += v;
        lazy += v;
    }

    friend info operator+(const info& a, const info& b) { // careful about lazy_copy
        info res;
        res.s = min(a.s, b.s);
        return res;
    }
};

using T = pair<ll, mint>;
const static T lazy_value = {INF, 0};
struct info { // set, add
    mint s;
    T lazy;

    info(ll v = 0) : s(v), lazy(lazy_value) {}

    bool have_lazy() const {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(T v, int len) {
        auto& [setv, addv] = v;
        auto& [lazy_set, lazy_add] = lazy;
        if(setv != INF) {
            lazy_set = setv;
            lazy_add = lazy_value.ss;
            s = setv * len;
        } else if(addv != lazy_value.ss) {
            if(lazy_set != INF) lazy_set = (ll)mint(lazy_set + (ll)addv);
            else lazy_add += addv;
            s += addv * len;
        }
    }

    friend info operator+(const info& a, const info& b) { // careful with copy lazy_tag when merge
        info res;
        res.s = a.s + b.s;
        return res;
    }
};

const ar(3) lazy_value = {0, 1, 2};
struct inversion_info {
	// https://atcoder.jp/contests/abc265/tasks/abc265_g
    ll inv[3][3];
    ll c[3];
    ar(3) lazy;
    inversion_info(ll v = -1) : lazy(lazy_value) { 
        memset(inv, 0, sizeof(inv));
        memset(c, 0, sizeof(c));
        if(v == -1) return;
        c[v] = 1;
    }

    int have_lazy() {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(const ar(3)& v, int len) {
        ll f[3] = {};
        f[v[0]] += c[0];
        f[v[1]] += c[1];
        f[v[2]] += c[2];
        ll nc[3][3] = {};
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                nc[v[i]][v[j]] += inv[i][j];
            }
        }
        for(int i = 0; i < 3; i++) {
            c[i] = f[i];
            for(int j = 0; j < 3; j++) {
                inv[i][j] = nc[i][j];
            }
        }
        ar(3) nlazy;
        for(int i = 0; i < 3; i++) {
            nlazy[i] = v[lazy[i]];
        }
        lazy = nlazy;
    }

    friend inversion_info operator+(const inversion_info& a, const inversion_info& b) { // careful about lazy_copy
        inversion_info res;
        for(int i = 0; i < 3; i++) {
            res.c[i] = a.c[i] + b.c[i];
        }
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                res.inv[i][j] = a.inv[i][j] + b.inv[i][j] + a.c[i] * b.c[j];
            }
        }
        return res;
    }

    ll get() {
        ll res = 0;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < i; j++) {
                res += inv[i][j];
            }
        }
        return res;
    }
};

struct max_consecutive_one { // maximum len of consecutive ones
    const static ll lazy_value = 0;
    ll s;
    ll l1, r1, l0, r0, ans1, ans0, len;
    ll lazy;
    max_consecutive_one(ll v = -1) : s(v), lazy(lazy_value), len(v != -1), l1(v == 1), r1(v == 1), l0(v == 0), r0(v == 0), ans1(v == 1), ans0(v == 0) { }

    int have_lazy() {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(ll v, int _len) {
        if(v) {
            swap(l1, l0);
            swap(r1, r0);
            swap(ans1, ans0);
        }
        lazy ^= v;
    }

    friend max_consecutive_one operator+(const max_consecutive_one& a, const max_consecutive_one& b) { // careful about lazy_copy
        max_consecutive_one res;
        if(a.len == 0) {
            res = b;
        } else if(b.len == 0) {
            res = a;
        } else {
            res.ans1 = max({a.ans1, b.ans1, a.r1 + b.l1});
            res.ans0 = max({a.ans0, b.ans0, a.r0 + b.l0});
            res.len = a.len + b.len;
            res.l0 = a.l0 + (a.l0 == a.len ? b.l0 : 0);
            res.l1 = a.l1 + (a.l1 == a.len ? b.l1 : 0);
            res.r0 = b.r0 + (b.r0 == b.len ? a.r0 : 0);
            res.r1 = b.r1 + (b.r1 == b.len ? a.r1 : 0);
        }
        res.lazy = lazy_value;
        return res;
    }
};

template<typename T, typename F = function<T(const T, const T)>>
class arithmetic_segtree { // add a + d * (i - left) to [left, right] 
    public: 
    int n;  
    vt<T> root;
    vpll lazy;
    T DEFAULT;
    F func;
    bool is_prefix, inclusive;
	arithmetic_segtree(int n, T DEFAULT, F func = [](const T a, const T b) {return a + b;}, bool is_prefix = true, bool inclusive = true) : n(n), DEFAULT(DEFAULT), is_prefix(is_prefix), inclusive(inclusive), func(func) {    
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1);    
        lazy.rsz(k << 1); 
    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
        pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, pll val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, pll val) {    
        pushDown;
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, MP(val.ss * (ll)(is_prefix ? left - start : end - right) + val.ff, val.ss));
			// apply(curr, left, right, {val.ss * (is_prefix ? (left - start) : (end - left)) + val.ff, is_prefix ? val.ss : -val.ss});
            pushDown;
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
        pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	T get() {
		return root[0];
	}
	
	void print() {  
        print(entireTree);
        cout << endl;
    }

    void apply(iter, pll v) {
        ll len = right - left + 1;
        root[i] += len * v.ff + (inclusive ? len * (len + 1) / 2 : len * (len - 1) / 2) * v.ss;
        lazy[i].ff += v.ff;
        lazy[i].ss += v.ss;
    }

    void push(iter) {
        pll zero = MP(0, 0);
        if(lazy[i] != zero && left != right) {
            int middle = midPoint;
            if(is_prefix) {
                apply(lp, lazy[i]);
                pll right_lazy = lazy[i];
                right_lazy.ff += lazy[i].ss * (ll)(middle - left + 1);
                apply(rp, right_lazy);
            } else {
                int middle = midPoint;
                apply(rp, lazy[i]);
                pll left_lazy = lazy[i];
                left_lazy.ff += lazy[i].ss * (ll)(right - middle);
                apply(lp, left_lazy);
            }
            lazy[i] = zero;
        }
    }
};

template<typename T>
struct merge_sort_tree {
    int n;
    vvi arr;
    vt<T> root;
    int res = inf;
    merge_sort_tree(const vi& a) : n(a.size()) {
        int k = 1;
        while(k < n) k <<= 1;
        arr.rsz(k * 2);
        root.rsz(k * 2);
        build(entireTree, a);
    }

    void build(iter, const vi& a) {
        root[i] = inf;
        for(int j = left; j <= right; j++) arr[i].pb(a[j]);
        srt(arr[i]);
        if(left == right) return;
        int middle = midPoint;
        build(lp, a);
        build(rp, a);
    }

    void update_range(int start, int end, int x) {
        update_range(entireTree, start, end, x);
    }

    void update_range(iter, int s, int e, int x) {
        if(left > e || s > right) return;
        if(s <= left && right <= e) {
            auto it = lb(all(arr[i]), x);
            int t = inf;
            if(it != end(arr[i])) t = min(t, abs(*it - x));
            if(it != begin(arr[i])) t = min(t, abs(*--it - x));
            root[i] = min(root[i], t);
            if(t >= res) return;
        }
        if(left == right) {
            res = min(res, root[i]);
            return;
        }
        int middle = midPoint;
        update_range(rp, s, e, x);
        res = min(res, root[rc]);
        update_range(lp, s, e, x);
        root[i] = min(root[lc], root[rc]);
        res = min(res, root[i]);
    }

    int queries_range(int left, int right) {
        return queries_range(entireTree, left, right);
    }

    int queries_range(iter, int s, int e) {
        if(left > e || s > right) return inf;
        if(s <= left && right <= e) return root[i];
        int middle = midPoint;
        return min(queries_range(lp, s, e), queries_range(rp, s, e));
    }
};

class bad_subarray_segtree { 
    // nlog^2n run time
    // for each r, how many l is bad
    // https://codeforces.com/contest/1736/problem/C2
    struct info {
        ll bad;
        int mn, mx;
        info(int x = 0) : bad(x), mn(x), mx(x) {}
    };
    public: 
    int n;  
    vt<info> root;
	bad_subarray_segtree(int n) {    
        this->n = n;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, info());    
    }
    
    void update_at(int id, int val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, int val) {  
        if(left == right) { 
            root[i] = info(val);
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = merge(i, left, right);
    }

    ll query_right(iter, int threshold) {
        if(root[i].mn >= threshold) return root[i].bad;
        if(root[i].mx <= threshold) return ((ll)right - left + 1) * threshold;
        int middle = midPoint;
        if(root[lc].mx > threshold) return query_right(lp, threshold) + root[i].bad - root[lc].bad; // the right part got global update by the root[lc].mx already so no need to call it
        return query_right(lp, threshold) + query_right(rp, threshold);
    }

    info merge(iter) {
        int middle = midPoint;
        info res;
        res.mn = min(root[lc].mn, root[rc].mn);
        res.mx = max(root[lc].mx, root[rc].mx);
        res.bad = root[lc].bad + query_right(rp, root[lc].mx);
        return res;
    }

    ll bad_subarray() {
        return root[0].bad; // answer for good subarray is n * (n + 1) / 2 - root[0].bad;
    }
};

struct wavelet_psgt {
    private:
    struct Node {
        int cnt;
        ll sm;
        Node(int cnt = 0, ll sm = 0) : cnt(cnt), sm(sm) {}
        friend Node operator+(const Node& x, const Node& y) { return {x.cnt + y.cnt, x.sm + y.sm}; };
        friend Node operator-(const Node& x, const Node& y) { return {x.cnt - y.cnt, x.sm - y.sm}; };
    };
    int n;
    vt<Node> root;
    vi t;
    vpii child;
    vi a;
    int new_node() { root.pb(Node(0, 0)); child.pb({0, 0}); return root.size() - 1; }
    int get_id(ll x) { return int(lb(all(a), x) - begin(a)); }
    public:
    wavelet_psgt() {}

    wavelet_psgt(const vi& arr) : a(arr) {
        t.rsz(arr.size());
        new_node(); 
        srtU(a);
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
        if(id <= middle) child[curr].ff = new_node(), update(child[curr].ff, child[prev].ff, id, delta, left, middle); 
        else child[curr].ss = new_node(), update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = root[child[curr].ff] + root[child[curr].ss];
    }

    int kth(int l, int r, int k) {
        return kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    ll sum_kth(int l, int r, int k) {
        return sum_kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    int kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(left == right) return a[left];
        int middle = (left + right) >> 1;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return kth(child[l].ff, child[r].ff, k, left, middle);
        return kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
    }

    ll sum_kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(k <= 0) return 0;
        if(left == right) return (ll)k * a[left];
        int middle = (left + right) >> 1;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return sum_kth(child[l].ff, child[r].ff, k, left, middle); 
        return root[child[r].ff].sm - root[child[l].ff].sm + sum_kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
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

    Node queries_range(int l, int r, ll low, ll high) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], get_id(low), get_id(high + 1) - 1, 0, n - 1);
    }

    Node query(int l, int r, int start, int end, int left, int right) {
        if(left > end || right < start || left > right) return Node();
        if(start <= left && right <= end) return root[r] - root[l];
        int middle = (left + right) >> 1;
        return query(child[l].ff, child[r].ff, start, end, left, middle) + query(child[l].ss, child[r].ss, start, end, middle + 1, right);
    }
	
	ll first_missing_number(int l, int r) { // https://cses.fi/problemset/task/2184/
        ll s = 1;
        return first_missing_number(l == 0 ? 0 : t[l - 1], t[r], 0, n - 1, s);
    }

    ll first_missing_number(ll l, ll r, ll left, ll right, ll &s) {
        if(s < a[left]) return s;
        Node seg = root[r] - root[l];
        if(a[right] <= s) {
            s += seg.sm;
            return s;
        }
        ll middle = (left + right) >> 1;
        first_missing_number(child[l].ff, child[r].ff, left, middle, s);
        first_missing_number(child[l].ss, child[r].ss, middle + 1, right, s);
        return s;
    }

    pii kth_in_range(int l, int r, int start, int end, int k, int left, int right) {
        int C = root[r].cnt - root[l].cnt;
        if(left > end || right < start || left > right || C == 0) return {-1, 0};
        if(start <= left && right <= end) {
            if(C < k) return {-1, C};
        }
        if(left == right) {
            return {a[left], C};
        }
        int middle = (left + right) >> 1;
        auto [lv, lc] = kth_in_range(child[l].ff, child[r].ff, start, end, k, left, middle);
        if(lv != -1) {
            return {lv, -1};
        }
        auto [rv, rc] = kth_in_range(child[l].ss, child[r].ss, start, end, k - lc, middle + 1, right);
        if(rv != -1) {
            return {rv, -1};
        }
        return {-1, lc + rc};
    }

    int kth_in_range(int l, int r, ll left, ll right, int k) {
		// https://atcoder.jp/contests/abc324/tasks/abc324_g
        return kth_in_range(l == 0 ? 0 : t[l - 1], t[r], get_id(left), get_id(right + 1) - 1, k, 0, n - 1).ff; 
    }
};

template<class T>
struct PSGT {
    struct Node {
        int l, r;
        T key;
        Node(T key) : key(key), l(0), r(0) {}
    };
    int new_node(int prev) {
        F.pb(F[prev]);
        return F.size() - 1;
    }

    int new_node() {
        F.pb(0);
        return F.size() - 1;
    }
    vt<Node> F;
    vi t;
    int n;
    T DEFAULT;
    PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n) {
        F.reserve(n * 20);
        F.pb(Node(DEFAULT));
    }

	int update(int prev, int id, T delta, int left, int right) {  
        int curr = new_node(prev);
        if(left == right) { 
            F[curr].key = merge(F[curr].key, delta);
            return curr;
        }
        int middle = (left + right) >> 1;
        if(id <= middle) F[curr].l = update(F[prev].l, id, delta, left, middle);
        else F[curr].r = update(F[prev].r, id, delta, middle + 1, right);
        F[curr].key = merge(F[F[curr].l].key, F[F[curr].r].key);
        return curr;
    }

	T queries_at(int curr, int start, int end, int left, int right) { 
        if(!curr || left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return F[curr].key;
        int middle = (left + right) >> 1;
		return merge(queries_at(F[curr].l, start, end, left, middle), queries_at(F[curr].r, start, end, middle + 1, right));
    };
        
	T get(int curr, int prev, int k, int left, int right) {    
        if(left == right) return left;
        int leftCount = F[F[curr].l].key - F[F[prev].l].key;
        int middle = (left + right) >> 1;
        if(leftCount >= k) return get(F[curr].l, F[prev].l, k, left, middle);
        return get(F[curr].r, F[prev].r, k - leftCount, middle + 1, right);
    }

    T get(int l, int r, int k) {
        return get(t[r], t[l - 1], k, 0, n - 1);
    }
	
	int find_k(int i, int k) {
        return find_k(t[i], k, 0, n - 1);
    }

    int find_k(int curr, int k, int left, int right) {
        if(F[curr].key < k) return inf;
        if(left == right) return left;
        int middle = (left + right) >> 1;
        if(F[F[curr].l].key >= k) return find_k(F[curr].l, k, left, middle);
        return find_k(F[curr].r, k - F[F[curr].l].key, middle + 1, right);
    }

    void update_at(int i, int& prev, int id, T delta) { 
        t[i] = update(prev, id, delta, 0, n - 1); 
        prev = t[i];
//            while(i < n) { 
//                t[i] = update(t[i], id, delta, 0, n - 1);
//                i |= (i + 1);
//            }

    }

    T queries_at(int i, int start, int end) {
        return queries_at(t[i], start, end, 0, n - 1);
    }

	T queries_range(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        auto L = (l == 0 ? DEFAULT : queries_at(l - 1, low, high));
        auto R = queries_at(r, low, high);
        return R - L;
		
//            T res = 0;
//            while(i >= 0) {
//                res += queries_at(t[i], start, end, 0, n - 1);
//                i = (i & (i + 1)) - 1;
//            }
//            return res;

    }

    T merge(T left, T right) {
        return left + right;
    }
};

struct mex_tree {
    // change merge to min(left, right)
    // change the update to be root[curr] = delta;
    PSGT<int> seg;
    int n;

    mex_tree(const vi& a, int max_value, int starting_mex = 0) : n(max_value), seg(max_value, inf) {
        int prev = 0;
        seg.update_at(0, prev, 0, starting_mex == 0 ? -1 : inf);
        for(int i = 1; i < n; i++) {
            seg.update_at(0, prev, i, -1);
        }
        for (int i = 0; i < (int)a.size(); ++i) {
            int v = min(a[i], n - 1);
            seg.update_at(i + 1, prev, v, i);
        }
    }

    int mex(int l, int r, int k = 1) { // find_kth_mex
        return find_mex(seg.t[r + 1], 0, n - 1, l, k);
    }

    int mex_descending(int l, int r, int lim, int k = 1) {
        return mex_descending(seg.t[r + 1], 0, n - 1, l, lim, k);
    }

private:
    int find_mex(int curr, int L, int R, int bound, int& k) {
        if (L == R) {
            if(--k == 0) return L;
            return -1;
        }
        int M = (L + R) >> 1;
        const auto& F = seg.F;
        if(F[F[curr].l].key < bound) {
            int t = find_mex(F[curr].l, L, M, bound, k);
            if(t != -1) return t;
        }
        if(F[F[curr].r].key < bound) {
            int t = find_mex(F[curr].r, M + 1, R, bound, k);
            if(t != -1) return t;
        }
        return -1;
    }

    int mex_descending(int curr, int L, int R, int bound, int lim, int& k) {
        if(L > lim) return -1;
        if (L == R) {
            if(--k == 0) return L;
            return -1;
        }
        int M = (L + R) >> 1;
        const auto& F = seg.F;
        if(F[F[curr].r].key < bound) {
            int t = mex_descending(F[curr].r, M + 1, R, bound, lim, k);
            if(t != -1) return t;
        }
        if(F[F[curr].l].key < bound) {
            int t = mex_descending(F[curr].l, L, M, bound, lim, k);
            if(t != -1) return t;
        }
        return -1;
    }
};

struct distinct_tree { // range distinct element online
    // modify merging to left + right;
    PSGT<int> root;
    distinct_tree(const vi& a) : root(a.size(), 0) {
        int n = a.size();
        map<int, int> last;
        for(int i = 0, prev = 0; i < n; i++) {
            int x = a[i];
            if(last.count(x)) {
                root.update_at(i, prev, last[x], -1);
            } 
            root.update_at(i, prev, i, 1);
            last[x] = i;
        }
    }  

    int query(int l, int r) {
        return root.queries_at(r, l, r);
    }
};

struct good_split {
    // determine if in [l, r], there's an index such that max([l, i]) < min([i + 1, r])
	// for minimum index l, https://www.codechef.com/problems/SSS7?tab=statement
    vi a;
    int n;
    PSGT<int> Tree;
    // merge is min, and root[curr] = delta
    good_split(const vi& a) : n(a.size()), a(a), Tree(n, inf) {
        // https://codeforces.com/contest/1887/problem/D
        auto L = closest_left(a, less<int>());
        linear_rmq<int> t(a, [](const int& x, const int& y) {return x > y;});
        int prev = 0;
        for(int i = 0; i < n; i++) {
            Tree.update_at(0, prev, i, inf);
        }
        set<int> s;
        for(int r = 1; r < n; r++) {
            for(auto it = s.lb(L[r]); it != end(s);) {
                Tree.update_at(r, prev, *it, inf);
                it = s.erase(it);
            }
            int left = 0, right = L[r] - 1, right_most = 0;
            while(left <= right) {
                int middle = midPoint;
                if(t.query(middle, L[r] - 1) > a[r]) right_most = middle, left = middle + 1;
                else right = middle - 1;
            }
            if(L[r] - 1 > 0) {
                Tree.update_at(r, prev, L[r] - 1, right_most);
                s.insert(L[r] - 1);
            }
        }
    }

    int query(int l, int r) {
        return Tree.queries_at(r, l, r) < l;
    }
};

struct LCM_tree {
    // do merge as left * right
    // careful with the memory, memory should be MX * 240
    // MX initializer should be meeting the constraint
    // have an init variable in the psgt to init everything with 1
    // do the DIV as vpii for [prime, cnt]
    // https://codeforces.com/contest/1422/problem/F
    PSGT<mint> Tree;
    LCM_tree(const vi& a) {
        Tree.reset();
        int n = a.size();
        Tree.assign(n, 1);
        int prev = 0;
        for(int i = 0; i < n; i++) {
            Tree.add(0, prev, i, 1);
        }
        Tree.init = false;
        const int N = MAX(a);
        stack<pii> s[N + 1];
        for(int i = 1; i <= n; i++) {
            t[i] = prev;
            int X = a[i - 1];
            for(auto& [x, cnt] : DIV[X]) {
                auto& curr = s[x];
                int last = 0;
                while(!curr.empty() && curr.top().ss <= cnt) {
                    auto [j, c] = curr.top(); curr.pop();
                    assert(c >= last);
                    Tree.add(i, prev, j, mint(1) / mint(x).pow(c - last));
                    last = c;
                }
                auto now = mint(x).pow(cnt);
                if(!curr.empty() && cnt > last) {
                    auto [j, oldCnt] = curr.top();
                    Tree.add(i, prev, j, mint(1) / mint(x).pow(cnt - last));
                }
                Tree.add(i, prev, i - 1, now);
                curr.push({i - 1, cnt});
            } 
        }
    } 

    mint query(int l, int r) {
        return Tree.queries_at(r + 1, l, r);
    }
};

struct min_abs_tree { // min of abs(a[i] - a[j]) in [l, r] where i ! = j
    // https://codeforces.com/problemset/problem/765/F
    vi a;
    PSGT<int> root;
    min_abs_tree(const vi& a) : a(a), root(a, inf) {
        int prev = 0;
        for(int i = 0; i < a.size(); i++) {
            root.update_at(i, prev, i);
        }
    }

    int query(int l, int r) {
        return root.queries_at(r, l, r);
    }
};

struct mod_tree { // n*sqrtn*logn for printing queries [l, r, x] sum a[i] % x for [l, r], can eliminate the log by doing sqrt decomp
    // https://codeforces.com/gym/105009/problem/L
    // assign F[curr].key = delta;
    vi base;
    PSGT<ll> tree;
    vll prefix;
    mod_tree(const vi& a) : tree(1, 1, 0) {
        min_heap<pii> q;
        int n = a.size();
        prefix.rsz(n + 1);
        for(int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + a[i];
            q.push({1, i});
            q.push({a[i] + 1, i});
            int curr = 1;
            while(curr <= a[i]) {
                base.pb(curr);
                int add = a[i] / curr;
                int last = a[i] / add;
                curr = last + 1;
            }
            base.pb(1);
            base.pb(a[i] + 1);
        }
        srtU(base);
        const int N = base.size();
        tree = PSGT<ll>(n, N, 0);
        int prev = 0;
        while(!q.empty()) {
            auto [curr, id] = q.top(); q.pop(); 
            int x = a[id];
            if(curr <= x) {
                int add = x / curr;
                int last = x / add;
                tree.update_at(get_id(curr), prev, id, add);
                curr = last + 1;
                if(curr <= x) q.push({curr, id});
            } else {
                tree.update_at(get_id(curr), prev, id, 0);
            }
        }
    } 

    int get_id(int x) {
        return int(lb(all(base), x) - begin(base));
    }

    ll query(int l, int r, int x) {
        int p = get_id(x + 1) - 1;
        ll s = prefix[r + 1] - prefix[l];
        ll floor_sum = (ll)x * tree.queries_at(p, l, r);
        return s - floor_sum;
    }
};

// you have to set up by assigning size and updating from 0 to n - 1 first
template<typename T>
struct lazy_PSGT {
	struct Node {
        T s;
        ll lazy;
        int l, r;
        Node(T key = 0) : s(key), lazy(0), l(0), r(0) { }

        friend Node operator+(const Node& a, const Node& b) {
            Node res;
            res.s = a.s + b.s;
            return res;
        }
    };

    vt<Node> F;
    vi t;
    int n;
    T DEFAULT;
    lazy_PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n + 5) {
        F.reserve(n * 20);
        F.pb(Node(DEFAULT));
    }

    int new_node(int prev) {
        F.pb(F[prev]);
        return int(F.size()) - 1;
    }

    void pull(int curr) {
        int l = F[curr].l;
        int r = F[curr].r;
        F[curr] = F[l] + F[r];
        F[curr].l = l;
        F[curr].r = r;
    }

	void apply(int curr, int left, int right, T val) {
        auto& x = F[curr];
        x.s += (right - left + 1) * val;
        x.lazy += val;
    }

    void push_down(int curr, int left, int right) {
        if(left == right || !curr || F[curr].lazy == 0) return;
        int middle = (left + right) >> 1;
        F[curr].l = new_node(F[curr].l);
        apply(F[curr].l, left, middle, F[curr].lazy);
        F[curr].r = new_node(F[curr].r);
        apply(F[curr].r, middle + 1, right, F[curr].lazy);
        F[curr].lazy = 0;
    }

    int update_range(int prev, int start, int end, T delta) {
        return update_range(prev, start, end, 0, n - 1, delta);
    }

    int update_range(int prev, int start, int end, int left, int right, T delta) {
        if(left > end || start > right) return prev;
        int curr = new_node(prev);
        push_down(curr, left, right);
        if(start <= left && right <= end) {
            apply(curr, left, right, delta);
            push_down(curr, left, right);
            return curr;
        }
        int middle = midPoint;
        F[curr].l = update_range(F[curr].l, start, end, left, middle, delta);
        F[curr].r = update_range(F[curr].r, start, end, middle + 1, right, delta);
        pull(curr);
        return curr;
    }

    Node queries_range(int i, int start, int end) {
        return queries_range(i, start, end, 0, n - 1);
    }

    Node queries_range(int curr, int start, int end, int left, int right) {
        push_down(curr, left, right);
        if(!curr || start > right || left > end) return Node();
        if(start <= left && right <= end) return F[curr];
        int middle = midPoint;
        return queries_range(F[curr].l, start, end, left, middle) + queries_range(F[curr].r, start, end, middle + 1, right);
    }

    int update_at(int prev, int id, T delta) {
        return update_at(prev, id, delta, 0, n - 1);
    }

    int update_at(int prev, int id, T delta, int left, int right) {
        int curr = new_node(prev);
        if(left == right) {
            F[curr] = Node(delta);
            return curr;
        }
        int middle = midPoint;
        if(id <= middle) {
            F[curr].l = update_at(F[prev].l, id, delta, left, middle);
        } else {
            F[curr].r = update_at(F[prev].r, id, delta, middle + 1, right);
        }
        pull(curr);
        return curr;
    }
};

template<class T>
struct SGT_2D {
    vt<vt<T>> root;
    int n, m, N;           
    T DEFAULT;             

    SGT_2D(int n, int m, T DEFAULT) {
        this->n = n;
        this->m = m;
        this->DEFAULT = DEFAULT;
        this->N = max(n, m); 
        root.resize(N * 2, vt<T>(N * 2)); // do 4 * N for recursive segtreee
    }

    void update_at(int x, int y, T value) {
        x += N; y += N;
        root[x][y] = value;
        for (int ty = y; ty > 1; ty >>= 1) {
            root[x][ty >> 1] = merge(root[x][ty], root[x][ty ^ 1]);
        }
        for (int tx = x; tx > 1; tx >>= 1) {
            for (int ty = y; ty >= 1; ty >>= 1) {
                root[tx >> 1][ty] = merge(root[tx][ty], root[tx ^ 1][ty]);
            }
        }
    }

    T queries_range(int start_x, int end_x, int start_y, int end_y) {
        start_x += N; end_x += N;    
        start_y += N; end_y += N;   
        T result = DEFAULT;

        while (start_x <= end_x) {
            if (start_x & 1) { 
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[start_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[start_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                start_x++;
            }
            if (!(end_x & 1)) {
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[end_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[end_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                end_x--;
            }
            start_x >>= 1;
            end_x >>= 1;
        }
        return result;
    }

	T queries_at(int r, int c) {
        return queries_range(r, r, c, c);
    }

    T merge(T A, T B) {
    }

//    void update_at(int x, int y, T v) {
//        update_at(x, y, v, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    void update_at(int x, int y, T val, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (left_x == right_x && left_y == right_y) {
//            root[node_x][node_y] = val;
//            return;
//        }
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        if (x <= mid_x && y <= mid_y) update_at(x, y, val, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y); 
//        else if (x <= mid_x) update_at(x, y, val, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1);
//        else if (y <= mid_y) update_at(x, y, val, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y);
//        else update_at(x, y, val, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1);
//        root[node_x][node_y] = merge(
//            root[2 * node_x][2 * node_y],
//            root[2 * node_x][2 * node_y + 1],
//            root[2 * node_x + 1][2 * node_y],
//            root[2 * node_x + 1][2 * node_y + 1]
//        );
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y) {
//        return queries_range(start_x, end_x, start_y, end_y, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (start_x > right_x || end_x < left_x || start_y > right_y || end_y < left_y) return DEFAULT;
//        if (start_x <= left_x && right_x <= end_x && start_y <= left_y && right_y <= end_y) return root[node_x][node_y];
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        return merge(
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1)
//        );
//    }
// 
//    T merge(T A, T B, T C, T D) {
//        return min(A, min(B, min(C, D)));
//    }
};

int root[MX * 120], lazy[MX * 120], ptr; // MX should be 1e5
pii child[MX * 120];
class implicit_segtree {
    public:
    int n;
    implicit_segtree(int n) {
        this->n = n;
        root[0] = a.queries(0, n - 1); // initialize
        lazy[0] = -1;
    } 

    void create_node(int& node, int left, int right) {
        if(node) return;
        node = ++ptr;
        lazy[node] = -1;
        root[node] = a.queries(left, right);
    }

    void update(int start, int end, int x) {
        update(entireTree, start, end, x);
    }

    void update(iter, int start, int end, int x) {
        pushDown;
        if(left > end || start > right) return;
        if(start <= left && right <= end) {
			apply(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        update(child[i].ff, left, middle, start, end, x);
        update(child[i].ss, middle + 1, right, start, end, x);
        root[i] = merge(root[child[i].ff], root[child[i].ss]);
    }

	void apply(iter, int val) {
        root[i] = val * (right - left + 1);
        lazy[i] = val;
    }

    void push(iter) {
        if(lazy[i] != -1 && left != right) {
            int middle = midPoint;
            create_node(child[i].ff, left, middle);
            create_node(child[i].ss, middle + 1, right);
			apply(child[i].ff, left, middle, lazy[i]);
            apply(child[i].ss, middle + 1, right, lazy[i]);
            lazy[i] = -1;
        }
    }

    int merge(int left, int right) {
        return min(left, right);
    }

    int queries(int start, int end) {
        return queries(entireTree, start, end);
    }

    int queries(iter, int start, int end) {
        pushDown;
        if(start > right || left > end) return inf;
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        return merge(queries(child[i].ff, left, middle, start, end), queries(child[i].ss, middle + 1, right, start, end));
    }
	
	void update(int& i, int x) {
        update(i, 0, n - 1, x);
    }

    void update(int& i, int left, int right, int x) {
        if(!i) i = ++ptr;
        if(left == right) return;
        int middle = midPoint;
        if(x <= middle) update(child[i].ff, left, middle, x);
        else update(child[i].ss, middle + 1, right, x);
    }


    int merge_two_tree(int i, int j) {
        if(!i || !j) return i + j;
        child[i].ff = merge_two_tree(child[i].ff, child[j].ff);
        child[i].ss = merge_two_tree(child[i].ss, child[j].ss);
        return i;
    }

    void modify_two_tree(int& i, int& j, int start, int end) {
        modify(i, j, 0, n - 1, start, end);
    }

    void modify(int& i, int& j, int left, int right, int start, int end) {
        if(!i || left > end || start > right) return;
        if(!j) j = ++ptr;
        if(left >= start && right <= end) {
            j = merge_two_tree(i, j);
            i = 0;
            return;
        }
        int middle = midPoint;
        modify(child[i].ff, child[j].ff, left, middle, start, end);
        modify(child[i].ss, child[j].ss, middle + 1, right, start, end);
    }

};

template<class T>
class SGT_BEAT {
    private:
    struct Node {
        T mx1, mx2, mn1, mn2, mx_cnt, mn_cnt, sm, ladd, lval;
        Node(T x = INF) : mx1(x), mx2(-INF), mn1(x), mn2(INF), mx_cnt(1), mn_cnt(1), sm(x), lval(INF), ladd(0) {}
    };

    Node merge(const Node& left, const Node& right) {
        if(left.mx1 == INF) return right;
        if(right.mx1 == INF) return left;
        Node res;
        res.sm = left.sm + right.sm;
        if(left.mx1 > right.mx1) {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt;
            res.mx2 = max(left.mx2, right.mx1);
        } else if(left.mx1 < right.mx1) {
            res.mx1 = right.mx1;
            res.mx_cnt = right.mx_cnt;
            res.mx2 = max(left.mx1, right.mx2);
        } else {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt + right.mx_cnt;
            res.mx2 = max(left.mx2, right.mx2);
        }

        if(left.mn1 < right.mn1) {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt;
            res.mn2 = min(left.mn2, right.mn1);
        } else if(left.mn1 > right.mn1) {
            res.mn1 = right.mn1;
            res.mn_cnt = right.mn_cnt;
            res.mn2 = min(right.mn2, left.mn1);
        } else {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt + right.mn_cnt;
            res.mn2 = min(left.mn2, right.mn2);
        }
        return res;
    }

    void update_at(iter, int id, T x) {
        pushDown;
        if(left == right) {
            root[i] = Node(x);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update_at(lp, id, x);
        else update_at(rp, id, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_min(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end || root[i].mx1 <= x) return;
        if(start <= left && right <= end && root[i].mx2 < x) {
            update_node_max(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_min(lp, start, end, x);
        update_min(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_max(iter, int start, int end, T x) {
        pushDown;
        if(left > end || start > right || x <= root[i].mn1) return;
        if(start <= left && right <= end && x < root[i].mn2) {
			update_node_min(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_max(lp, start, end, x);
        update_max(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
        
    }

    void update_node_min(iter, T x) {
        root[i].sm += (x - root[i].mn1) * root[i].mn_cnt;  
        if(root[i].mn1 == root[i].mx1) {
            root[i].mn1 = root[i].mx1 = x;
        } else if(root[i].mn1 == root[i].mx2) {
            root[i].mn1 = root[i].mx2 = x;
        } else {
            root[i].mn1 = x;
        }
    }

    void update_node_max(iter, T x) {
        root[i].sm += (x - root[i].mx1) * root[i].mx_cnt;
        if(root[i].mx1 == root[i].mn1) {
            root[i].mx1 = root[i].mn1 = x;
        } else if(root[i].mx1 == root[i].mn2) {
            root[i].mx1 = root[i].mn2 = x;
        } else {
            root[i].mx1 = x;
        }
    }

    void update_val(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            update_all(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_val(lp, start, end, x);
        update_val(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_all(iter, T x) {
        root[i] = Node(x);
        T len = right - left + 1;
        root[i].sm = len * x;
        root[i].mx_cnt = root[i].mn_cnt = len;
        root[i].lval = x;
    }

    void update_add(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            add_val(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_add(lp, start, end, x);
        update_add(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void add_val(iter, T x) {
        root[i].mx1 += x;
        if(root[i].mx2 != -INF) root[i].mx2 += x;
        root[i].mn1 += x;
        if(root[i].mn2 != INF) root[i].mn2 += x;
        root[i].sm += x * (right - left + 1);
        if(root[i].lval != INF) root[i].lval += x;
        else root[i].ladd += x;
    }

    void push(iter) {
        if(left == right) return;    
        int middle = midPoint;
        if(root[i].lval != INF) {
            update_all(lp, root[i].lval);
            update_all(rp, root[i].lval);
            root[i].lval = INF;
            return;
        }
        if(root[i].ladd) {
            add_val(lp, root[i].ladd);
            add_val(rp, root[i].ladd);
            root[i].ladd = 0;
        }
        if(root[i].mx1 < root[lc].mx1) update_node_max(lp, root[i].mx1);
        if(root[i].mn1 > root[lc].mn1) update_node_min(lp, root[i].mn1);
        if(root[i].mx1 < root[rc].mx1) update_node_max(rp, root[i].mx1);
        if(root[i].mn1 > root[rc].mn1) update_node_min(rp, root[i].mn1);
    }

    Node queries_range(iter, int start, int end) {
        pushDown;
        if(left > end || start > right) return Node();
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        return merge(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	Node queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    public:
    int n;
    vt<Node> root;
    SGT_BEAT(int n) {
        this->n = n;
        int k = 1;
        while(k < n) k <<= 1;
        root.rsz(k << 1);
    }

    void update_at(int id, T x) { update_at(entireTree, id, x); }
    void update_min(int start, int end, T x) { update_min(entireTree, start, end, x); }
    void update_max(int start, int end, T x) { update_max(entireTree, start, end, x); }
    void update_val(int start, int end, T x) { update_val(entireTree, start, end, x); }
    void update_add(int start, int end, T x) { update_add(entireTree, start, end, x); }
    Node queries_range(int start, int end) { return queries_range(entireTree, start, end); }
	Node queries_at(int id) { return queries_at(entireTree, id); }
	
    template<typename OP>
    // call by update_unary(l, r, x, [](const int& a, const int& b) {return a % b;});
    void update_unary(int start, int end, T x, OP c) { // update range and, range or, range divide, range mod, ... anything that's unary
        update_unary(entireTree, start, end, x, c);
    }
    
    template<typename OP>
    void update_unary(iter, int start, int end, T x, OP op) {
        pushDown;
        if(start > right || left > end) return; // for range mod do a return if root[i].mx1 < x
        if(start <= left && right <= end && root[i].mx1 == root[i].mn1) {
            T nv = op(root[i].mx1, x);
            update_all(i, left, right, nv);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_unary(lp, start, end, x, op);
        update_unary(rp, start, end, x, op);
        root[i] = merge(root[lc], root[rc]);
    }
};

struct HISTORICAL_SGT_BEAT {
    // having two same array a and b at the start
    // store historical mn and historical mx 
    // meaning it's the lowest a[i] gets to at any point, same for mx in b[i]
    // https://uoj.ac/problem/169
    struct node {
        int mn, hmn, se;
        int mx, hmx, le, hle;
        int tag1, htag1, tag2, htag2, tag3, htag3, tag4, htag4;
        node(ll val = inf)
            : mn(val), hmn(val), se(inf),
              mx(val), hmx(val), le(val), hle(val),
              tag1(0), htag1(0), tag2(0), htag2(0), tag3(0), htag3(0), tag4(0), htag4(0) {}
    };
    vt<node> tree;
    int n;

    HISTORICAL_SGT_BEAT(int _n = 0) : n(_n) {
        int k = 1;
        while(k < n) k <<= 1;
        tree.rsz(k << 1);
    }

    node merge(const node &L, const node &R) {
        if(L.mn == inf) return R;
        if(R.mn == inf) return L;
        node res;
        res.mn = min(L.mn, R.mn);
        res.hmn = min(L.hmn, R.hmn);
        if(L.mn == R.mn) res.se = min(L.se, R.se);
        else if(L.mn < R.mn) res.se = min(L.se, R.mn);
        else res.se = min(L.mn, R.se);
        res.mx = max(L.mx, R.mx);
        res.hmx = max(L.hmx, R.hmx);
        if(L.mx == R.mx) res.le = max(L.le, R.le);
        else if(L.mx > R.mx) res.le = max(L.le, R.mx);
        else res.le = max(L.mx, R.le);
        res.hle = max(L.hle, R.hle);
        res.tag1 = res.htag1 = res.tag2 = res.htag2 = 0;
        res.tag3 = res.htag3 = res.tag4 = res.htag4 = 0;
        return res;
    }

    void push_up(int i) {
        tree[i].mn = min(tree[lc].mn,  tree[rc].mn);
        tree[i].hmn = min(tree[lc].hmn, tree[rc].hmn);
        if(tree[lc].mn == tree[rc].mn) tree[i].se = min(tree[lc].se, tree[rc].se);
        else if(tree[lc].mn < tree[rc].mn) tree[i].se = min(tree[lc].se, tree[rc].mn);
        else tree[i].se = min(tree[lc].mn, tree[rc].se);
        tree[i].mx = max(tree[lc].mx,  tree[rc].mx);
        tree[i].hmx = max(tree[lc].hmx, tree[rc].hmx);
        if(tree[lc].mx == tree[rc].mx) tree[i].le = max(tree[lc].le, tree[rc].le);
        else if (tree[lc].mx > tree[rc].mx) tree[i].le = max(tree[lc].le, tree[rc].mx);
        else tree[i].le = max(tree[lc].mx, tree[rc].le);
        tree[i].hle = max(tree[lc].hle, tree[rc].hle);
    }

    void push_tag1(int i, int tag, int htag) {
        tree[i].hmn = min(tree[i].hmn, tree[i].mn + htag);
        tree[i].mn += tag;
        tree[i].htag1 = min(tree[i].htag1, tree[i].tag1 + htag);
        tree[i].tag1 += tag;
    }

    void push_tag2(int i, int tag, int htag) {
        if(tree[i].se != inf) tree[i].se += tag;
        tree[i].htag2 = min(tree[i].htag2, tree[i].tag2 + htag);
        tree[i].tag2 += tag;
    }

    void push_tag3(int i, int tag, int htag) {
        tree[i].hmx = max(tree[i].hmx, tree[i].mx + htag);
        tree[i].mx += tag;
        tree[i].htag3 = max(tree[i].htag3, tree[i].tag3 + htag);
        tree[i].tag3 += tag;
    }

    void push_tag4(int i, int tag, int htag) {
        tree[i].hle = max(tree[i].hle, tree[i].le + htag);
        tree[i].le += tag;
        tree[i].htag4 = max(tree[i].htag4, tree[i].tag4 + htag);
        tree[i].tag4 += tag;
    }

    void push(iter) {
        if(left == right) return;
        int middle = midPoint;
        int mv = min(tree[lc].mn, tree[rc].mn);
        if(tree[lc].mn <= mv) push_tag1(lc, tree[i].tag1, tree[i].htag1);
        else push_tag1(lc, tree[i].tag2, tree[i].htag2);
        push_tag2(lc, tree[i].tag2, tree[i].htag2);
        push_tag3(lc, tree[i].tag3, tree[i].htag3);
        push_tag4(lc, tree[i].tag4, tree[i].htag4);
        if(tree[rc].mn <= mv) push_tag1(rc, tree[i].tag1, tree[i].htag1);
        else push_tag1(rc, tree[i].tag2, tree[i].htag2);
        push_tag2(rc, tree[i].tag2, tree[i].htag2);
        push_tag3(rc, tree[i].tag3, tree[i].htag3);
        push_tag4(rc, tree[i].tag4, tree[i].htag4);
        tree[i].tag1 = tree[i].htag1 = tree[i].tag2 = tree[i].htag2 = 0;
        tree[i].tag3 = tree[i].htag3 = tree[i].tag4 = tree[i].htag4 = 0;
    }

    void update_add(iter, int l, int r, int k) {
        if(l <= left && right <= r) {
            push_tag1(i, k, k);
            push_tag2(i, k, k);
            push_tag3(i, k, k);
            push_tag4(i, k, k);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_add(lp, l, r, k);
        if(r > middle)  update_add(rp, l, r, k);
        push_up(i);
    }

    void update_max(iter, int l, int r, int k) {
        if(tree[i].mn >= k) return;
        if(l <= left && right <= r && tree[i].se > k) {
            int delta = k - tree[i].mn;
            push_tag1(i, delta, delta);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_max(lp, l, r, k);
        if(r > middle) update_max(rp, l, r, k);
        push_up(i);
    }

    void update_at(iter, int pos, int val) {
        if(left == right) {
            tree[i] = node(val);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(pos <= middle) update_at(lp, pos, val);
        else update_at(rp, pos, val);
        push_up(i);
    }

    void update_min(iter, int l, int r, int x) {
        if(tree[i].mx <= x) return;
        if(l <= left && right <= r && tree[i].le < x) {
            int delta = x - tree[i].mx;
            push_tag3(i, delta, delta);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_min(lp, l, r, x);
        if(r > middle) update_min(rp, l, r, x);
        push_up(i);
    }

    node queries_at(iter, int pos) {
        if(left == right) return tree[i];
        pushDown;
        int middle = midPoint;
        return pos <= middle ? queries_at(lp, pos)
                              : queries_at(rp, pos);
    }

    node queries_range(iter, int l, int r) {
        if (l <= left && right <= r) return tree[i];
        pushDown;
        int middle = midPoint;
        if (r <= middle) return queries_range(lc, left, middle, l, r);
        if (l > middle)  return queries_range(rc, middle + 1, right, l, r);
        node L = queries_range(lc, left, middle, l, r);
        node R = queries_range(rc, middle + 1, right, l, r);
        return merge(L, R);
    }

    void update_add(int l, int r, int x) { update_add(entireTree, l, r, x); }
    void update_max(int l, int r, int x) { update_max(entireTree, l, r, x); }
    void update_min(int l, int r, int x) { update_min(entireTree, l, r, x);};
    void update_at(int pos, int x) { update_at(entireTree, pos, x); }
    node queries_at(int pos) { return queries_at(entireTree, pos); }
    node queries_range(int l, int r) { return queries_range(entireTree, l, r); }
};

struct hash_info {
    const static int K = 26;
    int cnt[K], len;
    ll fwd[2], rev[2];

    hash_info(ll v = -1, int _len = 1) : len(_len) {
        if(v == -1) {
            fwd[0] = -1;
            return;
        }
        set(v);
    }

    bool is_palindrome() const {
        return len > 0 && fwd[0] == rev[0] && fwd[1] == rev[1];
    }

    friend hash_info operator+(const hash_info& a, const hash_info& b) {
        if(a.fwd[0] == -1) return b;
        if(b.fwd[0] == -1) return a;
        hash_info r;
        r.len = a.len + b.len;
        for(int i = 0; i < 2;i++){
            r.fwd[i] = (a.fwd[i] * p[i][b.len] + b.fwd[i]) % mod[i];
            r.rev[i] = (b.rev[i] * p[i][a.len] + a.rev[i]) % mod[i];
        }
        for(int i = 0; i < K; i++) {
            r.cnt[i] = a.cnt[i] + b.cnt[i];
        }
        return r;
    }

    void set(int x) {
        for (int h = 0; h < 2; ++h) {
            ll val = (ll)x * geom[h][len] % mod[h];
            fwd[h] = rev[h] = val;
        }
        mset(cnt, 0);
        cnt[x] = len;
    }
};

struct max_subarray_info {
    ll ans, prefix, suffix, sm;
    max_subarray_info(ll x = -INF) : ans(max(0LL, x)), prefix(max(0LL, x)), suffix(max(0LL, x)), sm(x) {}

    friend max_subarray_info operator+(const max_subarray_info& a, const max_subarray_info& b) {
        if(a.sm == -INF) return b;
        if(b.sm == -INF) return a;
        max_subarray_info res;
        res.ans = max({a.ans, b.ans, a.suffix + b.prefix});
        res.sm = a.sm + b.sm;
        res.prefix = max(a.prefix, a.sm + b.prefix);
        res.suffix = max(b.suffix, b.sm + a.suffix);
        return res;
    }
};

struct info_0_1_0 { // maximum of segment of left0 + right0 where mid is a one
    // https://leetcode.com/problems/maximize-active-section-with-trade-ii/submissions/1590475876/
    int one_left, one_right, zero_left, zero_right,
        ans, sm, zero_left2, zero_right2, one_left2, one_right2;
    info_0_1_0(int x = -1)
        : one_left(x == 1),
          one_right(x == 1),
          zero_left(x == 0),
          zero_right(x == 0),
          ans(0),
          sm(x != -1),
          zero_left2(x == 0),
          zero_right2(x == 0),
          one_left2(x == 1),
          one_right2(x == 1)
    {}

    friend info_0_1_0 operator+(const info_0_1_0& a, const info_0_1_0& b) {
        if(a.sm == 0) return b;
        if(b.sm == 0) return a;
        info_0_1_0 res;
        res.sm = a.sm + b.sm;
        res.ans = max(a.ans, b.ans);
        res.ans = max(a.ans, b.ans);
        if(b.one_left && a.one_right && a.zero_right2 && b.zero_left2) res.ans = max(res.ans, a.zero_right2 + b.zero_left2);
        else if(a.one_right && a.zero_right2 && b.zero_left) res.ans = max(res.ans, a.zero_right2 + b.zero_left);
        else if(b.one_left && a.zero_right && b.zero_left2) res.ans = max(res.ans, a.zero_right + b.zero_left2);
        else {
            if(a.one_right2 && a.zero_right2 && b.zero_left) {
                res.ans = max(res.ans, a.zero_right + b.zero_left + a.zero_right2);
            }
            if(b.one_left2 && b.zero_left2 && a.zero_right) {
                res.ans = max(res.ans, a.zero_right + b.zero_left + b.zero_left2);
            }
        }
        if(a.one_left) {
            res.one_left = a.one_left;
            if(a.one_left == a.sm) {
                if(b.one_left) {
                    res.one_left += b.one_left;
                    res.zero_left2 = b.zero_left2;
                } else {
                    res.zero_left2 = b.zero_left;
                }
                res.one_left2 = b.one_left2;
            } else {
                res.zero_left2 = a.zero_left2;
                if(a.one_left + a.zero_left2 == a.sm) {
                    if(b.zero_left) {
                        res.zero_left2 += b.zero_left;
                        res.one_left2 = b.one_left2;
                    } else {
                        res.one_left2 = b.one_left;
                    }
                } else {
                    res.one_left2 = a.one_left2 + (a.one_left + a.zero_left2 + a.one_left2 == a.sm ? b.one_left : 0);
                }
            }
        }
        else {
            res.zero_left = a.zero_left;
            if(a.zero_left == a.sm) {
                if(b.zero_left) {
                    res.zero_left += b.zero_left;
                    res.one_left2 = b.one_left2;
                } else {
                    res.one_left2 = b.one_left;
                }
                res.zero_left2 = b.zero_left2;
            } else {
                res.one_left2 = a.one_left2;
                if(a.zero_left + a.one_left2 == a.sm) {
                    if(b.one_left) {
                        res.one_left2 += b.one_left;
                        res.zero_left2 = b.zero_left2;
                    } else {
                        res.zero_left2 = b.zero_left;
                    }
                } else {
                    res.zero_left2 = a.zero_left2 + (a.zero_left + a.one_left2 + a.zero_left2 == a.sm ? b.zero_left : 0);
                }
            }
        }
        if(b.one_right) {
            res.one_right = b.one_right;
            if(b.one_right == b.sm) {
                if(a.one_right) {
                    res.one_right += a.one_right;
                    res.zero_right2 = a.zero_right2;
                } else {
                    res.zero_right2 = a.zero_right;
                }
                res.one_right2 = a.one_right2;
            } else {
                res.zero_right2 = b.zero_right2;
                if(b.one_right + b.zero_right2 == b.sm) {
                    if(a.zero_right) {
                        res.zero_right2 += a.zero_right;
                        res.one_right2 = a.one_right2;
                    } else {
                        res.one_right2 = a.one_right;
                    }
                } else {
                    res.one_right2 = b.one_right2 + (b.one_right + b.one_right2 + b.zero_right2 == b.sm ? a.one_right : 0);
                }
            }
        } else {
            res.zero_right = b.zero_right;
            if(b.zero_right == b.sm) {
                if(a.zero_right) {
                    res.zero_right += a.zero_right;
                    res.one_right2 = a.one_right2;
                } else {
                    res.one_right2 = a.one_right;
                }
                res.zero_right2 = a.zero_right2;
            } else {
                res.one_right2 = b.one_right2;
                if(b.zero_right + b.one_right2 == b.sm) {
                    if(a.one_right) {
                        res.one_right2 += a.one_right;
                        res.zero_right2 = a.zero_right2;
                    } else {
                        res.zero_right2 = a.zero_right;
                    }
                } else {
                    res.zero_right2 = b.zero_right2 + (b.zero_right + b.one_right2 + b.zero_right2 == b.sm ? a.zero_right : 0);
                }
            }
        }
        return res;
    }
};

struct dp_info { // knapsack pick not pick
    ll dp[2][2];
    dp_info(ll x = -INF) {
        mset(dp, 0);
        if(x > 0) dp[1][1] = x;
    }

    friend dp_info operator+(const dp_info& a, const dp_info& b) {
        dp_info res;
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                res.dp[i][j] = max({res.dp[i][j], 
                                    a.dp[i][0] + b.dp[0][j], 
                                    a.dp[i][1] + b.dp[0][j], 
                                    a.dp[i][0] + b.dp[1][j]});
            }
        }
        return res;
    }
};

struct bracket_info { // balance bracket sequence
    int sm, min_prefix;
    bracket_info(int x = 0) : sm(x), min_prefix(min(x, 0)) {}

    bool is_balance() {
        return sm == 0 && min_prefix >= 0;
    }

    friend bracket_info operator+(const bracket_info& a, const bracket_info& b) {
        bracket_info res;
        res.sm = a.sm + b.sm;
        res.min_prefix = min(a.min_prefix, a.sm + b.min_prefix);
        return res;
    }
};

struct bracket_subsequence_info { // maximum balance bracket [l, r]
    // https://codeforces.com/contest/380/problem/C
    int open, close, ans;
    bracket_subsequence_info(int x = 0) : ans(0), open(x == 1), close(x == -1) {}

    friend bracket_subsequence_info operator+(const bracket_subsequence_info& a, const bracket_subsequence_info& b) {
        bracket_subsequence_info res;
        int mn = min(a.open, b.close);
        res.ans = a.ans + b.ans + mn * 2;
        res.open = a.open + b.open - mn;
        res.close = a.close + b.close - mn;
        return res;
    }
};

struct good_index_info { // determine the minimum and maximum index where a good index is 
                         // a[0, i] all <= i and i <= a[i, n - 1]
                         // basically it's a index that left half is less than i and right half all greater than i
                         // for(int i = 0; i < n; i++) {
                         //     if(min(0, i) <= i && i <= max(i, n - 1)) {
                         //         return true
                         //     }
                         // }
    // https://www.codechef.com/problems/DOUBLEFLIPQ?tab=statement
    // update i with info(i + 1, a[i] == i ? i : -1)
    // then lazy segtree update(max(a[x], pos[a[x]])) to n - 1 with -1, the one that holds the mn == 0 is the good index we're looking for
    // careful cause [2, 1] doesn't have good index but it's the edge case of mn == 0 as well
    // update min_id and max_id each time the index changes

    int mn, min_id, max_id;
    good_index_info(int x = -1, int id = -1) : mn(x), max_id(id == -1 ? -inf : id), min_id(id == -1 ? inf : id) {} // id is -1 if a[i] != i and i if a[i] == i

    bool good() { // determine if this good_index_info is good enough
        return mn == 0 && min_id < inf && max_id >= 0;
    }

    friend good_index_info operator+(const good_index_info& a, const good_index_info& b) {
        if(a.mn == -1) return b;
        if(b.mn == -1) return a;
        good_index_info res;
        res.mn = min(a.min_id != inf ? a.mn : inf, b.min_id != inf ? b.mn : inf);
        if(a.mn == res.mn) res = a;
        if(b.mn == res.mn) {
            res.min_id = min(res.min_id, b.min_id);
            res.max_id = max(res.max_id, b.max_id);
        }
        return res;
    }
};

struct diameter_info {
    // https://codeforces.com/contest/1192/problem/B
    // find max diameter of the tree
    // careful with tin, tout set up for distinct subtree
    // max(d[i] + d[j] - 2 * d[lca(i, j)])
    ll diameter, plus_max, minus_max, left_mix, right_mix;
    // plus_max = d[i]
    // minus_max = - 2 * d[i]
    // left_mix = d[i] - 2 * d[j]
    // right_mix = -2 * d[i] + d[j]
    diameter_info() : diameter(0), plus_max(0), minus_max(0), left_mix(0), right_mix(0) {}

    diameter_info& operator+=(const ll v) {
        plus_max += v;
        minus_max -= 2 * v;
        left_mix -= v;
        right_mix -= v;
        return *this;
    }

    friend diameter_info operator+(const diameter_info& a, const diameter_info& b) {
        diameter_info res;
        res.plus_max = max(a.plus_max, b.plus_max);
        res.minus_max = max(a.minus_max, b.minus_max);
        res.left_mix = max({a.left_mix, b.left_mix, a.plus_max + b.minus_max});
        res.right_mix = max({b.right_mix, a.right_mix, b.plus_max + a.minus_max});
        res.diameter = max({a.diameter, b.diameter, a.left_mix + b.plus_max, a.plus_max + b.right_mix});
        return res;
    }
};

struct sorted_info {
    // https://codeforces.com/contest/1982/problem/F
    int mn, mx, R, L;
    bool sorted;
    
    sorted_info(int x = inf) : mn(x), mx(x == inf ? -inf : x), R(x), L(x), sorted(true) {}

    friend sorted_info operator+(const sorted_info& a, const sorted_info& b) {
        if(a.mx == -inf) return b;
        if(b.mx == -inf) return a;
        sorted_info res;  
        res.mn = min(a.mn, b.mn);
        res.mx = max(a.mx, b.mx);
        res.L = a.L;
        res.R = b.R;
        res.sorted = a.sorted && b.sorted && a.R <= b.L;
        return res;
    }
};

struct power_sum_info { // keep track of sum of a^5 segtree sum
    mint s1, s2, s3, s4, s5;
    power_sum_info(mint x = 0)
      : s1(x),
        s2(x * x),
        s3(x * x * x),
        s4(x * x * x * x),
        s5(x * x * x * x * x)
    {}

    friend power_sum_info operator+(const power_sum_info& a, const power_sum_info& b) {
        power_sum_info res;
        res.s1 = a.s1 + b.s1;
        res.s2 = a.s2 + b.s2;
        res.s3 = a.s3 + b.s3;
        res.s4 = a.s4 + b.s4;
        res.s5 = a.s5 + b.s5;
        return res;
    }

    void apply(mint v, int len) {
        mint v2 = v * v;
        mint v3 = v2 * v;
        mint v4 = v3 * v;
        mint v5 = v4 * v;
        // s5 = s5 + 5*s4*v + 10*s3*v^2 + 10*s2*v^3 + 5*s1*v^4 + len*v^5
        s5 = s5 + 5 * s4 * v + 10 * s3 * v2 + 10 * s2 * v3 + 5 * s1 * v4 + mint(len) * v5;
        // s4 = s4 + 4*s3*v + 6*s2*v^2 + 4*s1*v^3 + len*v^4
        s4 = s4 + 4 * s3 * v + 6 * s2 * v2 + 4 * s1 * v3 + mint(len) * v4;
        // s3 = s3 + 3*s2*v + 3*s1*v^2 + len*v^3
        s3 = s3 + 3 * s2 * v + 3 * s1 * v2 + mint(len) * v3;
        // s2 = s2 + 2*s1*v + len*v^2
        s2 = s2 + 2 * s1 * v + mint(len) * v2;
        // s1 = s1 + len*v
        s1 = s1 + mint(len) * v;
    }
};

struct bad_pair_info {
    // count number of [l, r] such that their [for(int i..) for(int j...) s += a[i] * a[j], s is odd]
    // we work on prefix and it's bad when prefix[r] - prefix[l - 1] % 4 == {2, 3}
    // when query, we do queries_range(l - 1, r) not normal [l, r] bc we're working with prefix
    // https://codeforces.com/group/o09Gu2FpOx/contest/541484/problem/K
    // when update a[i] to v, we update the prefix [i, n] with a[i] == 0 ? 1 : -1
    int dp[4];
    ll bad;
    bad_pair_info(int p = -1) : bad(0) {
        mset(dp, 0);
        if(p == -1) return;
        dp[p] = 1;
    }
    
    friend bad_pair_info operator+(const bad_pair_info& a, const bad_pair_info& b) {
        bad_pair_info res;
        res.bad = a.bad + b.bad;
        for(int i = 0; i < 4; i++) {
            res.dp[i] = a.dp[i] + b.dp[i];
            res.bad += (ll)a.dp[i] * (b.dp[(i + 2) % 4] + b.dp[(i + 3) % 4]);
        }
        return res;
    }

    void apply(int x) {
        x = ((x % 4) + 4) % 4;
        int now[4];
        for(int i = 0; i < 4; i++) {
            now[(i + x) % 4] = dp[i];
        }
        for(int i=  0; i < 4; i++) {
            dp[i] = now[i];
        }
    }
};

struct max_k_subarray_info {
    // max k non-overlapping subarray sum
    // https://codeforces.com/contest/280/problem/D
    const static int K = 20;
    ll L[K + 1], R[K + 1], LR[K + 1], best[K + 1];
    max_k_subarray_info(ll x = -INF) {
        for(int i = 0; i <= K; i++) {
            L[i] = R[i] = LR[i] = (i ? x : -INF);
            best[i] = i ? max(0LL, x) : 0;
        }
    }

    friend max_k_subarray_info operator+(const max_k_subarray_info& a, const max_k_subarray_info& b) {
        if(a.L[1] == -INF) return b;
        if(b.L[1] == -INF) return a;
        max_k_subarray_info res;
        for(int l = 0; l <= K; l++) {
            for(int r = 0; l + r <= K; r++) {
                int k = l + r;
                res.L[k] = max(res.L[k], a.L[l] + b.best[r]);
                res.R[k] = max(res.R[k], a.best[l] + b.R[r]);
                res.LR[k] = max(res.LR[k], a.L[l] + b.R[r]);
                res.best[k] = max(res.best[k], a.best[l] + b.best[r]);
				if(r + 1 <= K) {
                    res.best[k] = max(res.best[k], a.R[l] + b.L[r + 1]);
                    res.L[k] = max(res.L[k], a.LR[l] + b.L[r + 1]);
                    res.R[k] = max(res.R[k], a.R[l] + b.LR[r + 1]);
                    res.LR[k] = max(res.LR[k], a.LR[l] + b.LR[r + 1]);
                }
//                if(l + 1 <= K) {
//                    res.best[k] = max(res.best[k], a.R[l + 1] + b.L[r]);
//                    res.L[k] = max(res.L[k], a.LR[l + 1] + b.L[r]);
//                    res.R[k] = max(res.R[k], a.R[l + 1] + b.LR[r]);
//                    res.LR[k] = max(res.LR[k], a.LR[l + 1] + b.LR[r]);
//                }

            }
        }
        return res;
    }
};

struct binomial_info {
    // https://codeforces.com/problemset/problem/266/E
    mint s[6] = {};
    binomial_info() {}
    binomial_info(int x, int pos) {
        mint v = x;
        for(int p = 0; p < 6; p++)
            s[p] = v * mint(pos + 1).pow(p);
    }
    friend binomial_info operator+(binomial_info const &A, binomial_info const &B) {
        binomial_info R;
        for(int p = 0; p < 6; p++)
            R.s[p] = A.s[p] + B.s[p];
        return R;
    }
    void apply(int l, int r, mint x) {
        for(int p = 0; p < 6; p++) {
            mint sum_ip = pre[p][r + 1] - pre[p][l];
            s[p] = sum_ip * x;
        }
    }
    mint get_res(int l, int k) const {
        mint ans = 0;
        mint base = mint(-l);
        for(int p = 0; p <= k; p++)
            ans += comb.nCk(k, p) * base.pow(k - p) * s[p];
        return ans;
    }
};

struct dsu_info {
    // https://codeforces.com/contest/687/problem/D
    vi res;

    dsu_info(int l = -1) {
        res.clear();
        if(l == -1) return;
        dsu.rollBack();
        bool ok = true;
        sort(edges.begin() + pos[l], edges.begin() + pos[l + 1], [](const ar(3)& a, const ar(3)& b) {return a[2] > b[2];});
        for(int i = pos[l]; i < pos[l + 1] && ok; i++) {
            if(!merge(i, res)) {
                ok = false; 
            }
        }
        if(ok) res.pb(-1);
    }

    friend dsu_info operator+(const dsu_info& a, const dsu_info& b) {
        if(a.res.empty()) return b;
        if(b.res.empty()) return a;
        dsu_info res;
        const auto& A = a.res, &B = b.res;
        const int N = A.size(), M = B.size();
        bool ok = true;
        dsu.rollBack();
        int i = 0, j = 0;
        while(ok && ((i < N && A[i] != -1) || (j < M && B[j] != -1))) {
            if(i == N || A[i] == -1) ok = merge(B[j++], res.res);
            else if(j == M || B[j] == -1) ok = merge(A[i++], res.res);
            else if(edges[A[i]][2] > edges[B[j]][2]) ok = merge(A[i++], res.res);
            else ok = merge(B[j++], res.res);
        }
        if(ok) res.res.pb(-1);
        return res;
    }

    int get() {
        if(res.empty() || res.back() == -1) return -1;
        return edges[res.back()][2];
    }
};

struct swap_info {
    // if one condition is bad, then swap the two array
    // https://www.codechef.com/START196A/problems/SWAPABK
    int ai, bi, bad1, bad2;
    ll s1, s2;
    swap_info(int _ai = 0, int _bi = 0) : ai(_ai), bi(_bi) {
        if(ai + k < bi) {
            bad1 = 1;
            s1 = bi;
        } else {
            bad1 = 0;
            s1 = ai;
        }
        if(bi + k < ai) {
            bad2 = 0;
            s2 = ai;
        } else {
            bad2 = 1;
            s2 = bi;
        }
    }
    
    friend swap_info operator+(const swap_info& a, const swap_info& b) {
        swap_info res;
        res.bad1 = a.bad1 == 0 ? b.bad1 : b.bad2;
        res.s1 = a.s1 + (a.bad1 == 0 ? b.s1 : b.s2);
        res.bad2 = a.bad2 == 0 ? b.bad1 : b.bad2;
        res.s2 = a.s2 + (a.bad2 == 0 ? b.s1 : b.s2);
        return res;
    }
};

struct dikstra_info {
	// https://oj.uz/problem/view/BOI17_toll
    ll dp[5][5];
    dikstra_info() {
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {	
                dp[i][j] = INF;
            }
        }
    }

    friend dikstra_info operator+(const dikstra_info& a, const dikstra_info& b) {
        dikstra_info res;
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                for(int m = 0; m < k; m++) {
                    res.dp[i][j] = min(res.dp[i][j], a.dp[i][m] + b.dp[m][j]);
                }
            }
        }
        return res;
    }

    void set(int r, int c, ll cost) {
        dp[r][c] = min(dp[r][c], cost);
    }
};

const int k = 3;
struct dikstra_info {
    // https://atcoder.jp/contests/abc429/tasks/abc429_f
    ll dp[3][3];
    int ok[3];
    int empty;
    dikstra_info(int _empty = 1) : empty(_empty) {
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {	
                ok[i] = 0;
                dp[i][j] = INF;
            }
        }
    }

    friend dikstra_info operator+(const dikstra_info& a, const dikstra_info& b) {
        if(a.empty) return b;
        if(b.empty) return a;
        dikstra_info res;
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                for(int m = 0; m < k; m++) {
                    res.dp[i][j] = min(res.dp[i][j], a.dp[i][m] + 1 + b.dp[m][j]);
                }
            }
        }
        res.empty = 0;
        return res;
    }

    void flip(int r) {
        empty = 0;
        ok[r] ^= 1;
        for(auto& it : dp) {
            for(auto& i : it) i = INF;
        }
        for(int i = 0; i < k; i++) {
            if(ok[i]) dp[i][i] = 0;
            if(i + 1 < k && ok[i] && ok[i + 1]) {
                dp[i][i + 1] = dp[i + 1][i] = 1;
            }
        }
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                for(int v = 0; v < k; v++) {
                    dp[i][j] = min(dp[i][j], dp[i][v] + dp[v][j]);
                }
            }
        }
    }
};


struct fraction_info {
    // https://dmoj.ca/problem/dmopc19c7p4
    mint dp[2][2];
    fraction_info(int x = -1) { // numerator is dp[0][0], denominator is dp[1][0]
        mset(dp, 0);
        if (x < 0) {
            dp[0][0] = 1;
            dp[1][1] = 1;
        } else {
            dp[0][0] = x;
            dp[0][1] = 1;
            dp[1][0] = 1;
            dp[1][1] = 0;
        }
    }

    friend fraction_info operator+(fraction_info const &A, fraction_info const &B) {
        fraction_info C;
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                mint sum = 0;
                for(int k = 0; k < 2; k++) {
                    sum += A.dp[i][k] * B.dp[k][j];
                }
                C.dp[i][j] = sum;
            }
        }
        return C;
    }
};

struct top_k_mode_info {
    // https://codeforces.com/contest/840/problem/d
    const static int k = 4;
    pii slot[k];

    top_k_mode_info(int value = -inf) {
        for(auto &p : slot) p = {-1, 0};
        if(value != -inf) slot[0] = {value, 1};
    }

    void add(int value, int count) {
        for(auto& [x, c] : slot) {
            if(x == value) {
                c += count;
                return;
            }
        }
        for(auto& [x, c] : slot) {
            if(x == -1) {
                x = value;
                c = count;
                return;
            }
        }
        pii cur = {value, count};
        for(auto &p : slot) 
            if(p.ss < cur.ss) swap(p, cur);

        for(auto &[x, c] : slot) {
            c -= cur.ss;
            if(c == 0) {
                x = -1;
                c = 0;
            }
        }
    }

    friend top_k_mode_info operator+(const top_k_mode_info &left, const top_k_mode_info &right) {
        top_k_mode_info res = left;
        for(const auto &p : right.slot)
            if(p.ff != -1) res.add(p.ff, p.ss);
        return res;
    }
};

struct at_most_one_swap_max_subarray_info {
    // https://codeforces.com/group/o09Gu2FpOx/contest/541481/problem/B
    // section : dp medium
    ll sm, pre, suff;
    ll pre0, suff0;
    ll pre1, suff1;
    ll mx, mn;
    ll best0, best1, best_minus;
    ll pre_with_mx, suff_with_mx;
    at_most_one_swap_max_subarray_info(ll x = -INF) : 
        sm(x), pre(max(0LL, x)), suff(max(0LL, x)), 
        pre0(max(0LL, x)), suff0(max(0LL, x)), 
        pre1(max(0LL, x)), suff1(max(0LL, x)),
        mx(x), mn(x),
        best0(max(0LL, x)), best1(max(0LL, x)), best_minus(max(0LL, x)),
        pre_with_mx(max(0LL, x)), suff_with_mx(max(0LL, x)) {} 

    friend at_most_one_swap_max_subarray_info operator+(const at_most_one_swap_max_subarray_info& a, const at_most_one_swap_max_subarray_info& b) {
        if(a.sm == -INF) return b;
        if(b.sm == -INF) return a;
        at_most_one_swap_max_subarray_info res;
        res.sm = a.sm + b.sm;
        res.pre = max(a.pre, a.sm + b.pre);
        res.suff = max(b.suff, b.sm + a.suff);
        res.best0 = max({a.best0, b.best0, a.suff + b.pre});
        res.mn = min(a.mn, b.mn);
        res.mx = max(a.mx, b.mx);
        res.best_minus = max({a.best_minus, b.best_minus, a.suff0 + b.pre, b.pre0 + a.suff});
        res.pre0 = max({a.pre, a.pre0, a.sm - a.mn + b.pre, a.sm + b.pre0});
        res.suff0 = max({b.suff, b.suff0, b.sm - b.mn + a.suff, b.sm + a.suff0});
        res.pre1 = max({res.pre, a.pre1, a.pre0 + b.mx, a.sm + b.pre1, a.pre0, a.sm - a.mn + b.pre_with_mx});
        res.suff1 = max({res.suff, b.suff1, b.suff0 + a.mx, b.sm + a.suff1, b.suff0, b.sm - b.mn + a.suff_with_mx});
        res.pre_with_mx = max({a.pre_with_mx, a.pre + max(0LL, b.mx), a.sm + b.pre_with_mx});
        res.suff_with_mx = max({b.suff_with_mx, b.suff + max(0LL, a.mx), b.sm + a.suff_with_mx});
        res.best1 = max({ a.best1, 
                b.best1,
                res.best0,
                res.best_minus,
                a.best_minus + b.mx,
                b.best_minus + a.mx,
                a.suff + b.pre_with_mx,
                a.suff_with_mx + b.pre,
                a.suff + b.pre0,
                b.pre + a.suff0,
                a.suff0 + b.mx,
                b.pre0 + a.mx,
                a.pre0 + b.mx,
                b.suff0 + a.mx,
                a.suff + b.pre1,
                b.pre + a.suff1,
                b.pre0 + a.suff_with_mx,
                a.suff0 + b.pre_with_mx,
                a.best0 + b.mx,
                b.best0 + a.mx,
                });
        return res;
    }
};

struct poly_info {
    // https://codeforces.com/group/o09Gu2FpOx/contest/541486/problem/O
    mint sum_1, sum_i, sum_i2, sum_val;
    poly_info(mint _sum_val = 0, mint _sum_1 = 0, mint _sum_i = 0, mint _sum_i2 = 0) : sum_1(_sum_1), sum_i(_sum_i), sum_i2(_sum_i2), sum_val(_sum_val) {}
    
    friend poly_info operator+(const poly_info& a, const poly_info& b) {
        poly_info res;
        res.sum_1 = a.sum_1 + b.sum_1;
        res.sum_i = a.sum_i + b.sum_i;
        res.sum_i2 = a.sum_i2 + b.sum_i2;
        res.sum_val = a.sum_val + b.sum_val;
        return res;
    }

    void apply(mint a, mint b, mint c, int len) {
        sum_val += a * sum_i2 + b * sum_i + c * sum_1;
    }
};

struct min_max_info {
    // optimizing this 
    // int res = inf;
    // for(int i = 0; i < n; i++) {
    //      if(i % 2 == 0) res = min(res, a[i]);
    //      else res = max(res, a[i]);
    // }
    // this struct works for odd n,
    // for even n and range [l, r]
    // do queries_range(l, r - 1) ans manually do the last index
    // https://www.codechef.com/problems/DEQMNMX
    int mn, mx;
    min_max_info(int _mn = 1, int _mx = inf) : mn(_mn), mx(_mx) {}

    static int get(const min_max_info& a, int x) {
        return min(a.mx, max(a.mn, x));
    }

    friend min_max_info operator+(const min_max_info& a, const min_max_info& b) {
        min_max_info res;
        res.mn = get(b, a.mn);
        res.mx = get(b, a.mx);
        return res;
    }
};

struct subarray_parity_info { // number of subarray divisiable by 3 when concatnating the subarray in decial, 0 <= a[i] <= 9
    // https://www.codechef.com/problems/QSET?tab=statement
    const static int K = 3;
    ll ans[K], suff[K], pre[K], s;
    subarray_parity_info(int x = -1) {
        s = 0;
        memset(ans, 0, sizeof(ans));
        memset(suff, 0, sizeof(suff));
        memset(pre, 0, sizeof(pre));
        if(x != -1) {
            ans[x % K]++, suff[x % K]++, pre[x % K]++;
            s = x % K;
        }
    }

    friend subarray_parity_info operator+(const subarray_parity_info& a, const subarray_parity_info& b) {
        subarray_parity_info res;
        for(int i = 0; i < K; i++) {
            for(int j = 0; j < K; j++) {
                res.ans[(i + j) % K] += a.suff[i] * b.pre[j];
            }
        }
        for(int i = 0; i < K; i++) {
            res.ans[i] += a.ans[i] + b.ans[i];
            res.pre[i] += a.pre[i];
            res.pre[(a.s + i) % K] += b.pre[i];
            res.suff[i] += b.suff[i];
            res.suff[(b.s + i) % K] += a.suff[i];
        }
        res.s = (a.s + b.s) % K;
        return res;
    }
};

#define P pair<ld, ld>
struct mul_add_info { // [mul, add]
    // https://codeforces.com/contest/895/problem/E
    constexpr static P lazy_value = {1.0, 0};
    ld s;
    P lazy;

    mul_add_info(ld v = 0) : s(v), lazy(lazy_value) { }

    bool have_lazy() const {
        return lazy != lazy_value;
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(P v, int len) {
        const auto& [mul, add] = v;
        auto& [lazy_mul, lazy_add] = lazy;
        s = s * mul + add * len;
        lazy_mul *= mul;
        lazy_add = lazy_add * mul + add;
    }

    friend mul_add_info operator+(const mul_add_info& a, const mul_add_info& b) {
        return mul_add_info(a.s + b.s);
    }
};

struct all_subarray_or_info {
    // https://codeforces.com/contest/1004/problem/F
    const static int K = 20;
    pii L[K], R[K];
    int Lsz, Rsz;
    int OR;
    ll ans;
    int len;
    bool empty;

    all_subarray_or_info() : Lsz(0), Rsz(0), OR(0), ans(0), len(0), empty(true) {}
    explicit all_subarray_or_info(int x) : Lsz(1), Rsz(1), OR(x), ans((x >= X) ? 1 : 0), len(1), empty(false) {
        L[0] = R[0] = {x, 1};
    }

    friend all_subarray_or_info operator+(const all_subarray_or_info& a, const all_subarray_or_info& b) {
        if(a.empty) return b;
        if(b.empty) return a;

        all_subarray_or_info res;
        res.empty = false;
        res.len = a.len + b.len;
        res.OR = a.OR | b.OR;
        res.ans = a.ans + b.ans;

        ll suff = a.len;
        for(int j = b.Lsz - 1, i = 0; j >= 0 && i < a.Rsz; j--) {
            while(i < a.Rsz && ((a.R[i].ff | b.L[j].ff) < X)) {
                suff -= a.R[i++].ss;
            }
            res.ans += 1LL * b.L[j].ss * suff;
        }

        res.Lsz = 0;
        for(int i = 0; i < a.Lsz; i++) 
            res.L[res.Lsz++] = a.L[i];
        for(int j = 0; j < b.Lsz; ++j) {
            int nx = b.L[j].ff | a.OR;
            int cnt = b.L[j].ss;
            if(res.Lsz && res.L[res.Lsz - 1].ff == nx) res.L[res.Lsz - 1].ss += cnt;
            else res.L[res.Lsz++] = {nx, cnt};
        }

        res.Rsz = 0;
        for(int i = 0; i < b.Rsz; i++) 
            res.R[res.Rsz++] = b.R[i];
        for(int i = 0; i < a.Rsz; i++) {
            int nx = a.R[i].ff | b.OR;
            int cnt = a.R[i].ss;
            if(res.Rsz && res.R[res.Rsz - 1].ff == nx) res.R[res.Rsz - 1].ss += cnt;
            else res.R[res.Rsz++] = {nx, cnt};
        }
        return res;
    }
};

struct max_vs_sum_info { // maximum subarray with 2 * max < sum
    // https://codeforces.com/contest/1990/problem/F
    struct part {
        ll s, mx, edge;
        int len;
    };
 
    vector<part> pre, suff;
    ll s = 0, mx = -1, pref, suf;
    int len = 0, best = 0;
 
    max_vs_sum_info() = default;
 
    max_vs_sum_info(ll x) : s(x), mx(x), len(1), best(0), pref(x), suf(x) {
        pre.pb({x, x, -1, 1});
        suff.pb({x, x, -1, 1});
    }
 
    friend max_vs_sum_info operator+(const max_vs_sum_info& A, const max_vs_sum_info& B) {
        if(A.len == 0) return B;
        if(B.len == 0) return A;
 
        auto ok = [](const part& x) {
            return x.edge == -1 || x.edge * 2 >= x.s;
        };
 
        max_vs_sum_info C;
        C.best = max(A.best, B.best);
        C.s = A.s + B.s;
        C.len = A.len + B.len;
        C.pref = A.pref;
        C.suf = B.suf;
        C.mx = max(A.mx, B.mx);
        C.pre = A.pre;
        if(!C.pre.empty() && C.pre.back().edge == -1) {
            C.pre.back().edge = B.pref;
            if(!ok(C.pre.back())) C.pre.pop_back();
        }
        for(auto x : B.pre) {
            x.mx = max(x.mx, A.mx);
            x.s += A.s;
            x.len += A.len;
            if(ok(x)) C.pre.pb(x);
        }
 
        C.suff = B.suff;
        if(!C.suff.empty() && C.suff.back().edge == -1) {
            C.suff.back().edge = A.suf;
            if(!ok(C.suff.back())) C.suff.pop_back();
        }
        for(auto x : A.suff) {
            x.mx = max(x.mx, B.mx);
            x.s += B.s;
            x.len += B.len;
            if(ok(x)) C.suff.push_back(x);
        }
        int i = 0;
        for(auto& x : A.suff) {
            while(i < int(B.pre.size()) && B.pre[i].mx <= x.mx) i++;
            if(i) {
                const auto& y = B.pre[i - 1];
                ll M = max(x.mx, y.mx);
                ll sm = x.s + y.s;
                if(M * 2 < sm) C.best = max(C.best, x.len + y.len);
            }
        }
        i = 0;
        for(auto& x : B.pre) {
            while(i < int(A.suff.size()) && A.suff[i].mx <= x.mx) i++;
            if(i) {
                const auto& y = A.suff[i - 1];
                ll M = max(x.mx, y.mx);
                ll sm = x.s + y.s;
                if(M * 2 < sm) C.best = max(C.best, x.len + y.len);
            }
        }
 
        return C;
    }
};

const int K = 3;
struct component_grid_info { // count number of connected component in grid from [l, r]
    // https://codeforces.com/contest/1661/problem/E
    int ans, empty;
    int f[2 * K + 1];
 
    component_grid_info() : empty(true) {}
 
    component_grid_info(const ar(K)& col) {
        empty = false;
        ans = 0;
        memset(f, 0, sizeof(f));
        for(int r = 0; r < K; ++r) {
            if(!col[r]) continue;
            if(r > 0 && col[r - 1]) {
                f[r + 1] = f[r];
                f[K + r + 1] = f[r];
            } else {
                ++ans;
                f[r + 1] = r + 1;
                f[K + r + 1] = r + 1;
            }
        }
    }
 
    friend component_grid_info operator+(const component_grid_info& A, const component_grid_info& B) {
        if(A.empty) return B;
        if(B.empty) return A;
        int p[4 * K + 1];
        memset(p, 0, sizeof(p));
        int res = A.ans + B.ans;
        for(int i = 1; i <= 2 * K; ++i) {
            if(A.f[i]) p[i] = A.f[i];
            if(B.f[i]) p[i + 2 * K] = B.f[i] + 2 * K;
        }
 
        auto Find = [&](int x) {
            while(p[x] != x) x = p[x] = p[p[x]];
            return x;
        };
 
        for(int r = 1; r <= K; ++r) {
            int u = K + r;
            int v = 2 * K + r;
            if(p[u] && p[v]) {
                int x = Find(u), y = Find(v);
                if(x != y) { p[x] = y; --res; }
            }
        }
 
        component_grid_info out;
        out.empty = false;
        out.ans = res;
        memset(out.f, 0, sizeof(out.f));
 
        for(int i = 1; i <= K; ++i) {
            if(!p[i]) { out.f[i] = 0; continue; }
            int ri = Find(i);
            for(int j = 1; j <= i; ++j) {
                if(Find(j) == ri) { out.f[i] = j; break; }
            }
        }
 
        for(int idx = 3 * K + 1; idx <= 4 * K; ++idx) {
            int pos = K + (idx - 3 * K);
            if(!p[idx]) { out.f[pos] = 0; continue; }
            int ri = Find(idx);
            bool linked = false;
            for(int j = 1; j <= K; ++j) {
                if(Find(j) == ri) { out.f[pos] = j; linked = true; break; }
            }
            if(linked) continue;
            for(int j = 3 * K + 1; j <= idx; ++j) {
                if(Find(j) == ri) { out.f[pos] = K + (j - 3 * K); break; }
            }
        }
 
        return out;
    }
};

const static pll lazy_value = {0, 0};
struct nc2_tree_info { // track sum of nc2 for whole tree
    // https://www.codechef.com/problems/SUMLCA?tab=statement
    ll s, sub, sub2, c, sc;
    pll lazy;
    int empty;
    nc2_tree_info() : empty(true) { }

    nc2_tree_info(ll _sub) : empty(false), s(0), sub(_sub), sub2(_sub * _sub), c(0), lazy(lazy_value), sc(0) { }

    int have_lazy() {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(pll v, int len) {
        if(empty) return;
        const auto& [flip, add] = v;
        if(flip) {
            s += 2 * c - 2 * sc + sub2 - sub; 
            c = sub - c;
            // sc = sub * (sub - c) = sub^2 - subc
            sc = sub2 - sc;
            lazy.ff ^= 1;
            lazy.ss *= -1;
        }
        s += 2 * c * add + (add * add - add) * len;
        c += add * len;
        // sc = sub * (c + add)
        sc += sub * add;
        lazy.ss += add;
    }

    friend nc2_tree_info operator+(const nc2_tree_info& a, const nc2_tree_info& b) { // careful about lazy_copy
        if(a.empty) return b;
        if(b.empty) return a;
        nc2_tree_info res;
        res.lazy = lazy_value;
        res.sub = a.sub + b.sub;
        res.sub2 = a.sub2 + b.sub2;
        res.c = a.c + b.c;
        res.sc = a.sc + b.sc;
        res.s = a.s + b.s;
        res.empty = 0;
        return res;
    }
};
