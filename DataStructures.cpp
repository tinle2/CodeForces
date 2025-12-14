template<typename T>
class Treap {
public:
    struct TreapNode {
        int pri, size, reverse;
        T key, ans, lazy;
        ll pref[2], suff[2];
        TreapNode* left;
        TreapNode* right;
        TreapNode* par;
        TreapNode* cor; // https://codeforces.com/contest/455/problem/D
        
		TreapNode(T key) : reverse(0), lazy(0), key(key), ans(key), pri(rng()), size(1), left(nullptr), right(nullptr), par(nullptr), cor(nullptr) {
//            for(int i = 0; i < HASH_COUNT; i++) {
//                pref[i] = suff[i] = key;
//            }
        }

    };

    TreapNode* root;

    int size(TreapNode* treap) {
        return !treap ? 0 : treap->size;
    }
    
    T get_ans(TreapNode* treap) {
        return !treap ? 0 : treap->ans;
    }
    
    ll get_pref(TreapNode* treap, int i) {
        return !treap ? 0 : (treap->reverse ? treap->suff[i] : treap->pref[i]);
    }

    ll get_suff(TreapNode* treap, int i) {
        return !treap ? 0 : (treap->reverse ? treap->pref[i] : treap->suff[i]);
    }
    
	void unite(TreapNode* treap) {  
        if(!treap) return;
        treap->size = size(treap->left) + size(treap->right) + 1;
        if(treap->left) treap->left->par = treap;
        if(treap->right) treap->right->par = treap;
//        for(int i = 0; i < HASH_COUNT; i++) {
//            treap->pref[i] = (get_pref(treap->right, i) + (p[i][size(treap->right)] * treap->key) % mod[i] + (get_pref(treap->left, i) * p[i][size(treap->right) + 1]) % mod[i]) % mod[i];
//            treap->suff[i] = (get_suff(treap->left, i) + (p[i][size(treap->left)] * treap->key) % mod[i] + (get_suff(treap->right, i) * p[i][size(treap->left) + 1]) % mod[i]) % mod[i];
//        }
    }

	void apply(TreapNode*treap, int flag, T val = 0) {
        if(!treap) return;
        treap->key += val;
        treap->lazy += val;
        treap->reverse ^= flag;
    }

    void push(TreapNode* treap) {  
        if(!treap) return;
        if(treap->reverse) swap(treap->left, treap->right);
        apply(treap->left, treap->reverse, treap->lazy);
        apply(treap->right, treap->reverse, treap->lazy);
        treap->lazy = 0;
        treap->reverse = 0;
        unite(treap);
    }

    void split(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        push(treap);
        if ((by_value ? (treap->key > k) : (size(treap->left) >= k))) { 
            split(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split(treap->right, treap->right, right, k - (by_value ? 0 : (size(treap->left) + 1)));
            left = treap;
        }
        unite(treap);
    }

	void merge(TreapNode*& treap, TreapNode* left, TreapNode* right) {
        if (!left || !right) {
            treap = left ? left : right;
            return;
        }
        if(left->pri < right->pri) {
            push(left);
            merge(left->right, left->right, right);
            treap = left;
        } else {
            push(right);
            merge(right->left, left, right->left);
            treap = right;
        }
        unite(treap);
    }

    void destroy(TreapNode* treap) {
        if (!treap) return;
        destroy(treap->left);
        destroy(treap->right);
        delete treap;
    }

    bool by_value;
    Treap(bool by_value = false) : root(nullptr), by_value(by_value) {} // all one base indexing
    
    ~Treap() {
        destroy(root);
    }

    TreapNode* insert(T key) { 
        TreapNode* t = new TreapNode(key);
        if(!by_value) {
            merge(root, root, t);
            return t;
        }
        TreapNode* A;
        split(root, root, A, key - 1);
        merge(root, root, t);
        merge(root, root, A);
        return t;
    }
    
	void insert_at(int k, T x) {
        if(size(root) < k) {
            insert(x);
            return;
        }
        TreapNode* A;
        split(root, root, A, k - 1);
        merge(root, root, new TreapNode(x));
        merge(root, root, A);
    }

    int find_index(TreapNode* treap) {
        int pos = 0;
        bool left = true;
        while(treap) {
            if(left) pos += size(treap->left) + 1;
            if(treap->par && treap->par->right == treap) left = true;
            else left = false;
            treap = treap->par;
        }
        return pos;
    }

    int order_of_key(int r) {
        int cnt = 0;
        TreapNode* treap = root;
        while(treap) {
            if(find_index(treap->cor) <= r) {
                cnt += size(treap->left) + 1;
                treap = treap->right;
            } else {
                treap = treap->left;
            }
        }
        return cnt;
    }

    void split_and_insert_at(int l, int r, int k) {
        // split s[l, r], concatnate t = s[1, l - 1] + s[r + 1, n], then insert s[i, j] at kth position of the t string
        TreapNode *A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r - l + 1);
        merge(root, root, B);
        if(size(root) < k) {
            merge(root, root, A);
            return;
        }
        split(root, root, B, k);
        merge(root, root, A);
        merge(root, root, B);
    }

	bool is_palindrome(int l, int r) { // https://csacademy.com/contest/archive/task/strings/statement/
        TreapNode *L = nullptr, *M = nullptr, *R = nullptr;
        split(root, L, M, l - 1);
        split(M, M, R, r - l + 1);
        push(M);  
        bool ok = (M->pref[0] == M->suff[0]) && (M->pref[1] == M->suff[1]);
        merge(M, M, R);
        merge(root, L, M);
        return ok;
    }

    void split_and_apply(int l, int r, T k = 0) { 
        if(by_value) {
            TreapNode* A, *B;
            split(root, root, A, l - 1); 
            if(A) { 
                apply(A, 1, k);
                push(A);
            }
            split(A, A, B, l - 1);
            root = merge_treap(root, A);
            merge(root, root, B);
            return;
        }
        TreapNode* A, *B;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            apply(A, 1, k);
            push(A);
        }
        merge(root, root, A);
        merge(root, root, B);
    }

	void erase_at(int k) { 
        assert(size(root) >= k);
        TreapNode*A, *B;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        merge(root, root, B);
        delete(A);
    }
	
	void update_at(int k, T x) {
        if(size(root) < k) return;
        TreapNode*A, *B;
        split(root, root, A, k - 1);
        split(A, A, B, by_value ? k : 1);
        A = new TreapNode(x);
        merge(root, root, A);
        merge(root, root, B);
    }

    T queries_at(int k) {  
        TreapNode* A, *B, *C;   
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = B->key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	T queries_range(int l, int r) {
        TreapNode*A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, (!by_value ? r - l + 1 : r));
        T res = get_ans(A);
        merge(root, root, A);
        merge(root, root, B);
        return res == INF ? -1 : res;
    }

	TreapNode* erase_range(int l, int r) {
        TreapNode* A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

	TreapNode* merge_treap(TreapNode* A, TreapNode* B) {
        if(!A) return B;
        if(!B) return A;
        if(A->pri < B->pri) swap(A, B);
        push(B);
        TreapNode *L = nullptr, *R = nullptr;
        split(A, L, R, B->key);
        B->left = merge_treap(L, B->left);
        if(B->left) B->left->par  = B;
        B->right = merge_treap(R, B->right);
        if(B->right) B->right->par = B;
        unite(B);
        return B;
    }

    void merge_treap(TreapNode* other) {
        root = merge_treap(root, other);
    }
    
    void print() {  
        print(root);
        cout << endl;
    }
	
    void print(TreapNode* treap) {  
        push(treap);
        if(!treap) return;
        print(treap->left); 
        cout << char(treap->key + 'a');
        print(treap->right);
    }
	
	void print_substring(int pos, int len) { // 1 base index
        TreapNode*A, *B;
        split(root, root, A, pos - 1);
        split(A, A, B, len);
        print(A);
        cout << endl;
        merge(root, root, A);
        merge(root, root, B);
    }
};

struct TreapNode {
    int pri, size, reverse, flip;
    int key;
    ll ans, lazy_add, lazy_set;
    ll pref[2], suff[2];
    int left;
    int right;
    TreapNode(int key = 0) : reverse(0), key(key), ans(key), lazy_add(0), lazy_set(-INF), flip(0), pri(rand()), size(1), left(0), right(0) { 
//            for(int i = 0; i < HASH_COUNT; i++) {
//                pref[i] = suff[i] = key;
//            }
    }
    bool empty() { return flip == 0 && lazy_add == 0 && reverse == 0 && lazy_set == -INF; }
    void reset() { flip = lazy_add = reverse = 0; lazy_set = -INF; }
};

static TreapNode nodes[MX];
int ptr = 0;
int new_node(int key) {
    int node = ++ptr;
    nodes[node] = TreapNode(key);
    return node;
}
template<typename T>
class Treap {
private:
    int root;

    int get_size(int treap) {
        return !treap ? 0 : nodes[treap].size;
    }
    
    T get_ans(int treap) {
        return !treap ? 0 : nodes[treap].ans;
    }
    
    ll get_pref(int treap, int i) {
        return !treap ? 0 : (nodes[treap].reverse ? nodes[treap].suff[i] : nodes[treap].pref[i]);
    }

    ll get_suff(int treap, int i) {
        return !treap ? 0 : (nodes[treap].reverse ? nodes[treap].pref[i] : nodes[treap].suff[i]);
    }
    
	void unite(int treap) {  
        if(!treap) return;
        nodes[treap].size = get_size(nodes[treap].left) + get_size(nodes[treap].right) + 1;
//        for(int i = 0; i < HASH_COUNT; i++) {
//            nodes[treap].pref[i] = (get_pref(nodes[treap].right, i) + (p[i][get_size(nodes[treap].right)] * nodes[treap].key) % mod[i] + (get_pref(nodes[treap].left, i) * p[i][get_size(nodes[treap].right) + 1]) % mod[i]) % mod[i];
//            nodes[treap].suff[i] = (get_suff(nodes[treap].left, i) + (p[i][get_size(nodes[treap].left)] * nodes[treap].key) % mod[i] + (get_suff(nodes[treap].right, i) * p[i][get_size(nodes[treap].left) + 1]) % mod[i]) % mod[i];
//        }
    }

    void push(int treap) {  
        if(!treap || nodes[treap].empty()) return;
        if(nodes[treap].reverse) {
            swap(nodes[treap].left, nodes[treap].right);
            auto& L = nodes[treap].left;
            auto& R = nodes[treap].right;
            if(L) nodes[L].reverse ^= 1;
            if(R) nodes[R].reverse ^= 1;
        }
        if(nodes[treap].flip) {
            auto&L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].flip ^= 1;
            if(R) nodes[R].flip ^= 1;
        }
        if(nodes[treap].lazy_set != -INF) {
            nodes[treap].key = nodes[treap].lazy_set;
            auto& L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].lazy_set = nodes[treap].lazy_set;
            if(R) nodes[R].lazy_set = nodes[treap].lazy_set;
            nodes[treap].reset();
            unite(treap);
            return;
        }
        if(nodes[treap].lazy_add) {
            nodes[treap].key += nodes[treap].lazy_add;
            auto&L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].lazy_add += nodes[treap].lazy_add;
            if(R) nodes[R].lazy_add += nodes[treap].lazy_add;
        }
        nodes[treap].reset();
        unite(treap);
    }

    void split(int treap, int& left, int& right, int k) {
        if(!treap) {
            left = right = 0;
            return;
        }
        push(treap);
        if((by_value ? (nodes[treap].key > k) : (get_size(nodes[treap].left) >= k))) { 
            split(nodes[treap].left, left, nodes[treap].left, k);
            right = treap;
        } else {
            split(nodes[treap].right, nodes[treap].right, right, k - (by_value ? 0 : (get_size(nodes[treap].left) + 1)));
            left = treap;
        }
        unite(treap);
    }

	void merge(int& treap, int left, int right) {
        if(!left || !right) {
            treap = left ? left : right;
            return;
        }
        if(nodes[left].pri < nodes[right].pri) {
            push(left);
            merge(nodes[left].right, nodes[left].right, right);
            treap = left;
        } else {
            push(right);
            merge(nodes[right].left, left, nodes[right].left);
            treap = right;
        }
        unite(treap);
    }
public:
    bool by_value;
    Treap(bool by_value) : root(0), by_value(by_value) {} // all one base indexing
    
    void insert(T key) { 
        if(!by_value) {
            merge(root, root, new_node(key));
            return;
        }
        int A = 0;
        split(root, root, A, key - 1);
        merge(root, root, new_node(key));
        merge(root, root, A);
    }
    
	void insert_at(int k, T x) {
        if(get_size(root) < k) {
            insert(x);
            return;
        }
        int A = 0;
        split(root, root, A, k - 1);
        merge(root, root, new_node(x));
        merge(root, root, A);
    }

    void split_and_insert_at(int l, int r, int k) {
        // split s[l, r], concatnate t = s[1, l - 1] + s[r + 1, n], then insert s[i, j] at kth position of the t string
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r - l + 1);
        merge(root, root, B);
        if(get_size(root) < k) {
            merge(root, root, A);
            return;
        }
        split(root, root, B, k);
        merge(root, root, A);
        merge(root, root, B);
    }

	bool is_palindrome(int l, int r) { // https://csacademy.com/contest/archive/task/strings/statement/
        int L = 0, M = 0, R = 0;
        split(root, L, M, l - 1);
        split(M, M, R, r - l + 1);
        push(M);  
        bool ok = (nodes[M].pref[0] == nodes[M].suff[0]) && (nodes[M].pref[1] == nodes[M].suff[1]);
        merge(M, M, R);
        merge(root, L, M);
        return ok;
    }

	void split_and_apply(int l, int r, T k = 0) { 
        if(by_value) {
            int A = 0, B = 0;
            split(root, root, A, l - 1); 
            if(A) { 
                push(A);
            }
            split(A, A, B, l - 1);
            root = merge_treap(root, A);
            merge(root, root, B);
            return;
        }
        int A = 0, B = 0;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            push(A);
        }
        merge(root, root, A);
        merge(root, root, B);
    }

    void split_and_swap(int k) {
        if(k == 0 || k == get_size(root)) return; 
        int A = 0, B = 0, C = 0;
        split(root, root, A, k);
        if(!A) return;
        merge(root, A, root);
    }

    void shift_right(int l, int r) { // [1 2 3 4] -> [4 1 2 3] and off_set by 1
        r = r - l + 1;
        int A = 0, B = 0, C = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        split(A, A, C, r - 1);
        merge(root, root, C);
        merge(root, root, A);
        merge(root, root, B);
    }

    void erase_at(int k) { 
        int A = 0, B = 0;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        merge(root, root, B);
    }
	
	void update_at(int k, T x) {
        if(get_size(root) < k) return;
        int A = 0, B = 0;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        A = new_node(x);
        merge(root, root, A);
        merge(root, root, B);
    }

    T queries_at(int k) {  
        int A = 0, B = 0, C = 0;
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = nodes[B].key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	T queries_range(int l, int r) {
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, (!by_value ? r - l + 1 : r));
        T res = get_ans(A);
        merge(root, root, A);
        merge(root, root, B);
        return res == INF ? -1 : res;
    }

	int erase_range(int l, int r) {
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

	int merge_treap(int A, int B) {
        if(!B) return A;
        if(!A) return B;
        push(B);
        A = merge_treap(A, nodes[B].left);
        A = merge_treap(A, nodes[B].right);
        nodes[B].left = nodes[B].right = 0;
        int L = 0, R = 0;
        split(A, L, R, nodes[B].key);
        merge(L, L, B);
        merge(A, L, R);
        unite(A);
        return A;
    }

    void merge_treap(int other) {
        root = merge_treap(root, other);
    }
    
    void reset() {
        for(int i = 0; i <= ptr; i++) {
            nodes[i] = TreapNode(0);
        }
        ptr = 0;
        root = 0;
    }

    void print() {  
        print(root);
        cout << endl;
    }
	
    void print(int treap) {  
        push(treap);
        if(!treap) return;
        print(nodes[treap].left); 
        cout << char(nodes[treap].key + 'a');
        print(nodes[treap].right);
    }
	
	void print_substring(int pos, int len) { // 1 base index
        int A = 0, B = 0;
        split(root, root, A, pos - 1);
        split(A, A, B, len);
        print(A);
        cout << endl;
        merge(root, root, A);
        merge(root, root, B);
    }
};

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

template<typename T>
struct range_fenwick {
    int n; 
    FW<T> B1, B2;
    range_fenwick(int n): n(n), B1(n, 0), B2(n, 0) {}

    inline void update_range(int l, int r, T v){
        B1.update_at(l, v);        
        B1.update_at(r + 1, -v);
        B2.update_at(l, v * (l - 1));  
        B2.update_at(r + 1, -v * r);
    }

    inline T prefix(int i){
        return i * B1.get(i) - B2.get(i);
    }

    inline T queries_range(int l,int r){
        return prefix(r) - prefix(l - 1);
    }

    inline T queries_at(int p) {
        return prefix(p) - prefix(p - 1);
    }
};

template<class T>
class FW_2D {
    public:
    int n;
    vector<vector<T>> root;
	vvi coord;
    T DEFAULT;
    FW_2D(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT) {
        coord.resize(n), root.resize(n);
    }
 
    void go_up(int& id) {
        id |= (id + 1);
    }
 
    void go_down(int& id) {
        id = (id & (id + 1)) - 1;
    }
 
    void add_coord(int i, int x, bool is_up = true) {
        while(i >= 0 && i < n) {
            coord[i].pb(x);
            if(is_up) go_up(i);
            else go_down(i);
        }
    }

    void update_coord_query_range(int l, int r, int low, int high) {
        add_coord(l - 1, low - 1, false);
        add_coord(l - 1, high, false);
        add_coord(r, low - 1, false);
        add_coord(r, high, false);
    }
 
    void update_coord(int i, int l, int r, bool is_up = true) {
        add_coord(i, l - 1, is_up);
        add_coord(i, r, is_up);
    }

    void add_rectangle(int r1, int c1, int r2, int c2, bool is_up = true) {
        add_coord(r1, c1, is_up);
        add_coord(r1, c2 + 1, is_up);
        add_coord(r2 + 1, c1, is_up);
        add_coord(r2 + 1, c2 + 1, is_up);
    }

    void add_point(int r, int c, bool is_up = false) { // for queries on a specific point so is_up is false
        update_coord(r, c, c, is_up);
    }
 
    void build() {
        for(int i = 0; i < n; i++) {
            srtU(coord[i]);
            root[i].resize(coord[i].size(), DEFAULT);
        }
    }
 
    int get_id(int i, int x) {
        return int(lb(all(coord[i]), x) - begin(coord[i]));
    }
 
    void update_at(int i, int x, T delta) {
        while(i < n) {
            int p = get_id(i, x);
            while(p < coord[i].size()) {
                root[i][p] = merge(root[i][p], delta);
                go_up(p);
            }
            go_up(i);
        }
    }
 
    void update_range(int i, int l, int r, T v) {
        update_at(i, l, v); 
        update_at(i, r + 1, -v);
    }
 
    void update_rectangle(int r1, int c1, int r2, int c2, T v) {
        update_range(r1, c1, c2, v);
        update_range(r2 + 1, c1, c2, -v);
    }
 
    T point_query(int i, int x) {
        T res = DEFAULT;
        while(i >= 0) {
            int p = get_id(i, x);
            while(p >= 0) {
                res = merge(res, root[i][p]);
                go_down(p);
            }
            go_down(i);
        }
        return res;
    }
 
    T bit_range_queries(int i, int low, int high) {
        if(low > high) return DEFAULT;
        return point_query(i, high) - point_query(i, low - 1);
    }
 
    T range_queries(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        return bit_range_queries(r, low, high) - bit_range_queries(l - 1, low, high);
    }

    T merge(T left, T right) {
        return left + right;
    }
};

template<class T, typename F = function<T(const T&, const T&)>>
class compress_FW {  
    public: 
    int n;
    vector<T> root;    
    T DEFAULT;
    F func;
    vll coord;
    compress_FW() {}
    compress_FW(T DEFAULT, F func) : func(func), DEFAULT(DEFAULT) {}
    
    void build() {
        srtU(coord);
        n = coord.size();
        root.resize(n + 1, DEFAULT);
    }

    void add_coord(ll x) {
        coord.pb(x);
    }

    int get_id(ll x) {
        return int(lb(all(coord), x) - begin(coord));
    }

    void update_at(ll id, T val) {  
        id = get_id(id);
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    T get(ll id) {   
        id = min(n - 1, get_id(id + 1) - 1);
        T res = DEFAULT;
        while(id >= 0) { 
            assert(id < n);
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    T queries_range(ll left, ll right) {  
        return get(right) - get(left - 1);
    }

    T queries_at(ll i) {
        return queries_range(i, i);
    }

    void update_range(ll l, ll r, T val) {
        update_at(l, val), update_at(r + 1, -val);
    }
	
	void reset() {
		root.assign(n, DEFAULT);
	}

    int select(int x) { // get pos where sum >= x
        int global = get(n - 1), curr = 0;
        for(int i = log2(n) - 1; i >= 0; i--) {
            int t = curr ^ (1LL << i);
            if(t < n && global - root[t] >= x) {
                swap(curr, t);
                global -= root[curr];
            }
        }
        return curr + 1;
    }
};

template<typename T>
struct BIT2D {
    vector<vector<T>> val;
    int n, m;

    BIT2D(int _n, int _m) : n(_n), m(_m) {
        val.assign(n, vector<T>(m, 0));
    }

    T query(int x, int y) {
        T res = 0;
        for(int i = x + 1; i > 0; i -= i & -i) {
            for(int j = y + 1; j > 0; j -= j & -j) {
                res += val[i - 1][j - 1];
            }
        }
        return res;
    }

    void update(int x, int y, T delta) {
        for(int i = x + 1; i <= n; i += i & -i) {
            for(int j = y + 1; j <= m; j += j & -j) {
                val[i - 1][j - 1] += delta;
            }
        }
    }
};


template<typename T> // for queries and updating rectangle
struct BIT2D {
    int n, m;
    vector<vector<T>> B1, B2, B3, B4;
    
    BIT2D(int n, int m) : n(n), m(m) {
        B1.assign(n + 1, vector<T>(m + 1, 0));
        B2.assign(n + 1, vector<T>(m + 1, 0));
        B3.assign(n + 1, vector<T>(m + 1, 0));
        B4.assign(n + 1, vector<T>(m + 1, 0));
    }
    
    void add(vector<vector<T>> &B, int x, int y, T v) {
        for (int i = x; i <= n; i += i & -i)
            for (int j = y; j <= m; j += j & -j)
                B[i][j] += v;
    }
    
    void update_range(int x1, int y1, int x2, int y2, T v) {
        add(B1, x1, y1, v);
        add(B1, x1, y2 + 1, -v);
        add(B1, x2 + 1, y1, -v);
        add(B1, x2 + 1, y2 + 1, v);
        
        add(B2, x1, y1, v * (x1 - 1));
        add(B2, x1, y2 + 1, -v * (x1 - 1));
        add(B2, x2 + 1, y1, -v * x2);
        add(B2, x2 + 1, y2 + 1, v * x2);
        
        add(B3, x1, y1, v * (y1 - 1));
        add(B3, x1, y2 + 1, -v * y2);
        add(B3, x2 + 1, y1, -v * (y1 - 1));
        add(B3, x2 + 1, y2 + 1, v * y2);
        
        add(B4, x1, y1, v * (x1 - 1) * (y1 - 1));
        add(B4, x1, y2 + 1, -v * (x1 - 1) * y2);
        add(B4, x2 + 1, y1, -v * x2 * (y1 - 1));
        add(B4, x2 + 1, y2 + 1, v * x2 * y2);
    }
    
    T query(vector<vector<T>> &B, int x, int y) {
        T sum = 0;
        for (int i = x; i > 0; i -= i & -i)
            for (int j = y; j > 0; j -= j & -j)
                sum += B[i][j];
        return sum;
    }
    
    T prefix_sum(int x, int y) {
        return query(B1, x, y) * x * y
             - query(B2, x, y) * y
             - query(B3, x, y) * x
             + query(B4, x, y);
    }
    
    T queries_range(int x1, int y1, int x2, int y2) {
        return prefix_sum(x2, y2)
             - prefix_sum(x1 - 1, y2)
             - prefix_sum(x2, y1 - 1)
             + prefix_sum(x1 - 1, y1 - 1);
    }
    
    void update_at(int r, int c, T v) {
        update_range(r, c, r, c, v);
    }
    
    T queries_at(int r, int c) {
        return queries_range(r, c, r, c);
    }
};

template<class T>
struct BIT2D_XOR {
    int n, m;
    vector<vector<T>> data0, data1, data2, data3;
    BIT2D_XOR() { }
    BIT2D_XOR(int n, int m) : n(n), m(m), data0(n, vector<T>(m, 0)), data1(n, vector<T>(m, 0)), data2(n, vector<T>(m, 0)), data3(n, vector<T>(m, 0)) { }
    BIT2D_XOR(int n, int m, T init) : BIT2D_XOR(vector<vector<T>>(n, vector<T>(m, init))) { }
    BIT2D_XOR(const vector<vector<T>> &v) : n((int)v.size()), m((int)v[0].size()), data0(n, vector<T>(m, 0)), data1(n, vector<T>(m, 0)), data2(n, vector<T>(m, 0)), data3(v) {
        for(auto i = 1; i <= n; ++i) 
            if(i + (i & -i) <= n) 
                for(auto j = 0; j < m; ++j) 
                    data3[i + (i & -i) - 1][j] ^= data3[i - 1][j];
        for(auto i = 0; i < n; ++i) 
            for(auto j = 1; j <= m; ++j) 
                if(j + (j & -j) <= m) 
                    data3[i][j + (j & -j) - 1] ^= data3[i][j - 1];
    }
    void update_range(int xl, int xr, int yl, int yr, T x) {
        xl--, yl--;
        assert(0 <= xl && xl <= xr && xr <= n);
        assert(0 <= yl && yl <= yr && yr <= m);
        if(xl == xr || yl == yr) return;
        for(auto i = xl + 1; i <= n; i += i & -i) 
            for(auto j = yl + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xl & 1) * x;
                data2[i - 1][j - 1] ^= (yl & 1) * x;
                data3[i - 1][j - 1] ^= (xl & yl & 1) * x;
            }
        for(auto i = xl + 1; i <= n; i += i & -i) 
            for(auto j = yr + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xl & 1) * x;
                data2[i - 1][j - 1] ^= (yr & 1) * x;
                data3[i - 1][j - 1] ^= (xl & yr & 1) * x;
            }
        for(auto i = xr + 1; i <= n; i += i & -i) 
            for(auto j = yl + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xr & 1) * x;
                data2[i - 1][j - 1] ^= (yl & 1) * x;
                data3[i - 1][j - 1] ^= (xr & yl & 1) * x;
            }
        for(auto i = xr + 1; i <= n; i += i & -i) 
            for(auto j = yr + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xr & 1) * x;
                data2[i - 1][j - 1] ^= (yr & 1) * x;
                data3[i - 1][j - 1] ^= (xr & yr & 1) * x;
            }
    }
    void update_at(int x, int y, T x_val) {
        update_range(x, x + 1, y, y + 1, x_val);
    }
    T pref(int xr, int yr) const {
        assert(0 <= xr && xr <= n);
        assert(0 <= yr && yr <= m);
        T sum0 = {}, sum1 = {}, sum2 = {}, sum3 = {};
        for(auto x = xr; x > 0; x -= x & -x) 
            for(auto y = yr; y > 0; y -= y & -y) {
                sum0 ^= data0[x - 1][y - 1];
                sum1 ^= data1[x - 1][y - 1];
                sum2 ^= data2[x - 1][y - 1];
                sum3 ^= data3[x - 1][y - 1];
            }
        return (xr & yr & 1) * sum0 ^ (yr & 1) * sum1 ^ (xr & 1) * sum2 ^ sum3;
    }
    T queries_range(int xl, int xr, int yl, int yr) const {
        xl--, yl--;
        assert(0 <= xl && xl <= xr && xr <= n);
        assert(0 <= yl && yl <= yr && yr <= m);
        if(xl == xr || yl == yr) return {};
        return pref(xr, yr) ^ pref(xl, yr) ^ pref(xr, yl) ^ pref(xl, yl);
    }
    T queries_at(int x, int y) const {
        x--, y--;
        return queries_range(x, x + 1, y, y + 1);
    }
    template<class output_stream>
    friend output_stream &operator<<(output_stream &out, const BIT2D_XOR<T> &solver) {
        for(auto i = 0; i < solver.n; ++i) {
            out << "\n[";
            for(auto j = 0; j < solver.m; ++j) {
                out << solver.queries_range(i, i + 1, j, j + 1);
                if(j != solver.m - 1) out << ", ";
            }
            out << "]\n";
        }
        return out;
    }
};

//https://atcoder.jp/contests/abc223/submissions/65149999
// https://codeforces.com/contest/938/problem/G
// this one is faster since it iterating over the # of value instead of BITS
template<typename T, int BITS = 30>
struct xor_basis {
    T basis[BITS];
    int r = 0;

    xor_basis() {
        r = 0;
        for(int b = 0; b < BITS; b++) {
            basis[b] = 0;
        }
    }
 
    bool insert(T x) {
        if (x == 0) return false;
        x = min_value(x);
        if (x == 0) return false;
        for (int i = 0; i < r; ++i)
            if ((basis[i] ^ x) < basis[i]) basis[i] ^= x;
        basis[r++] = x;
        int k = r - 1;
        while (k > 0 && max_bit(basis[k]) > max_bit(basis[k - 1])) {
            swap(basis[k], basis[k - 1]);
            --k;
        }
        return true;
    }
 
    bool contains(T x) const {
        return min_value(x) == 0;
    }
 
    T min_value(T x) const {
        if(r == BITS) return 0;
        for (int i = 0; i < r; ++i) x = min(x, x ^ basis[i]);
        return x;
    }
 
    T max_value(T x = 0) const {
        for (int i = 0; i < r; ++i) x = max(x, x ^ basis[i]);
        return x;
    }
 
    int rank() const { return r; }
 
    uint64_t size() const { return (r >= 64 ? 0ULL : 1ULL << r); }

 
	friend xor_basis operator+(const xor_basis& a, const xor_basis& b) {
        xor_basis res = a;
        for(auto& x : b.basis) {
            if(x) res.insert(x);
        }
        return res;
    }
 
    xor_basis& operator^=(T k) {
        for(int i = r - 1; i >= 0; --i)
            if(basis[i] & 1) {
                basis[i] ^= k;
            }
        return *this;
    }
 
    bool operator==(const xor_basis &o) const {
        if(r != o.r) return false;
        for(int i = 0; i < r; ++i)
            if(basis[i] != o.basis[i]) return false;
        return true;
    }
};

template<typename T = int, int BITS = 30> 
struct xor_basis {
    T basis[BITS];
    int pos[BITS];
    int zeroes = 0, r = 0;

    xor_basis() {
        for(int i = 0; i < BITS; i++) {
            basis[i] = 0; 
            pos[i] = inf;
        }
    }

    bool insert(T x, int id = -1) { // https://atcoder.jp/contests/abc141/submissions/me
        if(contains(x)) return false;
        for(int b = BITS - 1; b >= 0; b--) {
            if(x >> b & 1) {
                if(id != -1) {
                    if(!basis[b]) {
                        basis[b] = x;
                        pos[b] = id;
                        r++;
                        return true;
                    }
                    if(pos[b] > id) {
                        swap(pos[b], id);
                        swap(x, basis[b]);
                    }
                    x ^= basis[b];
                } else {
                    if(basis[b]) {
                        x ^= basis[b];
                    } else {
                        basis[b] = x;
                        r++;
                        return true;
                    }
                }
            }
        }
        if(x == 0) {
            zeroes++;
        }
        return false;
    }
	
	bool contains(T x) const {
        return min_value(x) == 0;
    }
 
    T min_value(T x) const {
        if(r == BITS) return 0;
        for(int i = BITS - 1; i >= 0; i--) x = min(x, x ^ basis[i]);
        return x;
    }
 
    T max_value(T x = 0) const {
        for(int i = BITS - 1; i >= 0; i--) {
            x = max(x, x ^ basis[i]);
        }
        return x;
    }

    int reduce(T pref, int from_bit) const {
        for(int b = BITS - 1; b >= from_bit; b--) {
            if(pref >> b & 1 && basis[b]) {
                pref ^= basis[b];
            }
        }
        return pref;
    }
 
    int can_make_prefix(T ai, T pref, int from_bit) const {
        T now = ai ^ pref;
        T v = reduce(now, from_bit);
        return (v >> from_bit) == 0;
    }
 
    T max_value(T ai, T bi) const { // find maximum x <= bi such that ai ^ x exists
        // https://codeforces.com/gym/644977/problem/C
        T x = 0;
        T mx = ai <= bi ? ai : -1;
        for(int i = BITS - 1; i >= 0; i--) {
            int b_bit = bi >> i & 1;
            if(can_make_prefix(ai, x, i)) {
                T pref = reduce(x ^ ai, i);
                T nx = x;
                for(int j = i - 1; j >= 0; j--) {
                    int change = 0;
                    if((nx | (1ll << j)) <= bi) {
                        change = 1;
                        nx |= 1ll << j;
                        pref ^= 1ll << j;
                    }
                    if(pref >> j & 1) {
                        if(basis[j]) {
                            pref ^= basis[j];
                        } else {
                            pref ^= 1ll << j;
                            nx ^= change << j;
                        }
                    }
                } 
                if(can_make_prefix(ai, nx, 0)) {
                    mx = max(mx, nx);
                }
            }
            if(b_bit) {
                x |= 1ll << i;
                if(!can_make_prefix(ai, x, i)) break;
            }
        }
        return mx;
    }

    T get_kth_smallest(ll k) { // 0 base index
        if(k < 0) return -1;
        // k >>= zeroes; // kth distinct comment this out
        if(k >= (1LL << r)) return -1;
        int b = r - 1;
        T res = 0;
        for(int j = BITS - 1; j >= 0; j--) {
            if(basis[j]) {
                if((k >> b & 1) != (res >> j & 1)) {
                    res ^= basis[j];
                } 
                b--;
            }
        }
        return res;
    }

    T get_kth_largest(ll k) {
        k = (1LL << BITS) - k - 1;
        // k = (1LL << r) - k - 1;
        return get_kth_smallest(k);
    }

    T get_ord(T x) {
        T ord = 0;
        int b = r - 1;
        for(int i = BITS - 1; i >= 0; i--) {
            if(basis[i]) {
                if(x >> i & 1) {
                    ord |= 1LL << b;
                }
                b--;
            } 
        }
        // ord <<= zero; // for duplicates
        return ord;
    }

    template<typename F>
    F subsequence_xor_contain_x(ll x, int len) { // https://codeforces.com/contest/959/problem/F
        // there are rank() important value, (len - rank()) non-important value, which can be expressed as some subset
        // can skip or take, either way we can still express them as a valid xor == 0 then xor with x
        return contains(x) ? F(2).pow(len - rank()) : 0;
    }

    int rank() {
        return r;
    }

    friend xor_basis operator+(const xor_basis& a, const xor_basis& b) {
        xor_basis res = a;
        for(auto& x : b.basis) {
            if(x) res.insert(x);
        }
        return res;
    }

    bool operator==(const xor_basis &o) const {
        for(int b = 0; b < BITS; b++)
            if(basis[b] != o.basis[b])
                return false;
        return true;
    }

    void reduced_row_echelon_form() { // https://codeforces.com/gym/102979/problem/F
        for(int i = 0; i < BITS; i++) {
            if(!basis[i]) continue;
            for(int j = i + 1; j < BITS; j++) {
                if(basis[j] >> i & 1) basis[j] ^= basis[i];
            }
        }
    }
};

struct BasisNode {
    int value;
    int id;
    int mask;
    int high;
};

inline int highest_bit(int x) {
    if (x == 0) return -1;
    return 31 - __builtin_clz(x);
}

inline void swap_mask_bits(int &x, int a, int b) {
    int bit_a = (x >> a) & 1;
    int bit_b = (x >> b) & 1;
    int delta = (bit_a ^ bit_b);
    x ^= (delta << a);
    x ^= (delta << b);
}

struct XorBasis {
    vector<BasisNode> basis;

    static inline bool minimize(int &a, int b) {
        if (b < a) { a = b; return true; }
        return false;
    }

    inline bool add(int x, int id) {
        int comb_mask = 0;
        for(auto &row : basis) {
            int reduced = x ^ row.value;
            if(minimize(x, reduced)) comb_mask ^= row.mask;
        }

        if(x == 0) return false;
        int col = int(basis.size());
        comb_mask |= (1 << col);
        for(auto &row : basis) {
            int reduced = row.value ^ x;
            if(minimize(row.value, reduced)) {
                row.mask ^= comb_mask;
                row.high = highest_bit(row.value);
            }
        }

        basis.pb(BasisNode{ .value = x, .id = id, .mask = comb_mask, .high = highest_bit(x) });
        return true;
    }
    inline int get_row_with_odd_parity_on_bits(int bits, const vi &vals_by_id) const {
        for(const auto &row : basis) {
            int v = vals_by_id[row.id];
            if(__builtin_popcount(v & bits) & 1) return row.id;
        }
        return -1;
    }

    inline int rebuild_and_delete(int id) {
        // Find its position in 'basis'
        int pos = -1;
        for(int i = 0; i < (int)basis.size(); ++i) {
            if(basis[i].id == id) { pos = i; break; }
        }
        assert(pos != -1);
        int affected_high_bits_mask = 0;
        int best_row = pos;
        int best_high = INT_MAX;
        for(int i = 0; i < (int)basis.size(); ++i) {
            if(basis[i].mask & (1 << pos)) {
                affected_high_bits_mask ^= (1 << basis[i].high);
                if(basis[i].high < best_high) {
                    best_high = basis[i].high;
                    best_row = i;
                }
            }
        }

        if(best_row != pos) {
            swap(basis[best_row].id, basis[pos].id);
            for(auto &row : basis) swap_mask_bits(row.mask, pos, best_row);
            pos = best_row;
        }

        for(int i = 0; i < (int)basis.size(); ++i) {
            if(i == pos) continue;
            if(basis[i].mask & (1 << pos)) {
                basis[i].value ^= basis[pos].value;
                basis[i].mask ^= basis[pos].mask;
                basis[i].high = highest_bit(basis[i].value);
            }
        }

        int left_mask  = (1 << pos) - 1;
        int right_mask = ((1 << (int)basis.size()) - 1) ^ (left_mask | (1 << pos));

        basis.erase(basis.begin() + pos);

        for(auto &row : basis) {
            int new_mask = (row.mask & left_mask) | ((row.mask & right_mask) >> 1);
            row.mask = new_mask;
        }

        return affected_high_bits_mask;
    }
};

template<int MAX_BITS>
class XorBasisOnline { // support deletion
	// https://eolymp.com/en/problems/11732
public:
    XorBasisOnline() : max_layer_(0), current_id_(0) {}

    int add(int x) {
        values_.pb(x);
        location_.pb({-1, -1});
        int id = current_id_++;

        if(x == 0) return id;

        for(int layer = max_layer_; layer >= 0; --layer) {
            if(layers_[layer].empty()) continue;
            if(layers_[layer].back().add(x, id)) {
                layers_[layer + 1].pb(layers_[layer].back());
                layers_[layer].pop_back();
                max_layer_ = max(max_layer_, layer + 1);

                for(const auto &row : layers_[layer + 1].back().basis) {
                    location_[row.id] = {layer + 1, (int)layers_[layer + 1].size() - 1};
                }
                return id;
            }
        }

        layers_[1].emplace_back();
        layers_[1].back().add(x, id);
        location_[id] = {1, (int)layers_[1].size() - 1};
        max_layer_ = max(max_layer_, 1);
        return id;
    }

    void erase(int id) {
        if(values_[id] == 0) return;

        int layer, idx;
        tie(layer, idx) = location_[id];
        idx = move_to_back(layer, idx);

        while(true) {
            int affected_bits = layers_[layer].back().rebuild_and_delete(id);
            int lower = layer - 1;
            while(lower > 0 && layers_[lower].empty()) --lower;
            int replacement_id = -1;
            if(lower > 0) {
                replacement_id = layers_[lower].back().get_row_with_odd_parity_on_bits(affected_bits, values_);
            }
            if (replacement_id == -1) {
                if (layer > 1) {
                    layers_[layer - 1].pb(layers_[layer].back());
                    for (const auto &row : layers_[layer - 1].back().basis) {
                        location_[row.id] = {layer - 1, (int)layers_[layer - 1].size() - 1};
                    }
                }
                layers_[layer].pop_back();
                if(max_layer_ > 0 && layers_[max_layer_].empty()) --max_layer_;
                return;
            }

            int val = values_[replacement_id];
            bool ok = layers_[layer].back().add(val, replacement_id);
            (void)ok;
            assert(ok);

            int new_layer = lower;
            int new_idx = (int)layers_[lower].size() - 1;
            location_[replacement_id] = {layer, idx};

            id = replacement_id;
            layer = new_layer;
            idx = new_idx;
        }
    }

    int rank() const {
        return max_layer_;
    }

private:
    array<vector<XorBasis>, MAX_BITS + 1> layers_;
    int max_layer_;

    vpii location_;
    vi values_;

    int current_id_;

    int move_to_back(int layer, int idx) {
        int to = (int)layers_[layer].size() - 1;
        if(to == idx) return idx;

        for(const auto &row : layers_[layer][idx].basis) {
            location_[row.id].ss = to;
        }
        for(const auto &row : layers_[layer][to].basis) {
            location_[row.id].ss = idx;
        }
        swap(layers_[layer][idx], layers_[layer][to]);
        return to;
    }
};


class MO {  
    public: 
    int n, q;  
    int block;
    const static int K = 3;
    vi a;   
    var(K) Q;
    MO(vi& a, var(K)& Q) {  // 1 base index array
        n = a.size();
        q = Q.size();
        this->a = a;    
        this->Q = Q;
		block = max(1, n / (int)sqrt(q));
    }
	
	inline ll hilbertOrder(int x, int y, int pow, int rotate) {
        if(pow == 0) return 0;
        int hpow = 1 << (pow - 1);
        int seg = (x < hpow) ? ((y < hpow) ? 0 : 3) : ((y < hpow) ? 1 : 2);
        static const int rotateDelta[4] = {3, 0, 0, 1};
        seg = (seg + rotate) & 3;
        int nx = x & (x ^ hpow), ny = y & (y ^ hpow);
        int nrot = (rotate + rotateDelta[seg]) & 3;
        ll subSquare = 1ll << (2*pow - 2);
        ll ans = seg * subSquare;
        ll add = hilbertOrder(nx, ny, pow - 1, nrot);
        if (seg == 1 || seg == 2) return ans + add;
        return ans + (subSquare - add - 1);
    }

    vll queries() {    
		// don't forget the sorting, you might accidentally remove it
//        sort(Q.begin(), Q.end(), [&](auto &A, auto &B){
//            return hilbertOrder(A[0], A[1], 18, 0)
//                 < hilbertOrder(B[0], B[1], 18, 0);
//        });
        auto cmp = [&](const ar(K)& a, const ar(K)& b) -> bool {    
            if(a[0] / block != b[0] / block) return a[0] / block < b[0] / block;
            int d = a[0] / block;   
            if(d & 1) return a[1] > b[1];
            return a[1] < b[1];
        }; sort(all(Q), cmp);
        vi pos(a);  
        srtU(pos); 
        const int N = pos.size();
        auto get_id = [&](ll x) -> int {
            return int(lb(all(pos), x) - begin(pos));
        };
        for(auto& x : a) x = get_id(x);
        vll dp(N);
        ll ans = 0;
        auto modify = [&](int x, int v) -> void {    
            auto& curr = dp[x];
            if(v == 1) {
                ans += curr * (curr - 1) / 2;
                curr++;
            }
            else {
                curr--;
                ans -= curr * (curr - 1) / 2;
            }
        };

        vll res(q);
        int l = 0, r = -1;    // modify to 0 as needed "left = 0"
        for(auto& [ql, qr, id] : Q) { // 1 base index
			while(r < qr) modify(a[++r], 1);
			while(l > ql) modify(a[--l], 1);
			while(r > qr) modify(a[r--], -1);
			while(l < ql) modify(a[l++], -1);
            res[id] = ans;
        }
        return res;
    }
	
	pair<vvar(3), vi> get_iter() { 
	// https://marisaoj.com/problem/604
        auto cmp = [&](const ar(K)& a, const ar(K)& b) -> bool {    
            if(a[0] / block != b[0] / block) return a[0] / block < b[0] / block;
            int d = a[0] / block;   
            if(d & 1) return a[1] > b[1];
            return a[1] < b[1];
        }; sort(all(Q), cmp);
        vvar(3) arr(n);
        vi I;
        int l = 1, r = 0;    // modify to 0 as needed "left = 0"
        for(int i = 0; i < q; i++) { // 1 base index
            const auto& [ql, qr, id] = Q[i];
            I.pb(id);
            while(r < qr) {
                r++;
                arr[l - 1].pb({a[r], i, -1});
                arr[r - 1].pb({a[r], i, 1});
            }
            while(l > ql) {
                l--;
                arr[l].pb({a[l], i, -1});
                arr[r].pb({a[l], i, +1});
            }
            while(r > qr) {
                arr[l - 1].pb({a[r], i, 1});
                arr[r - 1].pb({a[r], i, -1});
                r--;
            }
            while(l < ql) {
                arr[l].pb({a[l], i, 1});
                arr[r].pb({a[l], i, -1});
                l++;
            }
        }
        return {arr, I};
    }
};

struct Mo_Update {
    struct Update { int pos, oldv, newv; };
    struct Query  { int l, r, t, idx; };
    int n, B, cl, cr, ct;
    vi A, orig, freq;
    ll ans;
    vector<Update> ups;
    vector<Query> qs;

    Mo_Update(vi &arr, var(3) &ops) { // 1 base index
        n = arr.size();
        cl = 1; cr = 0; ct = 0; ans = 0; // change to cl = 0, cr = -1 if needed for 0 index
        auto curr(arr);
        for(auto &[op, l, r] : ops) {
            if(op == 0) { // update
                ups.pb({l, curr[l], r});
                curr[l] = r;
            } else { // query
                if(l > r) swap(l, r);
                qs.pb({l, r, (int)ups.size(), (int)qs.size()});
            }
        }
        orig = arr;
        for(auto &u : ups) orig.pb(u.newv);
        srtU(orig);
        A.resize(n);
        auto get_id = [&](ll x) -> int {
            return int(lb(all(orig), x) - begin(orig));
        };
        for(int i = 0; i < n; i++)
            A[i] = get_id(arr[i]);
        for (auto &u : ups) {
            u.oldv = get_id(u.oldv);
            u.newv = get_id(u.newv);
        }
        freq.assign(orig.size(), 0);
        B = max(1, int(pow(n, 2.0 / 3.0)));
    }

    inline void add(int i) {
        int v = A[i];
        if(++freq[v] == 1) ans += orig[v];
    }

    inline void remove_(int i) {
        int v = A[i];
        if(--freq[v] == 0) ans -= orig[v];
    }

    inline void apply(int t) {
        auto &u = ups[t];
        int p = u.pos;
        if(cl <= p && p <= cr) {
            remove_(p);
            A[p] = u.newv;
            add(p);
        } else A[p] = u.newv;
    }

    inline void rollback(int t) {
        auto &u = ups[t];
        int p = u.pos;
        if(cl <= p && p <= cr) {
            remove_(p);
            A[p] = u.oldv;
            add(p);
        } else A[p] = u.oldv;
    }

    vll run() {
        sort(all(qs), [&](const auto &a, const auto &b) {
            int ab = a.l / B, bb = b.l / B;
            if(ab != bb) return ab < bb;
            int ar = a.r / B, br = b.r / B;
            if(ar != br) return ar < br;
            return a.t < b.t;
        });
        vll res(qs.size());
        for(auto &q : qs) {
            while(ct < q.t) apply(ct++);
            while(ct > q.t) rollback(--ct);
            while(cl > q.l) add(--cl);
            while(cr < q.r) add(++cr);
            while(cl < q.l) remove_(cl++);
            while(cr > q.r) remove_(cr--);
            res[q.idx] = ans;
        }
        return res;
    }
};

struct Mo4D {
    // https://codeforces.com/contest/1767/problem/F
    struct Query {
        int l1, r1, l2, r2, id;
    };
    int n, B;
    vector<Query> qs;
    vi a, pos;
    vvi cc;
    vi mx, c;

    Mo4D(vi& a, int _B = 0) : n(a.size()), a(a), B(_B ? _B : int(sqrt(n)) + 1) {}

    void add_query(int l1, int r1, int l2, int r2, int id) {
        qs.push_back({l1, r1, l2, r2, id});
    }

    void add(int u) {
        int x = a[u];
        int bl = x / B;
        cc[bl][c[x]]--;
        ++c[x];
        cc[bl][c[x]]++;
        mx[bl] = max(mx[bl], c[x]);
    }

    void del(int u) {
        int x = a[u];
        int bl = x / B;
        if(mx[bl] == c[x] && cc[bl][c[x]] == 1) mx[bl]--;
        cc[bl][c[x]]--;
        --c[x];
        cc[bl][c[x]]++;
    }

    ll get() {
        int i = max_element(all(mx)) - begin(mx);
        for(int j = 0; ; j++) {
            if(c[i * B + j] == mx[i]) {
                return pos[i * B + j];
            }
        }
        return 0;
    }

    vi query() {
        sort(all(qs), [&](const Query &a, const Query &b) {
            int A[3] = {a.l1 / B, a.r1 / B, a.l2 / B};
            int Bv[3] = {b.l1 / B, b.r1 / B, b.l2 / B};
            if(A[0] != Bv[0]) return A[0] < Bv[0];
            if(A[1] != Bv[1]) return A[1] < Bv[1];
            if(A[2] != Bv[2]) return A[2] < Bv[2];
            return a.r2 < b.r2;
        });
        pos = a;
        srtU(pos);
        const int N = a.size();
        mx.resize(N / B + 1);
        cc.resize(N / B + 1, vi(N));
        c.resize(N);
        auto get_id = [&](int x) -> int {
            return int(lb(all(pos), x) - begin(pos));
        };
        for(auto& x : a) x = get_id(x);

        vi res(qs.size());
        int cl1 = 0, cr1 = -1, cl2 = 0, cr2 = -1;

        for(auto &q : qs) {
            while(cr1 < q.r1) add(++cr1);
            while(cl1 > q.l1) add(--cl1);
            while(cr1 > q.r1) del(cr1--);
            while(cl1 < q.l1) del(cl1++);

            while(cr2 < q.r2) add(++cr2);
            while(cl2 > q.l2) add(--cl2);
            while(cr2 > q.r2) del(cr2--);
            while(cl2 < q.l2) del(cl2++);

            res[q.id] = get();
        }
        return res;
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
class SparseTable {
public:
    int n, m;
    vector<vector<T>> st;
    vi log_table;
    F func;
    
    SparseTable() {}

    SparseTable(const vector<T>& a, F func) : n(a.size()), func(func) {
        m = floor(log2(n)) + 1;
        st.resize(m);
        for (int j = 0; j < m; j++) st[j].resize(n - (1 << j) + 1);
        log_table.resize(n + 1);
        for (int i = 2; i <= n; i++) log_table[i] = log_table[i / 2] + 1;
        for (int i = 0; i < n; i++) st[0][i] = a[i];
        for (int j = 1; j < m; j++) {
            for (int i = 0; i + (1 << j) <= n; i++)
                st[j][i] = func(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);
        }
    }
    
    T query(int L, int R) {
        int j = log_table[R - L + 1];
        return func(st[j][L], st[j][R - (1 << j) + 1]);
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
class SparseTable2D {
public:
    int n, m, LOGN, LOGM;
    vector<vector<vector<vector<T>>>> st;
    vi logn, logm;
    F f;
    T DEFAULT;
    SparseTable2D(const vector<vector<T>> &a, T DEFAULT, F func) : f(func), DEFAULT(DEFAULT) {
        n = a.size();
        m = a[0].size();
        LOGN = floor(log2(n)) + 1;
        LOGM = floor(log2(m)) + 1;
        logn.resize(n + 1, 0);
        logm.resize(m + 1, 0);
        for (int i = 2; i <= n; i++) logn[i] = logn[i / 2] + 1;
        for (int j = 2; j <= m; j++) logm[j] = logm[j / 2] + 1;
        st.assign(LOGN, vector<vector<vector<T>>>(LOGM));
        for (int k = 0; k < LOGN; k++) {
            int rows = n - (1 << k) + 1;
            for (int l = 0; l < LOGM; l++) {
                int cols = m - (1 << l) + 1;
                st[k][l].resize(rows, vector<T>(cols, DEFAULT));
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                st[0][0][i][j] = a[i][j];
        for (int l = 1; l < LOGM; l++) {
            for (int i = 0; i < n; i++) {
                int cols = m - (1 << l) + 1;
                for (int j = 0; j < cols; j++)
                    st[0][l][i][j] = f(st[0][l-1][i][j], st[0][l-1][i][j + (1 << (l-1))]);
            }
        }
        for (int k = 1; k < LOGN; k++) {
            int rows = n - (1 << k) + 1;
            for (int l = 0; l < LOGM; l++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < m - (1 << l) + 1; j++)
                        st[k][l][i][j] = f(st[k-1][l][i][j], st[k-1][l][i + (1 << (k-1))][j]);
                }
            }
        }
    }
    T query(int r1, int c1, int r2, int c2) {
        int h = r2 - r1 + 1, w = c2 - c1 + 1;
        int k = logn[h], l = logm[w];
        T a1 = st[k][l][r1][c1],
          a2 = st[k][l][r2 - (1 << k) + 1][c1],
          a3 = st[k][l][r1][c2 - (1 << l) + 1],
          a4 = st[k][l][r2 - (1 << k) + 1][c2 - (1 << l) + 1];
        return f(f(a1, a2), f(a3, a4));
    }
};

template <typename T, typename F = function<bool(const T&, const T&)>> // only handle max, min
struct linear_rmq {
    vector<T> values;
    F compare;
    vi head;
    vector<array<unsigned,2>> masks;

    linear_rmq() {}

    linear_rmq(const vector<T>& arr, F cmp = F{})
      : values(arr), compare(cmp),
        head(arr.size()+1),
        masks(arr.size())
    {
        vi monoStack{-1};
        int n = arr.size();
        for (int i = 0; i <= n; i++) {
            int last = -1;
            while (monoStack.back() != -1 &&
                   (i == n || !compare(values[monoStack.back()], values[i])))
            {
                if (last != -1) head[last] = monoStack.back();
                unsigned diffBit = __bit_floor(unsigned(monoStack.end()[-2] + 1) ^ i);
                masks[monoStack.back()][0] = last = (i & -diffBit);
                monoStack.pop_back();
                masks[monoStack.back() + 1][1] |= diffBit;
            }
            if (last != -1) head[last] = i;
            monoStack.pb(i);
        }
        for (int i = 1; i < n; i++) {
            masks[i][1] = (masks[i][1] | masks[i-1][1])
                        & -(masks[i][0] & -masks[i][0]);
        }
    }

    T query(int L, int R) const {
        unsigned common = masks[L][1] & masks[R][1]
                        & -__bit_floor((masks[L][0] ^ masks[R][0]) | 1);
        unsigned k = masks[L][1] ^ common;
        if (k) {
            k = __bit_floor(k);
            L = head[(masks[L][0] & -k) | k];
        }
        k = masks[R][1] ^ common;
        if (k) {
            k = __bit_floor(k);
            R = head[(masks[R][0] & -k) | k];
        }
        return compare(values[L], values[R]) ? values[L] : values[R];
    }
};

template<typename T>
class TWO_DIMENSIONAL_RANGE_QUERY {   
    public: 
    vector<vector<T>> prefix;
    int n, m;
    TWO_DIMENSIONAL_RANGE_QUERY(const vvi& grid) {  
        n = grid.size(), m = grid[0].size();
        prefix.assign(n + 1, vector<T>(m + 1));  
        for(int i = 1; i <= n; i++) {  
            T sm = 0;
            for(int j = 1; j <= m; j++) {  
                sm += grid[i - 1][j - 1];
                prefix[i][j] = sm + prefix[i - 1][j];
            }
        }
    }
    
    T get(int r1, int c1, int r2, int c2) {   
        if(r2 < r1 || c2 < c1 || r1 <= 0 || r2 > n || c1 <= 0 || c2 > m) return -inf;
        T bottomRight = prefix[r2][c2];   
        T topLeft = prefix[r1 - 1][c1 - 1];
        T topRight = prefix[r1 - 1][c2];  
        T bottomLeft = prefix[r2][c1 - 1];
        return bottomRight - topRight - bottomLeft + topLeft;
    }
};

struct nd_prefix_sum { // prefix sum on multidimensional
    // example usage
    // int n, m, k; cin >> n >> m >> k;
    // int D = n * m * k;
    // vll a(D);
    // for(int i = 0, p = 0; i < n; i++) for(int j = 0; j < m; j++) for(int kk = 0; kk < k; kk++, p++) cin >> a[p];
    // nd_prefix_sum t(a, {n, m, k});
    // while(q--) {
    //      vi low, high; cin >> low >> high;
    // }
    int D;
    vi dims, stride;
    vll pref;

    nd_prefix_sum(const vll& data, const vi& dims_) {
        dims = dims_;
        D = dims.size();
        int N = 1;
        for (int x : dims) N *= x;
        stride.assign(D, 1);
        for (int i = D - 2; i >= 0; i--) {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        pref = data;
        for (int d = 0; d < D; d++) {
            for (int i = 0; i < N; i++) {
                int idx = (i / stride[d]) % dims[d];
                if (idx > 0) {
                    pref[i] += pref[i - stride[d]];
                }
            }
        }
    }

    ll get_flat(const vi& idx) const {
        int pos = 0;
        for (int i = 0; i < D; i++) {
            pos += idx[i] * stride[i];
        }
        return pref[pos];
    }

    ll query(const vi& lo, const vi& hi) const {
        ll res = 0;
        int maskN = 1 << D;
        vi idx(D);
        for (int mask = 0; mask < maskN; mask++) {
            int bits = pct(mask);
            ll sign = (bits % 2 ? -1 : 1);
            bool ok = true;
            for (int d = 0; d < D; d++) {
                idx[d] = (mask & (1 << d)) ? lo[d] - 1 : hi[d];
                if (idx[d] < 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            res += sign * get_flat(idx);
        }
        return res;
    }
};

struct interval_solver {
    vpii a;
    vi contain, is_contained, intersect;
    vvi graph;
    vi dp;
    int n;
    interval_solver(const vpii& a) : a(a), n(a.size()) {
        // all inclusive(include itself), -1 if needed
        contain.resize(n); // how many segments does the ith contained
        is_contained.resize(n); // how many segment does the ith be in
        intersect.resize(n); // how many segment does the ith intersect with
        graph.resize(n + 1);
        solve_contain();
        solve_is_contained();
        solve_intersect();
    }

    static bool cmp(const ar(3)& a, const ar(3)& b) { // [l, r, id]
        if(a[0] != b[0]) return a[0] < b[0];
        return a[1] > b[1];
    }

    void solve_is_contained() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        ordered_set<pii> s;
        for(auto& [l, r, id] : A) {
            s.insert({r, id});
            is_contained[id] = s.size() - s.order_of_key(MP(r, -1));
        }
    }
    
    void solve_contain() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        ordered_set<pii> s;
        for(int i = n - 1; i >= 0; i--) {
            auto& [l, r, id] = A[i];
            s.insert({r, id});
            contain[id] = s.order_of_key(MP(r + 1, -1)); 
        }
    }

    void solve_intersect() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        min_heap<int> q;
        for(int i = 0; i < n; i++) {
            auto& [l, r, id] = A[i];
            while(!q.empty() && q.top() < l) q.pop();
            int left = i, right = n - 1, extra = 0;
            while(left <= right) {
                int middle = midPoint;
                if(A[middle][0] <= r) extra = middle - i + 1, left = middle + 1;
                else right = middle - 1;
            }
            intersect[i] = q.size() + extra;
            q.push(r);
        }
    }
};

template<class T>
class Splay_Tree {
public:
    struct Node {
        Node *left, *right, *parent;
        T key;   
        int delta;  
        int need_rev;
        int offset;
        int size; 
        ll Sum[5];

        Node(int x = 0) 
            : left(nullptr), right(nullptr), parent(nullptr), key(x),
              delta(0), offset(0), size(1), need_rev(0)
        {
            mset(Sum, 0);
            Sum[0] = x;
        }

        void fix() {
            if (left)  left->parent = this;
            if (right) right->parent = this;
        }

        void push_down() {
            if(need_rev) {
                swap(left, right);
                if(left) left->need_rev ^= 1;
                if(right) right->need_rev ^= 1;
                need_rev = 0;
            }
            if(delta) {
                if (left) left->delta += delta;
                if (right) right->delta += delta;
                delta = 0;
            }
        }

        void unite() {
            memset(Sum, 0, sizeof(Sum));
            size = 1;
            if (left) {
                size += left->size;
                for (int i = 0; i < 5; i++){
                    Sum[i] = left->Sum[i];
                }
            }
            Sum[(size - 1) % 5] += key;
            if (right) {
                for (int i = 0; i < 5; i++){
                    Sum[(i + size) % 5] += right->Sum[i];
                }
                size += right->size;
            }
        }
    };

private:
    Node* root;

    void rotate_right(Node* x) {
        Node* p = x->parent;
        if (!p) return;
        p->push_down();
        x->push_down();
        Node* b = x->right;
        x->right = p;
        p->left = b;
        if (b) b->parent = p;
        x->parent = p->parent;
        if (p->parent) {
            if (p->parent->left == p) p->parent->left = x;
            else p->parent->right = x;
        }
        p->parent = x;
        p->fix();
        x->fix();
        p->unite();
        x->unite();
    }

    void rotate_left(Node* x) {
        Node* p = x->parent;
        if (!p) return;
        p->push_down();
        x->push_down();
        Node* b = x->left;
        x->left = p;
        p->right = b;
        if (b) b->parent = p;
        x->parent = p->parent;
        if (p->parent) {
            if (p->parent->left == p) p->parent->left = x;
            else p->parent->right = x;
        }
        p->parent = x;
        p->fix();
        x->fix();
        p->unite();
        x->unite();
    }

    void splay(Node* x, Node* goal = nullptr) {
        if (!x) return;
        while (x->parent != goal) {
            Node* p = x->parent;
            Node* g = p->parent;
            if (g == goal) {  // Zig step.
                if (p->left == x) rotate_right(x);
                else rotate_left(x);
            }
            else if (g->left == p && p->left == x) { // Zig-Zig
                rotate_right(p);
                rotate_right(x);
            }
            else if (g->right == p && p->right == x) { // Zig-Zig
                rotate_left(p);
                rotate_left(x);
            }
            else if (g->left == p && p->right == x) { // Zig-Zag
                rotate_left(x);
                rotate_right(x);
            }
            else { // g->right == p && p->left == x, Zig-Zag.
                rotate_right(x);
                rotate_left(x);
            }
        }
        if (goal == nullptr) root = x;
    }

public:
    Splay_Tree() : root(nullptr) {}

    void Insert(int x) {
        root = merge(root, new Node(x));
    }

    Node* merge(Node* L, Node* R) {
        if(!L) return R;
        if(!R) return L;
        Node* maxL = L;
        while (maxL->right) maxL = maxL->right; // find the right_most node of left_children
        splay(maxL);
        maxL->right = R;
        if (R) R->parent = maxL;
        maxL->unite();
        return maxL;
    }
    // Find the node with key x. Splay it to the root if found.
    Node* Find(int x) {
        Node* cur = root;
        while (cur) {
            if (x == cur->key) break;
            else if (x < cur->key) cur = cur->left;
            else cur = cur->right;
        }
        if (cur) splay(cur);
        return cur;
    }

    bool Erase(int x) {
        Node* target = Find(x);
        if (!target) return false;
        splay(target);
        Node* L = target->left;
        Node* R = target->right;
        if (L) L->parent = nullptr;
        if (R) R->parent = nullptr;
        delete target;
        root = merge(L, R);
        return true;
    }

    ll get() {
        if (!root) return 0;
        root->unite();
        return root->Sum[2];
    }

    int get_k(int k) {
        if (!root || k < 0 || k >= root->size)
            throw out_of_range("k is out of bounds");
        Node* node = get_k(root, k);
        splay(node);
        return node->key;
    }

    Node* get_k(Node* curr, int k) { // index start at 0
        int left_size = (curr->left ? curr->left->size : 0);
        if (k < left_size) return get_k(curr->left, k);
        else if (k == left_size) return curr;
        else return get_k(curr->right, k - left_size - 1);
    }
};

struct LCT {
    struct Node {
        int p = 0;
        int sz = 1;
        int c[2] = {0, 0};
        ll val = 0, sm = 0, mn = INF, mx = -INF, lazy_set = INF, lazy_add = 0;
        bool flip = false;
        Node() {}
        Node(ll v)
            : p(0), sz(1), val(v), sm(v), mn(v), mx(v),
            lazy_set(INF), lazy_add(0), flip(false) {
                c[0] = c[1] = 0;
            }
    };
    vector<Node> T;
    LCT() {}
    LCT(int N) : T(N + 1) {}
    LCT(int N, const vll& A)
      : T(N + 1) {
        for (int i = 1; i <= N; ++i) {
            T[i] = Node(A[i]);
        }
    }

    bool notRoot(int x) {
        int p = T[x].p;
        return p && (T[p].c[0] == x || T[p].c[1] == x);
    }

    void push(int x) {
        if (!x) return;
        int l = T[x].c[0], r = T[x].c[1];

        if (T[x].flip) {
            swap(T[x].c[0], T[x].c[1]);
            if (l) apply_flip(l);
            if (r) apply_flip(r);
            T[x].flip = false;
        }

        if (T[x].lazy_set != INF) {
            if (l) apply_set(l, T[x].lazy_set);
            if (r) apply_set(r, T[x].lazy_set);
            T[x].lazy_set = INF;
        }

        if (T[x].lazy_add) {
            if (l) apply_add(l, T[x].lazy_add);
            if (r) apply_add(r, T[x].lazy_add);
            T[x].lazy_add = 0;
        }
    }

    void pull(int x) {
        push(T[x].c[0]);
        push(T[x].c[1]);
        int l = T[x].c[0], r = T[x].c[1];
        T[x].sz = 1 + (l ? T[l].sz : 0) + (r ? T[r].sz : 0);
        T[x].sm = T[x].val + (l ? T[l].sm : 0) + (r ? T[r].sm : 0);
        T[x].mn = min({T[x].val, l ? T[l].mn : INF, r ? T[r].mn : INF});
        T[x].mx = max({T[x].val, l ? T[l].mx : -INF, r ? T[r].mx : -INF});
    }

    void apply_add(int x, ll v) {
        if (!x) return;
        T[x].lazy_add += v;
        T[x].val += v;
        T[x].sm += v * T[x].sz;
        T[x].mn += v;
        T[x].mx += v;
    }

    void apply_set(int x, ll v) {
        if (!x) return;
        T[x].lazy_set = v;
        T[x].lazy_add = 0;
        T[x].val = v;
        T[x].sm = v * T[x].sz;
        T[x].mn = T[x].mx = v;
    }

    void apply_flip(int x) {
        if(x) T[x].flip = !T[x].flip;
    }

    void rotate(int x) {
        int p = T[x].p;
        int g = T[p].p;
        int d = (T[p].c[1] == x);
        if(notRoot(p)) T[g].c[T[g].c[1] == p] = x;
        T[x].p = g;
        T[p].c[d] = T[x].c[d ^ 1];
        if (T[p].c[d]) T[T[p].c[d]].p = p;
        T[x].c[d ^ 1] = p;
        T[p].p = x;
        pull(p);
        pull(x);
    }

    void splay(int x) {
        static vi stk;
        int y = x;
        stk.pb(y);
        while (notRoot(y)) {
            y = T[y].p;
            stk.pb(y);
        }
        while(!stk.empty()) {
            push(stk.back());
            stk.pop_back();
        }
        while (notRoot(x)) {
            int p = T[x].p;
            int g = T[p].p;
            if (notRoot(p)) {
                bool dx = (T[p].c[0] == x);
                bool dy = (T[g].c[0] == p);
                if (dx ^ dy) rotate(x);
                else rotate(p);
            }
            rotate(x);
        }
    }

    int access(int x) {
        int last = 0;
        for (int y = x; y; y = T[y].p) {
            splay(y);
            T[y].c[1] = last;
            pull(y);
            last = y;
        }
        splay(x);
        return last;
    }

    void makeRoot(int x) {
        access(x);
        apply_flip(x);
        push(x);
    }

	int find_root(int u) {
        access(u);
        while(true) {
            push(u);
            int c = T[u].c[0];
            if(!c) break;
            u = c;
        }
        splay(u);
        return u;
    }

    bool is_connected(int u, int v) {
        return find_root(u) == find_root(v);
    }

    bool link(int u, int v) {
        if(find_root(u) == find_root(v)) return false;
        makeRoot(u);
        T[u].p = v;
        return true;
    }

    bool cut(int u, int v) {
        makeRoot(u);
        access(v);
        if (T[v].c[0] == u) {
            T[T[v].c[0]].p = 0;
            T[v].c[0] = 0;
            pull(v);
            return true;
        }
        return false;
    }

    int get_path(int u, int v) {
        makeRoot(u);
        access(v);
        return v;
    }

    void update_path(int u, int v, ll k, int type) {
        int x = get_path(u, v);
        if(type == 1) apply_set(x, k);
        else apply_add(x, k);
        pull(x);
    }

    Node path_queries(int u, int v) {
        int x = get_path(u, v);
        return T[x];
    }

    int rt = 1;
    void assign_root(int r) {
        rt = r;
    }

    void change_parent(int x, int y) {
        if (x == lca(x, y)) return;
        cut(rt, x);
        link(x, y);
    }

    int lca(int x, int y) {
        makeRoot(rt);
        access(x);
        return access(y);
    }
	
	bool is_ancestor(int par, int child) {
        return lca(par, child) == par;
    }
};

struct node {
    int p = 0, c[2] = {0, 0}, pp = 0;
    bool flip = 0;
    int sz = 0, ssz = 0, vsz = 0;
    ll val = 0, sum = 0, lazy = 0, subsum = 0, vsum = 0;
    node() {}
    node(int x) {
        val = x; sum = x;
        sz = 1; lazy = 0;
        ssz = 1; vsz = 0;
        subsum = x; vsum = 0;
    }
};

struct LCT {
    vector<node> t;
    LCT() {}
    LCT(int n) : t(n + 1) {}

    int dir(int x, int y) { return t[x].c[1] == y; }

    void set(int x, int d, int y) {
        if (x) t[x].c[d] = y, pull(x);
        if (y) t[y].p = x;
    }

    void pull(int x) {
        if (!x) return;
        int &l = t[x].c[0], &r = t[x].c[1];
        push(l); push(r);
        t[x].sum    = t[l].sum    + t[r].sum    + t[x].val;
        t[x].sz     = t[l].sz     + t[r].sz     + 1;
        t[x].ssz    = t[l].ssz    + t[r].ssz    + t[x].vsz + 1;
        t[x].subsum = t[l].subsum + t[r].subsum + t[x].vsum + t[x].val;
    }

    void push(int x) {
        if (!x) return;
        int &l = t[x].c[0], &r = t[x].c[1];
        if (t[x].flip) {
            swap(l, r);
            if (l) t[l].flip ^= 1;
            if (r) t[r].flip ^= 1;
            t[x].flip = 0;
        }
        if (t[x].lazy) {
            t[x].val    += t[x].lazy;
            t[x].sum    += t[x].lazy * t[x].sz;
            t[x].subsum += t[x].lazy * t[x].ssz;
            t[x].vsum   += t[x].lazy * t[x].vsz;
            if (l) t[l].lazy += t[x].lazy;
            if (r) t[r].lazy += t[x].lazy;
            t[x].lazy = 0;
        }
    }

    void rotate(int x, int d) {
        int y = t[x].p, z = t[y].p, w = t[x].c[d];
        swap(t[x].pp, t[y].pp);
        set(y, !d, w);
        set(x, d, y);
        set(z, dir(z, y), x);
    }

    void splay(int x) {
        for (push(x); t[x].p;) {
            int y = t[x].p, z = t[y].p;
            push(z); push(y); push(x);
            int dx = dir(y, x), dy = dir(z, y);
            if (!z)           rotate(x, !dx);
            else if (dx == dy) rotate(y, !dx), rotate(x, !dx);
            else               rotate(x, dy), rotate(x, dx);
        }
    }

    void make_root(int u) {
        access(u);
        int l = t[u].c[0];
        t[l].flip ^= 1;
        swap(t[l].p, t[l].pp);
        t[u].vsz  += t[l].ssz;
        t[u].vsum += t[l].subsum;
        set(u, 0, 0);
    }

    int access(int _u) {
        int last = _u;
        for (int v = 0, u = _u; u; u = t[v = u].pp) {
            splay(u); splay(v);
            t[u].vsz  -= t[v].ssz;
            t[u].vsum -= t[v].subsum;
            int r = t[u].c[1];
            t[u].vsz  += t[r].ssz;
            t[u].vsum += t[r].subsum;
            t[v].pp = 0;
            swap(t[r].p, t[r].pp);
            set(u, 1, v);
            last = u;
        }
        splay(_u);
        return last;
    }

    void link(int u, int v) {
        make_root(v);
        access(u); splay(u);
        t[v].pp  = u;
        t[u].vsz += t[v].ssz;
        t[u].vsum += t[v].subsum;
    }

    void cut(int u) {
        access(u);
        assert(t[u].c[0] != 0);
        t[t[u].c[0]].p = 0;
        t[u].c[0] = 0;
        pull(u);
    }

    int get_parent(int u) {
        access(u); splay(u); push(u);
        u = t[u].c[0]; push(u);
        while (t[u].c[1]) {
            u = t[u].c[1]; push(u);
        }
        splay(u);
        return u;
    }

    int find_root(int u) {
        access(u); splay(u); push(u);
        while (t[u].c[0]) {
            u = t[u].c[0]; push(u);
        }
        splay(u);
        return u;
    }

    bool connected(int u, int v) { return find_root(u) == find_root(v); }

    int depth(int u) { access(u); splay(u); return t[u].sz; }

    int lca(int u, int v) {
        if (u == v) return u;
        if (depth(u) > depth(v)) swap(u, v);
        access(v);
        return access(u);
    }

    int is_root(int u) { return get_parent(u) == 0; }

    int component_size(int u) { return t[find_root(u)].ssz; }

    int subtree_size(int u) {
        int p = get_parent(u);
        if (p == 0) return component_size(u);
        cut(u);
        int ans = component_size(u);
        link(p, u);
        return ans;
    }

    ll component_sum(int u) { return t[find_root(u)].subsum; }

    ll subtree_sum(int u) {
        int p = get_parent(u);
        if (p == 0) return component_sum(u);
        cut(u);
        ll ans = component_sum(u);
        link(p, u);
        return ans;
    }

    ll subtree_query(int u, int root) {
        int cur = find_root(u);
        make_root(root);
        ll ans = subtree_sum(u);
        make_root(cur);
        return ans;
    }

    ll query(int u, int v) {
        int cur = find_root(u);
        make_root(u); access(v);
        ll ans = t[v].sum;
        make_root(cur);
        return ans;
    }

    void cut(int u, int v) {
        make_root(u);
        cut(v);
    }

    int comp_size(int u) {
        int z = find_root(u);
        int c = t[z].c[1];
        return t[c].ssz;
    }

    void update_at(int u, ll v) {
        access(u);          
        t[u].val += v;     
        pull(u);          
        access(u);       
    }
};

class BITSET {
public:
    using ubig = unsigned long long;
    int sz;
    vector<ubig> blocks;
    BITSET(int n) : sz(n) {
        int len = (n + 8 * (int)sizeof(ubig) - 1) / (8 * (int)sizeof(ubig));
        blocks.assign(len, 0ULL);
    }
    void set(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] |= (1ULL << offset);
    }
    BITSET& set() {
        for (auto &blk : blocks)
            blk = ~0ULL;
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET& set(int l, int r) {
        if (l < 0) l = 0;
        if (r >= sz) r = sz - 1;
        if (l > r) return *this;
        const int B = 8 * sizeof(ubig);
        int startBlock = l / B;
        int endBlock   = r / B;
        int startOff   = l % B;
        int endOff     = r % B;

        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            blocks[startBlock] |= mask;
        } else {
            ubig firstMask = (~0ULL << startOff);
            blocks[startBlock] |= firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                blocks[b] = ~0ULL;
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            blocks[endBlock] |= lastMask;
        }
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig tailMask = ~0ULL >> extra;
            blocks.back() &= tailMask;
        }
        return *this;
    }
    bool test(int i) const {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        return (blocks[block] >> offset) & 1ULL;
    }
    BITSET& reset() {
        fill(blocks.begin(), blocks.end(), 0ULL);
        return *this;
    }
    void reset(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] &= ~(1ULL << offset);
    }
    BITSET& reset(int l, int r) {
        if (l < 0)       l = 0;
        if (r >= sz)     r = sz - 1;
        if (l > r)       return *this;
        const int B = 8 * sizeof(ubig);
        int startBlock = l / B;
        int endBlock   = r / B;
        int startOff   = l % B;
        int endOff     = r % B;
        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            blocks[startBlock] &= ~mask;
        } else {
            ubig firstMask = (~0ULL << startOff);
            blocks[startBlock] &= ~firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                blocks[b] = 0ULL;
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            blocks[endBlock] &= ~lastMask;
        }
        return *this;
    }
    BITSET& flip() {
        for (auto &blk : blocks)
            blk = ~blk;
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    void flip(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] ^= (1ULL << offset);
    }
    int count() const {
        int cnt = 0;
        for (auto blk : blocks)
            cnt += __builtin_popcountll(blk);
        return cnt;
    }
    bool any() const {
        for (auto blk : blocks)
            if (blk != 0) return true;
        return false;
    }
    bool none() const {
        return !any();
    }
    bool all() const {
        int fullBlocks = sz / (8 * (int)sizeof(ubig));
        for (int i = 0; i < fullBlocks; i++)
            if (blocks[i] != ~0ULL) return false;
        int remaining = sz % (8 * (int)sizeof(ubig));
        if (remaining > 0) {
            ubig mask = (1ULL << remaining) - 1;
            if (blocks[fullBlocks] != mask) return false;
        }
        return true;
    }
    BITSET extract(int l, int r) const { // turn off every bit not in range [l, r]
        BITSET res(sz);
        if (l < 0) l = 0;
        if (r >= sz) r = sz - 1;
        if (l > r) return res;
        const int B = 8 * sizeof(ubig);

        int startBlock = l / B, endBlock = r / B;
        int startOff = l % B, endOff = r % B;

        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            res.blocks[startBlock] = blocks[startBlock] & mask;
        } else {
            ubig firstMask = ~0ULL << startOff;
            res.blocks[startBlock] = blocks[startBlock] & firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                res.blocks[b] = blocks[b];
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            res.blocks[endBlock] = blocks[endBlock] & lastMask;
        }
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig tailMask = ~0ULL >> extra;
            res.blocks.back() &= tailMask;
        }
        return res;
    }
    string to_string() const {
        string s;
        s.resize(sz);
        for (int i = 0; i < sz; i++)
            s[sz - 1 - i] = test(i) ? '1' : '0';
        return s;
    }
    BITSET& operator|=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] |= other.blocks[i];
        return *this;
    }
    BITSET& operator&=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] &= other.blocks[i];
        return *this;
    }
    BITSET& operator^=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] ^= other.blocks[i];
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator|(const BITSET& other) const {
        BITSET res(*this);
        res |= other;
        return res;
    }
    BITSET operator&(const BITSET& other) const {
        BITSET res(*this);
        res &= other;
        return res;
    }
    BITSET operator^(const BITSET& other) const {
        BITSET res(*this);
        res ^= other;
        return res;
    }
    BITSET operator~() const {
        BITSET res(*this);
        res.flip();
        return res;
    }
    BITSET& operator<<=(int shift) {
        if(shift >= sz) {
            fill(blocks.begin(), blocks.end(), 0ULL);
            return *this;
        }
        const int B = 8 * (int)sizeof(ubig);
        int blockShift = shift / B;
        int bitShift = shift % B;
        int nblocks = blocks.size();
        vector<ubig> newBlocks(nblocks, 0ULL);
        for (int i = nblocks - 1; i >= 0; i--) {
            int srcIndex = i - blockShift;
            if (srcIndex < 0) continue;
            newBlocks[i] |= blocks[srcIndex] << bitShift;
            if (bitShift > 0 && srcIndex - 1 >= 0)
                newBlocks[i] |= blocks[srcIndex - 1] >> (B - bitShift);
        }
        blocks.swap(newBlocks);
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator<<(int shift) const {
        BITSET res(*this);
        res <<= shift;
        return res;
    }
    BITSET& operator>>=(int shift) {
        if (shift >= sz) {
            fill(blocks.begin(), blocks.end(), 0ULL);
            return *this;
        }
        const int B = 8 * (int)sizeof(ubig);
        int blockShift = shift / B;
        int bitShift = shift % B;
        int nblocks = blocks.size();
        vector<ubig> newBlocks(nblocks, 0ULL);
        for (int i = 0; i < nblocks; i++) {
            int srcIndex = i + blockShift;
            if (srcIndex >= nblocks) continue;
            newBlocks[i] |= blocks[srcIndex] >> bitShift;
            if (bitShift > 0 && srcIndex + 1 < nblocks)
                newBlocks[i] |= blocks[srcIndex + 1] << (B - bitShift);
        }
        blocks.swap(newBlocks);
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator>>(int shift) const {
        BITSET res(*this);
        res >>= shift;
        return res;
    }
    
    int find_first() const {
        const int B = 8 * (int)sizeof(ubig);
        for (size_t b = 0; b < blocks.size(); b++) {
            if (blocks[b] != 0ULL) {
                int tz = __builtin_ctzll(blocks[b]);
                int pos = b * B + tz;
                if (pos < sz)
                    return pos;
                else
                    return -1;
            }
        }
        return -1;
    }
    int find_prev_set_bit(int pos) const {
        if(pos < 0) return -1;
        if(pos >= sz) pos = sz - 1;
        if(test(pos)) return pos;
        const int B = 8 * (int)sizeof(ubig);
        int block = pos / B, offset = pos % B;
        for (int b = block; b >= 0; b--) {
            ubig mask = (b == block) ? ((1ULL << offset) - 1ULL) : ~0ULL;
            ubig curr = blocks[b] & mask;
            if (curr != 0ULL) {
                int lz = __builtin_clzll(curr);
                return b * B + (B - 1 - lz);
            }
        }
        return -1;
    }
    int find_next_set_bit(int pos) const {
        if(pos < 0) pos = 0;
        if(pos < sz && test(pos)) return pos;
        const int B = 8 * (int)sizeof(ubig);
        int block = pos / B, offset = pos % B;
        ubig mask = ~((1ULL << (offset + 1)) - 1ULL);
        ubig curr = blocks[block] & mask;
        if(curr != 0ULL) {
            int tz = __builtin_ctzll(curr);
            int res = block * B + tz;
            return (res < sz ? res : -1);
        }
        for(size_t b = block + 1; b < blocks.size(); b++) {
            if(blocks[b] != 0ULL) {
                int tz = __builtin_ctzll(blocks[b]);
                int res = b * B + tz;
                return (res < sz ? res : -1);
            }
        }
        return -1;
    }
    bool operator==(const BITSET& other) const {
        return blocks == other.blocks;
    }
    bool operator!=(const BITSET& other) const {
        return !(*this == other);
    }
};

// Computes all possible subset sums from 0 to n that can be made using values from sizes. Runs in O(n sqrt n / 64) if
// the sum of sizes is bounded by n, and O(n^2 / 64) otherwise.
const int K = 1000;
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

template<int K>
bitset<K> apply_shifts(const bitset<K>& dp, vector<int> b) { // apply ndp |= dp << shift in log times
	// https://www.facebook.com/codingcompetitions/hacker-cup/2025/round-3/problems/C
    bitset<K> ndp;
    int sz = (int)b.size();
    if(sz == 0) return ndp;
    sort(b.begin(), b.end());
    int beg = 0;
    while(beg < sz) {
        int end = beg;
        while(end + 1 < sz && (end == beg || b[end + 1] + b[end - 1] == 2 * b[end])) {
            end++;
        }

        bitset<K> add = dp << b[beg];

        if(beg < end) {
            int diff  = b[beg + 1] - b[beg];
            int steps = end - beg;
            for(int q = 0; (1 << q) <= steps; q++) {
                add |= add << (diff * (1 << q));
                steps -= 1 << q;
            }
            add |= add << (diff * steps);
        }

        ndp |= add;
        beg = end + 1;
    }
    return ndp;
}


vi min_knapsack(int n, const vi& a) { // return a vector which a[i] is min_element to reach sum_i, sum is bounded by n, giving n * sqrt(n) * log(n) sometime faster
    // https://codeforces.com/contest/95/problem/E
    vi count(n + 1, 0);
    for(int sz : a) {
        if(sz > 0 && sz <= n) count[sz]++;
    }

    vi dp(n + 1, inf), next_dp;
    dp[0] = 0;

    for(int s = 1; s <= n; ++s) {
        int cnt = count[s];
        if(cnt <= 0) continue;

        for(int r = 0; r < s; ++r) {
            deque<pii> dq; // (q, value)
            for(int j = r, q = 0; j <= n; j += s, ++q) {
                int val = dp[j] - q;
                while(!dq.empty() && dq.front().ff < q - cnt) dq.pop_front();
                while(!dq.empty() && dq.back().ss >= val) dq.pop_back(); // for max just reverse this sign and init as -inf
                dq.emplace_back(q, val);
                dp[j] = min(dp[j], dq.front().ss + q);
            }
        }
    }
    return dp;
}

template<typename T>
vector<T> sum_knapsack(int n, const vi& a) {
    // return the # of subset sum to each dp[i]
    // nqrt(n) 
    vi cnt(n + 1, 0);
    for(int x : a) {
        if(x > 0 && x <= n) cnt[x]++;
    }
    vector<T> dp(n + 1, T(0)), ndp(n + 1, T(0));
    dp[0] = T(1);
    for(int v = 1; v <= n; ++v) {
        int c = cnt[v];
        if(c == 0) continue;
        fill(ndp.begin(), ndp.end(), T(0));
        for(int r0 = 0; r0 < v; ++r0) {
            T window_sum = T(0);
            int maxK = (n - r0) / v;
            for(int k = 0; k <= maxK; ++k) {
                int j = r0 + k * v;
                window_sum = window_sum + dp[j];
                if(k - (c + 1) >= 0) {
                    int jrem = r0 + (k - (c + 1)) * v;
                    window_sum = window_sum - dp[jrem];
                }
                ndp[j] = window_sum;
            }
        }
        dp.swap(ndp);
    }
    int zero = count(all(a), 0);
    for(auto& x : dp) {
        x *= (zero + 1);
    }
    return dp;
}

class median_tree {
public:
    void insert(int num) {
        if(left.empty() || num <= left.top()) {
            left.push(num);
            left_sum += num;
        } else {
            right.push(num);
            right_sum += num;
        }
        balance();
        clear();
    }
    
    void remove(int num) {
        if(num <= left.top()) {
            left_sum -= num;
            left_removed.push(num);
        } else {
            right_sum -= num;
            right_removed.push(num);
        }
        balance();
        clear();
    }
    
    int get_median() {
        return left.top();
    }
	
	pii get_range_median() {
        balance();
        clear();
        int ls = (int)left.size() - (int)left_removed.size();
        int rs = (int)right.size() - (int)right_removed.size();

        if (ls == rs) {
            return {left.top(), right.top()};
        } else {
            return {left.top(), left.top()};
        }
    }

    
    ll get_cost() {
        ll median = get_median();
        return median * (left.size() - left_removed.size()) - left_sum + right_sum - median * (right.size() - right_removed.size());
    }
    
private:
    void balance() {
        if(left.size() - left_removed.size() >= right.size() - right_removed.size() + 2) {
            right_sum += left.top();
            left_sum -= left.top();
            right.push(left.top());
            left.pop();
        }
        if(right.size() - right_removed.size() > left.size() - left_removed.size()) {
            left_sum += right.top();
            right_sum -= right.top();
            left.push(right.top());
            right.pop();
        }
    }
    
    void clear() {
        while(!left_removed.empty() && !left.empty() && left_removed.top() > left.top()) {
            right_removed.push(left_removed.top());
            left_removed.pop();
        }
        while(!right_removed.empty() && !left.empty() && right_removed.top() < left.top()) {
            left_removed.push(right_removed.top());
            right_removed.pop();
        }
        while(!left_removed.empty() && !left.empty() && left.top() == left_removed.top()) {
            left.pop();
            left_removed.pop();
        }
        while(!right_removed.empty() && !right.empty() && right.top() == right_removed.top()) {
            right.pop();
            right_removed.pop();
        }
    }
    
    max_heap<int> left, left_removed;
    min_heap<int> right, right_removed;
    ll left_sum = 0, right_sum = 0;
};

template<typename T>
struct Mat {
    int R, C;
    vector<vector<T>> a;
    T DEFAULT; 

    Mat(const vector<vector<T>>& m, T _DEFAULT = 0) : R((int)m.size()), C(m.empty() ? 0 : (int)m[0].size()), a(m), DEFAULT(_DEFAULT) {}

    Mat(int _R, int _C, T _DEFAULT = 0) : R(_R), C(_C), DEFAULT(_DEFAULT), a(R, vector<T>(C, _DEFAULT)) {}

    static Mat identity(int n, T _DEFAULT) {
        Mat I(n, n, _DEFAULT);
        for (int i = 0; i < n; i++)
            I.a[i][i] = T(1); // for min max do 0 instead of 1
        return I;
    }

    Mat operator*(const Mat& o) const {
        Mat r(R, o.C, DEFAULT);
        for(int i = 0; i < R; i++) {
            for(int k = 0; k < C; k++) {
                T v = a[i][k];
                if(v == DEFAULT) continue;
                for(int j = 0; j < o.C; j++) {
                    T w = o.a[k][j];
                    if(w == o.DEFAULT) continue;
                    r.a[i][j] = r.a[i][j] + v * o.a[k][j];
                }
            }
        }
        return r;
    }

    Mat pow(ll e) const {
        Mat res = identity(R, DEFAULT), base = *this;
        while(e > 0) {
            if(e & 1) res = res * base;
            base = base * base;
            e >>= 1;
        }
        return res;
    }

    friend ostream& operator<<(ostream& os, const Mat& M) {
        for(int i = 0; i < M.R; i++) {
            for(int j = 0; j < M.C; j++) {
                os << M.a[i][j];
                if(j + 1 < M.C) os << ' ';
            }
            if(i + 1 < M.R) os << '\n';
        }
        return os;
    }
	
	bool operator==(const Mat& o) const {
        if(R != o.R || C != o.C) return false;
        for(int i = 0; i < R; i++)
            for(int j = 0; j < C; j++)
                if(a[i][j] != o.a[i][j]) return false;
        return true;
    }
};

template<typename T, int R, int C>
struct Mat {
    T a[R][C];
    T DEFAULT; 

    Mat(const T (&m)[R][C], T _DEFAULT = 0) : DEFAULT(_DEFAULT) {
        for(int i = 0; i < R; i++) {
            for(int j = 0; j < C; j++) {
                a[i][j] = m[i][j];
            }
        }
    }

    Mat(int _R, int _C, T _DEFAULT = 0) : DEFAULT(_DEFAULT) {
        for(auto& it : a) for(auto& x : it) x = DEFAULT;
    }

    static Mat identity(int n, T _DEFAULT) {
        Mat I(n, n, _DEFAULT);
        for (int i = 0; i < n; i++)
            I.a[i][i] = T(1);
        return I;
    }

    Mat operator*(const Mat& o) const {
        Mat r(R, C, DEFAULT);
        for(int i = 0; i < R; i++) {
            for(int k = 0; k < C; k++) {
                T v = a[i][k];
                if(v == DEFAULT) continue;
                for(int j = 0; j < C; j++)
                    r.a[i][j] = r.a[i][j] + v * o.a[k][j];
            }
        }
        return r;
    }

    Mat pow(ll e) const {
        Mat res = identity(R, DEFAULT), base = *this;
        while(e > 0) {
            if(e & 1) res = res * base;
            base = base * base;
            e >>= 1;
        }
        return res;
    }

    friend ostream& operator<<(ostream& os, const Mat& M) {
        for(int i = 0; i < M.R; i++) {
            for(int j = 0; j < M.C; j++) {
                os << M.a[i][j];
                if(j + 1 < M.C) os << ' ';
            }
            if(i + 1 < M.R) os << '\n';
        }
        return os;
    }

    bool operator==(const Mat& o) const {
        for(int i = 0; i < R; i++)
            for(int j = 0; j < C; j++)
                if(a[i][j] != o.a[i][j]) return false;
        return true;
    }
};

template<typename T>
struct range_unique { // determine if a[l, r] contain all unique value
    vi safe;
    int n;
    range_unique(const vector<T>& a) : n(a.size()), safe(a.size()) {
        map<int, int> last;
        for(int i = 0, l = -1; i < n; i++) {
            if(last.count(a[i])) l = max(l, last[a[i]] + 1);
            safe[i] = l;
            last[a[i]] = i;
        }
    }

    bool all_unique(int l, int r) {
        return safe[r] <= l;
    }
};

struct bracket {
    vi prefix;
    int n;
    linear_rmq<int> rq;
    vi right_most; // longest balance bracket sequence starting at this index
    bracket() {}

    bracket(const string& s) {
        n = s.size();
        prefix.resize(n + 1);
        right_most.resize(n, -1);
        for(int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + (s[i] == '(' ? 1 : -1);
        }
        rq = linear_rmq<int>(prefix, [](const int& a, const int& b) {return a < b;});
        stack<int> st;
        for(int i = 0; i < n; i++) {
            if(s[i] == '(') {
                st.push(i);
                continue;
            }
            if(!st.empty() && s[st.top()] == '(') {
                right_most[st.top()] = i;
                st.pop();
            }
            else st.push(i);
        }
        for(int i = n - 1; i >= 0; i--) {
            int r = right_most[i];
            if(r == -1) continue;
            if(r + 1 < n && right_most[r + 1] != -1) right_most[i] = right_most[r + 1];
        }
    }

    bool is_balanced(int l, int r) {
        return rq.query(l + 1, r + 1) - prefix[l] >= 0 && prefix[r + 1] - prefix[l] == 0;
    }

    int longest_balance_bracket_starting_at(int i) {
        if(right_most[i] == -1) return -inf;
        return right_most[i] - i + 1;
    }

    int max_balanced_substring_len(vector<string>& A) { // given an array of bracket string, find the max subarray over all permutation
                                                        // TC : A.size() * sum(str)
        // https://open.kattis.com/problems/piecesofparentheses
        struct Piece { int diff, mn, len; };
        int n = A.size();
        vector<Piece> pieces;
        int sumPos = 0;
        for(auto& s : A) {
            int bal = 0, m = 0;
            for(char c : s) {
                bal += (c == '(' ? 1 : -1);
                m = min(m, bal);
            }
            pieces.pb({bal, m, (int)s.size()});
            if(bal > 0) sumPos += bal;
        }
        sort(all(pieces), [&](auto &a, auto &b){
                if(a.mn != b.mn) return a.mn > b.mn;
                if(a.diff != b.diff) return a.diff > b.diff;
                return a.len < b.len;
                });
        int M = sumPos;
        vi dp(M + 1, -1), highest(M + 1, -1);
        dp[0] = 0;
        highest[0] = 0;
        for(auto &p : pieces) {
            int d = p.diff, m = p.mn, t = p.len;
            if(d >= 0) {
                for(int j = M; j >= d; j--) {
                    int prev = j - d;
                    if(dp[prev] < 0) continue;
                    if(highest[prev] + m < 0) continue;
                    int candLen = dp[prev] + t;
                    int candHigh = highest[prev] + d;
                    if(candLen > dp[j]) {
                        dp[j] = candLen;
                        highest[j] = candHigh;
                    } else if(candLen == dp[j] && candHigh > highest[j]) {
                        highest[j] = candHigh;
                    }
                }
            } else {
                for(int j = 0; j <= M + d; j++) {
                    int prev = j - d;
                    if(dp[prev] < 0) continue;
                    if(highest[prev] + m < 0) continue;
                    int candLen = dp[prev] + t;
                    int candHigh = highest[prev];
                    if(candLen > dp[j]) {
                        dp[j] = candLen;
                        highest[j] = candHigh;
                    } else if(candLen == dp[j] && candHigh > highest[j]) {
                        highest[j] = candHigh;
                    }
                }
            }
        }
        return max(0, dp[0]);
    }
};

struct square_root_decomp {
    int block_size;
    vi a;
    vll blocks;
    vll lazy;
    int n;
    
    square_root_decomp(const vi& arr) : a(arr), n(arr.size()) {
        block_size = sqrt(n);
        int num_blocks = (n + block_size - 1) / block_size;
        blocks.assign(num_blocks, 0);
        lazy.assign(num_blocks, 0);
        for (int i = 0; i < n; i++) {
            blocks[i / block_size] += a[i];
        }
    }
    
    int start_id(int i) {
        return (i / block_size) * block_size;
    }
    
    int end_id(int i) {
        return min(((i / block_size) + 1) * block_size - 1, n - 1);
    }
    
    int id(int i) {
        return i / block_size;
    }
    
    void update(int i, int x) {
        int b = id(i);
        if (lazy[b] != 0) {
            for (int j = start_id(i); j <= end_id(i); j++) {
                a[j] += lazy[b];
            }
            lazy[b] = 0;
        }
        if (a[i] == x) return;
        blocks[b] -= a[i];
        a[i] = x;
        blocks[b] += a[i];
    }
    
    ll get(int r) {
        ll res = 0;
        int b = id(r);
        for (int i = 0; i < b; i++) {
            res += blocks[i];
        }
        for (int i = b * block_size; i <= r; i++) {
            res += a[i] + lazy[b];
        }
        return res;
    }
    
    ll queries_range(int l, int r) {
        if(l == 0) return get(r);
        return get(r) - get(l - 1);
    }
    
    void update_range(int l, int r, ll x) {
        int startBlock = id(l);
        int endBlock = id(r);
        if(startBlock == endBlock) {
            if(lazy[startBlock] != 0) {
                for (int i = start_id(l); i <= end_id(l); i++) {
                    a[i] += lazy[startBlock];
                }
                lazy[startBlock] = 0;
            }
            for (int i = l; i <= r; i++) {
                a[i] += x;
                blocks[startBlock] += x;
            }
            return;
        }
        if(lazy[startBlock] != 0) {
            for (int i = start_id(l); i <= end_id(l); i++) {
                a[i] += lazy[startBlock];
            }
            lazy[startBlock] = 0;
        }
        for (int i = l; i <= end_id(l); i++) {
            a[i] += x;
            blocks[startBlock] += x;
        }
        if(lazy[endBlock] != 0) {
            for (int i = start_id(r); i <= end_id(r); i++) {
                a[i] += lazy[endBlock];
            }
            lazy[endBlock] = 0;
        }
        for (int i = start_id(r); i <= r; i++) {
            a[i] += x;
            blocks[endBlock] += x;
        }
        for (int block = startBlock + 1; block <= endBlock - 1; block++) {
            lazy[block] += x;
            int blockStart = block * block_size;
            int blockEnd = min((block + 1) * block_size - 1, n - 1);
            blocks[block] += x * (blockEnd - blockStart + 1);
        }
    }
    
    ll queries_at(int i) {
        return a[i] + lazy[id(i)];
    }
};

struct square_root_decomp_max {
    int B;
    vi a;
    vi mx;
    vi cnt;
    vector<map<int, int>> freq;
    int n;

    square_root_decomp_max(int n) : n(n) {
        B = sqrt(n);
        int num_blocks = (n + B - 1) / B;
        mx.resize(num_blocks);
        cnt = vi(n);
        freq = vector<map<int, int>>(num_blocks);
    }
    
    int start_id(int i) {
        return (i / B) * B;
    }
    
    int end_id(int i) {
        return min(((i / B) + 1) * B - 1, n - 1);
    }
    
    int id(int i) {
        return i / B;
    }

    void insert(int i) {
        int I = id(i);
        if(cnt[i]) freq[I][cnt[i]]--;
        mx[I] = max(mx[I], ++cnt[i]);
        freq[I][cnt[i]]++;
    }

    void remove(int i) {
        int I = id(i);
        if(mx[I] == cnt[i] && freq[I][cnt[i]] == 1) mx[I]--;
        freq[I][cnt[i]]--;
        if(--cnt[i]) freq[I][cnt[i]]++;
    }
    
    pii queries_range(int l, int r) {
        if(r - l + 1 <= B) {
            int m = l;
            for(int i = l + 1; i <= r; i++) {
                if(cnt[i] > cnt[m]) {
                    m = i;
                }
            }
            return {m, cnt[m]};
        }
        int bl = id(l), br = id(r);
        int m = 0;
        for(int i = l; i <= end_id(l); i++) {
            m = max(m, cnt[i]);
        }
        for(int i = start_id(r); i <= r; i++) {
            m = max(m, cnt[i]);
        }
        for(int i = bl + 1; i < br; i++) {
            m = max(m, mx[i]);
        } 
        for(int i = l; i <= end_id(l); i++) {
            if(cnt[i] == m) {
                return {i, m};
            }
        }
        for(int i = bl + 1; i < br; i++) {
            if(mx[i] == m) {
                for(int j = i * B; j <= min((i + 1) * B - 1, n - 1); j++) {
                    if(cnt[j] == m) {
                        return {j, m};
                    }
                }
                assert(false);
            }
        }
        for(int i = start_id(r); i <= r; i++) {
            if(cnt[i] == m) {
                return {i, m};
            }
        }
        assert(false);
    }
};

struct cartesian_tree {
    int n, root;
    vi parent, l, r;
    cartesian_tree(const vi& a) : n(a.size()),
        parent(n, -1), l(n, -1), r(n, -1) {
        vi st;
        for (int i = 0; i < n; ++i) {
            int last = -1;
            while(!st.empty() && a[i] < a[st.back()]) {
                last = st.back();
                st.pop_back();
            }
            if(!st.empty()) {
                parent[i] = st.back();
                r[st.back()] = i;
            }
            if(last != -1) {
                parent[last] = i;
                l[i] = last;
            }
            st.pb(i);
        }
        root = st.front();
        while(parent[root] != -1) root = parent[root];
    }
};

template <bool smallest_tie = true>
struct static_range_mode_query {
    int n;
    int bk;
    int U;
    vi occur, start, pos, A, V, arr, freq, seen;
    vvpii mode_table;

    static bool better(int candidate, int current_best, int cnt_cand, int cnt_best) {
        if(cnt_cand != cnt_best) return cnt_cand > cnt_best;
        if(smallest_tie) return candidate < current_best;
        return candidate > current_best;
    }

    static_range_mode_query(const vi& input) {
        arr = input;
        n = arr.size();
        bk = max(1, (int)sqrt(n));

        V = arr;
        srtU(V);
        U = V.size();
        freq.resize(U);
        seen.resize(U);
        map<int, int> comp;
        for (int i = 0; i < U; i++) comp[V[i]] = i;

        A.resize(n);
        for (int i = 0; i < n; i++) A[i] = comp[arr[i]];

        occur.assign(n, 0);
        start.assign(U + 1, 0);
        for (int i = 0; i < n; i++) {
            int a = A[i];
            occur[i] = start[a];
            start[a]++;
        }
        for(int i = 1; i <= U; i++) start[i] += start[i - 1];

        pos.assign(n, 0);
        for(int i = n - 1; i >= 0; i--) {
            int a = A[i];
            start[a]--;
            pos[start[a]] = i;
        }

        int blocks = (n + bk - 1) / bk;
        mode_table.assign(blocks, vpii(blocks, {0, 0}));
        for(int bi = 0; bi < blocks; bi++) {
            fill(all(freq), 0);
            int bestVal = 0;
            int bestCnt = 0;
            for(int bj = bi; bj < blocks; bj++) {
                int lo = bj * bk;
                int hi = min(n, lo + bk);
                for(int k = lo; k < hi; k++) {
                    int v = A[k];
                    freq[v]++;
                    if(better(v, bestVal, freq[v], bestCnt)) {
                        bestVal = v;
                        bestCnt = freq[v];
                    }
                }
                mode_table[bi][bj] = {bestVal, bestCnt};
            }
        }
        fill(all(freq), 0);
    }

    int test = 0;

    pii query(int left, int right) {
        test++;
        if (left > right) return {-1, 0};
        int r_excl = right + 1;
        int lb = (left == 0) ? 0 : ((left - 1) / bk + 1);
        int rb = r_excl / bk;

        if(lb >= rb) {
            int bestVal = 0;
            int bestCnt = 0;
            for(int i = left; i < r_excl; i++) {
                int a = A[i];
                if(seen[a] != test) {
                    freq[a] = 0;
                }
                seen[a] = test;
                int cnt = ++freq[a];
                if(better(a, bestVal, cnt, bestCnt)) {
                    bestVal = a;
                    bestCnt = cnt;
                }
            }
            return {V[bestVal], bestCnt};
        }

        int freqVal = mode_table[lb][rb - 1].ff;
        int freqCnt = mode_table[lb][rb - 1].ss;

        for (int i = left; i < lb * bk; i++) {
            int a = A[i];
            int j = occur[i] + start[a];
            if(j + freqCnt - 1 < start[a + 1] && pos[j + freqCnt - 1] < r_excl && better(a, freqVal, freqCnt, freqCnt)) {
                freqVal = a;
            }
            while(j + freqCnt < start[a + 1] && pos[j + freqCnt] < r_excl) {
                freqCnt++;
                freqVal = a;
            }
        }

        for(int i = rb * bk; i < r_excl; i++) {
            int a = A[i];
            int j = occur[i] + start[a];
            if(j - freqCnt + 1 >= start[a] && pos[j - freqCnt + 1] >= left && better(a, freqVal, freqCnt, freqCnt)) {
                freqVal = a;
            }
            while(j - freqCnt >= start[a] && pos[j - freqCnt] >= left) {
                freqCnt++;
                freqVal = a;
            }
        }

        return {V[freqVal], freqCnt};
    }
};

struct RangeSorter {
    // use 1 base index array
    // works for permutation
    int POOLSZ;
    int free_top;
    var(2) ch;
    vi rots, rs, typ, cnt, free_list;
    set<int> runs;
    int n;

    RangeSorter(int _n, const vi& a)
        : POOLSZ(_n * ((int)log2(_n) + 3)), free_list(POOLSZ), ch(POOLSZ), cnt(POOLSZ),
          rots(_n + 2), rs(_n + 2), typ(_n + 2), n(_n)
    {
        free_top = POOLSZ - 1;
        for(int i = 1; i < POOLSZ; i++) free_list[i] = i;
        runs.clear();
        for(int i = 1; i <= n; i++) {
            rs[i] = i;
            typ[i] = false;
            runs.insert(i);
            build_node(rots[i], 1, n, a[i]);
        }
    }

    int alloc_node() {
        int x = free_list[free_top--];
        ch[x][0] = ch[x][1] = cnt[x] = 0;
        return x;
    }

    void build_node(int &T, int L, int R, int p) {
        T = alloc_node();
        cnt[T] = 1;
        if(L == R) return;
        int M = (L + R) >> 1;
        if(p <= M) build_node(ch[T][0], L, M, p);
        else build_node(ch[T][1], M + 1, R, p);
    }

    int merge_tree(int t1, int t2) {
        if(!t1 || !t2) return t1 ^ t2;
        ch[t1][0] = merge_tree(ch[t1][0], ch[t2][0]);
        ch[t1][1] = merge_tree(ch[t1][1], ch[t2][1]);
        cnt[t1] += cnt[t2];
        free_list[++free_top] = t2;
        return t1;
    }

    void split_tree(int t1, int &t2, int k) {
        t2 = alloc_node();
        int ls = cnt[ch[t1][0]];
        if(k > ls) split_tree(ch[t1][1], ch[t2][1], k - ls);
        else swap(ch[t1][1], ch[t2][1]);
        if(k < ls) split_tree(ch[t1][0], ch[t2][0], k);
        cnt[t2] = cnt[t1] - k;
        cnt[t1] = k;
    }

    int ask_kth(int T, int L, int R, int k) {
        if(L == R) return L;
        int ls = cnt[ch[T][0]];
        int M = (L + R) >> 1;
        if(k <= ls) return ask_kth(ch[T][0], L, M, k);
        else return ask_kth(ch[T][1], M + 1, R, k - ls);
    }

    void split_at(int x) {
        if(x < 1 || x > n) return;
        auto it = prev(runs.ub(x));
        int st = *it;
        if (st == x) return;
        int en = rs[st];
        int len_left = x - st;
        bool desc = typ[st];
        int root = rots[st];
        runs.erase(it);

        int left_root, right_root;
        if(!desc) {
            split_tree(root, right_root, len_left);
            left_root = root;
        } else {
            split_tree(root, left_root, (en - st + 1) - len_left);
            right_root = root;
        }
        rots[st] = left_root;
        rs[st] = x - 1;
        typ[st] = desc;
        rots[x] = right_root;
        rs[x] = en;
        typ[x] = desc;

        runs.insert(st);
        runs.insert(x);
    }

    void merge_runs(int a, int b) {
        runs.erase(b);
        rots[a] = merge_tree(rots[a], rots[b]);
        rs[a] = rs[b];
    }

    void sort_range(int L, int R, bool descending) {
        split_at(L);
        split_at(R + 1);
        auto itL = runs.lb(L);
        auto itR = runs.ub(R);
        vi to_merge;
        for(auto it = itL; it != itR; ++it)
            to_merge.pb(*it);
        int base = to_merge[0];
        for(size_t i = 1; i < to_merge.size(); i++)
            merge_runs(base, to_merge[i]);
        typ[base] = descending;
    }

    void sort_ascending(int L, int R) { sort_range(L, R, false); }
    void sort_descending(int L, int R) { sort_range(L, R, true); }

    int get(int pos) {
        auto it = prev(runs.ub(pos));
        int st = *it;
        int offset = pos - st + 1;
        bool desc = typ[st];
        int length = rs[st] - st + 1;
        if(!desc) return ask_kth(rots[st], 1, n, offset);
        else return ask_kth(rots[st], 1, n, length - offset + 1);
    }

    vi final_array() {
        vi ans(n);
        for (int i = 1; i <= n; i++)
            ans[i - 1] = get(i);
        return ans;
    }
};

struct range_set_distinct { // 1 base index
    // https://www.luogu.com.cn/problem/P4690
private:
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
        vector<Node> F;
        vi t;
        int n;
        T DEFAULT;
        PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n) { F.reserve(n * 20); F.pb(Node(DEFAULT)); }
        int update(int prev, int id, T delta, int left, int right) {  
            int curr = new_node(prev);
            if(left == right) { 
                F[curr].key = merge(F[curr].key, delta);
                return curr;
            }
            int middle = midPoint;
            if(id <= middle) F[curr].l = update(F[prev].l, id, delta, left, middle);
            else F[curr].r = update(F[prev].r, id, delta, middle + 1, right);
            F[curr].key = merge(F[F[curr].l].key, F[F[curr].r].key);
            return curr;
        }
        T queries_at(int curr, int start, int end, int left, int right) { 
            if(!curr || left > end || start > right) return DEFAULT;
            if(left >= start && right <= end) return F[curr].key;
            int middle = midPoint;  
            return merge(queries_at(F[curr].l, start, end, left, middle), queries_at(F[curr].r, start, end, middle + 1, right));
        };
        void update_at(int i, int id, T delta) { 
            while(i < n) { 
                t[i] = update(t[i], id, delta, 0, n - 1);
                i |= (i + 1);
            }
        }
        T queries_at(int i, int start, int end) {
            T res = 0;
            while(i >= 0) {
                res += queries_at(t[i], start, end, 0, n - 1);
                i = (i & (i + 1)) - 1;
            }
            return res;
        }
        T queries_range(int l, int r, int low, int high) {
            if(l > r || low > high) return DEFAULT;
            auto L = (l == 0 ? DEFAULT : queries_at(l - 1, low, high));
            auto R = queries_at(r, low, high);
            return R - L;
        }
        T merge(T left, T right) { return left + right; }
    };
    struct info {
        int l, r, x;
        info(int _l = 0, int _r = 0, int _x = 0) : l(_l), r(_r), x(_x) {}
        bool operator < (const info& a) const {
            return l < a.l;
        }
    };
    map<int, int> mp;
    vector<set<info>> each;
    set<info> global;
    int n, m;
    vi pre;
    PSGT<int> seg;
    void insert(info x) {
        global.insert(x);
        each[x.x].insert(x);
    }

    void erase(info x) {
        global.erase(x);
        each[x.x].erase(x);
    }

    void split(int p) {
        if(p >= n) return;
        auto it = global.lb({p + 1, -1, -1}); 
        it--;
        if(it->l == p) return;
        int L = it->l, R = it->r, x = it->x;
        erase(*it);
        insert({L, p - 1, x});
        insert({p, R, x});
    }

    void update(int x, int v) {
        if(pre[x] == v) return;
        seg.update_at(x, pre[x], -1);
        pre[x] = v;
        seg.update_at(x, pre[x], 1);
    }

public:
    int N;
    range_set_distinct(vi& a, int q) : n(a.size()), pre(a.size()), m(q + a.size() + 5), seg(n, 0) {
        each.resize(m);
        N = 0;
        for(int i = 1; i < n; i++) {
            if(!mp.count(a[i])) {
                mp[a[i]] = N++;
                each[mp[a[i]]].insert(info());
            }
            a[i] = mp[a[i]];
            auto it = prev(each[a[i]].end());
            pre[i] = it->l;
            seg.update_at(i, pre[i], 1);
            insert({i, i, a[i]});
        }
    }

    void range_set(int l, int r, int x) {
        if(!mp.count(x)) {
            mp[x] = N++;
            each[mp[x]].insert(info());
        }
        x = mp[x];
        split(l), split(r + 1);
        set<int> col = {x};
        while(true) {
            auto it = global.lb(info(l, -1, -1));
            if(it == end(global) || it->l > r) break;
            auto t = *it;
            col.insert(t.x);
            if(t.l > l) {
                update(t.l, t.l - 1);
            }
            erase(*it);
        }
        insert({l, r, x});
        {
            auto it = prev(each[x].lb({l, -1, -1}));
            update(l, it->r);
        }
        for(auto& c : col) {
            auto it = each[c].lb({r + 1, -1, -1});
            if(it != end(each[c])) {
                update(it->l, prev(it)->r);
            }
        }
    }

    int query(int l, int r) {
        return seg.queries_range(l, r, 0, l - 1);
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
struct MonoQueue {
    // can handle anything that's associative
    // sum, xor, min, max, and, or, product, gcd
    // careful with the default
    F op;
    T e;
    stack<pair<T,T>> in, out;

    MonoQueue(T DEFAULT, F _op) : op(_op), e(DEFAULT) {}

    void push(T x) {
        T agg = in.empty() ? x : op(in.top().ss, x);
        in.emplace(x, agg);
    }

    void pop() {
        if(out.empty()) {
            while(!in.empty()) {
                T v = in.top().ff;
                in.pop();
                T agg = out.empty() ? v : op(v, out.top().ss);
                out.emplace(v, agg);
            }
        }
        if(!out.empty()) out.pop();
    }

    T top() const {
        if(in.empty() && out.empty()) return e;
        if(in.empty()) return out.top().ss;
        if(out.empty()) return in.top().ss;
        return op(in.top().ss, out.top().ss);
    }

    bool empty() const {
        return in.empty() && out.empty();
    }
	
	int size() {
        return int(in.size() + out.size());
    }
    
    void clear() {
        while(!empty()) pop();
    }
};

template<int K>
struct ODT {
    // https://atcoder.jp/contests/abc237/tasks/abc237_g
    struct Node {
        int l, r, v;
        bool operator<(Node const &o) const { return l < o.l; }
    };

    set<Node> s;

    ODT(int n, const vector<int> &a) {
        for (int i = 1; i <= n; i++) s.insert({i, i, a[i]});
    }

    auto split(int pos) {
        auto it = s.lower_bound({pos, 0, 0});
        if (it != s.end() && it->l == pos) return it;
        --it;
        Node cur = *it;
        s.erase(it);
        s.insert({cur.l, pos - 1, cur.v});
        return s.insert({pos, cur.r, cur.v}).first;
    }

    void sort_increasing(int l, int r) {
        auto itr = split(r + 1), itl = split(l);
        array<int, K> cnt{};
        for (auto it = itl; it != itr; ++it) cnt[it->v] += it->r - it->l + 1;
        s.erase(itl, itr);
        int cur = l;
        for (int v = 0; v < K; v++) {
            int c = cnt[v];
            if (!c) continue;
            s.insert({cur, cur + c - 1, v});
            cur += c;
        }
    }

    void sort_descending(int l, int r) {
        auto itr = split(r + 1), itl = split(l);
        array<int, K> cnt{};
        for (auto it = itl; it != itr; ++it) cnt[it->v] += it->r - it->l + 1;
        s.erase(itl, itr);
        int cur = l;
        for (int v = K - 1; v >= 0; v--) {
            int c = cnt[v];
            if (!c) continue;
            s.insert({cur, cur + c - 1, v});
            cur += c;
        }
    }

    int get(int pos) const {
        auto it = s.upper_bound({pos, 0, 0});
        --it;
        return it->v;
    }
};

template<typename T>
struct arithmetic_prefix { // 0 index
    vector<T> a;
    vll PREFIX, prefix;
    int n;
    arithmetic_prefix(const vector<T>& a) : a(a), n(a.size()) {
        prefix.resize(n + 1);
        PREFIX.resize(n + 1);
        for(int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + a[i - 1];
            PREFIX[i] = PREFIX[i - 1] + ((ll)a[i - 1] * i);
        }
    }

    ll query_prefix(int l, int r, bool inclusive = true) {
        l++, r++;
        ll big = PREFIX[r] - PREFIX[l - 1];
        ll small = prefix[r] - prefix[l - 1];
        return big - (inclusive ? small * (l - 1) : small * l);
    }
    
    ll query_suffix(int l, int r, bool inclusive = true) {
        l++, r++;
        ll big = PREFIX[r] - PREFIX[l - 1];
        ll small = prefix[r] - prefix[l - 1];
        ll base = inclusive ? r + 1 : r;
        return small * base - big;
    }

    ll median_split_inward(int l, int r, bool inclusive = true) { // find best point to split between [l, r]
                                                                  // the cost bring inward
        int left = l - 1, right = r;
        ll res = INF;
        while(left <= right) {
            int middle = midPoint;
            ll A = middle < l ? 0 : query_prefix(l, middle, inclusive);
            ll B = middle == r ? 0 : query_suffix(middle + 1, r, inclusive);
            res = min(res, A + B);
            if(A < B) left = middle + 1;
            else right = middle - 1;
        }
        return res;
    }

    ll get_prefix(int l, int r) {
        return prefix[r + 1] - prefix[l];
    }
    
    ll median_split_outward(int l, int r) { // find the best point to split between [l, r]
                                            // then cost is query_suffix(l, p, false) + query_prefix(p + 1, r)
        ll tot  = get_prefix(l, r);
        ll half = (tot + 1) / 2;
        int left = l, right = r;
        while(left < right) {
            int middle = midPoint;
            if(get_prefix(l, middle) < half) left = middle + 1;
            else right = middle;
        }
        int m = left;;
        ll leftCost = (m >= l) ? query_suffix(l, m, false) : 0;
        ll rightCost = (m + 1 <= r) ? query_prefix(m + 1, r, true) : 0;
        return leftCost + rightCost;
    }
};

struct Alien_trick {
    struct state {
        ll seg;
        ll val;
        state(ll val = -INF, ll seg = 0) : val(val), seg(seg) {}
    };

    Alien_trick() {}

    state cmp(const state& a, const state& b) {
        if(a.val != b.val) return a.val > b.val ? a : b;
        return a.seg < b.seg ? a : b;
    }
    
    ll run(const vi& a, ll k) {
        int n = a.size();
        auto f = [&](ll cost) -> state {
            state out(0, 0), in; // either continue a segment, infer no cost, or starting a new segment, costing cost extra
            for(int i = 0; i < n; i++) {
                state next_out = cmp(out, in), next_in;
                state op1(out.val + a[i] - cost, out.seg + 1);
                state op2(in.val + a[i], in.seg);
                next_in = cmp(op1, op2);
                swap(out, next_out);
                swap(in, next_in);
            }
            return cmp(in, out);
        };
        ll left = -1, right = INF;
        while(left + 1 < right) {
            ll middle = midPoint;
            auto now = f(middle);
            if(now.seg > k) left = middle;
            else right = middle;
        }
        auto st = f(right);
        auto res =  st.val + right * k;
        return res;
    }

    ll run(const vi& a, ll k, ll len) {
        int n = a.size();
        vll prefix(n + 1);
        for(int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + a[i - 1];
        }
        auto f = [&](ll cost) -> state {
            vector<state> dp(n + 1);
            dp[0] = state(0, 0);
            for(int i = 1; i <= n; i++) {
                dp[i] = cmp(dp[i], dp[i - 1]);
                if(i >= len) {
                    int j = i - len;
                    ll cover = prefix[i] - prefix[j];
                    state now(dp[j].val + cover - cost, dp[j].seg + 1);
                    dp[i] = cmp(dp[i], now);
                }
                if(i == n) {
                    for(int j = n - len; j < n; j++) {
                        if(j < 0) continue;
                        ll cover = prefix[i] - prefix[j];
                        state now(dp[j].val + cover - cost, dp[j].seg + 1);
                        dp[i] = cmp(dp[i], now);
                    }
                }
            }
            return dp[n];
        };
        ll left = -1, right = INF;
        while(left + 1 < right) {
            ll middle = midPoint;
            auto now = f(middle);
            if(now.seg > k) left = middle;
            else right = middle;
        }
        auto st = f(right);
        auto res =  st.val + right * k;
        return res;
    }
};

struct DNC {
	// iterative impl : https://cses.fi/problemset/result/13062862/
    int n;
    vll &dp;
    vll next;
    DNC(vll &_dp) : dp(_dp), n((int)_dp.size() - 1), next(_dp.size()) { }

    template<typename Eval>
    void run(Eval eval) {
        fill(all(next), INF);
        dfs(1, n, 1, n, eval);
        swap(dp, next);
        
    }
    
    template<typename Eval>
    void dfs(int l, int r, int idl, int idr, Eval eval) {
        if(l > r) return;
        int mid = (l + r) >> 1;
        ll best = INF;
        int best_k = idl;
        for(int p = idl; p <= min(mid, idr); p++) {
            ll v = eval(p, mid);
            if(v < best) {
                best = v;
                best_k = p;
            }
        }
        next[mid] = best;
        dfs(l, mid - 1, idl, best_k, eval);
        dfs(mid + 1, r, best_k, idr, eval);
    }
};

struct range_lis_query_impl { // only works for permutation
    struct wavelet_matrix_impl {
        using uint = unsigned int;
        static constexpr int w = CHAR_BIT * sizeof(uint);

        static int popcount(uint x) {
#ifdef __GNUC__
            return __builtin_popcount(x);
#else
            static_assert(w == 32, "");
            x -= (x >> 1) & 0x55555555;
            x  = (x & 0x33333333) + ((x >> 2) & 0x33333333);
            x  = (x + (x >> 4)) & 0x0F0F0F0F;
            return (x * 0x01010101 >> 24) & 0x3F;
#endif
        }

        class bit_vector {
            struct node_type { uint bit = 0; int sum = 0; };
            vector<node_type> v;
        public:
            explicit bit_vector(uint n) : v(n / w + 1) {}
            void set(uint i) { v[i / w].bit |= uint(1) << (i % w); ++v[i / w].sum; }
            void build() { for (size_t i = 1; i < v.size(); ++i) v[i].sum += v[i - 1].sum; }
            int rank(uint i) const { return v[i / w].sum - popcount(v[i / w].bit & (~uint(0) << (i % w))); }
            int one()  const { return v.back().sum; }
        };

        class wavelet_matrix {
            template <class I> static bool test(I x, int k) { return (x & (I(1) << k)) != 0; }
            vector<bit_vector> mat;
        public:
            template <class I>
            wavelet_matrix(int bit_len, vector<I> a) : mat(bit_len, bit_vector(a.size())) {
                int n = a.size();
                vector<I> tmp; tmp.reserve(n);
                for (int p = bit_len - 1; p >= 0; --p) {
                    bit_vector &bv = mat[p];
                    auto it = a.begin();
                    for (int i = 0; i < n; ++i) {
                        if (test(a[i], p)) { bv.set(i); *it++ = a[i]; }
                        else               { tmp.push_back(a[i]); }
                    }
                    bv.build();
                    copy(tmp.begin(), tmp.end(), it);
                    tmp.clear();
                }
            }

            int count_less_than(int l, int r, ll key) const {
                int ret = r - l;
                for (int p = int(mat.size()) - 1; p >= 0; --p) {
                    const bit_vector &bv = mat[p];
                    int rl = bv.rank(l), rr = bv.rank(r);
                    if (test(key, p)) { l = rl; r = rr; }
                    else {
                        ret -= rr - rl;
                        int o = bv.one();
                        l += o - rl;
                        r += o - rr;
                    }
                }
                return ret - (r - l);
            }
        };
    };

    using wavelet_matrix = wavelet_matrix_impl::wavelet_matrix;
    using ptr  = vi::iterator;
    static constexpr int none = -1;

    static vi inverse(const vi &p) {
        int n = p.size();
        vi q(n, none);
        for (int i = 0; i < n; ++i) if (p[i] != none) q[p[i]] = i;
        return q;
    }

    static void unit_monge_dmul(int n, ptr st, ptr a, ptr b) {
        if (n == 1) { st[0] = 0; return; }

        ptr c_row = st; st += n;
        ptr c_col = st; st += n;

        auto map_fn = [&](int len, auto f, auto g) {
            ptr a_h = st + 0 * len;
            ptr a_m = st + 1 * len;
            ptr b_h = st + 2 * len;
            ptr b_m = st + 3 * len;

            auto split = [&](ptr v, ptr vh, ptr vm) {
                for (int i = 0; i < n; ++i)
                    if (f(v[i])) { *vh++ = g(v[i]); *vm++ = i; }
            };

            split(a, a_h, a_m);
            split(b, b_h, b_m);

            ptr c = st + 4 * len;
            unit_monge_dmul(len, c, a_h, b_h);

            for (int i = 0; i < len; ++i) {
                int row = a_m[i];
                int col = b_m[c[i]];
                c_row[row] = col;
                c_col[col] = row;
            }
        };

        int mid = n / 2;
        map_fn(mid,     [mid](int x){ return x <  mid; }, [](int x){ return x;       });
        map_fn(n - mid, [mid](int x){ return x >= mid; }, [mid](int x){ return x - mid; });

        struct d_itr { int delta = 0; int col = 0; } neg, pos;
        int row = n;

        auto move_right = [&](d_itr &it) {
            if (b[it.col] < mid ? c_col[it.col] >= row : c_col[it.col] < row) ++it.delta;
            ++it.col;
        };

        auto up = [&](d_itr &it) {
            if (a[row] < mid ? c_row[row] >= it.col : c_row[row] < it.col) --it.delta;
        };

        while (row) {
            while (pos.col != n) {
                d_itr t = pos;
                move_right(t);
                if (!t.delta) pos = t; else break;
            }
            --row;
            up(neg);
            up(pos);
            while (neg.delta) move_right(neg);
            if (neg.col > pos.col) c_row[row] = pos.col;
        }
    }

    static vi subunit_monge_dmul(vi a, vi b) {
        int n = a.size();
        vi a_inv = inverse(a), b_inv = inverse(b);
        swap(b, b_inv);

        vi a_map, b_map;
        for (int i = n - 1; i >= 0; --i) if (a[i] != none) { a_map.push_back(i); a[n - a_map.size()] = a[i]; }
        reverse(a_map.begin(), a_map.end());

        int cnt = 0;
        for (int i = 0; i < n; ++i) if (a_inv[i] == none) a[cnt++] = i;

        for (int i = 0; i < n; ++i) if (b[i] != none) { b[b_map.size()] = b[i]; b_map.push_back(i); }
        cnt = b_map.size();
        for (int i = 0; i < n; ++i) if (b_inv[i] == none) b[cnt++] = i;

        int stack_size = [](int m){ int ret = 0; while (m > 1) { ret += 2 * m; m = (m + 1) / 2; ret += 4 * m; } return ret + 1; }(n);

        vi c(stack_size);
        unit_monge_dmul(n, c.begin(), a.begin(), b.begin());

        vi c_pad(n, none);
        for (int i = 0; i < (int)a_map.size(); ++i) {
            int t = c[n - a_map.size() + i];
            if (t < (int)b_map.size()) c_pad[a_map[i]] = b_map[t];
        }
        return c_pad;
    }

    static vi seaweed_doubling(const vi &p) {
        int n = p.size();
        if (n == 1) return vi{none};
        int mid = n / 2;

        vi lo, hi, lo_map, hi_map;
        for (int i = 0; i < n; ++i) {
            int e = p[i];
            if (e < mid) { lo.push_back(e); lo_map.push_back(i); }
            else         { hi.push_back(e - mid); hi_map.push_back(i); }
        }

        lo = seaweed_doubling(lo);
        hi = seaweed_doubling(hi);

        vi lo_pad(n), hi_pad(n);
        iota(lo_pad.begin(), lo_pad.end(), 0);
        iota(hi_pad.begin(), hi_pad.end(), 0);

        for (int i = 0; i < mid; ++i)          lo_pad[lo_map[i]] = (lo[i] == none) ? none : lo_map[lo[i]];
        for (int i = 0; mid + i < n; ++i)     hi_pad[hi_map[i]] = (hi[i] == none) ? none : hi_map[hi[i]];

        return subunit_monge_dmul(move(lo_pad), move(hi_pad));
    }

    static bool is_permutation(const vi &p) {
        int n = p.size();
        vector<bool> used(n, false);
        for (int e : p) {
            if (e < 0 || e >= n || used[e]) return false;
            used[e] = true;
        }
        return true;
    }

    static wavelet_matrix convert(const vi &p) {
        assert(is_permutation(p));
        int n = p.size();
        vi row = n ? seaweed_doubling(vi(p.begin(), p.end())) : vi();
        for (int &e : row) if (e == none) e = n;
        int bit_len = 0; for (int t = n; t; t >>= 1) ++bit_len;
        return wavelet_matrix(bit_len, move(row));
    }

    class range_lis_query {
        int n;
        wavelet_matrix wm;
    public:
        range_lis_query() = default;
        explicit range_lis_query(const vector<int> &p) : n(p.size()), wm(convert(p)) {}
        int query(int l, int r) const {
            assert(0 <= l && l <= r && r < n);
            return (r - l + 1) - wm.count_less_than(l, n, r + 1);
        }
    };
}; using range_lis = range_lis_query_impl::range_lis_query;

struct static_range_palindrome {
    int n;
    vi a, b, c, d, d1, d2;
    wavelet_psgt oddl, oddr, evenl, evenr;

    static_range_palindrome(const string &s) {
        n = (int)s.size();
        d1.assign(n, 0);
        d2.assign(n, 0);
        build_manachers(s);

        a.resize(n);
        b.resize(n);
        c.resize(n);
        d.resize(n);
        for (int i = 0; i < n; i++) {
            a[i] = d1[i] - (i + 1);
            b[i] = d1[i] + (i + 1);
            c[i] = d2[i] - (i + 1);
            d[i] = d2[i] + (i + 1);
        }

        oddl = wavelet_psgt(a);
        oddr = wavelet_psgt(b);
        evenl = wavelet_psgt(c);
        evenr = wavelet_psgt(d);
    }

    // number of odd-length palindromes fully inside [l..r]
    ll query_odd(int l, int r) {
        return compute_odd(l, r);
    }
    // number of even-length palindromes fully inside [l..r]
    ll query_even(int l, int r) {
        return compute_even(l, r);
    }

    ll query_all(int l, int r) {
        return query_odd(l, r) + query_even(l, r);
    }

private:
    void build_manachers(const string &s) {
        for(int i = 0, L = 0, R = -1; i < n; i++) {
            int k = (i > R ? 1 : min(d1[L + R - i], R - i + 1));
            while(i - k >= 0 && i + k < n && s[i - k] == s[i + k]) k++;
            d1[i] = k--;
            if(i + k > R) { L = i - k; R = i + k; }
        }
        for(int i = 0, L = 0, R = -1; i < n; i++) {
            int k = (i > R ? 0 : min(d2[L + R - i + 1], R - i + 1));
            while(i - k - 1 >= 0 && i + k < n && s[i - k - 1] == s[i + k]) k++;
            d2[i] = k--;
            if(i + k > R) { L = i - k - 1; R = i + k; }
        }
    }

    inline ll get(int L, int R) {
        return (ll)R * (R + 1) / 2 - (ll)(L - 1) * L / 2;
    }

    ll compute_odd(int l, int r) {
        int m = (l + r) >> 1;
        int c1 = -l;
        auto n1 = oddl.query_leq(l, m, c1);
        ll less1 = n1.cnt, sum1 = n1.sm;
        ll left  = get(l + 1, m + 1) + sum1 + (ll)(m - l + 1 - less1) * c1;

        int c2 = r + 2;
        auto n2 = oddr.query_leq(m + 1, r, c2);
        ll less2 = n2.cnt, sum2 = n2.sm;
        ll right = -get((m + 1) + 1, (r) + 1) + sum2 + (ll)(r - m - less2) * c2;
        return left + right;
    }

    ll compute_even(int l, int r) {
        int m = (l + r) >> 1;
        int c1 = -(l + 1);
        auto n1  = evenl.query_leq(l, m, c1);
        ll less1 = n1.cnt, sum1 = n1.sm;
        ll left  = get(l + 1, m + 1) + sum1 + (ll)(m - l + 1 - less1) * c1;

        int c2 = r + 2;
        auto n2  = evenr.query_leq(m + 1, r, c2);
        ll less2 = n2.cnt, sum2 = n2.sm;
        ll right = -get((m + 1) + 1, (r) + 1) + sum2 + (ll)(r - m - less2) * c2;
        return left + right;
    }
};

struct PASCAL {
    vvll prefix;
    vpii coord;
    vvi pascal;
    int limit;
    PASCAL(int _N) : limit(_N), coord(_N) {
        int cnt = 1;
        for(int r = 0; cnt < limit; r++) {
            pascal.pb(vi(r + 1));
            auto& curr = pascal.back();
            for(int i = 0; i < (int)curr.size(); i++) {
                curr[i] = cnt++;
                coord[curr[i]] = MP(r, i);
                if(cnt == limit) break;
            }
            const int N = curr.size();
            prefix.pb(vll(N + 1));
            auto& p = prefix.back();
            for(int i = 1; i <= N; i++) {
                p[i] = p[i - 1] + (ll)curr[i - 1] * curr[i - 1];
            }
        }
    }
};

template <typename T>
struct StaticRangeInversion {
    // https://judge.yosupo.jp/submission/28427
    const int N, bs, nb_bc;
    vi vals, sufG, preH;
    vpii vals_sorted;
    vvi presuf;
    vvll R;

    StaticRangeInversion(const vector<T> &sequence) : N(sequence.size()), bs(ceil(sqrt(std::max(N, 1)))), nb_bc((N + bs - 1) / bs) {
        vector<T> dict = sequence;
        srtU(dict);
        const int D = dict.size();
        vals.reserve(N), vals_sorted.reserve(N);
        for (auto x : sequence) {
            vals.emplace_back(lb(all(dict), x) - dict.begin());
            vals_sorted.emplace_back(vals.back(), int(vals.size()) - 1);
        }

        presuf.assign(nb_bc, vi(N));
        sufG.resize(N), preH.resize(N);

        for(int ibc = 0; ibc < nb_bc; ibc++) {
            const int L = ibc * bs, R = std::min(L + bs, N);
            sort(vals_sorted.begin() + L, vals_sorted.begin() + R);
            vi cnt(D + 1);
            for(int i = L; i < R; i++) {
                cnt[vals[i] + 1]++;
            }
            for(int i = 0; i < D; i++) {
                cnt[i + 1] += cnt[i];
            }
            for(int b = 0; b < ibc; b++) {
                for(int i = (b + 1) * bs - 1; i >= b * bs; i--) {
                    presuf[ibc][i] = presuf[ibc][i + 1] + cnt[vals[i]];
                }
            }
            for(int b = ibc + 1; b < bs; b++) {
                for(int i = b * bs; i < min((b + 1) * bs, N); i++) {
                    presuf[ibc][i] = (i == b * bs ? 0 : presuf[ibc][i - 1]) + cnt.back() - cnt[vals[i] + 1];
                }
            }
            for(int i = L + 1; i < R; i++) {
                preH[i] = preH[i - 1] + count_if(vals.begin() + L, vals.begin() + i, [&](int x) { return x > vals[i]; });
            }
            for(int i = R - 2; i >= L; i--) {
                sufG[i] = sufG[i + 1] + count_if(vals.begin() + i + 1, vals.begin() + R, [&](int x) { return x < vals[i]; });
            }
        }

        R.resize(nb_bc, vll(nb_bc));
        for(int i = nb_bc - 1; i >= 0; i--) {
            R[i][i] = sufG[i * bs];
            for(int j = i + 1; j < nb_bc; j++) {
                R[i][j] = R[i][j - 1] + R[i + 1][j] - R[i + 1][j - 1] + presuf[j][i * bs];
            }
        }
    }

    ll query(int l, int r) const {
        r++;
        const int lb = (l + bs - 1) / bs, rb = (r == N ? nb_bc : r / bs) - 1;
        ll ret = 0;
        if(l / bs == (r - 1) / bs) {
            const int b = l / bs;
            ret += preH[r - 1] - (l % bs ? preH[l - 1] : 0);
            int less_cnt = 0;
            for(int p = b * bs, q = min((b + 1) * bs, N); p < q; p++) {
                less_cnt += (vals_sorted[p].ss >= l && vals_sorted[p].ss < r);
                ret -= less_cnt * (vals_sorted[p].ss < l);
            }
            return ret;
        }
        ret += R[lb][rb];
        if(bs * lb > l) {
            ret += sufG[l];
            for(int b = lb; b <= rb; b++) {
                ret += presuf[b][l];
            }
        }
        if(bs * (rb + 1) < r) {
            ret += preH[r - 1];
            for(int b = lb; b <= rb; b++) {
                ret += presuf[b][r - 1];
            }
        }
        int less_cnt = 0, j = (rb + 1) * bs;
        for(int p = max(0, (lb - 1) * bs), q = lb * bs; p < q; p++) {
            if(vals_sorted[p].ss >= l) {
                while(j < min(N, (rb + 2) * bs) && (vals_sorted[j].ss >= r || vals_sorted[j].ff < vals_sorted[p].ff)) {
                    less_cnt += (vals_sorted[j].ss < r), j++;
                }
                ret += less_cnt;
            }
        }
        return ret;
    }
};

struct static_range_interval_query {
    // https://codeforces.com/group/o09Gu2FpOx/contest/541484/problem/T
    int n;
    vi far, far_block, cnt, next;
    int B;
    static_range_interval_query(int _n) : B(sqrt(_n)), n(_n) { 
        far.resize(n, -inf);
        cnt.resize(n, -inf);
        next.resize(n, -inf);
        far_block.resize(n / B + 1, -inf);
    }

    int start_id(int i) {
        return (i / B) * B;
    }

    int end_id(int i) {
        return min(((i / B) + 1) * B - 1, n - 1);
    }

    int id(int i) {
        return i / B;
    }

    void rebuild(int block) {
        int st = block * B;
        int en = min(n - 1, (block + 1) * B - 1);
        int mx = far_block[block];
        for(int i = st; i <= en; i++) {
            mx = max(mx, far[i]);
            far[i] = mx;
        }
        for(int i = en; i >= st; i--) {
            int j = far[i] + 1;
            if(j <= i) {
                next[i] = -inf;
                cnt[i] = 0;
                continue;
            }
            if(j > en) {
                next[i] = j;
                cnt[i] = 1;
            } else {
                next[i] = next[j];
                cnt[i] = cnt[j] + 1;
            }
        }
    }

    void insert(int l, int r) {
        if(far[l] >= r || far_block[id(l)] >= r) return;
        int bl = id(l), br = id(r);
        if(bl == br) {
            for(int i = l; i <= r; i++) {
                far[i] = max(far[i], r);
            }
            rebuild(bl);
            return;
        }
        for(int i = l; i <= end_id(l); i++) {
            far[i] = max(far[i], r);
        }
        for(int i = bl + 1; i < br; i++) {
            far_block[i] = max(far_block[i], r);
        }
        for(int i = start_id(r); i <= r; i++) {
            far[i] = max(far[i], r);
        }
        rebuild(bl);
        rebuild(br);
    }

    int query(int l, int r) {
        int res = 0;
        while(l <= r) {
            int reach = max(far[l], far_block[id(l)]) + 1;
            if(reach > end_id(l)) {
                l = reach;
                res++;
            } else if(next[l] > l && next[l] < r) {
                res += cnt[l];
                l = next[l];
            } else if(far[l] + 1 > l) {
                res++;
                l = far[l] + 1;
            } else {
                return -1;
            }
        }
        return res;
    }
};

const static int BIT = 20;
const static int half = 10;
bitset<1 << BIT> s, have;
int ans[1 << BIT], last[1 << BIT];
int timer = 0;
struct max_and_set { // insert x into s, query max(x & a) where a in s
    void insert(int x) {
        if(s.test(x)) return;
        timer++;
        int high = x >> half;
        int low = x & ((1 << half) - 1);
        for(int sub = high;; sub = (sub - 1) & high) {
            have[sub] = true;
            s.set((sub << half) | low);
            if(sub == 0) break;
        }
    }

    int query(int x) {
        if(last[x] == timer) {
            return ans[x];
        }
        last[x] = timer;
        int mx = 0;
        int high = (x >> half);
        for(int mask = high; mask; mask = (mask - 1) & high) {
            if(have.test(mask)) {
                mx = max(mx, mask);
            }
        }
        int res = 0;
        for(int low = (1 << half) - 1; low >= 0; low--) {
            int now = (mx << half) | low;
            if(s.test(now)) {
                res = max(res, x & now);
            }
        }
        return ans[x] = res;
    }
};

template<typename T>
struct sweep_2d {
    int n, m;
    vector<vector<T>> arr;
    sweep_2d(int n, int m) : n(n), m(m) {
        arr.assign(n + 2, vector<T>(m + 2));
    }
    void update_rect(int r1, int c1, int r2, int c2, T delta) {
        if(r1 > r2 || c1 > c2) return;
        arr[r1][c1] += delta;
        arr[r1][c2 + 1] -= delta;
        arr[r2 + 1][c1] -= delta;
        arr[r2 + 1][c2 + 1] += delta;
    }
    void finalize() {
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++) {
                arr[i][j] += arr[i - 1][j] + arr[i][j - 1] - arr[i - 1][j - 1];
            }
        }
    }
};

template<typename T>
struct leftist_tree { // 1 base index
    struct Node {
        int ls, rs, td, fa; 
        T vl;
        Node(int _ls = 0, int _rs = 0, T _vl = 0, int _td = -1, int _fa = 0) : ls(_ls), rs(_rs), vl(_vl), td(_td), fa(_fa) {}
    };

    vi t;
    vector<Node> F;
    int n, src, dest;
    leftist_tree(int _n, int _src, int _dest) : t(_n + 1), n(_n), src(_src), dest(_dest) { F.emplace_back(); }

    int new_node(int ff = 0, T v = T()) {
        F.pb(Node(0, 0, v, 0, ff));
        return int(F.size()) - 1;
    }

    int merge(int a, int b) {
        if(!a || !b) return a ^ b;
        if(F[a].vl > F[b].vl) swap(a, b);
        int node = new_node();
        F[node] = F[a];
        F[node].rs = merge(F[node].rs, b);
        if(F[F[node].ls].td < F[F[node].rs].td) swap(F[node].ls, F[node].rs);
        F[node].td = F[node].rs ? F[F[node].rs].td + 1 : 0;
        return node;
    }

    void insert(int rt, int ff, T v) {
        t[rt] = merge(t[rt], new_node(ff, v));
    }

    void merge_two_tree(int a, int b) {
        t[a] = merge(t[a], t[b]);
    }

    using info = pair<T, int>;
    vector<T> dp;
    vpii par;
    void preprocess(const vector<vector<tuple<int, int, T>>>& f, const vector<vector<tuple<int, int, T>>>& g) { // f is reverse graph, g is normal graph, [nei, id, T], 1 base index
        dp.resize(n + 1, INF);
        par.resize(n + 1);
        min_heap<info> q;
        auto process = [&](int node, T cost, int id, int p) -> void {
            if(dp[node] > cost) {
                dp[node] = cost;
                par[node] = {id, p};
                q.push({cost, node});
            }
        };
        process(dest, 0, 0, 0);
        while(!q.empty()) {
            auto [cost, node] = q.top(); q.pop();
            if(dp[node] != cost) continue;
            for(auto& [nei, id, w] : f[node]) {
                process(nei, cost + w, id, node);
            }
        }
        vi id(n);
        iota(all(id), 1);
        sort(all(id), [&](int i, int j) {return dp[i] < dp[j];});
        for(auto& x : id) {
            if(dp[x] >= INF) continue;
            for(auto& [nei, id, w] : g[x]) {
                if(dp[nei] >= INF) continue;
                if(id != par[x].ff) {
                    T delta = w + dp[nei] - dp[x];
                    insert(x, nei, delta);
                }
            }
            merge_two_tree(x, par[x].ss);
        }
    }

    vector<T> run_dijkstra(int k) {
        // https://www.luogu.com.cn/problem/P2483
        // https://judge.yosupo.jp/problem/k_shortest_walk
        vector<T> ans;
        if(dp[src] >= INF) return ans;
        ans.pb(dp[src]);
        min_heap<info> q;
        if(t[src]) q.push({F[t[src]].vl, t[src]});
        while(!q.empty() && ans.size() < k) {
            auto [cost, node] = q.top(); q.pop();
            ans.pb(cost + dp[src]);
            if(F[node].ls) q.push({cost - F[node].vl + F[F[node].ls].vl, F[node].ls});
            if(F[node].rs) q.push({cost - F[node].vl + F[F[node].rs].vl, F[node].rs});
            int p = t[F[node].fa];
            if(p) q.push({cost + F[p].vl, p});
        }
        return ans;
    }
};

template<typename T>
struct sweepline_update {
    int n;
    vector<T> a, diff, slope;
 
    sweepline_update(int n_) : n(n_) {
        a.assign(n, 0);
        diff.assign(n + 1, 0);
        slope.assign(n + 1, 0);
    }
 
    void update_add(int l, int r, T val) {
        if(l > r) return;
        diff[l] += val;
        if(r + 1 < n) diff[r + 1] -= val;
    }
 
    void update_arithmetic_prefix(int l, int r, T k) { // add k, k * 2, k * 3
        if(l > r) return;
        int sz = r - l + 1;
        diff[l] += k;
        slope[l] += k;
        if(r + 1 < n) {
            T last = (k * (sz + 1));
            diff[r + 1] = (diff[r + 1] - last);
            slope[r + 1] = (slope[r + 1] - k);
        }
    }
 
    void update_arithmetic_suffix(int l, int r, T k) { // add k, k * 2, k * 3 but on suffix
        if(l > r) return;
        int sz = r - l + 1;
        diff[l] += k * sz;
        slope[l] -= k;
        if(r + 1 < n) {
            diff[r + 1] -= k * sz;
            slope[r + 1] += k;
        }
    }
 
    void build() {
        T val = 0, s = 0;
        for(int i = 0; i < n; i++) {
            s += slope[i];
            val += diff[i];
            a[i] = val;
            val += s;
        }
    }
 
    vector<T> get() {
        build();
        return a;
    }
};

struct non_monotone_triples { // return the longest chain without monotone triples in [l, r]
    // https://codeforces.com/contest/1332/problem/G
    vi a;
    int n;
    var(3) a3;
    var(4) a4;
    non_monotone_triples(const vi& _a) : n(_a.size()), a(_a) {
        a3.assign(n, {-1, -1, -1});
        a4.assign(n, {-1, -1, -1, -1});
        build();
    }

    void build() {
        set<int> non_highs, non_lows, neither;
        vi highs, lows, in_highs(n), in_lows(n);
        for(int i = 0; i < n; i++) {
            if(i) {
                a4[i] = a4[i - 1];
                a3[i] = a3[i - 1];
            }
            while(!highs.empty() && a[highs.back()] < a[i]) {
                int j = highs.back(); highs.pop_back();
                non_highs.insert(j);
                in_highs[j] = false;
                if(!in_lows[j]) neither.insert(j);
            } 
            while(!lows.empty() && a[lows.back()] > a[i]) {
                int j = lows.back(); lows.pop_back();
                non_lows.insert(j);
                in_lows[j] = false;
                if(!in_highs[j]) neither.insert(j);
            }
            in_lows[i] = in_highs[i] = true;
            highs.pb(i);
            lows.pb(i);
            auto highs_it = lb(all(highs), i, [&](const int& x, const int& y) {return a[x] > a[y];});
            auto lows_it = lb(all(lows), i, [&](const int& x, const int& y) {return a[x] < a[y];});
            int last_high = highs_it == begin(highs) ? -1 : *prev(highs_it);
            int last_low = lows_it == begin(lows) ? -1 : *prev(lows_it);
            {
                auto it = neither.lb(min(last_high, last_low));
                if(it != begin(neither)) {
                    it--;
                    debug(lows, highs, neither, last_high, last_low, *it);
                    int mx = *lb(all(highs), *it);
                    int mn = *lb(all(lows), *it);
                    a4[i] = max(a4[i], {*it, min(mn, mx), max(mn, mx), i});
                }
            }
            {
                auto it = non_highs.lb(last_high);
                if(it != begin(non_highs)) {
                    it--;
                    int mx = *lb(all(highs), *it);
                    a3[i] = max(a3[i], {*it, mx, i});
                }
            }
            {
                auto it = non_lows.lb(last_low);
                if(it != begin(non_lows)) {
                    it--;
                    int mn = *lb(all(lows), *it);
                    a3[i] = max(a3[i], {*it, mn, i});
                }
            }
        }
    }

    vi query(int l, int r) {
        if(a4[r][0] >= l) {
            return {a4[r][0], a4[r][1], a4[r][2], a4[r][3]};
        }
        if(a3[r][0] >= l) {
            return {a3[r][0], a3[r][1], a3[r][2]};
        }
        return {};
    }
};

struct paint_grid {
    // https://codeforces.com/contest/1080/problem/C
    struct Paint { ll x1, y1, x2, y2; bool is_black; };

    int n, m;
    ll white, black;
    vector<Paint> paints;

    paint_grid(int _n, int _m) : n(_n), m(_m) {
        ll total = 1LL * n * m;
        white = (total + 1) / 2;
        black = total / 2;
    }

    static inline bool inter(ll x1, ll y1, ll x2, ll y2, ll a1, ll b1, ll a2, ll b2, ll &ix1, ll &iy1, ll &ix2, ll &iy2) {
        ix1 = max(x1, a1);
        iy1 = max(y1, b1);
        ix2 = min(x2, a2);
        iy2 = min(y2, b2);
        return ix1 <= ix2 && iy1 <= iy2;
    }

    static inline ll area(ll x1, ll y1, ll x2, ll y2) {
        return (x2 - x1 + 1) * 1LL * (y2 - y1 + 1);
    }

    static inline pll count_white_black_base(ll x1, ll y1, ll x2, ll y2) {
        ll h = y2 - y1 + 1;
        ll w = x2 - x1 + 1;
        ll a = h * w;
        ll whites = a / 2;
        ll blacks = a / 2;
        if(a & 1) {
            if((x1 + y1) & 1) blacks++;
            else whites++;
        }
        return {whites, blacks};
    }

    ll unique_area_from(int i, ll x1, ll y1, ll x2, ll y2, int last) {
        ll ix1, iy1, ix2, iy2;
        if(!inter(paints[i].x1, paints[i].y1, paints[i].x2, paints[i].y2, x1, y1, x2, y2, ix1, iy1, ix2, iy2)) return 0;
        ll res = area(ix1, iy1, ix2, iy2);
        for(int j = i + 1; j <= last; ++j)
            res -= unique_area_from(j, ix1, iy1, ix2, iy2, last);
        return res;
    }

    ll unique_base_white_from(int i, ll x1, ll y1, ll x2, ll y2, int last) {
        ll ix1, iy1, ix2, iy2;
        if(!inter(paints[i].x1, paints[i].y1, paints[i].x2, paints[i].y2, x1, y1, x2, y2, ix1, iy1, ix2, iy2)) return 0;
        auto [w0, b0] = count_white_black_base(ix1, iy1, ix2, iy2);
        ll res = w0;
        for(int j = i + 1; j <= last; ++j)
            res -= unique_base_white_from(j, ix1, iy1, ix2, iy2, last);
        return res;
    }

    void paint(ll x1, ll y1, ll x2, ll y2, bool is_black) {
        int last = (int)paints.size() - 1;

        auto [baseW, baseB] = count_white_black_base(x1, y1, x2, y2);
        ll white_in_rect = baseW;

        for(int i = 0; i <= last; ++i) {
            ll bw = unique_base_white_from(i, x1, y1, x2, y2, last);
            ll ua = unique_area_from(i, x1, y1, x2, y2, last);
            if(paints[i].is_black)
                white_in_rect -= bw;
            else
                white_in_rect += (ua - bw);
        }

        ll rect_area = area(x1, y1, x2, y2);
        if(is_black) {
            white -= white_in_rect;
            black += white_in_rect;
        } else {
            ll black_in_rect = rect_area - white_in_rect;
            white += black_in_rect;
            black -= black_in_rect;
        }

        paints.pb({x1, y1, x2, y2, is_black});
    }

    void paint_white(ll x1, ll y1, ll x2, ll y2) { paint(x1, y1, x2, y2, false); }
    void paint_black(ll x1, ll y1, ll x2, ll y2) { paint(x1, y1, x2, y2, true); }
};

const int B = 316;
const int K = 360;
int val[MX][K], cnt[K][MX], pre_val[K][K], label[K][MX], color[K][MX], N = 0;
int cnt_now[MX], cnt_block[K], tmp[B];
struct range_set_range_kth {
    // https://marisaoj.com/problem/99
    int n;
    vi a;
    range_set_range_kth(const vi& _a) : n((int)_a.size() - 1), a(_a) {
        // 1-index everything
        for(int i = 1, b = 1; i <= n; i += B, b++) {
            N = b;
            for(int j = LB(b); j <= RB(b); j++) {
                int x = a[j];
                color[b][x] = label[b][x] = x;
                cnt[b][x]++;
                val[x][b]++;
                pre_val[BB(x)][b]++;
            }
        }
        for(int b = 1; b <= N; b++) {
            for(int x = 1; x < MX; x++) {
                val[x][b] += val[x][b - 1];
            }
            for(int x = 1; x < K; x++) {
                pre_val[x][b] += pre_val[x][b - 1];
            }
        }
    }
    int LB(int b) { return (b - 1) * B + 1; };
    int RB(int b) { return min(n, b * B); };
    int BB(int i) { return (i - 1) / B + 1; };
    void reset(int b) {
        for(int i = LB(b); i <= RB(b); i++) {
            a[i] = label[b][a[i]];
        }
        for(int i = LB(b); i <= RB(b); i++) {
            color[b][a[i]] = label[b][a[i]] = a[i];
        }
    };
    void change_block(int l, int r, int x, int y) {
        int b = BB(l);
        if(!cnt[b][x]) return;
        reset(b);
        int c = 0;
        for(int i = l; i <= r; i++) {
            if(a[i] == x) {
                c++;
                a[i] = y;
            }
        }
        color[b][y] = label[b][y] = y;
        cnt[b][x] -= c;
        cnt[b][y] += c;
        val[x][b] -= c;
        val[y][b] += c;
        pre_val[BB(x)][b] -= c;
        pre_val[BB(y)][b] += c;
    };
 
    int brute(int l, int r, int k) {
        int m = 0;
        for(int i = l; i <= r; i++) tmp[m++] = a[i];
        nth_element(tmp, tmp + (k - 1), tmp + m);
        return tmp[k - 1];
    };
    int query(int l, int r, int k) {
        int bl = BB(l);
        int br = BB(r);
        if(bl == br) {
            reset(bl);
            return brute(l, r, k);
        }
        reset(bl);
        reset(br);
        for(int i = l; i <= RB(bl); i++) {
            int x = a[i];
            cnt_now[x]++;
            cnt_block[BB(x)]++;
        }
        for(int i = LB(br); i <= r; i++) {
            int x = a[i];
            cnt_now[x]++;
            cnt_block[BB(x)]++;
        }
        int b = 1;
        int res = 0;
        while(b <= N && k > (pre_val[b][br - 1] + cnt_block[b]) - pre_val[b][bl]) {
            k -= (pre_val[b][br - 1] + cnt_block[b]) - pre_val[b][bl];
            b++;
        }
        assert(B <= N);
        for(int i = LB(b);; i++) {
            assert(i < LB(b) + B);
            int C = (val[i][br - 1] + cnt_now[i]) - val[i][bl];
            if(k <= C) {
                res = i;
                break;
            }
            k -= C;
        }
        for(int i = l; i <= RB(bl); i++) {
            int x = a[i];
            cnt_now[x] = 0;
            cnt_block[BB(x)] = 0;
        }
        for(int i = LB(br); i <= r; i++) {
            int x = a[i];
            cnt_now[x] = 0;
            cnt_block[BB(x)] = 0;
        }
        return res;
    };
 
    void update(int l, int r, int x, int y) {
        int bl = BB(l), br = BB(r);
        if(x == y || val[x][br] - val[x][bl - 1] == 0) return;
        for(int i = N; i > bl; i--) {
            val[x][i] -= val[x][i - 1];
            pre_val[BB(x)][i] -= pre_val[BB(x)][i - 1];
            val[y][i] -= val[y][i - 1];
            pre_val[BB(y)][i] -= pre_val[BB(y)][i - 1];
        }
        if(bl == br) {
            change_block(l, r, x, y);
        } else {
            change_block(l, RB(bl), x, y); 
            change_block(LB(br), r, x, y);
            for(int i = bl + 1; i < br; i++) {
                if(!cnt[i][x]) continue;
                if(cnt[i][y] == 0) {
                    swap(color[i][x], color[i][y]);
                    label[i][color[i][y]] = y;
                    int c = cnt[i][x];
                    cnt[i][y] += c;
                    cnt[i][x] = 0;
                    pre_val[BB(x)][i] -= c;
                    pre_val[BB(y)][i] += c;
                    val[x][i] -= c;
                    val[y][i] += c;
                    continue;
                }
                change_block(LB(i), RB(i), x, y);
            }
        }
        for(int i = bl + 1; i <= N; i++) {
            val[x][i] += val[x][i - 1];
            pre_val[BB(x)][i] += pre_val[BB(x)][i - 1];
            val[y][i] += val[y][i - 1];
            pre_val[BB(y)][i] += pre_val[BB(y)][i - 1];
        }
    };

	int get_val(int id) {
        int b = BB(id);   
        return label[b][a[id]];
    }
};

struct FastSet {
    static constexpr unsigned B = 64;
    int n;
    int levels;
    vector<vector<ull>> seg;

    FastSet() : n(0), levels(0) {}

    FastSet(int n_) { build(n_); }

    int size() const { return n; }

    void build(int m) {
        seg.clear();
        n = m;
        int cur = m;
        while(true) {
            int len = (cur + B - 1) / B;
            seg.pb(vector<ull>(len, 0));
            if(len == 1) break;
            cur = len;
        }
        levels = (int)seg.size();
    }

    template <class F>
    void build(int m, F f) {
        build(m);
        for(int i = 0; i < m; i++) {
            if(f(i)) {
                seg[0][i / B] |= (1ULL << (i % B));
            }
        }
        for(int h = 0; h + 1 < levels; h++) {
            for(int i = 0; i < (int)seg[h].size(); i++) {
                if(seg[h][i]) {
                    seg[h + 1][i / B] |= (1ULL << (i % B));
                }
            }
        }
    }

    bool operator[](int i) const {
        if(i < 0 || i >= n) return false;
        return (seg[0][i / B] >> (i % B)) & 1ULL;
    }

    void insert(int i) {
        if(i < 0 || i >= n) return;
        for(int h = 0; h < levels; h++) {
            seg[h][i / B] |= (1ULL << (i % B));
            i /= B;
        }
    }

    void erase(int i) {
        if(i < 0 || i >= n) return;
        unsigned long long carry = 0;
        for(int h = 0; h < levels; h++) {
            int idx = i / B;
            ull &v = seg[h][idx];
            v &= ~(1ULL << (i % B));
            v |= (carry << (i % B));
            carry = (v != 0);
            i = idx;
        }
    }

    // first index x >= i with bit 1, or n if none
    int next(int i) const {
        if(i < 0) i = 0;
        if(i >= n) return n;
        int pos = i;
        for(int h = 0; h < levels; h++) {
            int idx = pos / B;
            if(idx >= (int)seg[h].size()) break;
            ull v = seg[h][idx];
            v >>= (pos % B);
            if(!v) {
                idx++;
                while(idx < (int)seg[h].size() && seg[h][idx] == 0) {
                    idx++;
                }
                if(idx == (int)seg[h].size()) return n;
                pos = idx * B;
                v = seg[h][idx];
            }
            int offset = __builtin_ctzll(v);
            pos += offset;
            for(int g = h - 1; g >= 0; g--) {
                int lowerIdx = pos;
                pos = lowerIdx * B;
                ull vv = seg[g][lowerIdx];
                int off2 = __builtin_ctzll(vv);
                pos += off2;
            }
            return pos;
        }
        return n;
    }

    // last index x <= i with bit 1, or -1 if none
    int prev(int i) const {
        if(i < 0) return -1;
        if(i >= n) i = n - 1;
        int pos = i;
        for(int h = 0; h < levels; h++) {
            if(pos < 0) break;
            int idx = pos / B;
            if(idx >= (int)seg[h].size()) {
                idx = (int)seg[h].size() - 1;
                pos = idx * B + (B - 1);
            }
            ull v = seg[h][idx];
            unsigned shift = (unsigned)(pos % B);
            ull mask = ~0ULL >> (63 - shift);
            v &= mask;
            while(!v) {
                idx--;
                if(idx < 0) return -1;
                v = seg[h][idx];
                pos = idx * B + (B - 1);
            }
            int offset = 63 - __builtin_clzll(v);
            pos = idx * B + offset;
            for(int g = h - 1; g >= 0; g--) {
                int lowerIdx = pos;
                pos = lowerIdx * B + (B - 1);
                ull vv = seg[g][lowerIdx];
                int off2 = 63 - __builtin_clzll(vv);
                pos = lowerIdx * B + off2;
            }
            return pos;
        }
        return -1;
    }

    bool any(int l, int r) const {
        return next(l) < r;
    }

    template <class F>
    void enumerate(int l, int r, F f) const {
        if(l < 0) l = 0;
        if(r > n) r = n;
        int x = next(l);
        while(x < r) {
            f(x);
            x = next(x + 1);
        }
    }

    void clear() {
        enumerate(0, n, [&](int x) {erase(x);});
    }

    string to_string() const {
        string s(n, '0');
        for(int i = 0; i < n; i++) {
            if((*this)[i]) s[i] = '1';
        }
        return s;
    }
};

