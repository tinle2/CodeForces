struct Point {
    ld x, y;
    Point() : x(0), y(0) {}
    Point(ld x, ld y) : x(x), y(y) {}
    double distance(const Point &other) const {
        return std::sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    ld slope_to(const Point &other) const {
        if(fabs(other.x - x) < 1e-9) return std::numeric_limits<ld>::infinity();
        return (other.y - y) / (other.x - x);
    }

    // parallel if they share same {A, B}
    bool on_same_line(array<ll, 3> line) {
        ll A = line[0], B = line[1], C = line[2];
        return A * (ll)x + B * (ll)y + C == 0;
    }
};

Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y}; }
Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y}; }
Point operator*(ll t, Point b) { return {t * b.x, t * b.y}; }

ll operator*(Point a, Point b) { return a.x * b.x + a.y * b.y; } // dot product
ll operator%(Point a, Point b) { return a.x * b.y - a.y * b.x; } // cross product

bool operator<(Point a, Point b) { // lexicographical compare
	if (a.x != b.x) return a.x < b.x;
	return a.y < b.y;
}

ostream &operator<<(ostream &out, Point a) { // for debugging
	return out << "(" << a.x << "," << a.y << ")";
}

ll abs2(Point a) { return a * a; }
Point perp(Point a) { return {-a.y, a.x}; } // rotate 90 degrees counterclockwise

int ccw(Point a, Point b, Point c) { // returns 1|0|-1 if c is left|straight|right of ab
	ll res = (b-a) % (c-a);
	return res ? (res > 0 ? 1 : -1) : 0;
}

bool on_segment(Point p, Point a, Point b) {
	return (a - p) * (b - p) <= 0 && ccw(a, b, p) == 0;
}

string Point_in_polygon(const vector<Point> &a, Point p) {
	int n = a.size();
	if(ccw(a[0], a[1], p) < 0 || ccw(a[0], a[n - 1], p) > 0) return "OUT";
	if(ccw(a[0], a[1], p) == 0) return on_segment(p, a[0], a[1]) ? "ON" : "OUT";
	if(ccw(a[0], a[n - 1], p) == 0) return on_segment(p, a[0], a[n - 1]) ? "ON" : "OUT";

	int lo = 1, hi = n - 1;
	while(hi - lo > 1) {
		int md = (lo + hi)/2;
		(ccw(a[0], a[md], p) >= 0 ? lo : hi) = md;
	}
	assert(ccw(a[0], a[lo], p) >= 0);
	assert(ccw(a[0], a[hi], p) < 0);
	int s = ccw(a[lo], a[hi], p);
	return s == 1 ? "IN" : (s == 0 ? "ON" : "OUT");
}

// Normalize (A, B, C) so that gcd(A, B, C) = 1 and A >= 0 (if A == 0, B >= 0)
void normalize_abc(ll &A, ll &B, ll &C) {
    ll g = std::gcd(std::gcd(std::abs(A), std::abs(B)), std::abs(C));
    if(g > 0) {
        A /= g; B /= g; C /= g;
    }
    if(A < 0 || (A == 0 && B < 0)) {
        A = -A; B = -B; C = -C;
    }
}

// Compute line coefficients Ax + By + C = 0 from two points
std::array<ll, 3> get_slope_intercept(const Point &p1, const Point &p2) {
    ll A = p2.y - p1.y;
    ll B = p1.x - p2.x;
    ll C = -A * p1.x - B * p1.y;
    normalize_abc(A, B, C);
    return {A, B, C};
}

bool is_collinear(const Point &p1, const Point &p2, const Point &p3) {
    ld cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return std::fabs(cross) < eps;
}

ld triangle_area(const Point &p1, const Point &p2, const Point &p3) {
    ld cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return std::fabs(cross) / 2.0;
}

struct Circle {
    ld x, y, r;
    Circle(ld x = 0, ld y = 0, ld r = 0) : x(x), y(y), r(r) {}

    ld intersect(const Circle& o) const {
        // https://codeforces.com/contest/600/problem/D
        ld dx = x - o.x;
        ld dy = y - o.y;
        ld d  = sqrt(dx * dx + dy * dy);
        if(d >= r + o.r) return 0.0L;
        if(d <= fabsl(r - o.r)) {
            ld rr = min(r, o.r);
            return M_PI * rr * rr;
        }
        ld r2 = r * r, R2 = o.r * o.r, d2 = d * d;
        ld alpha = acosl((d2 + r2 - R2) / (2 * d * r)) * 2;
        ld beta = acosl((d2 + R2 - r2) / (2 * d * o.r)) * 2;
        ld area1 = 0.5L * r2 * (alpha - sinl(alpha));
        ld area2 = 0.5L * R2 * (beta - sinl(beta));
        return area1 + area2;
    }
};


struct Rectangle {
    ld x, y, w, h;
    Rectangle(ld x = 0, ld y = 0, ld w = 0, ld h = 0) : x(x), y(y), w(w), h(h) {}
};

bool circlesMergeWithinRect(const Circle &c1, const Circle &c2, const Rectangle &rect) { // check if the both circle intersect
                                                                                         // where the intersection is within the rectangle
    ld dx = c2.x - c1.x;
    ld dy = c2.y - c1.y;
    ld d = std::sqrt(dx * dx + dy * dy);
    if (d > c1.r + c2.r || d < fabsl(c1.r - c2.r)) return false;
    ld a = (c1.r * c1.r - c2.r * c2.r + d * d) / (2 * d);
    ld temp = c1.r * c1.r - a * a;
    if(temp < 0) temp = 0;
    ld h = std::sqrt(temp);
    ld px = c1.x + a * dx / d;
    ld py = c1.y + a * dy / d;
    if (fabs(h) < eps) {
        return (px >= rect.x && px <= rect.x + rect.w &&
                py >= rect.y && py <= rect.y + rect.h);
    } else {
        ld rx = -h * dy / d;
        ld ry = h * dx / d;
        ld ix1 = px + rx;
        ld iy1 = py + ry;
        ld ix2 = px - rx;
        ld iy2 = py - ry;
        bool inside1 = (ix1 >= rect.x && ix1 <= rect.x + rect.w &&
                        iy1 >= rect.y && iy1 <= rect.y + rect.h);
        bool inside2 = (ix2 >= rect.x && ix2 <= rect.x + rect.w &&
                        iy2 >= rect.y && iy2 <= rect.y + rect.h);
        return inside1 || inside2;
    }
}

bool circleRectangleIntersect(const Circle &c, const Rectangle &rect) { // check if circle intersect with the rectangle
    ld closestX = std::max(rect.x, std::min(c.x, rect.x + rect.w));
    ld closestY = std::max(rect.y, std::min(c.y, rect.y + rect.h));
    ld dx = c.x - closestX;
    ld dy = c.y - closestY;
    return (dx * dx + dy * dy) <= c.r * c.r;
}

bool circlesIntersect(const Circle &c1, const Circle &c2) { // check if two circle intersect
    ld dx = c1.x - c2.x;
    ld dy = c1.y - c2.y;
    ld distanceSq = dx * dx + dy * dy;
    ld radiusSum = c1.r + c2.r;
    return distanceSq <= radiusSum * radiusSum;
}

bool circleLineIntersect(const Circle &c, const Point &p1, const Point &p2) { // check if a circle intersect with a line
                                                                              // (x1, y1) to (x2, y2) inclusive
    long double cx = c.x, cy = c.y, cr = c.r;
    long double dx = p2.x - p1.x, dy = p2.y - p1.y;
    long double lenSq = dx * dx + dy * dy;
    if (lenSq == 0.0L) {
        long double distSq = (cx - p1.x) * (cx - p1.x) + (cy - p1.y) * (cy - p1.y);
        return distSq <= cr * cr;
    }
    long double t = ((cx - p1.x) * dx + (cy - p1.y) * dy) / lenSq;
    long double closestX, closestY;
    if (t < 0.0L) {
        closestX = p1.x;
        closestY = p1.y;
    } else if (t > 1.0L) {
        closestX = p2.x;
        closestY = p2.y;
    } else {
        closestX = p1.x + t * dx;
        closestY = p1.y + t * dy;
    }
    long double distSq = (cx - closestX) * (cx - closestX) + (cy - closestY) * (cy - closestY);
    return distSq <= cr * cr;
}

pll getMidpointKey(const pll& p1, const pll& p2) { // return the key to determine if two line are parallel
    return {p1.ff + p2.ff, p1.ss + p2.ss};
}

bool areParallelByMidpoint(const pll& p1, const pll& p2, const pll& q1, const pll& q2) {
    return getMidpointKey(p1,p2) == getMidpointKey(q1,q2);
}

--------------------------------------------------------------------------------------------------------------
template <typename T>
struct Point {
    T x, y;
};

template <typename P>
static inline ll cross(const P& a, const P& b, const P& c) {
    ll x1 = (ll)b.x - a.x;
    ll y1 = (ll)b.y - a.y;
    ll x2 = (ll)c.x - a.x;
    ll y2 = (ll)c.y - a.y;
    return x1 * y2 - y1 * x2;
}

template <typename P>
static inline int sgn(ll v) {
    return (v > 0) - (v < 0);
}

template <typename P>
static inline bool on_segment(const P& a, const P& b, const P& p) {
    if(cross(a, b, p) != 0) return false;
    return min(a.x, b.x) <= p.x && p.x <= max(a.x, b.x) &&
           min(a.y, b.y) <= p.y && p.y <= max(a.y, b.y);
}

template <typename P>
static inline bool seg_intersect(const P& a, const P& b, const P& c, const P& d) {
    int o1 = sgn<P>(cross(a, b, c));
    int o2 = sgn<P>(cross(a, b, d));
    int o3 = sgn<P>(cross(c, d, a));
    int o4 = sgn<P>(cross(c, d, b));
    if((o1 == 0 && on_segment(a, b, c)) ||
        (o2 == 0 && on_segment(a, b, d)) ||
        (o3 == 0 && on_segment(c, d, a)) ||
        (o4 == 0 && on_segment(c, d, b))) return true;
    return (o1 > 0) != (o2 > 0) && (o3 > 0) != (o4 > 0);
}

template <typename P>
static inline bool point_in_polygon_inclusive(const vector<P>& poly, const P& p) {
    const int n = (int)poly.size();
    for(int i = 0; i < n; ++i) {
        const P& a = poly[i];
        const P& b = poly[(i + 1) % n];
        if(on_segment(a, b, p)) return true;
    }
    bool inside = false;
    for(int i = 0; i < n; ++i) {
        P a = poly[i];
        P b = poly[(i + 1) % n];
        bool intersectY = ((a.y > p.y) != (b.y > p.y));
        if(intersectY) {
            double xint = a.x + (double)(b.x - a.x) * (double)(p.y - a.y) / (double)(b.y - a.y);
            if(xint > (double)p.x) inside = !inside;
        }
    }
    return inside;
}

template <typename P>
bool polygons_intersect(const vector<P>& A, const vector<P>& B) { // given 2 polygon, determine if they intersect
	// https://codeforces.com/contest/994/problem/C
    const int n = (int)A.size();
    const int m = (int)B.size();
    for(int i = 0; i < n; ++i) {
        P a1 = A[i];
        P a2 = A[(i + 1) % n];
        for(int j = 0; j < m; ++j) {
            P b1 = B[j];
            P b2 = B[(j + 1) % m];
            if(seg_intersect(a1, a2, b1, b2)) return true;
        }
    }
    if(point_in_polygon_inclusive(B, A[0])) return true;
    if(point_in_polygon_inclusive(A, B[0])) return true;
    return false;
}
--------------------------------------------------------------------------------------------------------------


struct manhattan {
    ll mx1 = -INF, mn1 = INF, mx2 = -INF, mn2 = INF;
    void update(ll x, ll y) {
        mx1 = max(mx1, x + y);
        mn1 = min(mn1, x + y);
        mx2 = max(mx2, x - y);
        mn2 = min(mn2, x - y);
    }

    ll query() { return max(mx1 - mn1, mx2 - mn2); }
};

struct Rect { ll x1, y1, x2, y2; };

struct SheetManager {
    // https://codeforces.com/contest/1216/problem/C
    vector<Rect> placed;
    bool insert(const Rect &r) {
        vector<Rect> clip;
        for (auto &q : placed) {
            Rect c{max(r.x1, q.x1), max(r.y1, q.y1), min(r.x2, q.x2), min(r.y2, q.y2)};
            if(c.x1 < c.x2 && c.y1 < c.y2) clip.push_back(c);
        }
        if(clip.empty()) {
            placed.pb(r);
            return true;
        }
        vll ys;
        ys.reserve(clip.size() * 2 + 2);
        ys.pb(r.y1);
        ys.pb(r.y2);
        for(auto &c : clip) {
            ys.pb(c.y1);
            ys.pb(c.y2);
        }
        srtU(ys);
        int Y = ys.size();
        struct Node { int cnt, len; };
        vector<Node> st(4 * Y);
        vector<tuple<int, int, int, int>> events;
        events.reserve(clip.size() * 2);
        for (auto &c : clip) {
            int y1 = lb(all(ys), c.y1) - ys.begin();
            int y2 = lb(all(ys), c.y2) - ys.begin();
            events.emplace_back(c.x1, y1, y2, 1);
            events.emplace_back(c.x2, y1, y2, -1);
        }
        sort(events.begin(), events.end(), [](auto &a, auto &b) { return get<0>(a) < get<0>(b); });
        auto update = [&](auto& update, int node, int l, int r, int ql, int qr, int v) {
            if(ql >= r || qr <= l) return;
            if(ql <= l && r <= qr) st[node].cnt += v;
            else {
                int m = (l + r) >> 1;
                update(update, node << 1, l, m, ql, qr, v);
                update(update, node << 1 | 1, m, r, ql, qr, v);
            }
            if(st[node].cnt > 0) st[node].len = ys[r] - ys[l];
            else if(r - l == 1) st[node].len = 0;
            else st[node].len = st[node << 1].len + st[node << 1 | 1].len;
        };
        ll covered = 0;
        int prevX = get<0>(events[0]);
        for(auto &e : events) {
            int x, y1, y2, tp;
            tie(x, y1, y2, tp) = e;
            covered += 1LL * (x - prevX) * st[1].len;
            update(update, 1, 0, Y - 1, y1, y2, tp);
            prevX = x;
        }
        ll total = 1LL * (r.x2 - r.x1) * (r.y2 - r.y1);
        if(covered < total) {
            placed.pb(r);
            return true;
        }
        return false;
    }
};

struct Line {
    mutable ll m, c, p;
    bool isQuery;
    bool operator<(const Line& o) const { if(o.isQuery) return p < o.p; return m < o.m; }
};

struct CHT : multiset<Line> { // cht max, for min just inverse the sign
    mutable iterator best;
    mutable bool init = false;

    ll div(ll a, ll b) { return a / b - ((a ^ b) < 0 && a % b); }

    bool isect(iterator x, iterator y) {
        if (y == end()) { x->p = INF; return false; }
        if (x->m == y->m) x->p = x->c > y->c ? INF : -INF;
        else x->p = div(y->c - x->c, x->m - y->m);
        return x->p >= y->p;
    }
	
//    inline ld div(const Line& X, const Line& Y) { // for ld comparison
//        return (Y.c - X.c) / (X.m - Y.m);
//    }
//
//    bool isect(iterator x, iterator y) {
//        if(y == end()) {
//            x->p = INF;
//            return false;
//        }
//        if(x->m == y->m) {
//            x->p = x->c > y->c ? INF : -INF;
//        } else {
//            x->p = div(*x, *y); 
//        }
//        return x->p >= y->p;
//    }

    void add(ll m, ll c) {
        auto z = insert({m, c, 0, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p) isect(x, erase(y));
    }

    ll query(ll x) {
        if(empty()) return -INF;
        Line q; q.p = x, q.isQuery = 1;
        auto l = *lower_bound(q);
        return l.m * x + l.c;
    }

    ll linear_query(ll x) {
        if(empty()) return -INF;
        if(!init) { best = begin(); init = true; }
        while(next(best) != end() && next(best)->m * x + next(best)->c >= best->m * x + best->c) best++;
        return best->m * x + best->c;
    }
};

class CHT_segtree { // max cht
public: 
    int n, base;
    vector<CHT> tree;

    CHT_segtree(int n) : n(n) {
        base = 1;
        while(base < n) base <<= 1;
        tree.resize(base << 1);
    }
    
    void update_at(int id, pll val) {  
        if(id >= n) return;
        int pos = id + base;
        while(pos > 0) {
            tree[pos].add(val.ff, val.ss);
            pos >>= 1;
        }
    }

    ll queries_at(int id, ll x) {
        if(id < 0 || id >= n) return -INF;
        ll ans = -INF;
        int pos = id + base;
        while(pos > 0) {
            ans = max(ans, tree[pos].query(x));
            pos >>= 1;
        }
        return ans;
    }

    void update_range(int l, int r, pll val) { // be careful it doesn't add downward, so only use this when you queries_at
        if(l < 0) l = 0;
        if(r >= n) r = n - 1;
        if(l > r) return;
        int L = l + base, R = r + base;
        while(L <= R) {
            if(L & 1) tree[L++].add(val.ff, val.ss);
            if(!(R & 1)) tree[R--].add(val.ff, val.ss);
            L >>= 1; R >>= 1;
        }
    }

    ll queries_range(int l, int r, ll x) { 
        if(l < 0 || r >= n) return -INF;
        ll ans = -INF;
        l += base, r += base;
        while(l <= r) {
            if(l & 1) ans = max(ans, tree[l++].query(x));
            if(!(r & 1)) ans = max(ans, tree[r--].query(x)); 
//            if (l & 1) ans = max(ans, tree[l++].linear_query(x));
//            if (!(r & 1)) ans = max(ans, tree[r--].linear_query(x)); 
            l >>= 1, r >>= 1;
        }
        return ans;
    }
};

struct Undo_CHT { // ll version
    struct Line {
        ll a, b;   // y = a*x + b
    };
    struct UndoEntry {
        Line prev;
        ll pos, old_sz;
    };

    vector<Line> A;
    vector<UndoEntry> undo;
    ll sz = 0;

    Undo_CHT(int max_n) {
        A.resize(max_n);
        undo.reserve(max_n);
    }

    static bool bad(const Line &l1, const Line &l2, const Line &l3) {
        i128 lhs = i128(l2.b - l1.b) * (l2.a - l3.a);
        i128 rhs = i128(l3.b - l2.b) * (l1.a - l2.a);
        return lhs >= rhs;
    }

    void add(ll u, ll v) {
        Line x = {u, v};
        ll l = 1, r = sz - 1, ans = sz;
        while(l <= r) {
            ll mid = (l + r) / 2;
            if(bad(A[mid - 1], A[mid], x)) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        undo.pb({A[ans], ans, sz});
        A[ans] = x;
        sz = ans + 1;
    }

    void roll_back() {
        auto e = undo.back(); undo.pop_back();
        sz = e.old_sz;
        A[e.pos] = e.prev;
    }

    ll cal(const Line &L, ll x) const {
        return L.a * x + L.b;
    }

    ll query(ll x) const {
        if(sz == 0) return -INF;
        ll ans = cal(A[0], x);
        ll l = 1, r = sz - 1;
        while(l <= r) {
            ll mid = (l + r) / 2;
            ll y_mid = cal(A[mid], x);
            ll y_prev = cal(A[mid - 1], x);
            if(y_mid > y_prev) {
                ans = max(ans, y_mid);
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
};

struct Undo_CHT { // ld version
    struct Line {
        ld a, b;   // y = a*x + b
    };
    struct UndoEntry {
        Line prev;
        int pos;
        int old_sz;
    };

    vector<Line> A;
    vector<UndoEntry> undo;
    int sz = 0;

    Undo_CHT(int maxn) {
        A.resize(maxn);
        undo.reserve(maxn);
    }

    static bool bad(const Line &l1, const Line &l2, const Line &l3) {
        return (l2.b - l1.b) * (l2.a - l3.a) >= (l3.b - l2.b) * (l1.a - l2.a);
    }

    void add(ld u, ld v) {
        Line x{u, v};
        int l = 1, r = sz - 1, pos = sz;
        while(l <= r) {
            int mid = (l + r) >> 1;
            if(bad(A[mid - 1], A[mid], x)) {
                pos = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        undo.pb({A[pos], pos, sz});
        A[pos] = x;
        sz = pos + 1;
    }

    void roll_back() {
        auto e = undo.back(); undo.pop_back();
        sz = e.old_sz;
        A[e.pos] = e.prev;
    }

    ld cal(const Line &L, ld x) const {
        return L.a * x + L.b;
    }

    ld query(ld x) const {
        if (sz == 0) return -1e300; 
        ld res = cal(A[0], x);
        int l = 1, r = sz - 1;
        while(l <= r) {
            int mid = (l + r) >> 1;
            ld y1 = cal(A[mid - 1], x);
            ld y2 = cal(A[mid],   x);
            if(y2 > y1) {
                res = max(res, y2);
                l = mid + 1;
            } else {
                res = max(res, y1);
                r = mid - 1;
            }
        }
        return res;
    }
};

struct Line { ll m, b; ll eval(ll x) const { return m * x + b; } };
struct MonoCHT { // max cht for monotonic function(prefix sum with all positive, ...)
    deque<Line> dq;
    bool increasing_query;
    MonoCHT(bool _inc = true) : increasing_query(_inc) {}

    bool bad(const Line &L1, const Line &L2, const Line &L3) {
        auto L = (L3.b - L1.b) * (L1.m - L2.m);
        auto R = (L2.b - L1.b) * (L1.m - L3.m);
        return increasing_query ? L <= R : L >= R;
    }
    void add(Line L) {
        while(dq.size() >= 2 && bad(dq[dq.size() - 2], dq[dq.size() - 1], L)) dq.pop_back();
        dq.pb(L);
    }
    ll query(ll x) {
        while(dq.size() >= 2 && dq[0].eval(x) <= dq[1].eval(x)) dq.pop_front();
        return dq[0].eval(x);
    }
};

struct LiChaoSegtree {
    struct LiChaoMax {
        struct Line {
            ll m, b;
            Line(ll m = 0, ll b = -INF) : m(m), b(b) {}
            inline ll eval(ll x) const { return m * x + b; }
        };
        struct Node {
            Line ln;
            Node* l = nullptr;
            Node* r = nullptr;
            Node(Line ln = Line()) : ln(ln) {}
        };

        ll lo, hi;
        Node* root;

        LiChaoMax() : lo(0), hi(0), root(nullptr) {}
        LiChaoMax(ll lo, ll hi) : lo(lo), hi(hi), root(nullptr) {}

        void add_line(ll m, ll b) { add_line(root, lo, hi, Line(m, b)); }
        void add_segment(ll m, ll b, ll L, ll R) {
            if (R < lo || hi < L) return;
            L = max(L, lo);
            R = min(R, hi);
            if (L > R) return;
            add_segment(root, lo, hi, L, R, Line(m, b));
        }
        ll query(ll x) const { return query(root, lo, hi, x); }

        private:
        void add_line(Node*& p, ll l, ll r, Line nw) {
            if(!p) { p = new Node(nw); return; }
            ll mid = (l + r) >> 1;
            bool lef = nw.eval(l) > p->ln.eval(l);
            bool midb = nw.eval(mid) > p->ln.eval(mid);
            if(midb) swap(nw, p->ln);
            if(l == r) return;
            if(lef != midb) add_line(p->l, l, mid, nw);
            else add_line(p->r, mid + 1, r, nw);
        }
        void add_segment(Node*& p, ll l, ll r, ll ql, ll qr, Line nw) {
            if(qr < l || r < ql) return;
            if(ql <= l && r <= qr) { add_line(p, l, r, nw); return; }
            if(!p) p = new Node();
            ll mid = (l + r) >> 1;
            add_segment(p->l, l, mid, ql, qr, nw);
            add_segment(p->r, mid + 1, r, ql, qr, nw);
        }
        ll query(Node* p, ll l, ll r, ll x) const {
            if(!p) return -INF;
            ll res = p->ln.eval(x);
            if(l == r) return res;
            ll mid = (l + r) >> 1;
            if(x <= mid) return max(res, query(p->l, l, mid, x));
            return max(res, query(p->r, mid + 1, r, x));
        }
    };
    int n, base;
    ll xlo, xhi;
    vector<LiChaoMax> seg;

    LiChaoSegtree(int n, ll xlo, ll xhi) : n(n), xlo(xlo), xhi(xhi) {
        base = 1;
        while(base < n) base <<= 1;
        seg.assign(base << 1, LiChaoMax(xlo, xhi));
    }

    void update_at(int id, pll line) {
        if(id < 0 || id >= n) return;
        int p = id + base;
        while(p) {
            seg[p].add_line(line.first, line.second);
            p >>= 1;
        }
    }

    ll queries_at(int id, ll x) {
        if(id < 0 || id >= n) return -INF;
        ll ans = -INF;
        int p = id + base;
        while(p) {
            ans = max(ans, seg[p].query(x));
            p >>= 1;
        }
        return ans;
    }

    void update_range(int l, int r, pll line) {
        if(l < 0) l = 0;
        if(r >= n) r = n - 1;
        if(l > r) return;
        int L = l + base, R = r + base;
        while(L <= R) {
            if (L & 1) seg[L++].add_line(line.first, line.second);
            if (!(R & 1)) seg[R--].add_line(line.first, line.second);
            L >>= 1;
            R >>= 1;
        }
    }

    ll queries_range(int l, int r, ll x) {
        if(l < 0) l = 0;
        if(r >= n) r = n - 1;
        if(l > r) return -INF;
        ll ans = -INF / 4;
        int L = l + base, R = r + base;
        while(L <= R) {
            if (L & 1) ans = max(ans, seg[L++].query(x));
            if (!(R & 1)) ans = max(ans, seg[R--].query(x));
            L >>= 1;
            R >>= 1;
        }
        return ans;
    }

    void update_at_segmentX(int id, pll line, ll Lx, ll Rx) {
        if(id < 0 || id >= n) return;
        Lx = max(Lx, xlo);
        Rx = min(Rx, xhi);
        if(Lx > Rx) return;
        int p = id + base;
        while(p) {
            seg[p].add_segment(line.first, line.second, Lx, Rx);
            p >>= 1;
        }
    }

    void update_range_segmentX(int l, int r, pll line, ll Lx, ll Rx) {
        if(l < 0) l = 0;
        if(r >= n) r = n - 1;
        if(l > r) return;
        Lx = max(Lx, xlo);
        Rx = min(Rx, xhi);
        if (Lx > Rx) return;
        int L = l + base, R = r + base;
        while(L <= R) {
            if (L & 1) seg[L++].add_segment(line.first, line.second, Lx, Rx);
            if (!(R & 1)) seg[R--].add_segment(line.first, line.second, Lx, Rx);
            L >>= 1;
            R >>= 1;
        }
    }
};

struct Hull {
    vpll hull;
    set<pll> s;

    void insert(ll a, ll b) {
        auto it = s.lb({a, -INF});
        if(it != s.begin() && prev(it)->ss <= b) return;
        while(it != s.end() && it->ss >= b) it = s.erase(it);
        s.insert({a, b});
    }

    void convert() {
        hull = vpll(all(s));
    }

    // Query minimum value of f(a,b) = a*x + b*y
    ll query(ll x, ll y) {
        if(hull.empty()) return -INF;
        auto calc = [&](pll it) -> ll { return it.ff * x + it.ss * y; };

        int l = 0, r = int(hull.size()) - 2;
        ll res = calc(hull.back());
        while(l <= r) {
            int m = l + (r - l) / 2;
            auto X = calc(hull[m]);
            auto Y = calc(hull[m + 1]);
            res = min({res, X, Y});
            if(X < Y) r = m - 1;
            else l = m + 1;
        }
        return res;
    }

    void merge(const Hull &other) {
        int i = 0, j = 0;
        vpll newHull;
        while(i < (int)hull.size() || j < (int)other.hull.size()) {
            pll p;
            if(i == (int)hull.size()) {
                p = other.hull[j++];
            } else if(j == (int)other.hull.size()) {
                p = hull[i++];
            } else {
                if(hull[i].ff < other.hull[j].ff) {
                    p = hull[i++];
                } else {
                    p = other.hull[j++];
                }
            }
            while(!newHull.empty() && newHull.back().ss <= p.ss) {
                newHull.pop_back();
            }
            newHull.pb(p);
        }
        hull = move(newHull);
    }
};

