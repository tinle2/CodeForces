#include<bits/stdc++.h>
#if defined(LOCAL) && __has_include("debug.h")
#include "debug.h"
#else
#define debug(...)
#endif
using i64 = long long;
using u64 = unsigned long long;
using i128 = __int128;

const int HASH_COUNT = 2;
const int MX = 1e5 + 5;
std::vector<i64> base, mod;
i64 p[HASH_COUNT][MX], geom[HASH_COUNT][MX];
void initGlobalHashParams() {
    if (!base.empty() && !mod.empty()) return;
	std::vector<i64> candidateBases = {
        10007ULL, 10009ULL, 10037ULL, 10039ULL, 10061ULL, 10067ULL, 10069ULL, 10079ULL, 10091ULL, 10093ULL,
        10099ULL, 10103ULL, 10111ULL, 10133ULL, 10139ULL, 10141ULL, 10151ULL, 10159ULL, 10163ULL, 10169ULL,
        10177ULL, 10181ULL, 10193ULL, 10211ULL, 10223ULL, 10243ULL, 10247ULL, 10253ULL, 10259ULL, 10267ULL,
        10271ULL, 10273ULL, 10289ULL, 10301ULL, 10303ULL, 10313ULL, 10321ULL, 10331ULL, 10333ULL, 10337ULL,
        10343ULL, 10357ULL, 10369ULL, 10391ULL, 10399ULL, 10427ULL, 10429ULL, 10433ULL, 10453ULL, 10457ULL,
        10459ULL, 10463ULL, 10477ULL, 10487ULL, 10499ULL, 10501ULL, 10513ULL, 10529ULL, 10531ULL, 10559ULL,
        10567ULL, 10589ULL, 10597ULL, 10601ULL, 10607ULL, 10613ULL, 10627ULL, 10631ULL, 10639ULL, 10651ULL,
        10657ULL, 10663ULL, 10667ULL, 10687ULL, 10691ULL, 10709ULL, 10711ULL, 10723ULL, 10729ULL, 10733ULL,
        10739ULL, 10753ULL, 10771ULL, 10781ULL, 10789ULL, 10799ULL, 10831ULL, 10837ULL, 10847ULL, 10853ULL,
        10859ULL, 10861ULL, 10867ULL, 10883ULL, 10889ULL, 10891ULL, 10903ULL, 10909ULL, 10937ULL, 10939ULL
    };

    std::vector<i64> candidateMods = {
        1000000007ULL, 1000000009ULL, 1000000021ULL, 1000000033ULL, 1000000087ULL, 1000000093ULL, 1000000097ULL, 1000000103ULL, 1000000123ULL, 1000000181ULL,
        1000000207ULL, 1000000223ULL, 1000000241ULL, 1000000271ULL, 1000000289ULL, 1000000297ULL, 1000000321ULL, 1000000349ULL, 1000000363ULL, 1000000403ULL,
        1000000409ULL, 1000000411ULL, 1000000427ULL, 1000000433ULL, 1000000439ULL, 1000000447ULL, 1000000453ULL, 1000000459ULL, 1000000483ULL, 1000000513ULL,
        1000000531ULL, 1000000579ULL, 1000000607ULL, 1000000613ULL, 1000000637ULL, 1000000663ULL, 1000000711ULL, 1000000753ULL, 1000000787ULL, 1000000801ULL,
        1000000829ULL, 1000000891ULL, 1000000901ULL, 1000000931ULL, 1000000951ULL, 1000000993ULL, 1000001011ULL, 1000001021ULL, 1000001053ULL, 1000001097ULL,
        1000001143ULL, 1000001161ULL, 1000001163ULL, 1000001179ULL, 1000001193ULL, 1000001231ULL, 1000001263ULL, 1000001303ULL, 1000001311ULL, 1000001351ULL,
        1000001369ULL, 1000001431ULL, 1000001453ULL, 1000001503ULL, 1000001531ULL, 1000001593ULL, 1000001617ULL, 1000001649ULL, 1000001663ULL, 1000001703ULL,
        1000001753ULL, 1000001783ULL, 1000001801ULL, 1000001853ULL, 1000001871ULL, 1000001957ULL, 1000001963ULL, 1000001969ULL, 1000002003ULL, 1000002029ULL,
        1000002043ULL, 1000002051ULL, 1000002061ULL, 1000002083ULL, 1000002101ULL, 1000002133ULL, 1000002161ULL, 1000002179ULL, 1000002253ULL, 1000002271ULL
    };
								 
	unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    shuffle(begin(candidateBases), end(candidateBases), std::default_random_engine(seed));
    shuffle(begin(candidateMods), end(candidateMods), std::default_random_engine(seed + 1));

    base.resize(HASH_COUNT);
    mod.resize(HASH_COUNT);
    for(int i = 0; i < HASH_COUNT; i++) {
        mod[i] = candidateMods[i];
        base[i] = candidateBases[i];
    }
	auto modExpo = [](i64 base, i64 exp, i64 mod) -> i64 {
        i64 res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; 
    };
    for(int j = 0; j < HASH_COUNT; j++) {
        i64 inv = modExpo(base[j] - 1, mod[j] - 2, mod[j]);
        p[j][0] = 1;
        geom[j][0] = 0;
        for(int i = 1; i < MX; i++) {
            p[j][i] = (p[j][i - 1] * base[j]) % mod[j];
            i64 num = (p[j][i] + mod[j] - 1) % mod[j];
            geom[j][i] = num * inv % mod[j];
        }
    }
}

static const bool _hashParamsInitialized = [](){
    initGlobalHashParams();
    return true;
}();

template<class T = std::string>
struct RabinKarp {
    // careful with the + 1 for the 0 hash
    std::vector<i64> prefix[HASH_COUNT], suffix[HASH_COUNT];
    int n;
    std::vector<int> s;

    RabinKarp() : n(0) {
        for(int i = 0; i < HASH_COUNT; i++) {
            prefix[i].push_back(0);
            suffix[i].push_back(0);
        }
    }

    RabinKarp(const T &s) : s(s) {
        n = s.size();
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].resize(n + 1, 0);
            suffix[i].resize(n + 1, 0);
        }
        for (int j = 1; j <= n; j++) {
            int x = s[j - 1] - 'a' + 1;
            int y = s[n - j] - 'a' + 1;
            for (int i = 0; i < HASH_COUNT; i++) {
                prefix[i][j] = (prefix[i][j - 1] * base[i] + x) % mod[i];
                suffix[i][j] = (suffix[i][j - 1] * base[i] + y) % mod[i];
            }
        }
    }

    void insert(int x) {
        for (int i = 0; i < HASH_COUNT; i++) {
            i64 v = (prefix[i].back() * base[i] + x) % mod[i];
            prefix[i].push_back(v);
        }
        n++;
    }

    void pop_back() {
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].pop_back();
        }
        n--;
    }

    int size() {
        return n;
    }
    
    i64 get() {
        return get_hash(0, n);
    }

    i64 get_hash(int l, int r) const {
        if (l < 0 || r > n || l > r) return 0;
        i64 hash0 = prefix[0][r] - (prefix[0][l] * p[0][r - l] % mod[0]);
        hash0 = (hash0 % mod[0] + mod[0]) % mod[0];
        i64 hash1 = prefix[1][r] - (prefix[1][l] * p[1][r - l] % mod[1]);
        hash1 = (hash1 % mod[1] + mod[1]) % mod[1];
        return (hash0 << 32) | hash1;
    }

    i64 get_rev_hash(int l, int r) const {
        std::tie(l, r) = std::make_tuple(n - r, n - l);
        if(l < 0 || r > n || l >= r) return 0;
        i64 h0 = suffix[0][r] - (suffix[0][l] * p[0][r - l] % mod[0]);
        i64 h1 = suffix[1][r] - (suffix[1][l] * p[1][r - l] % mod[1]);
        if(h0 < 0) h0 += mod[0];
        if(h1 < 0) h1 += mod[1];
        return (h0 << 32) | h1;
    }

    bool is_palindrome(int l, int r) const {
        if(l > r) return true;
        return get_hash(l, r + 1) == get_rev_hash(l, r + 1);
    }
    
    bool diff_by_one_char(RabinKarp &a, int offSet = 0) {
        int left = 0, right = n, rightMost = -1;
        while (left <= right) {
            int middle = left + (right - left) / 2;
            if (a.get_hash(offSet, middle + offSet) == get_hash(0, middle)) {
                rightMost = middle;
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
        return a.get_hash(rightMost + 1 + offSet, offSet + n) == get_hash(rightMost + 1, n);
    }
    
    i64 combine_hash(std::pair<i64, i64> a, std::pair<i64, i64> b, int len) {
        a.first = ((a.first * p[0][len]) + b.first) % mod[0];
        a.second = ((a.second * p[1][len]) + b.second) % mod[1];
        return (a.first << 32) | a.second;
    }

    int cmp(const RabinKarp& other) { // -1 : less, 0 = equal, 1 = greater
        if(n == 0 && other.n == 0) return 0;
        int left = 0, right = std::min(n, other.n) - 1, res = -1;
        while(left <= right) {
            int middle = (left + right) >> 1;
            if(get_hash(0, middle + 1) != other.get_hash(0, middle + 1)) res = middle, right = middle - 1;
            else left = middle + 1;
        }
        if(res == -1) {
            return n == other.n ? 0 : (n < other.n ? -1 : 1);
        }
        return s[res] < other.s[res] ? -1 : 1;
    }
};

class Solution {
public:
    int smallestUniqueSubarray(std::vector<int>& a) {
        initGlobalHashParams();
        const int n = a.size();
        RabinKarp<std::vector<int>> rk(a);
        auto f = [&](int k) -> bool {
            std::map<i64, int> mp;
            for (int i = 0; i + k <= n; i++) {
                auto hash = rk.get_hash(i, i + k);
                mp[hash]++;
            }
            for (const auto& [_, c] : mp) {
                if (c == 1) {
                    return true;
                }
            }
            return false;
        };
        int left = 1, right = n, res = n;
        while (left <= right) {
            int middle = (left + right) >> 1;
            if (f(middle)) {
                res = middle;
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        }
        return res;
    }
};
#ifdef LOCAL
void solve() {
    int n;
    std::cin >> n;
    std::vector<int> a(n);
    for (auto& x : a) {
        std::cin >> x;
    }
    std::cout << Solution().smallestUniqueSubarray(a) << '\n';
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int t = 1;
    //std::cin >> t;
    while(t--) {
        solve();
    }
    
    return 0;
}
#endif
