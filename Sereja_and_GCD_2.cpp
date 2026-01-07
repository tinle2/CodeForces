#include<bits/stdc++.h>
#if defined(LOCAL) && __has_include("debug.h")
#include "debug.h"
#else
#define debug(...)
#endif
using i64 = long long;
using u64 = unsigned long long;
using i128 = __int128;

template <int MOD>
struct mod_int {
    int value;
    
    mod_int(i64 v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((i64)value * other.value % MOD); return *this; }
    mod_int pow(i64 p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
    mod_int inv() const { return pow(MOD - 2); }
    mod_int& operator/=(const mod_int &other) { return *this *= other.inv(); }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const { return mod_int((i64)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((i64)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((i64)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((i64)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((i64)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator i64() const { return (i64)value; }
    explicit operator int() const { return value; }
	explicit operator double() const { return (double)value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend std::ostream& operator<<(std::ostream &os, const mod_int &a) { os << a.value; return os; }
    friend std::istream& operator>>(std::istream &is, mod_int &a) { i64 v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<MOD>;
const int MX = 1e6 + 5;

template<class T> 
class Combinatoric {    
    public: 
    int N;  
    std::vector<T> fact, inv;   
    Combinatoric(int _N) : N(_N), fact(N + 1), inv(N + 1) {   
        init();
    }
        
    void init() {   
        fact[0] = 1;
        for(int i = 1; i <= N; i++) {   
            fact[i] = fact[i - 1] * i;
        }
        inv[N] = fact[N].inv();
        for(int i = N - 1; i >= 0; i--) {   
            inv[i] = inv[i + 1] * (i + 1);
        }
    }
    
    T nCk(int a, int b) {  
        if(a < b) {
            return 0;
        }
        assert(std::max(a, b) <= N);
        return fact[a] * inv[b] * inv[a - b];
    }

    T nPk(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }
        return fact[n] * inv[n - k];
    }

    u64 nCk_no_mod(i64 n, i64 r) {
		if(n < r) {
            return 0;
        }
		r = std::min(r, n - r);
		u64 ans = 1;
		for(int i = 1; i <= r ; i++) {
			u64 d = std::gcd(ans, i);
			ans /= d;
			ans *= (n - i + 1) / (i / d);
		}
		return ans ;
	}

    mint lucas(i64 n, i64 r) { // call on Combinatoric comb(MOD - 1) for small PRIME MOD, log(C)
        if(r > n) {
            return 0;
        }
        if(r == 0) {
            return 1;
        }
        int ni = n % MOD;
        int ri = r % MOD;
        return ri > ni ? 0 : lucas(n / MOD, r / MOD) * nCk(ni, ri);
    }

    i64 derangement(int n) {
        if(n == 0) return 1;
        if(n == 1) return 0;
        std::vector<i64> D(n + 1);
        D[0] = 1;
        D[1] = 0;
        for(int i = 2; i <= n; ++i) {
            D[i] = (i - 1) * (D[i - 1] + D[i - 2]);
        }
        return D[n];
    }
	
    T catalan(int k) { // # of pair of balanced bracket of length n is catalan(n / 2)
        if(k == 0) return 1;
        return nCk(2 * k, k) - nCk(2 * k, k - 1);
    }

	T monotonic_array_count(int n, int m) {// len n, element from 1 to m increasing/decreasing
        return nCk(n + m - 1, n);
    }

//    i64 nCk_mod_Lucas_Theorem(int n, int r, int mod) {
//        if(r > n) {
//            return 0;
//        }
//        i64 res = 1;
//        while(n && r) {
//            res *= nCk(n % mod, r % mod) ;
//            res %= mod ;
//            n /= mod ;
//            r /= mod ; 
//        }
//        return res ;
//    }
//
//    int nCk_lucas(int n, int r, int mod) {
//        std::vector<int> ans;
//        for(auto& x : DIV[mod]) {
//            ans.push_back(nCk_mod_Lucas_Theorem(n, r, x));
//        }
//        i64 res = 0;
//        for(int i = 0; i < int(DIV[mod].size()); i++) {
//            int p = DIV[mod][i];
//            i64 m = mod / p;
//            i64 inv = modExpo(m, p - 2, p);
//            res = (res + ans[i] * m % mod * inv) % mod;
//        }
//        return res;
//    }
}; 
Combinatoric<mint> comb(MX - 1);

void solve() {
    int n, m;
    std::cin >> n >> m;
    std::vector<int> spf(m + 1), mask(m + 1), primes;
    int S = 0, B = 0;
    for(int i = 2, P = 0; i <= m; i++) {
        if(spf[i] == 0) {
            for(int j = i; j <= m; j += i) {
                if(spf[j] == 0) {
                    spf[j] = i;
                }
            }
            if(i * 2 > m) {
                B++;
            } else {
                for(int j = i; j <= m; j += i) {
                    mask[j] |= 1 << S;
                }
                S++;
            }
            P++;
        }
    }
    std::vector dp(S + 1, std::vector<mint>(1 << S));
    dp[0][0] = 1;
    for(int i = 2; i <= m; i++) {
        if(mask[i] == 0) continue;
        for(int x = 0; x < 1 << S; x++) {
            for(int sz = 1; sz <= S; sz++) {
                if(mask[i] == 0 || (mask[i] & x)) {
                    continue;
                }
                dp[sz][x | mask[i]] += dp[sz - 1][x];
            }
        }
    }
    std::vector<mint> f(S + B + 1);
    for(int sz = 0; sz <= S; sz++) {
        mint ways = accumulate(begin(dp[sz]), end(dp[sz]), mint(0));
        for(int b = 0; b <= B; b++) {
            f[sz + b] += ways * comb.nCk(B, b);
        }
    }
    mint res = 0;
    for(int i = 0; i <= std::min(n + 1, S + B); i++) {
        res += f[i] * comb.nPk(n, i);
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
