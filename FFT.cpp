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
using mint = mod_int<998244353>;

typedef complex<double> cd;
template<class T>
struct FFT { // given 2 array a and b, compute the number of pair that make up (a[i] * b[j] for all j) in nlogn
    static void fft(vector<cd>& a, bool invert) {
        int n = a.size();
        for(int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for(; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for(int len = 2; len <= n; len <<= 1) {
            double angle = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(angle), sin(angle));
            for(int i = 0; i < n; i += len) {
                cd w(1);
                for(int j = 0; j < len / 2; j++) {
                    cd u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if(invert) {
            for (cd & x : a) x /= n;
        }
    }
 
    static vector<T> convolve(const vector<T>& a, const vector<T>& b) {
        int a_sz = (int)a.size();
        int b_sz = (int)b.size();
        int needed = a_sz + b_sz - 1;
        int N = 1;
        while (N < needed) N <<= 1;
        vector<cd> fa(N, cd(0,0)), fb(N, cd(0,0));
        for(int i = 0; i < (int)a.size(); i++) fa[i] = cd(double(a[i]), 0);
        for(int j = 0; j < (int)b.size(); j++) fb[j] = cd(double(b[j]), 0);
        fft(fa, false);
        fft(fb, false);
        for(int i = 0; i < N; i++) fa[i] *= fb[i];
        fft(fa, true);
        vector<T> res(N);
        for(int i = 0; i < N; i++)
            res[i] = (T)round(fa[i].real());
        return res;
    }

    static vector<T> power(vector<T> poly, int exp) {
        vector<T> res(1, 1);
        while(exp > 0) {
            if (exp & 1) {
                res = convolve(res, poly);
                for(auto &c : res) c = c ? 1 : 0;
            }
            exp >>= 1;
            if(exp) {
                poly = convolve(poly, poly);
                for(auto &c : poly) c = c ? 1 : 0;
            }
        }
        return res;
    }
};

// https://www.spoj.com/status/ns=34520575
template<int mod = MOD>
struct Polynomial {
    vmint a;
    Polynomial() {}
    Polynomial(const vmint& a) : a(a) { normalize(); }
    void normalize() {
        while (!a.empty() && a.back().value == 0) a.pop_back();
    }
    int size() const { return a.size(); }
    int deg() const { return a.size() - 1; }
    mint at(int i) const { return (i < 0 || i >= a.size()) ? mint(0) : a[i]; }
    static void fft(vmint& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            mint wlen = mint(3).pow((mod - 1) / len);
            if (invert) wlen = wlen.inv();
            for (int i = 0; i < n; i += len) {
                mint w = 1;
                for (int j = 0; j < len / 2; j++) {
                    mint u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            mint inv_n = mint(n).inv();
            for (mint &x : a)
                x *= inv_n;
        }
    }
    static vmint multiply(const vmint& a, const vmint& b) {
        auto fa(a), fb(b);
        int n = 1;
        while (n < (int)a.size() + (int)b.size() - 1) n <<= 1;
        fa.rsz(n);
        fb.rsz(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++) fa[i] *= fb[i];
        fft(fa, true);
        vmint res(a.size() + b.size() - 1);
        for (int i = 0; i < res.size(); i++) res[i] = fa[i];
        return res;
    }
    Polynomial operator*(const Polynomial &other) const {
        vmint res = multiply(a, other.a);
        return Polynomial(res);
    }
};

struct BigCalculator {
    string num1;
    string num2;
    BigCalculator(const string n1, const string n2) : num1(n1), num2(n2) {}
    string add() { return bigAdd(num1, num2); }
    string mod() { return bigMod(num1, num2); }
    string minus() { return bigSub(num1, num2); }
    string multiply() { return bigMul(num1, num2); }
    string OR() { return bigBitOr(num1, num2); }
    string AND() { return bigBitAnd(num1, num2); }
    string XOR() { return bigBitXor(num1, num2); }
    string divide() { return bigDiv(num1, num2); }
    string operator|(const BigCalculator &other) const { return binToDec(bigBitOr(num1, other.num1)); }
    string operator&(const BigCalculator &other) const { return binToDec(bigBitAnd(num1, other.num1)); }
    string operator^(const BigCalculator &other) const { return binToDec(bigBitXor(num1, other.num1)); }
private:

    static string bigAdd(const string &a, const string &b) {
        int i = a.size() - 1, j = b.size() - 1, carry = 0;
        string result;
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) { sum += a[i] - '0'; i--; }
            if (j >= 0) { sum += b[j] - '0'; j--; }
            carry = sum / 10;
            result.pb('0' + sum % 10);
        }
        rev(result);
        return result;
    }
    static int compare(const string &a, const string &b) {
        if (a.size() != b.size())
            return (a.size() < b.size() ? -1 : 1);
        return a.compare(b);
    }
    static string bigSub(const string &a, const string &b) {
        if (compare(a, b) == 0) return "0";
        bool negative = false;
        string A = a, B = b;
        if (compare(a, b) < 0) { negative = true; swap(A, B); }
        int i = A.size() - 1, j = B.size() - 1, carry = 0;
        string result;
        while (i >= 0) {
            int diff = (A[i] - '0') - carry;
            if (j >= 0) { diff -= (B[j] - '0'); j--; }
            if (diff < 0) { diff += 10; carry = 1; }
            else { carry = 0; }
            result.pb('0' + diff);
            i--;
        }
        while (result.size() > 1 && result.back() == '0') result.pop_back();
        rev(result);
        if (negative) result.insert(result.begin(), '-');
        return result;
    }
    typedef complex<db> cd;
    static void fft(vector<cd> &a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            db ang = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                cd w(1);
                for (int j = 0; j < len / 2; j++) {
                    cd u = a[i + j];
                    cd v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (int i = 0; i < n; i++) a[i] /= n;
        }
    }
    static vector<int> multiplyFFT(const vector<int> &a, const vector<int> &b) {
        vt<cd> fa(all(a)), fb(all(b));
        int n = 1;
        while (n < int(a.size() + b.size())) n <<= 1;
        fa.resize(n); fb.resize(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        fft(fa, true);
        vi result(n);
        int carry = 0;
        for (int i = 0; i < n; i++) {
            ll t = static_cast<ll>(round(fa[i].real())) + carry;
            result[i] = t % 10;
            carry = t / 10;
        }
        while (carry) {
            result.pb(carry % 10);
            carry /= 10;
        }
        while (result.size() > 1 && result.back() == 0)
            result.pop_back();
        return result;
    }
    static string bigMul(const string &a, const string &b) {
        if (a == "0" || b == "0")
            return "0";
        vi A(a.size()), B(b.size());
        for (size_t i = 0; i < a.size(); i++) A[a.size() - 1 - i] = a[i] - '0';
        for (size_t i = 0; i < b.size(); i++) B[b.size() - 1 - i] = b[i] - '0';
        auto resultVec = multiplyFFT(A, B);
        string result;
        for (int i = resultVec.size() - 1; i >= 0; i--) result.pb(resultVec[i] + '0');
        size_t pos = result.find_first_not_of('0');
        return (pos != string::npos) ? result.substr(pos) : "0";
    }
    static string decToBin(const string &dec) {
        if (dec == "0") return "0";
        string number = dec, bin;
        while (number != "0") {
            int rem;
            number = divideByTwo(number, rem);
            bin.pb('0' + rem);
        }
        rev(bin);
        return bin;
    }
    static string binToDec(const string &bin) {
        string result = "0";
        for (char bit : bin) {
            result = bigMul(result, "2");
            if (bit == '1') result = bigAdd(result, "1");
        }
        return result;
    }
    static string divideByTwo(const string &num, int &rem) {
        string quotient;
        int carry = 0;
        for (char c : num) {
            int current = carry * 10 + (c - '0');
            int digit = current / 2;
            carry = current % 2;
            if (!quotient.empty() || digit != 0) quotient.pb('0' + digit);
        }
        if (quotient.empty()) quotient = "0";
        rem = carry;
        return quotient;
    }
    static string bigBitOr(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] == '1' || binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitAnd(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] == '1' && binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitXor(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] != binB[i]) ? '1' : '0');
        return binRes;
    }

    static string bigDiv(const string &a, const string &b) {
        if (b.size() == 1) {
            int d = b[0] - '0';
            if (2 <= d && d <= 10) {
                string quotient;
                quotient.reserve(a.size());
                int rem = 0;
                for (char c : a) {
                    int v     = rem * 10 + (c - '0');
                    int digit = v / d;
                    rem       = v % d;
                    if (!quotient.empty() || digit > 0)
                        quotient.pb(char('0' + digit));
                }
                return quotient.empty() ? "0" : quotient;
            }
        }

        if (compare(a, b) < 0) return "0";
        string quotient, remainder = "0";
        for (char c : a) {
            remainder = bigAdd(bigMul(remainder, "10"), string(1, c));
            int lo = 0, hi = 9, q = 0;
            while (lo <= hi) {
                int mid  = (lo + hi) / 2;
                string p = bigMulSingle(b, mid);
                if (compare(p, remainder) <= 0) {
                    q  = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            quotient.pb(char('0' + q));
            remainder = bigSub(remainder, bigMulSingle(b, q));
        }
        int pos = 0;
        while (pos + 1 < (int)quotient.size() && quotient[pos] == '0') pos++;
        return quotient.substr(pos);
    }

    static string bigMod(const string &a, const string &b) {
        if (b.size() == 1) {
            int d = b[0] - '0';
            if (2 <= d && d <= 10) {
                int rem = 0;
                for (char c : a) {
                    rem = (rem * 10 + (c - '0')) % d;
                }
                return to_string(rem);
            }
        }

        if (compare(a, b) < 0) return a;
        string remainder = "0";
        for (char c : a) {
            remainder = bigAdd(bigMul(remainder, "10"), string(1, c));
            int lo = 0, hi = 9, q = 0;
            while (lo <= hi) {
                int mid  = (lo + hi) / 2;
                string p = bigMulSingle(b, mid);
                if (compare(p, remainder) <= 0) {
                    q  = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            remainder = bigSub(remainder, bigMulSingle(b, q));
        }
        return remainder;
    }

    static string bigMulSingle(const string &a, int d) {
        int carry = 0;
        string result;
        for(int i = a.size() - 1; i >= 0; i--){
            int prod = (a[i] - '0') * d + carry;
            carry = prod / 10;
            result.pb('0' + prod % 10);
        }
        while(carry){
            result.pb('0' + carry % 10);
            carry /= 10;
        }
        rev(result);
        int pos = 0;
        while(pos < result.size() - 1 && result[pos]=='0') pos++;
        return result.substr(pos);
    }
};
