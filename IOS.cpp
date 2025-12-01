namespace FastIO {
    static const size_t BUF_SIZE = 1 << 16;

    static char ibuf[BUF_SIZE];
    static char obuf[BUF_SIZE];

    static size_t ipos = 0, ilen = 0, opos = 0;
    static bool use_fread = false;

    inline void init() {
        use_fread = false;
    }

    inline char next_getchar() {
        int c = getchar();
        return (c == EOF) ? 0 : (char)c;
    }

    inline void refill() {
        ilen = fread(ibuf, 1, BUF_SIZE, stdin);
        ipos = 0;
    }

    inline char next_fread() {
        if (ipos >= ilen) refill();
        if (ilen == 0) return 0;

        return ibuf[ipos++];
    }

    inline char next_char() {
        return use_fread ? next_fread() : next_getchar();
    }

    inline bool is_space(char c) {
        return c != 0 and c <= ' ';
    }

    inline int read_int() {
        char c; int x = 0, neg = 0;

        do c = next_char();
        while (is_space(c));

        if (c == 0) return 0;
        if (c == '-') neg = 1, c = next_char();

        for (; c >= '0' and c <= '9'; c = next_char()) x = x * 10 + (c - '0');

        return neg ? -x : x;
    }

    inline ll read_ll() {
        char c; ll x = 0, neg = 0;

        do c = next_char();
        while (is_space(c));

        if (c == 0) return 0;
        if (c == '-') neg = 1, c = next_char();

        for (; c >= '0' and c <= '9'; c = next_char()) x = x * 10 + (c - '0');

        return neg ? -x : x;
    }

    inline char read_char() {
        char c;

        do c = next_char();
        while (is_space(c));

        return c;
    }

    inline string read_string() {
        string s; char c;

        do c = next_char();
        while (is_space(c));

        for (; c != 0 and c > ' '; c = next_char()) s.push_back(c);

        return s;
    }

    inline void flush_output() {
        if (opos) fwrite(obuf, 1, opos, stdout), opos = 0;
    }

    inline void write_char(char c) {
        if (opos >= BUF_SIZE) flush_output();
        obuf[opos++] = c;
    }

    inline void print_int(int x, char endc = '\n') {
        if (x == 0) {
            write_char('0');
            if (endc) write_char(endc);

            return;
        }
        if (x < 0) write_char('-'), x = -x;

        char buf[20]; int i = 0;

        while (x) buf[i++] = '0' + (x % 10), x /= 10;
        while (i--) write_char(buf[i]);

        if (endc) write_char(endc);
    }

    inline void print_ll(ll x, char endc = '\n') {
        if (x == 0) {
            write_char('0');
            if (endc) write_char(endc);

            return;
        }
        if (x < 0) write_char('-'), x = -x;

        char buf[25]; int i = 0;

        while (x) buf[i++] = '0' + (x % 10), x /= 10;
        while (i--) write_char(buf[i]);

        if (endc) write_char(endc);
    }

    inline void print_char(char c, char endc = '\n') {
        write_char(c);
        if (endc) write_char(endc);
    }

    inline void print_string(const string& s, char endc = '\n') {
        for (char c : s) write_char(c);
        if (endc) write_char(endc);
    }

    struct Flusher {
        ~Flusher() {
            flush_output();
        }
    } static flusher;
}

using namespace FastIO;

namespace FastIO {
    struct FastInput {
        FastInput() {
            FastIO::init();
        }

        FastInput& operator>>(int &x) {
            x = FastIO::read_int();
            return *this;
        }

        FastInput& operator>>(long long &x) {
            x = FastIO::read_ll();
            return *this;
        }

        FastInput& operator>>(char &c) {
            c = FastIO::read_char();
            return *this;
        }

        FastInput& operator>>(std::string &s) {
            s = FastIO::read_string();
            return *this;
        }
    };

    struct FastOutput {
        FastOutput& operator<<(int x) {
            FastIO::print_int(x, 0);
            return *this;
        }

        FastOutput& operator<<(long long x) {
            FastIO::print_ll(x, 0);
            return *this;
        }

        FastOutput& operator<<(char c) {
            FastIO::write_char(c);
            return *this;
        }

        FastOutput& operator<<(const char *s) {
            while (*s) FastIO::write_char(*s++);
            return *this;
        }

        FastOutput& operator<<(const std::string &s) {
            FastIO::print_string(s, 0);
            return *this;
        }

        using Manip = FastOutput& (*)(FastOutput&);
        FastOutput& operator<<(Manip f) {
            return f(*this);
        }
    };

    inline FastOutput& endl(FastOutput &out) {
        FastIO::write_char('\n');
        return out;
    }

    static FastInput  fin;
    static FastOutput fout;

}
using FastIO::fin;
using FastIO::fout;


#pragma GCC optimize("Ofast")

#pragma GCC target("avx2")
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,sse4.1,sse4.2,popcnt,abm,mmx,avx,avx2,fma,tune=native")

// below is kinda fast
#pragma GCC optimize("-funsafe-loop-optimizations")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-fwhole-program")
#pragma GCC optimize("Ofast,no-stack-protector")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fdelete-null-pointer-checks")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC target("avx")