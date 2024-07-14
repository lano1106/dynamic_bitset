/*
 * dynamic bitset using AVX intrinsics for ultra fast implementation
 * https://github.com/lano1106/dynamic_bitset
 *
 * Olivier Langlois - May 2, 2024
 *
 * https://en.algorithmica.org/hpc/simd/intrinsics/
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *
 * Initial implementation is based on std::tr2::dynamic_bitset:
 * https://gcc.gnu.org/onlinedocs/gcc-13.1.0/libstdc++/api/a01627.html
 */

#ifndef BASE_DYNAMIC_BITSET_H_
#define BASE_DYNAMIC_BITSET_H_

#include <cstring> // for memset()
#include <vector>

#if defined(__AVX512BW__)
#include <x86intrin.h>
#include <immintrin.h>

namespace Base {
class bitsetImpl
{
public:
    using WordT = __m512i;
    static void do_or(WordT *lhs, const WordT *rhs, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm512_loadu_si512(lhs);
            B = _mm512_loadu_si512(rhs++);
            C = _mm512_or_si512(A, B);
            _mm512_storeu_si512(lhs++, C);
        }
    }
    static void do_sub(WordT *lhs, const WordT *rhs, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm512_loadu_si512(lhs);
            B = _mm512_loadu_si512(rhs++);
            C = _mm512_andnot_si512(B, A);
            _mm512_storeu_si512(lhs++, C);
        }
    }
    static void do_sub(const WordT *lhs, const WordT *rhs, WordT *res, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm512_loadu_si512(lhs++);
            B = _mm512_loadu_si512(rhs++);
            C = _mm512_andnot_si512(B, A);
            _mm512_storeu_si512(res++, C);
        }
    }
    /*
     * https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx512-vpopcnt.cpp
     */
    static uint64_t count(const WordT *lhs, int len)
    {
        WordT accumulator = _mm512_setzero_si512();
        WordT A, B;

        for (int i = 0; i < len; ++i) {
            A = _mm512_loadu_si512(lhs++);
            B = _mm512_popcnt_epi64(A);
            accumulator = _mm512_add_epi64(accumulator, B);
        }
        return _mm512_reduce_add_epi64(accumulator);
    }
};
}
#elifdef __SSE4_2__
#include <x86intrin.h>
#include <emmintrin.h>

namespace Base {
class bitsetImpl
{
public:
    using WordT = __m128i;
    static void do_or(WordT *lhs, const WordT *rhs, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm_loadu_si128(lhs);
            B = _mm_loadu_si128(rhs++);
            C = _mm_or_si128(A, B);
            _mm_storeu_si128(lhs++, C);
        }
    }
    static void do_sub(WordT *lhs, const WordT *rhs, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm_loadu_si128(lhs);
            B = _mm_loadu_si128(rhs++);
            C = _mm_andnot_si128(B, A); // C = A & ~B
            _mm_storeu_si128(lhs++, C);
        }
    }
    static void do_sub(const WordT *lhs, const WordT *rhs, WordT *res, int len)
    {
        WordT A, B, C;

        for (int i = 0; i < len; ++i) {
            A = _mm_loadu_si128(lhs++);
            B = _mm_loadu_si128(rhs++);
            C = _mm_andnot_si128(B, A); // C = A & ~B
            _mm_storeu_si128(res++, C);
        }
    }
    /*
     * https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-sse-cpu.cpp
     */
    static uint64_t count(const WordT *lhs, int len)
    {
        uint64_t result{};

        for (int i{}; i < len; ++i) {
            const __m128i v = _mm_loadu_si128(lhs++);

            result += _popcnt64(_mm_cvtsi128_si64(v));
            result += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(v, 8)));
        }
        return result;
    }
};
}
#else
namespace Base {
class bitsetImpl
{
public:
    using WordT = unsigned long long;
    static void do_or(WordT *lhs, const WordT *rhs, int len)
    {
        for (int i = 0; i < len; ++i)
            *lhs++ |= *rhs++;
    }
    static void do_sub(WordT *lhs, const WordT *rhs, int len)
    {
        for (int i = 0; i < len; ++i) {
            *lhs = *lhs & ~*rhs++;
            ++lhs;
        }
    }
    static void do_sub(const WordT *lhs, const WordT *rhs, WordT *res, int len)
    {
        for (int i = 0; i < len; ++i)
            *res++ = *lhs++ & ~*rhs++;
    }
    /*
     * https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-cpu.cpp
     */
    static uint64_t count(const WordT *lhs, int len)
    {
        uint64_t result{};

        for (int i{}; i < len; ++i)
            result += _popcnt64(*lhs++);
    
        return result;
    }
};
}
#endif

namespace Base
{
/*
 * class base_bitset
 *
 * CRTP core class for both dynamic_bitset and static_bitset classes
 *
 * Required concepts for Derived:
 * 1. a data member m_vec
 * 2. a method getEmpty() returning an empty Derived object.
 */
template<class Derived>
class base_bitset
{
public:
    using block_type = bitsetImpl::WordT;
    static constexpr size_t _S_bits_per_block = __CHAR_BIT__ * sizeof(block_type);

    void reset() noexcept
    {
        auto &vec{getDerived()->m_vec};

        memset(&vec[0], 0, std::size(vec)*sizeof(block_type));
    }
    Derived &operator|=(const Derived& rhs)
    {
        auto &vec{getDerived()->m_vec};

        bitsetImpl::do_or(&vec[0], &rhs.m_vec[0], std::size(vec));
        return *getDerived();
    }
    friend Derived operator-(const Derived &lhs,
                             const Derived &rhs)
    {
        auto res{lhs.getEmpty()};

        bitsetImpl::do_sub(&lhs.m_vec[0], &rhs.m_vec[0], &res.m_vec[0],
                           std::size(res.m_vec));
        return res;
    }
    Derived &operator-=(const Derived& rhs)
    {
        auto &vec{getDerived()->m_vec};

        bitsetImpl::do_sub(&vec[0], &rhs.m_vec[0], std::size(vec));
        return *getDerived();
    }
    Derived &operator|=(std::size_t pos)
    {
        return set(pos);
    }
    Derived &set(std::size_t pos)
    {
        _M_getword(pos) |= _S_maskbit(pos);
        return *getDerived();
    }
    Derived &set(std::size_t pos, bool val)
    {
        if (val)
            set(pos);
        else
            _M_getword(pos) &= ~_S_maskbit(pos);
        return *getDerived();
    }
    Derived &unset(std::size_t pos)
    {
        _M_getword(pos) &= ~_S_maskbit(pos);
        return *getDerived();
    }
    bool test(size_t pos) const
    { return _M_unchecked_test(pos); }

    /*
     * returns true if at least a bit is present in both bitmaps.
     */
    bool any_intersect(const Derived &rhs) const
    {
        // I am not too sure how and if this could be made faster using AVX
        auto &vec{getDerived()->m_vec};
        auto *lhs_bitmap = reinterpret_cast<const uint64_t *>(&vec[0]);
        auto *rhs_bitmap = reinterpret_cast<const uint64_t *>(&rhs.m_vec[0]);

        for (size_t k{}; k < m_64bitsIntNum; ++k) {
            if (lhs_bitmap[k] & rhs_bitmap[k])
                return true;
        }
        return false;
    }

    /*
     * https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
     */
    template <class CB>
    void iterate_bits(CB &callback) const
    {
        uint64_t     bitset;
        const auto &vec{getDerived()->m_vec};
        auto        *bitmap = reinterpret_cast<const uint64_t *>(&vec[0]);

        for (size_t k{}; k < m_64bitsIntNum; ++k) {
            bitset = bitmap[k];
            while (bitset != 0) {
                uint64_t t = bitset & -bitset;
                auto r = __builtin_ctzl(bitset);

                callback(k * 64 + r);
                bitset ^= t;
            }
        }
    }
    size_t nbits() const { return std::size(getDerived()->m_vec)*_S_bits_per_block; }
    size_t count() const
    {
        auto &vec{getDerived()->m_vec};

        return bitsetImpl::count(&vec[0], std::size(vec));
    }

protected:
    base_bitset(size_t bits64Num)
    : m_64bitsIntNum(bits64Num) {}
    base_bitset() = default;
    base_bitset(const base_bitset&) = default;
    base_bitset(base_bitset&& __b) = default;
    base_bitset& operator=(const base_bitset&) = default;
    base_bitset& operator=(base_bitset&&) = default;
    ~base_bitset() = default;

    void set64bitsIntNum(size_t bits64Num) { m_64bitsIntNum = bits64Num; }

private:
    Derived *getDerived() { return static_cast<Derived *>(this); }
    const Derived *getDerived() const { return static_cast<const Derived *>(this); }

    static size_t _S_whichword(size_t pos) noexcept
    { return pos / 64; }
    uint64_t &_M_getword(size_t pos) noexcept
    { return reinterpret_cast<uint64_t *>(&getDerived()->m_vec[0])[_S_whichword(pos)]; }
    uint64_t _M_getword(size_t pos) const noexcept
    { return reinterpret_cast<const uint64_t *>(&getDerived()->m_vec[0])[_S_whichword(pos)]; }
    static size_t _S_whichbit(size_t pos) noexcept
    { return pos % 64; }
    static uint64_t _S_maskbit(size_t pos) noexcept
    { return (static_cast<uint64_t>(1)) << _S_whichbit(pos); }
    bool _M_unchecked_test(size_t pos) const noexcept
    { return (_M_getword(pos) & _S_maskbit(pos)) != 0; }

    size_t m_64bitsIntNum{};
};

/*
 * class static_bitset
 */
template<size_t NBITS>
class static_bitset : public base_bitset<static_bitset<NBITS> >
{
    static constexpr size_t num64Bits() { return NBITS / 64 + (NBITS % 64 > 0); }
    using Parent = base_bitset<static_bitset<NBITS> >;
    friend Parent;
public:
    static_bitset() : Parent(num64Bits()) {}

private:
    static_bitset getEmpty() const
    { return static_bitset{}; }
    static constexpr size_t numBlocks()
    { return NBITS / Parent::_S_bits_per_block + (NBITS % Parent::_S_bits_per_block > 0); }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::array<typename Parent::block_type,numBlocks()> m_vec{}; __attribute__ ((aligned (64)));
#pragma GCC diagnostic pop
};

/*
 * class dynamic_bitset
 */
class dynamic_bitset : public base_bitset<dynamic_bitset>
{
public:
    using Parent = base_bitset<dynamic_bitset>;
    friend Parent;

    dynamic_bitset() = default;
    dynamic_bitset(const dynamic_bitset&) = default;
    dynamic_bitset(dynamic_bitset&& __b) = default;
    dynamic_bitset& operator=(const dynamic_bitset&) = default;
    dynamic_bitset& operator=(dynamic_bitset&&) = default;
    ~dynamic_bitset() = default;

    explicit dynamic_bitset(size_t nbits)
    : base_bitset(nbits / 64 + (nbits % 64 > 0)),
      m_vec(nbits / _S_bits_per_block + (nbits % _S_bits_per_block > 0))
    {}

    void swap(dynamic_bitset &b) noexcept
    {
        m_vec.swap(b.m_vec);
        /*
         * NOTE:
         * An assumption is made that only bitset of the same size are
         * swapped. So swapping the # of int64 is not needed.
         *
         * The assumption is currently valid among all the current class
         * clients.
         * 2024-06-14
         */
//        std::swap(m_64bitsIntNum, b.m_64bitsIntNum);
    }
    void resize(size_t nbits)
    {
        size_t sz = nbits / _S_bits_per_block;

        if (nbits % _S_bits_per_block > 0)
            ++sz;
        m_vec.resize(sz);
        set64bitsIntNum(nbits / 64 + (nbits % 64 > 0));
    }

private:
    dynamic_bitset getEmpty() const
    {
        return dynamic_bitset{nbits()};
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::vector<block_type> m_vec;
#pragma GCC diagnostic pop
};

/*
 * class fixed_dynamic_bitset
 */
template <size_t FIXED_SIZE>
class fixed_dynamic_bitset : public dynamic_bitset
{
public:
    fixed_dynamic_bitset()
    : dynamic_bitset(FIXED_SIZE) {}
};
}
#endif
