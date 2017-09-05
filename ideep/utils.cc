/*
 *COPYRIGHT
 *All modification made by Intel Corporation: © 2017 Intel Corporation.
 *Copyright (c) 2015 Preferred Infrastructure, Inc.
 *Copyright (c) 2015 Preferred Networks, Inc.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 *
 *######################################################################
 *# The CuPy is designed based on NumPy's API.
 *# CuPy's source code and documents contain the original NumPy ones.
 *######################################################################
 *Copyright (c) 2005-2016, NumPy Developers.
 *All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are
 *met:
 *
 *    * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    * Neither the name of the NumPy Developers nor the names of any
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *######################################################################
 */

#include "utils.hpp"
#include <stdint.h>

#if defined(_MSC_VER)
static inline uint64_t __cpuidXfeature()
{
#if (_MSC_VER > 1600)
    return _xgetbv(0);
#else
    uint32_t a, d;
    __asm {
        push edx
        push ecx
        push eax
        xor ecx, ecx
        _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0
        mov a, eax
        mov d, edx
        pop eax
        pop ecx
        pop edx
    } 
    return (((uint64_t)d << 32) | a);
#endif
}

#if (_MSC_VER < 1400)
static inline __declspec(naked) void __cpuid(int[4] result, int level)
{
    __asm {
        push    ebx
        push    edi
        mov     eax, dword ptr [esp + 4 * 4]    // level
        cpuid
        mov     edi, dword ptr [esp + 4 * 3]    // result
        mov     dword ptr [edi + 4 * 0], eax    // result[0]
        mov     dword ptr [edi + 4 * 1], ebx    // result[1]
        mov     dword ptr [edi + 4 * 2], ecx    // result[2]
        mov     dword ptr [edi + 4 * 3], edx    // result[3]
        pop     edi
        pop     ebx
        ret
    }
}

static inline __declspec(naked) void __cpuidex(int[4] result, int level, int count)
{
    __asm {
        push    ebx
        push    ecx
        push    edi
        mov     ecx, dword ptr [esp + 4 * 6]    // count
        mov     eax, dword ptr [esp + 4 * 5]    // level
        cpuid
        mov     edi, dword ptr [esp + 4 * 4]    // result
        mov     dword ptr [edi + 4 * 0], eax    // result[0]
        mov     dword ptr [edi + 4 * 1], ebx    // result[1]
        mov     dword ptr [edi + 4 * 2], ecx    // result[2]
        mov     dword ptr [edi + 4 * 3], edx    // result[3]
        pop     edi
        pop     ecx
        pop     ebx
        ret
    }
}

#else
#include <intrin.h>

#endif

#else   // Non-MSC
static inline uint64_t __cpuidXfeature()
{
    uint32_t eax, edx;
#if (((__GNUC__) > 4) || (((__GNUC__) == 4) && ((__GNUC_MINOR_) > 2)))
    __asm__ volatile("xgetbv"
            : "=a"(eax), "=d"(edx)
            : "c"(0));
#else
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
            : "=a"(eax), "=d"(edx)
            : "c"(0));
#endif
    return (((uint64_t)edx << 32) | eax);
}

#if defined(__APPLE__)
#define __cpuid(a, b, c, d, level) \
    __asm__ __volatile__(   \
            "pushl %%ebx\ncpuid\nmovl %%ebp, %%esi\npopl %%ebx"    \
                : "=a"(a), "=S"(b), "=c"(c), "=d"(d)    \
                : "0"(level))

#define __cpuid_count(a, b, c, d, level, count) \
    __asm__ __volatile__(   \
            "pushl %%ebx\ncpuid\nmovl %%ebp, %%esi\npopl %%ebx" \
                : "=a"(a), "=S"(b), "=c"(c), "=d"(d)    \
                : "0"(level), "2"(count))

#else   // Non-APPLE
#define __cpuid(a, b, c, d, level)  \
    __asm__ __volatile__(   \
            "cpuid\n"   \
                : "=a"(a), "=b"(b), "=c"(c), "=d"(d)    \
                : "0"(level))

#define __cpuid_count(a, b, c, d, level, count) \
    __asm__ __volatile__(   \
            "cpuid\n"   \
                : "=a"(a), "=b"(b), "=c"(c), "=d"(d)    \
                : "0"(level), "2"(count))
#endif
#endif

static inline void get_cpu_feature(uint32_t level, uint32_t result[4])
{
#ifdef _MSC_VER
        __cpuid(reinterpret_cast<int*>(result), level);
#else
        __cpuid(result[0], result[1], result[2], result[3], level);
#endif
}

static inline void get_cpu_feature_ext(uint32_t level, uint32_t count, uint32_t result[4])
{
#ifdef _MSC_VER
    __cpuidex(reinterpret_cast<int*>(result), level, count);
#else
    __cpuid_count(result[0], result[1], result[2], result[3], level, count);
#endif
}

class CpuFeatures
{
public:
    static const uint64_t f_NONE            = uint64_t(0);
    static const uint64_t f_MMX             = uint64_t(1) << 0;
    static const uint64_t f_MMX2            = uint64_t(1) << 1;
    static const uint64_t f_CMOV            = uint64_t(1) << 2;
    static const uint64_t f_SSE             = uint64_t(1) << 3;
    static const uint64_t f_SSE2            = uint64_t(1) << 4;
    static const uint64_t f_SSE3            = uint64_t(1) << 5;
    static const uint64_t f_SSSE3           = uint64_t(1) << 6;
    static const uint64_t f_SSE41           = uint64_t(1) << 7;
    static const uint64_t f_SSE42           = uint64_t(1) << 8;
    static const uint64_t f_POPCNT          = uint64_t(1) << 9;
    static const uint64_t f_AESNI           = uint64_t(1) << 10;
    static const uint64_t f_SSE5            = uint64_t(1) << 11;
    static const uint64_t f_OSXSAVE         = uint64_t(1) << 12;
    static const uint64_t f_PCLMULQDQ       = uint64_t(1) << 13;
    static const uint64_t f_AVX             = uint64_t(1) << 14;
    static const uint64_t f_FMA             = uint64_t(1) << 15;
    static const uint64_t f_SSE4a           = uint64_t(1) << 16;
    static const uint64_t f_RDTSCP          = uint64_t(1) << 17;
    static const uint64_t f_AVX2            = uint64_t(1) << 18;
    static const uint64_t f_BMI1            = uint64_t(1) << 19;
    static const uint64_t f_BMI2            = uint64_t(1) << 20;
    static const uint64_t f_LZCNT           = uint64_t(1) << 21;
    static const uint64_t f_ENHANCED_REP    = uint64_t(1) << 22;
    static const uint64_t f_RDRAND          = uint64_t(1) << 23;
    static const uint64_t f_ADX             = uint64_t(1) << 24;
    static const uint64_t f_RDSEED          = uint64_t(1) << 25;
    static const uint64_t f_SMAP            = uint64_t(1) << 26;
    static const uint64_t f_HLE             = uint64_t(1) << 27;
    static const uint64_t f_RTM             = uint64_t(1) << 28;
    static const uint64_t f_F16C            = uint64_t(1) << 29;
    static const uint64_t f_MOVBE           = uint64_t(1) << 30;
    static const uint64_t f_AVX512F         = uint64_t(1) << 31;
    static const uint64_t f_AVX512DQ        = uint64_t(1) << 32;
    static const uint64_t f_AVX512IFMA      = uint64_t(1) << 33;
    static const uint64_t f_AVX512PF        = uint64_t(1) << 34;
    static const uint64_t f_AVX512ER        = uint64_t(1) << 35;
    static const uint64_t f_AVX512CD        = uint64_t(1) << 36;
    static const uint64_t f_AVX512BW        = uint64_t(1) << 37;
    static const uint64_t f_AVX512VL        = uint64_t(1) << 38;
    static const uint64_t f_AVX512VBMI      = uint64_t(1) << 39;
    static const uint64_t f_AVX512_4VNNIW   = uint64_t(1) << 40;
    static const uint64_t f_AVX512_4FMAPS   = uint64_t(1) << 41;
    static const uint64_t f_PREFETCHWT1     = uint64_t(1) << 42;

    static const uint32_t any               = 0;
    static const uint32_t sse42             = 1;
    static const uint32_t avx               = 2;
    static const uint32_t avx2              = 3;
    static const uint32_t avx512_comm       = 4;
    static const uint32_t avx512_core       = 5;
    static const uint32_t avx512_mic        = 6;
    static const uint32_t avx512_mic_4ops   = 7;

    CpuFeatures()
    {
        features = f_NONE;
        uint32_t result[4] = {0};

        get_cpu_feature(0x80000001, result);
        if (result[2] & (1U << 5))  features |= f_LZCNT;
        if (result[3] & (1U << 27)) features |= f_RDTSCP;

        get_cpu_feature(1, result);
        if (result[2] & (1U << 0))  features |= f_SSE3;
        if (result[2] & (1U << 1))  features |= f_PCLMULQDQ;
        if (result[2] & (1U << 9))  features |= f_SSSE3;
        if (result[2] & (1U << 19)) features |= f_SSE41;
        if (result[2] & (1U << 20)) features |= f_SSE42;
        if (result[2] & (1U << 22)) features |= f_MOVBE;
        if (result[2] & (1U << 23)) features |= f_POPCNT;
        if (result[2] & (1U << 25)) features |= f_AESNI;
        if (result[2] & (1U << 27)) features |= f_OSXSAVE;
        if (result[2] & (1U << 30)) features |= f_RDRAND;
        if (result[2] & (1U << 29)) features |= f_F16C;
        if (result[3] & (1U << 15)) features |= f_CMOV;
        if (result[3] & (1U << 23)) features |= f_MMX;
        if (result[3] & (1U << 25)) features |= f_MMX2 | f_SSE;
        if (result[3] & (1U << 26)) features |= f_SSE2;

        if (features & f_OSXSAVE) {
            uint64_t x_enabled = __cpuidXfeature();
            if ((x_enabled & 0x6) == 0x6) {
                if (result[2] & (1U << 28)) features |= f_AVX;
                if (result[2] & (1U << 12)) features |= f_FMA;
                if (((x_enabled >> 5) & 0x7) == 0x7) {
                    get_cpu_feature_ext(0x7, 0x0, result);
                    if (result[1] & (1U << 16)) {
                        features |= f_AVX512F;
                        if (result[1] & (1U << 17)) features |= f_AVX512DQ;
                        if (result[1] & (1U << 21)) features |= f_AVX512IFMA;
                        if (result[1] & (1U << 26)) features |= f_AVX512PF;
                        if (result[1] & (1U << 27)) features |= f_AVX512ER;
                        if (result[1] & (1U << 28)) features |= f_AVX512CD;
                        if (result[1] & (1U << 30)) features |= f_AVX512BW;
                        if (result[1] & (1U << 31)) features |= f_AVX512VL;
                        if (result[2] & (1U << 1))  features |= f_AVX512VBMI;
                        if (result[3] & (1U << 2))  features |= f_AVX512_4VNNIW;
                        if (result[3] & (1U << 3))  features |= f_AVX512_4FMAPS;
                    }
                }
            }
        }

        get_cpu_feature(0x0, result);
        if (result[0] >= 7) {
            get_cpu_feature_ext(0x7, 0x0, result);
            if ((features & f_AVX) && (result[1] & 0x20)) features |= f_AVX2;
            if (result[1] & (1U << 3))  features |= f_BMI1;
            if (result[1] & (1U << 8))  features |= f_BMI2;
            if (result[1] & (1U << 9))  features |= f_ENHANCED_REP;
            if (result[1] & (1U << 18)) features |= f_RDSEED;
            if (result[1] & (1U << 19)) features |= f_ADX;
            if (result[1] & (1U << 20)) features |= f_SMAP;
            if (result[1] & (1U << 4))  features |= f_HLE;
            if (result[1] & (1U << 11)) features |= f_RTM;
            if (result[2] & (1U << 0))  features |= f_PREFETCHWT1;
        }

    }

    bool has_feature(uint64_t f)
    {
        return (features & f) ? true : false;
    }

    bool is_supported(const uint32_t cpu_isa)
    {
        switch (cpu_isa) {
            case sse42:
                return has_feature(f_SSE42);
            case avx:
                return has_feature(f_AVX);
            case avx2:
                return has_feature(f_AVX2);
            case avx512_comm:
                return has_feature(f_AVX512F);
            case avx512_core:
                return has_feature(f_AVX512F)
                    && has_feature(f_AVX512BW)
                    && has_feature(f_AVX512VL)
                    && has_feature(f_AVX512DQ);
            case avx512_mic:
                return has_feature(f_AVX512F)
                    && has_feature(f_AVX512CD)
                    && has_feature(f_AVX512ER)
                    && has_feature(f_AVX512PF);
            case avx512_mic_4ops:
                return is_supported(avx512_mic)
                    && has_feature(f_AVX512_4FMAPS)
                    && has_feature(f_AVX512_4VNNIW);
            case any:
                return true;
            default:
                return false;
        }

        return false;
    }

private:
    uint64_t features;
};

memory::format get_desired_format(int channel)
{
    CpuFeatures cpu_f;
    memory::format fmt_desired = memory::format::any;

    if (cpu_f.is_supported(CpuFeatures::avx512_comm) && (channel % 16) == 0) {
        fmt_desired = memory::format::nChw16c;
    } else if (cpu_f.is_supported(CpuFeatures::avx2) && (channel % 8) == 0) {
        fmt_desired = memory::format::nChw8c;
    } else {
        fmt_desired = memory::format::nchw;
    }
    return fmt_desired;
}

