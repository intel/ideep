#include <cassert>
#include <ctime>
#include <memory>
#include <random>

#include "mdarray.h"
#include "mkl_vsl.h"
#include "utils.hpp"

static void bernoulli_generate(long n, double p, int* r) {
    std::srand(std::time(0));
    int seed = 17 + std::rand() % 4096;
    const long my_amount = n;
    const long my_offset = 0;
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, seed);
    vslSkipAheadStream(stream, my_offset);
    viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
    vslDeleteStream(&stream);
}

template<typename T>
struct dropout {
    dropout(T ratio) : ratio_(ratio), scale_(1.0 / (1.0 - ratio)), first_malloc_(false) {}
    
    void forward(mdarray* x_in, mdarray* mask_in, mdarray* y) {
        auto x = x_in->get();
        auto mask_buf_ = static_cast<T*>(mask_in->get()->data());

        assert(x->size() == mask_in->get()->size());

        size_ = x->size();
        initialize(mask_buf_);
        auto data_ptr = static_cast<T*>(x->data());
        auto y_data_ptr = static_cast<T*>(y->get()->data());
        assert(x->size() == y->get()->size());

        eltwise_multiply(data_ptr, mask_buf_, y_data_ptr, size_);
    }
    
    void backward(mdarray* dy_in, mdarray* mask_in, mdarray* dx) {
        auto dy = dy_in->get();
        
        assert(size_ == dy->size());
        
        auto data_ptr = static_cast<T*>(dy->data());
        auto mask_buf_ = static_cast<T*>(mask_in->get()->data());
        auto dx_data_ptr = static_cast<T*>(dx->get()->data());

        eltwise_multiply(data_ptr, mask_buf_, dx_data_ptr, size_);
    }

private:
    void initialize(T* mask_buf_) {
        if (!first_malloc_) {
            bernouli_nums.reset(new int[size_]);
            first_malloc_ = true;
        }

        bernoulli_generate(size_, 1.0 - ratio_, bernouli_nums.get());

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size_; ++i) {
            mask_buf_[i] = bernouli_nums[i] * scale_;
        }
    }
    
    T ratio_, scale_;
    size_t size_;
    std::unique_ptr<int[]> bernouli_nums;
    bool first_malloc_;
};