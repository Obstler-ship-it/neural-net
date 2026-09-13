#include "../matrix/matrix.hpp"

#include <cassert>
#include <iostream>

void test_transpose_multiply(){
    Matrix<Layout::RowMajor> a(2, 2, Initialization::Random);
    a.print();

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 3.0f;

    Matrix<Layout::ColumnMajor> b(2, 2, Initialization::Zero);
    b(0, 0) = 3.0f;
    b(0, 1) = 2.0f;
    b(1, 0) = 1.0f;
    b(1, 1) = 5.0f;

    Matrix<Layout::ColumnMajor> c = a.T() * b;
    assert(c(0, 0) == 7.0f);
    assert(c(0, 1) == 22.0f);
    assert(c(1, 0) == 9.0f);
    assert(c(1, 1) == 19.0f);
}

void test_multiply_transpose(){
    Matrix<Layout::ColumnMajor> a(2, 2, Initialization::Uninitialized);

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 3.0f;

    Matrix<Layout::ColumnMajor> b(2, 2, Initialization::Zero);
    b(0, 0) = 3.0f;
    b(0, 1) = 2.0f;
    b(1, 0) = 1.0f;
    b(1, 1) = 5.0f;

    Matrix<Layout::ColumnMajor> c = a * b.T();
    assert(c(0, 0) == 7.0f);
    assert(c(0, 1) == 11.0f);
    assert(c(1, 0) == 18.0f);
    assert(c(1, 1) == 19.0f);
}

int main() {
    Matrix<Layout::RowMajor> a(2, 3, Initialization::Zero);

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;

    assert(a.rows == 2);
    assert(a.columns == 3);
    assert(a(0, 1) == 2.0f);

    Matrix<Layout::ColumnMajor> b(3, 2, Initialization::Zero);

    b(0, 0) = 1.0f;
    b(1, 0) = 2.0f;
    b(2, 0) = 3.0f;
    b(0, 1) = 4.0f;
    b(1, 1) = 5.0f;
    b(2, 1) = 6.0f;

    auto result = a * b;

    assert(result.rows == 2);
    assert(result.columns == 2);
    assert(result(0, 0) == 14.0f); // 1*1 + 2*2 + 3*3
    assert(result(0, 1) == 32.0f); // 1*4 + 2*5 + 3*6

    test_transpose_multiply();
    test_multiply_transpose();

    std::cout << "Alle Matrix-Tests erfolgreich.\n";
}
