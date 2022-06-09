///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief UT google test module
/// 
/// @file test.h
/// 
/// @author Jonathan
/// 
/// @date 2022-02-08
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE_CONFIG_TEST_H_
#define __SOFTWARE_CONFIG_TEST_H_

// System header
// C/C++ standard library header
#include <iostream>
#include <string>
// External library header
#include "gtest/gtest.h"
#include "glog/logging.h"
// Current module header
// Root directory header

class TestDemo : public ::testing::Test
{
public:
    TestDemo() = default;
    ~TestDemo() {}

    virtual void SetUp() override {}
    virtual void TearDown() override {}

    int Add(int a, int b)
    {
        return a + b;
    }

    double Divide(double a, double b)
    {
        if (0 == b)
        {
            LOG(FATAL) << "the dinominator is zero!!!";
        }
        return a / b;
    }

    std::string &SetString(std::string &str)
    {
        return str;
    }

private:
    int a;
    int b;
};

TEST_F(TestDemo, TestAdd)
{
    int a = 1;
    int b = 2;
    int sum = a + b;

    EXPECT_EQ(sum, Add(a, b));
}

TEST_F(TestDemo, TestDivide)
{
    double a  = 1.1111;
    double b = 2.22222222222;
    double divide = a / b;

    EXPECT_DOUBLE_EQ(divide, Divide(a, b));
}

TEST_F(TestDemo, TestDivide2)
{
    double a  = 1.1111;
    double b = 2.22222222222;
    double divide = a / b;

    const double EPSION = 0.000001;
    EXPECT_NEAR(divide, Divide(a, b), EPSION);
}

TEST_F(TestDemo, TestString)
{
    std::string str1 = "nano";
    std::string str2 = "nanovision";
    std::string str3 = "nanovision compoundeye";

    EXPECT_STREQ(str1.c_str(), SetString(str1).c_str());
    EXPECT_STRNE(str1.c_str(), SetString(str2).c_str());
}

#endif // __SOFTWARE_CONFIG_TEST_H_
