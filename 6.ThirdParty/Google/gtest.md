[TOC]

# Install 
`gtest` 没有在ubuntu的安装package中，需要下载源码然后编译成库，手动链接

repository address
```https
https://github.com/google/googletest.git
```

## Windows

## Linux

```shell
# clone to the target folder
git clone https://github.com/google/googletest.git

# cd to the google gtest folder
cd ./googletest/googlegtest
# create a build folder to build 
mkdir build
cd build

# build
cmake ..
make

# copy the library to the system library path
sudo cp gtest /usr/local/lib 
# copy the include file to the system include path
cd ..
sudo cp -r include/gest/ /usr/local/include/
```
![googletest](../.picture/2.CodingStyle/google-gtest-1.png) 
![googletest](../.picture/2.CodingStyle/google-gtest-2.png) 

# Usage

## `main()` config
```c++
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    RUN_ALL_TESTS();

    return 0;
}
```

## TEST

## TEST_F
```c++

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
};

TEST_F(TestDemo, Test)
{
    int a = 1;
    int b = 2;
    int sum = a + b;

    EXPECT_EQ(sum, Add(a, b));
}
```
result:
![googletest](../.picture/2.CodingStyle/google-gtest-3.png) 

