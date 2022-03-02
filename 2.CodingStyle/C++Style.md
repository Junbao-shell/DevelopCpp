[TOC]

本文参考 [ Google 开源风格指南 -- C++风格指南](https://github.com/zh-google-styleguide/zh-google-styleguide/releases) 

# C++ 编码规范

## 命名

命名包含对类型（包含C++的类，枚举等），常量，变量，函数，文件，文件夹（模块）等的命名规则

命名规范是最重要的**一致性规则**，可以让读者在不查找类型声明的情况下快速地了解某个名字代表的含义。

准则：

+ **命名要具有描述性**（有意义的命名）
+ **不要随意缩写** 

> 一些 **广为人知** 的缩写是允许的，如 `i` 作为迭代变量，`T` 作为模板参数

几种命名规则：

驼峰命名（AirImageNumber: 大驼峰，airImageNumber: 小驼峰），帕斯卡变量名，匈牙利命名法（sPath, iName)

### 常量的命名

+ 常量前加 `k` 前缀，单词首字母大写串接；

> 常量采用单词首字母大写串接，可以认为常量作为一个API的一部分让外部使用

```c++
constexpr int kMajorVersion = 1;
constexpr int kMinorVersion = 3;
```

### 变量的命名

+ 单词小写通过下划线连接
+ 有意义的命名
+ 不得所以缩写
+ 成员变量加 `m_` 前缀

+ 全局变量加 `g_` 前缀

> struct 结构体中的成员按照普通变量命名

```c++
int raw_image_width; // Y 可以
int w;               // X 不建议 不明确
int source_ID;       // X 不建议，int source_id; 
```

### 函数的命名

+ 单词首字母大写串接； 
+ 动宾结构； `int GetImageNumber();` `void SetImageNumber();` 
+ 返回值 `bool` 类型的函数 `bool IsReady();` 其他函数采用 `errorcode_t` 返回值；

```c++
int GetImageNumber();
```

### 类命名

+ 单词首字母大写串接；

```c++
class OffsetCorrect
{
// ...  
};
```

### 命名空间

+ 单词小写
+ 顶层命名空间的名字使用项目名称或者团队名称

```c++
namespace pap
{
namespace utility
{
// ...     
} // namespace utility
} // namespace pap
```

### 枚举命名

+ 全部大写
+ 加 `ENUM_` 前缀

```c++
enum ENUM_EXPOSURE_MODE
{
    NO = 0,
    SINGLE,
    DOUBLE,
    TRIPLE
};
```

### 文件命名

+ 单词小写，通过下划线连接；
+ `c++` 头文件使用 `.h` 扩展，源文件使用 `cpp` 扩展名；
+ `cuda` 代码头文件仍采用 `.h` 扩展，源文件采用 `.cu` 扩展；

## 注释

统一使用 `//` 注释符号，保持风格统一

注释使用中文注释还是英文注释没有严格的约束，如果使用中文注释要注意文件的编码格式；（最好使用英文注释）

包括文件注释，类注释，函数注释，块注释，关键变量注释

准则：

+ 不要描述显而易见的现象
+ 不要用自然语言翻译代码作为注释
+ 提供的注释应当解释代码，为什么要这么做和这么做的目的，最好让代码自文档化
+ 不要让复杂的东西简单化，不要让简单的东西复杂化

注释的通常写法是包含正确大小写和结尾句号的完整叙述性语句。

### 文件注释

文件注释应该以下内容

+ 版权公告，法律公告

+ 作者信息

+ 迭代信息

如果对原始作者的文件做了 **重大修改**，请考虑删除原作者信息  

```c++
//////////////////////////////////////////////////////////// 
/// @copyright: 
/// @brief: 
/// @file: 
/// @author: 
/// @date: 
/// @edit: 
//////////////////////////////////////////////////////////// 
```

### 函数注释 

> 函数注释采用 Doxygen 注释格式，方便直接生成说明文档

+ 函数声明处的注释描述函数功能；

  如果通过函数的名称可以直观确定函数的功能，可以省略函数注释，如 `OpenFile()` ；

+ 函数定义处的只是描述函数实现；

+ 对于类成员函数而言：函数调用期间对象是否需要保持引用参数，是否会释放这些参数；

+ 必要的函数的输入输出；

+ 重载的函数没有必要加重复的注释；

+ 其他的可能出现异常的解释

```c++
/**
* @ brief: reset the value in the xml file
* @ param xml_path: the target xml file path
* @ param node_path: the target node path relative root node
* @ param value: the target value to set
* @ return: if reset success reture true
*/
bool Configure::WriteConfig(const std::string &xml_path, const std::string &node_path, const std::string &value);
```

### 块注释

对于代码中巧妙的，晦涩的，有趣的地方加以注释 

```c++
int x = 100;
int count = 0;
// return the number of '1' in binary data type
while(x)
{
    x = x & (x - 1);
    ++count;
}
```

### 变量的注释

一般情况下，变量直接通过命名即可知其意。但是遇到一些关键变量比较难以理解是，请为当前变量加上注释；

### TODO注释

+ TODO注释要使用全大写的字符串 `TODO` ，并在随后的圆括号里写上你的名字，邮件地址，Bug ID等信息，

```c++
// TODO (@author @email) define anthor interface to meet reconstruct the image size after binning
void Reconstruction(ImageInfo<float>);
```

### FIXME注释

+ 对于某处可能引发某些错误，或者在未来扩展时可能存在的问题，加上全大写的字符串 `FIXME`，并在随后的表明需要修复的bug

```c++
// FIXME (@author @email) 1. fix the array boundary about memory fault; 2. user other more advanced interpolation function
void CheckDefectGainCoefficient(float *gain_image, const int image_size)
{
    const float upper_limit = 5.0f;
    for (int i = 0; i < image_size; ++i)
    {
        if (gain_image[i] > upper_limit)
        {
            gain_image[i] = (gain_image[i - 1] + gain_image[i + 1]);
        }
    }
}
```

> 注释中同样需要注意标点，空格，拼写和语法等细节问题；

## 格式

### 字符数和行数

+ 每行代码字符数不超过120；
+ 每个函数不要超过200行，最好将功能模块的代码控制在40行；
+ 每个文件不要超过2000行；

+ 尽量不适用非ASCII字符，如果必须要使用则需要使用UTF-8编码格式

### 空格

+ 只使用空格，使用4个空格缩进

  > 注释在使用的编辑器中将 `Tab` 转换为4个空格

设置空格

+ `if` 和 `while` 条件语句圆括号前加1个空格；
+ `for` 圆括号前后都加1个空格；
+ 运算符 `=`， `+ - x /`  前后加1个空格；
+ 逗号 `,` 以及 分号`;` 后加1个空格；
+ 注释符号 `//` 前后加1个空格；

没有空格

+ 函数名与做圆括号之间没有空格
+ 句点 `.` 和 `->`前后不能有空格, 取值 `*` ， 取地址 `&` 符号后不能有空格
+ `if` `whiile` 条件语句左圆括号右侧和右圆括号左侧不加空格
+ 命名空间内容不缩进
+ 预处理指令从行首开始，不要加空格
+ lambda 表达式捕获列表中括号和参数圆括号之间没有空格

```c++
namespace sapce_name
{    
void function()
{
#ifdef FLAG
    int condition_value = 10; // comment: 
    if (condition_value == 10)
    {
        ...;
    }
#endif // FLAG 
}
} // namespace space_name
```

### 换行

+ 函数返回值类型和函数名在同一行，左圆括号和函数名同行
+ 函数返回类型和参数名在同一行，参数也尽量放在同一行，如果参数过多必须分行，第一个参数紧接左圆括号，第二个参数换行与第一个参数对齐，以此类推。
+ `if()` 判断语句和 `else` 都单独在1行
+ 大括号换行，每个大括号都单独在1行
+ lambda 表达式的大括号，如果表达式语句非常简单，lambda表达式可以写在一行，否则左右大括号都独占1行。
+ `bool` 表达式较长时，逻辑运算符始终位于最右侧，每个条件换行并对齐
+ 类的构造函数初始化列表放在同1行，或者按4个缩进并空行
+ 不同逻辑语块之间空1行

```c++
// Foo.h
class Foo
{
public:
    Foo(double height, double score, int id, int age)
        : m_height(height), m_score(score), m_id(id), m_age(age)
    {}
public:
protected:
private:
protected:
private:
    
public:
protected:
private:
    double m_height;
    double m_score;
    int m_id;
    int m_age;
}

// Foo.cpp
void SetParameters(const int image_width,
                   const int image_height,
                   const int image_source_id,
                   const int image_table_position,
                   const int image_xray_angle,
                   ImageInfo image_info)
{
    if (first_condition &&
        second_condition && 
        third_condition)
    {
        ...;  
    }
    else
    {
        ....;
    }
}
```

### 括号

+ `return ` 表达式不要加上非必须圆括号；

+ 所有的大括号均不可省略；
+ `switch case` 每个 `case：` 语句都需要有大括号；
+ `if` `else` 语句只有一行，也需要有大括号；

## 变量

1. 变量声明即定义；
2. 变量声明的位置尽量紧挨着变量使用的位置，尽可能限制在较小的作用域；（构造较为复杂的对象除外）
3. 禁止使用类类型的全局变量；即除有必要，只允许使用POD类型的全局变量 (Plain Old Data)
4. 静态生存周期对象必须是POD类型；即包含全局变量，静态变量，静态成员函数，函数静态变量以及POD类型的指针，数组和结构体；
5. 使用 `constexpr` 进行常量初始化；
6. 使用 `<stdint.h>` 中的 `uint16_t`, `int16_t`, `int32_t`, `uint64_t`, `int64_t`， 禁止使用 `WORD` 等原生 `Windows` 类型；

## 函数

1. 禁止使用全局函数；如有必要可以在命名空间中使用非成员函数；
2. 函数参数：输入参数在先，输出函数在后，且输入参数一般是值或者 `const &`, 输出参数一般是指针；
3. 不使用异常；
4. 使用显示的类型转换，禁止使用C格式的类型转换；
5. 使用前置递增运算符；（`++i` 代替 `i++`）
6. 在可能使用 `const` 地方使用 `const` ;

## class

1. 禁止在构造函数中调用自身的虚函数；

2. 析构函数使用虚函数；

3. 所有的数据成员均需要设置为 `private` ;

4. 除非必要，否则不要使用静态成员函数；（静态成员函数应当和类的示例或静态数据紧密相关）

5. 构造函数，使用初始化列表初始化类成员变量；初始化复杂的构造函数，或者读取数据时，考虑使用 `Init()` 函数； 

6. 如果类需要支持拷贝和移动操作，需要显示定义拷贝构造函数和移动构造函数，拷贝/赋值，移动/赋值成对出现，如果确定不需要需要将这两个函数显示delete；

7. 禁止为基类提供拷贝/赋值操作，需要显示禁用；

8. 仅有数据成员时使用 `struct` 其他一律使用 `class` ; (`struct` 注意可能发生的字节对齐的问题)

9. 对于重载的虚函数或虚析构函数，使用`virtual` 和 `override` 关键字

10. 除非有必要使用继承，继承层次不要超过3级，可使用组合替代；

11. 除非有必要，禁止使用多重继承；（纯接口类除外，除继承的第一个类外，其他类必须是纯接口类）

12. 声明顺序

    ```c++
    class Foo
    {
    public:
    protected:
    private:
        
    public:
    protected:
    private:
    };
    ```

## 头文件

1. `.cpp` 与 `.h` 文件一一对应；`main()` 和 `UT` 测试可以例外

2. 通过 `#define ` 保护，禁止使用 `#pragma once` ；

   ```c++
   #ifndef <PROJECT>_<PATH>_<FILE>_H_
   #define <PROJECT>_<PATH>_<FILE>_H_
   ...
   #endif // <PROJECT>_<PATH>_<FILE>_H_    
   ```

3. 除模板在`h`文件中定义外，其他函数均在 `cpp`文件中定义；

4. 在头文件中禁止使用前置声明；

5. 只有当函数体小于10行时才可以使用内联函数；

6. `#include` 头文件路径顺序

   禁止使用相对目录

   ```c++
   // Current Cpp header
   // C/C++ standard library heaeder 
   // system library header
   // External library header
   // Current Module Cpp header
   // Root of project header
   ```

 	7.  `#include ` 头文件包含规则
     + 所有的源文件和头文件都可以包含根目录 `include` 文件夹下的头文件
     + `Common` 目录下各个模块，只能包含当前模块的头文件和根目录下的头文件，如果需要第三方库，可以包含第三方库的头文件；
     + `App` 目录下各个模块相互独立，每个模块只能包含当前模块下的头文件，Common目录下的各个模块的头文件，以及根目录下的头文件；

## 一些其他建议和约定

1. 除非必要，不要使用宏， 使用内敛函数，枚举和常量代替；
2. 尽量多实用 `auto` ;
3. 考虑使用 `lambda` 表达式；



# 附录

+ 基于 Clang-Format的代码格式配置 

+ 基于 vs code的代码格式配置

+ 静态代码检查工具配置

  

































