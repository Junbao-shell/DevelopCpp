///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief self string module
/// 
/// @file istring.cpp
/// 
/// @author GaoJunbao(junbao.gao@nanovision.com.cn)
/// 
/// @date 2022-05-12
///////////////////////////////////////////////////////////

#ifndef __SOFTWARE_CONFIG_MATH_INCLUDE_ISTRING_H_
#define __SOFTWARE_CONFIG_MATH_INCLUDE_ISTRING_H_

// System header
// C/C++ standard library header
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>
// External library header
#include <cuda_runtime.h>
// Current module header
// Root directory header

class iString
{
public:
    static inline void ToLower(std::string &str)
    {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    }

    static inline std::string ToLower(const std::string &str)
    {
        std::string res = "";
        res.resize(str.size());
        std::transform(str.begin(), str.end(), res.begin(), ::tolower);

        return res;
    }

    static inline void ToUpper(std::string &str)
    {
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    }

    static inline std::string ToUpper(const std::string &str)
    {
        std::string res = "";
        res.resize(str.size());
        std::transform(str.begin(), str.end(), res.begin(), ::toupper);
    
        return res;
    }

    static inline void CapitalLetters(std::string &str)
    {
        ToLower(str);
        str.at(0) -= 32;
    }

    static inline std::string CapitalLetters(const std::string &str)
    {
        std::string res = "";
        res.resize(str.size());
        res = ToLower(str);
        res.at(0) -= 32;
        return res;
    }

    static inline void EraseBothSideSpace(std::string &str)
    {
        str.erase(0, str.find_first_not_of(' '));
        str.erase(str.find_last_not_of(' ') + 1);
    }

    static inline std::vector<std::string> SplitString(const std::string &str, const char split)
    {
        std::vector<std::string> ret{};
        const int size = str.size();
        
        std::string substr = "";
        int i = 0;
        if (str.front() == split)
        {
            ++i;
        }
        for (; i < size;)
        {
            auto pos = str.find(split, i);
            if (std::string::npos == pos)
            {
                ret.emplace_back(str.substr(i, size - i));
                return ret;
            }
            ret.emplace_back(str.substr(i, (pos - i)));
            i = pos + 1;
        } 
        return ret;
    }

    static inline std::vector<std::string> SplitStringSpace(const std::string &str)
    {
        std::vector<std::string> res{};

        std::string temp;
        std::stringstream stream(str);
        while (stream >> temp)
        {
            res.emplace_back(temp);
        } 
        return res;
    }

    static inline std::vector<std::string> SplitString(const std::string &str)
    {
        std::vector<std::string> res{};

        // split by ',' first
        if (std::string::npos != str.find(','))
        {
            res = SplitString(str, ',');
            for (size_t i = 0; i < res.size(); ++i)
            {
                res.at(i).erase(0, res.at(i).find_first_not_of(' '));
                res.at(i).erase(res.at(i).find_last_not_of(' ') + 1);
            }
        }
        else
        {
            res = SplitStringSpace(str);
        }

        return res;
    }

    template<typename T>
    static inline std::string PrintArray(std::vector<T> arr)
    {
        std::stringstream stream;
        const int size = static_cast<int>(arr.size());
        int num_width = static_cast<int>(std::log10(*std::max_element(arr, arr + size))) + 1;
        num_width += 2;

        for (int i = 0; i < (size - 1); ++i)
        {
            stream << std::setw(num_width) << arr.at(i);
            stream << ' ';
        }
        stream << std::setw(num_width) << arr.back();
        return stream.str();
    }

    template<typename T>
    static inline std::string PrintArray(T *arr, const int size)
    {
        std::stringstream stream;
        int num_width = static_cast<int>(std::log10(*std::max_element(arr, arr + size))) + 1;
        num_width += 2;

        for (int i = 0; i < (size - 1); ++i)
        {
            stream << std::setw(num_width) << arr[i];
            stream << ' ';
        }
        stream << std::setw(num_width) << arr[size - 1];
        return stream.str();
    }

    template<typename T>
    static inline std::string PrintArray(T *arr, const int dimx, const int dimy)
    {
        std::stringstream stream;
        int num_width = static_cast<int>(std::log10(*std::max_element(arr, arr + dimx * dimy))) + 1;
        num_width += 2;

        for (int i = 0; i < dimy; ++i)
        {
            for (int j = 0, index = i * dimx; j < (dimx - 1); ++j, ++index)
            {
                stream << std::setw(num_width) << arr[index];
                stream << ' ';
            }
            stream << std::setw(num_width) << arr[i * dimx + dimx - 1];
            stream << "\n";
        }

        return stream.str();
    }

    template<typename T>
    static inline std::string PrintGpuArray(T *arr, const int size)
    {
        std::stringstream stream;
        
        T *h_arr;
        cudaMallocHost((void**)&h_arr, sizeof(T) * size);
        cudaMemcpy(h_arr, arr, sizeof(T) * size, cudaMemcpyDeviceToHost);

        int num_width = static_cast<int>(std::log10(*std::max_element(arr, arr + size))) + 1;
        num_width += 2;

        for (int i = 0; i < (size - 1); ++i)
        {
            stream << std::setw(num_width) << h_arr[i];
            stream << ' ';
        }
        stream << std::setw(num_width) << h_arr[size - 1];

        cudaFreeHost(h_arr);
        h_arr = nullptr;
        
        return stream.str();
    }

    template<typename T>
    static inline std::string PrintGpuArray(T *arr, const int dimx, const int dimy)
    {
        std::stringstream stream;
        const int size = dimx * dimy;

        T *h_arr;
        cudaMallocHost((void**)&h_arr, sizeof(T) * size);
        cudaMemcpy(h_arr, arr, sizeof(T) * size, cudaMemcpyDeviceToHost);

        int num_width = static_cast<int>(std::log10(*std::max_element(h_arr, h_arr + size))) + 1;
        num_width += 2;

        for (int i = 0; i < dimy; ++i)
        {
            for (int j = 0, index = i * dimx; j < (dimx - 1); ++j, ++index)
            {
                stream << std::setw(num_width) << h_arr[index];
                stream << ' ';
            }
            stream << std::setw(num_width) << h_arr[i * dimx + dimx - 1];
            stream << "\n";
        }

        cudaFreeHost(h_arr);
        h_arr = nullptr;
        
        return stream.str();
    }

protected:
private:

public:
protected:
private:
};

#endif // __SOFTWARE_CONFIG_MATH_INCLUDE_ISTRING_H_

