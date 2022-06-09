///////////////////////////////////////////////////////////
/// @copyright copyright description
///
/// @brief Configure module only for the scatter class 
///
/// @file configure.cpp
/// 
/// @author author
///
/// @date 2022-01-10
///////////////////////////////////////////////////////////

#ifndef __CUDACROSSSCATTER_INCLUDE_CONFIGURE_H_
#define __CUDACROSSSCATTER_INCLUDE_CONFIGURE_H_

// C/C++ standard library header
#include <string>
#include <vector>
// current module header
#include "parameters.h"
// External library header
#include <tinyxml2.h>

using namespace tinyxml2;

class Configure 
{
public:
    Configure(const std::string &path);
    Configure(const Configure &) = delete;
    Configure &operator=(const Configure &) = delete;
    ~Configure();

    /**
     * @brief writer the string to the xml file, 
     * only support write the string data type
     * note: only support write to the node that in the xml 
     * 
     * @return write success or not
     */
    int GetIntElement(const std::string &path);
    double GetDoubleElement(const std::string &path);
    std::string GetStringElement(const std::string &path);
    bool GetBoolElement(const std::string &path);
    
    bool ReadConfig(const std::string &xml_path, XmlStruct&);
    bool WriteConfig(const std::string &xml_path, const XmlStruct&);
    bool WriteConfig(const std::string &xml_path, const std::string &node_path, const std::string &str);

    // temp function for read all the xml value quickly
    int ReadScatterParameters(ScatterConfig &config, bool bowtieFlag);

private:
    /**
     * @brief get target node according the absolute node path
     * 
     * @param node_path 
     * @return std::vector<std::string> 
     */
    std::vector<std::string> GetNodePath(const std::string &node_path);
    XMLElement* GetNode(const std::string &node_path);
    void StringToDouble();
    void StringToVector();

private:
    XMLDocument *m_doc;
    XMLElement *m_rootElement;
};

#endif // __CUDACROSSSCATTER_INCLUDE_CONFIGURE_H_
