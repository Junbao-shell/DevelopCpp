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

// Current Cpp header
#include "configure.h"
// C/C++ standard library header
#include <string>
#include <iostream>
#include <vector>
// External library header
#include <tinyxml2.h>
#include <glog/logging.h>
// current module header
#include "parameters.h"

using namespace tinyxml2;

Configure::Configure(const std::string &file)
{
    VLOG(4) << "call one constructor";
    m_doc = new XMLDocument();
    m_doc->Clear();
    if (XML_SUCCESS != m_doc->LoadFile(file.c_str()))
    {
        LOG(ERROR) << "load xml file error.";
    }
    
    m_rootElement = m_doc->RootElement();
    if (nullptr == m_rootElement)
    {
        LOG(ERROR) << "open the xml file error.";
    }
}

Configure::~Configure()
{
    m_doc->Clear();
}

int Configure::GetIntElement(const std::string &path)
{
    return atoi(GetNode(path)->GetText());
}

double Configure::GetDoubleElement(const std::string &path)
{
    return atof(GetNode(path)->GetText());
}

std::string Configure::GetStringElement(const std::string &path)
{
    return GetNode(path)->GetText();
}

bool Configure::GetBoolElement(const std::string &path)
{
    return atoi(GetNode(path)->GetText());
}

bool Configure::ReadConfig(const std::string &xml_path, XmlStruct &xml_struct)
{
    const std::string scatter_protocol_node_path = "ScatterCorrectProtocol";
    xml_struct.scatter_correct_protocol = GetIntElement(scatter_protocol_node_path);

    // write bowtie flag
    const std::string bowtie_flag_node_path = "BowtieFlag";
    xml_struct.bowtie_flag = GetBoolElement(bowtie_flag_node_path);

    // write air image dir
    const std::string air_node_path = "AirImageDir";
    xml_struct.air_image_dir = GetStringElement(air_node_path);

    // write shape function dir
    const std::string shape_function_node_path = "ShapeFunctionDir";
    xml_struct.shape_function_dir = GetStringElement(shape_function_node_path);

    // write the source center offset value 
    const std::string source_node_path = "SourceOffsetID/ID";
    for (int i = 0; i < 24; ++i)
    {
        std::string source_id_node_path = source_node_path + std::to_string(i + 1);
        xml_struct.source_offset_ID[i] = GetIntElement(source_id_node_path);
    }

    // write the Scatter configure parameters
    const std::string scatter_config_node_path = "ScatterParameters/";
    {
        std::string mul_node_path = scatter_config_node_path + "mul";
        xml_struct.scatter_config.mul = GetDoubleElement(mul_node_path);
        std::string A1_node_path = scatter_config_node_path + "A1";
        xml_struct.scatter_config.A1 = GetDoubleElement(A1_node_path);
        std::string A2_node_path = scatter_config_node_path + "A2";
        xml_struct.scatter_config.A2 = GetDoubleElement(A2_node_path);
        std::string A3_node_path = scatter_config_node_path + "A3";
        xml_struct.scatter_config.A3 = GetDoubleElement(A3_node_path);
        std::string B1_node_path = scatter_config_node_path + "B1";
        xml_struct.scatter_config.B1 = GetDoubleElement(B1_node_path);
        std::string B2_node_path = scatter_config_node_path + "B2";
        xml_struct.scatter_config.B2 = GetDoubleElement(B2_node_path);
        std::string B3_node_path = scatter_config_node_path + "B3";
        xml_struct.scatter_config.B3 = GetDoubleElement(B3_node_path);
        std::string alpha1_node_path = scatter_config_node_path + "alpha1";
        xml_struct.scatter_config.alpha1 = GetDoubleElement(alpha1_node_path);
        std::string alpha2_node_path = scatter_config_node_path + "alpha2";
        xml_struct.scatter_config.alpha2 = GetDoubleElement(alpha2_node_path);
        std::string alpha3_node_path = scatter_config_node_path + "alpha3";
        xml_struct.scatter_config.alpha3 = GetDoubleElement(alpha3_node_path);
        std::string beta1_node_path = scatter_config_node_path + "beta1";
        xml_struct.scatter_config.beta1 = GetDoubleElement(beta1_node_path);
        std::string beta2_node_path = scatter_config_node_path + "beta2";
        xml_struct.scatter_config.beta2 = GetDoubleElement(beta2_node_path);
        std::string beta3_node_path = scatter_config_node_path + "beta3";
        xml_struct.scatter_config.beta3 = GetDoubleElement(beta3_node_path);
        std::string asca_node_path = scatter_config_node_path + "asca";
        xml_struct.scatter_config.asca = GetDoubleElement(asca_node_path);
        std::string tp1_node_path = scatter_config_node_path + "tp1";
        xml_struct.scatter_config.tp1 = GetIntElement(tp1_node_path);
        std::string tp2_node_path = scatter_config_node_path + "tp2";
        xml_struct.scatter_config.tp2 = GetIntElement(tp2_node_path);
        std::string tp3_node_path = scatter_config_node_path + "tp3";
        xml_struct.scatter_config.tp3 = GetIntElement(tp3_node_path);
        std::string tp4_node_path = scatter_config_node_path + "tp4";
        xml_struct.scatter_config.tp4 = GetIntElement(tp4_node_path);
        std::string tp5_node_path = scatter_config_node_path + "tp5";
        xml_struct.scatter_config.tp5 = GetIntElement(tp5_node_path);
    }

    return true;
}

bool Configure::WriteConfig(const std::string &xml_path, const XmlStruct &xml_struct)
{
    // check the data
    if (xml_struct.scatter_correct_protocol != 1 || xml_struct.scatter_correct_protocol != 3)
    {
        LOG(ERROR) << "the scatter configure of the protocol is illeagl, only can be set 1 or 3";
    }

    if (xml_struct.bowtie_flag != 0 || xml_struct.bowtie_flag != 1)
    {
        LOG(ERROR) << "the scatter configure of bowtie flag is illeagl, this is a bool value";
    }

    // write air image dir
    const std::string scatter_protocol_node_path = "ScatterCorrectProtocol";
    WriteConfig(xml_path, scatter_protocol_node_path, std::to_string(xml_struct.scatter_correct_protocol));

    // write bowtie flag
    const std::string bowtie_flag_node_path = "BowtieFlag";
    WriteConfig(xml_path, bowtie_flag_node_path, std::to_string(xml_struct.bowtie_flag));

    // write air image dir
    const std::string air_node_path = "AirImageDir";
    WriteConfig(xml_path, air_node_path, xml_struct.air_image_dir);

    // write shape function dir
    const std::string shape_function_node_path = "ShapeFunctionDir";
    WriteConfig(xml_path, shape_function_node_path, xml_struct.shape_function_dir);

    // write the source center offset value 
    const std::string source_node_path = "SourceOffsetID/ID";
    for (int i = 0; i < 24; ++i)
    {
        std::string source_id_node_path = source_node_path + std::to_string(i + 1);
        WriteConfig(xml_path, source_id_node_path, std::to_string(xml_struct.source_offset_ID[i]));
    }

    // write the Scatter configure parameters
    const std::string scatter_config_node_path = "ScatterParameters/";
    {
        std::string mul_node_path = scatter_config_node_path + "mul";
        WriteConfig(xml_path, mul_node_path, std::to_string(xml_struct.scatter_config.mul));
        std::string A1_node_path = scatter_config_node_path + "A1";
        WriteConfig(xml_path, A1_node_path, std::to_string(xml_struct.scatter_config.A1));
        std::string A2_node_path = scatter_config_node_path + "A2";
        WriteConfig(xml_path, A2_node_path, std::to_string(xml_struct.scatter_config.A2));
        std::string A3_node_path = scatter_config_node_path + "A3";
        WriteConfig(xml_path, A3_node_path, std::to_string(xml_struct.scatter_config.A3));
        std::string B1_node_path = scatter_config_node_path + "B1";
        WriteConfig(xml_path, B1_node_path, std::to_string(xml_struct.scatter_config.B1));
        std::string B2_node_path = scatter_config_node_path + "B2";
        WriteConfig(xml_path, B2_node_path, std::to_string(xml_struct.scatter_config.B2));
        std::string B3_node_path = scatter_config_node_path + "B3";
        WriteConfig(xml_path, B3_node_path, std::to_string(xml_struct.scatter_config.B3));
        std::string alpha1_node_path = scatter_config_node_path + "alpha1";
        WriteConfig(xml_path, alpha1_node_path, std::to_string(xml_struct.scatter_config.alpha1));
        std::string alpha2_node_path = scatter_config_node_path + "alpha2";
        WriteConfig(xml_path, alpha2_node_path, std::to_string(xml_struct.scatter_config.alpha2));
        std::string alpha3_node_path = scatter_config_node_path + "alpha3";
        WriteConfig(xml_path, alpha3_node_path, std::to_string(xml_struct.scatter_config.alpha3));
        std::string beta1_node_path = scatter_config_node_path + "beta1";
        WriteConfig(xml_path, beta1_node_path, std::to_string(xml_struct.scatter_config.beta1));
        std::string beta2_node_path = scatter_config_node_path + "beta2";
        WriteConfig(xml_path, beta2_node_path, std::to_string(xml_struct.scatter_config.beta2));
        std::string beta3_node_path = scatter_config_node_path + "beta3";
        WriteConfig(xml_path, beta3_node_path, std::to_string(xml_struct.scatter_config.beta3));
        std::string asca_node_path = scatter_config_node_path + "asca";
        WriteConfig(xml_path, asca_node_path, std::to_string(xml_struct.scatter_config.asca));
        std::string tp1_node_path = scatter_config_node_path + "tp1";
        WriteConfig(xml_path, tp1_node_path, std::to_string(xml_struct.scatter_config.tp1));
        std::string tp2_node_path = scatter_config_node_path + "tp2";
        WriteConfig(xml_path, tp2_node_path, std::to_string(xml_struct.scatter_config.tp2));
        std::string tp3_node_path = scatter_config_node_path + "tp3";
        WriteConfig(xml_path, tp3_node_path, std::to_string(xml_struct.scatter_config.tp3));
        std::string tp4_node_path = scatter_config_node_path + "tp4";
        WriteConfig(xml_path, tp4_node_path, std::to_string(xml_struct.scatter_config.tp4));
        std::string tp5_node_path = scatter_config_node_path + "tp5";
        WriteConfig(xml_path, tp5_node_path, std::to_string(xml_struct.scatter_config.tp5));
    }

    return true;
}

bool Configure::WriteConfig(const std::string &xml_path, const std::string &node_path, const std::string &value)
{
    auto node = GetNode(node_path);
    if (nullptr == node)
    {
        LOG(ERROR) << "the node to write is null.";
        return false;
    }

    node->SetText(value.c_str());

    if (XML_SUCCESS != m_doc->SaveFile(xml_path.c_str()))
    {
        LOG(ERROR) << "modify the text not success.";
    }

    return true;
}

std::vector<std::string> Configure::GetNodePath(const std::string &node_path)
{
    std::vector<std::string> ret;
    auto size = node_path.size();
    int i = 0;
    // check the first char, if start with '/', delete
    if ('/' == node_path.at(0)) 
    {
        ++i;
    }
    for (; i < size;)
    {
        auto pos = node_path.find('/', i);
        if (std::string::npos == pos)
        {
            if(size != i)
            {
                ret.emplace_back(node_path.substr(i, size - i));
            }
            return ret;
        }
        ret.emplace_back(node_path.substr(i, (pos - i)));
        i = pos + 1;
    }
    return ret;
}

XMLElement* Configure::GetNode(const std::string &node_dir)
{
    auto node_name = GetNodePath(node_dir);
    auto node = m_rootElement;
    for (int i = 0; i < node_name.size(); ++i)
    {
        node = node->FirstChildElement(node_name.at(i).c_str());
    }

    return node;
}
