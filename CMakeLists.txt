cmake_minimum_required(VERSION 3.5)

project(CONFIG)

set(PROJECT_SUB1 ${CMAKE_SOURCE_DIR}/1.SoftwareConfig/)
set(PROJECT_SUB2 ${CMAKE_SOURCE_DIR}/2.CodingStyle/)
set(PROJECT_SUB3 ${CMAKE_SOURCE_DIR}/3.ToolChain/)
set(PROJECT_SUB4 ${CMAKE_SOURCE_DIR}/4.Shell/)
set(PROJECT_SUB5 ${CMAKE_SOURCE_DIR}/5.DesignModel/)
set(PROJECT_SUB6 ${CMAKE_SOURCE_DIR}/6.ThirdParty/)

# subdirectory
# add_subdirectory(1.SoftwareConfig)
# add_subdirectory(2.CodingStyle)
# add_subdirectory(3.ToolChain)
# add_subdirectory(4.Shell)
# add_subdirectory(5.DesignModel)
add_subdirectory(6.ThirdParty)
