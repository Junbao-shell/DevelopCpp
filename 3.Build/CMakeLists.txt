# 3.Build/CMakeLists.txt

function(normal_variable var1)
	message(STATUS "var1 in function: ${var1}")
	set(var1 "666" PARENT_SCOPE)
    message(STATUS "var1 in function: ${var1}")
endfunction(normal_variable)

set(var1 "888")
message(STATUS "var1 in main: ${var1}")
normal_Variable(${var1})
message(STATUS "var1 in main: ${var1}")

set(vars "xxx" CACHE STRING INTERNAL)
message(STATUS "vars = ${vars}")
set(vars "ttt")
message(STATUS "vars = ${vars}")