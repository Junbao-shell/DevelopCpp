[TOC]

# Install

```cmake
install(TARGET my_lib
        EXPORT my_lib_targets
        LIBRARY DESTINITION lib
        ARCHIVE DESTINITION lib
        RUNTIME DESTINITION bin
        PUBLIC_HEADER DESTINITION include
)
```

