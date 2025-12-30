# dependencies

find_package(Threads REQUIRED)

# flags

llama_add_compile_flags()

# tools (stripped down: server + quantize only)

if (NOT EMSCRIPTEN)
    add_subdirectory(mtmd)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
endif()
