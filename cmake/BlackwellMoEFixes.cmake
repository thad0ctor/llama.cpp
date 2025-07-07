#
# CMake configuration for Unified Blackwell MoE System
# Handles compilation of integrated MoE optimizations
#

# Option to enable unified Blackwell MoE system
option(BLACKWELL_UNIFIED_MOE "Enable unified Blackwell MoE system for large models" ON)

if(BLACKWELL_UNIFIED_MOE)
    message(STATUS "Unified Blackwell MoE system enabled")
    
    # Add unified MoE headers
    set(BLACKWELL_MOE_HEADERS
        blackwell_moe_config.h
    )
    
    # Define unified MoE definitions
    set(BLACKWELL_MOE_DEFINITIONS
        -DBLACKWELL_UNIFIED_MOE_ENABLED=1
    )
    
    # Architecture-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND BLACKWELL_MOE_DEFINITIONS
            -DBLACKWELL_MOE_X64=1
        )
    endif()
    
    # Check for CUDA support for GPU-accelerated MoE processing
    if(GGML_CUDA)
        message(STATUS "CUDA detected - enabling GPU-accelerated unified MoE system")
        list(APPEND BLACKWELL_MOE_DEFINITIONS
            -DBLACKWELL_MOE_CUDA=1
        )
        
        # Add CUDA-specific MoE optimizations
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
            list(APPEND BLACKWELL_MOE_DEFINITIONS
                -DBLACKWELL_MOE_CUDA_12=1
            )
        endif()
    endif()
    
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND BLACKWELL_MOE_DEFINITIONS
            -DBLACKWELL_MOE_OPTIMIZED=1
        )
    endif()
    
else()
    message(STATUS "Unified Blackwell MoE system disabled")
    set(BLACKWELL_MOE_HEADERS "")
    set(BLACKWELL_MOE_DEFINITIONS "")
endif()

# Function to add unified MoE system to a target
function(add_unified_blackwell_moe target)
    if(BLACKWELL_UNIFIED_MOE)
        target_compile_definitions(${target} PRIVATE ${BLACKWELL_MOE_DEFINITIONS})
        
        # Add include directory for MoE headers
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        
        # Link with math library if needed
        if(UNIX AND NOT APPLE)
            target_link_libraries(${target} PRIVATE m)
        endif()
        
        message(STATUS "Added unified Blackwell MoE system to target: ${target}")
    endif()
endfunction()

# Export variables for use in main CMakeLists.txt
set(BLACKWELL_MOE_HEADERS ${BLACKWELL_MOE_HEADERS} PARENT_SCOPE)
set(BLACKWELL_MOE_DEFINITIONS ${BLACKWELL_MOE_DEFINITIONS} PARENT_SCOPE)

# Backward compatibility aliases (deprecated)
set(BLACKWELL_MOE_SOURCES "" PARENT_SCOPE)  # No separate sources needed
function(add_blackwell_moe_fixes target)
    add_unified_blackwell_moe(${target})
endfunction() 