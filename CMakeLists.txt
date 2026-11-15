cmake_minimum_required(VERSION 3.14)
project(scissor VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ======================================================
# Minimum compiled-in log level
# 0=DEBUG ... 5=FATAL, default 2=INFO
# ======================================================
set(SCISSOR_MIN_LOG_LEVEL "2" CACHE STRING
    "Minimum compiled-in log level (0=DEBUG, 1=PERF, 2=INFO, 3=WARN, 4=ERROR, 5=FATAL)")

# ======================================================
# User-controlled option flags
# ======================================================
option(SCISSOR_ENABLE_TESTS         "Build test executables"                         ON)
option(SCISSOR_ENABLE_TOOLS         "Build scissor-profile CLI tool"                 ON)
option(SCISSOR_USE_SYSTEM_LIBURING  "Use a system liburing (>= 2.6) found by pkg-config" ON)

find_package(Threads REQUIRED)

# ======================================================
# liburing >= 2.6
#   - If pkg-config finds a compatible install, that is used.
#   - Otherwise liburing is built from source:
#       https://github.com/axboe/liburing.git  @  liburing-2.6
#     The static archive is linked and installed alongside
#     scissor so the installed package is self-contained.
# ======================================================
set(SCISSOR_LIBURING_FROM_SYSTEM FALSE)
if(SCISSOR_USE_SYSTEM_LIBURING)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_liburing QUIET IMPORTED_TARGET "liburing>=2.6")
        if(PC_liburing_FOUND)
            set(SCISSOR_LIBURING_FROM_SYSTEM TRUE)
            message(STATUS "Found liburing ${PC_liburing_VERSION} via pkg-config")
        endif()
    endif()
endif()

if(NOT SCISSOR_LIBURING_FROM_SYSTEM)
    message(STATUS "liburing >= 2.6 not found, building from source (liburing-2.6)")
    include(ExternalProject)
    set(LIBURING_INSTALL_DIR "${CMAKE_BINARY_DIR}/liburing/install")
    ExternalProject_Add(liburing_ep
        GIT_REPOSITORY    https://github.com/axboe/liburing.git
        GIT_TAG           liburing-2.6
        GIT_SHALLOW       TRUE
        PREFIX            "${CMAKE_BINARY_DIR}/liburing"
        CONFIGURE_COMMAND ./configure --prefix=${LIBURING_INSTALL_DIR} --libdir=${LIBURING_INSTALL_DIR}/lib
        BUILD_COMMAND     make -C src
        INSTALL_COMMAND   make -C src install
        UPDATE_COMMAND    ""
        BUILD_IN_SOURCE   1
        BUILD_BYPRODUCTS  "${LIBURING_INSTALL_DIR}/lib/liburing.a"
    )
    # The include directory must exist at configure time for the
    # INTERFACE_INCLUDE_DIRECTORIES check.
    file(MAKE_DIRECTORY "${LIBURING_INSTALL_DIR}/include")
endif()

# ======================================================
# Default config directory: installed to
# ${prefix}/etc/scissor/ so the Regressor can locate
# models.conf without user intervention.
# ======================================================
set(SCISSOR_DEFAULT_CONFIG_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_SYSCONFDIR}/scissor"
    CACHE PATH "Default directory searched for models.conf")

# ======================================================
# Library sources
# ======================================================
set(SCISSOR_SOURCES
    src/aggregator.cpp
    src/regressor.cpp
    src/loader.cpp
    src/common/host_cache.cpp
    src/reader/liburing_reader.cpp
    src/reader/posix_reader.cpp
    src/reader/mmap_reader.cpp
)

add_library(scissor ${SCISSOR_SOURCES})
add_library(scissor::scissor ALIAS scissor)

set_target_properties(scissor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(scissor PRIVATE -Wall)

target_compile_definitions(scissor PUBLIC
    SCISSOR_MIN_LOG_LEVEL=${SCISSOR_MIN_LOG_LEVEL}
)

target_compile_definitions(scissor PRIVATE
    SCISSOR_DEFAULT_CONFIG_DIR="${SCISSOR_DEFAULT_CONFIG_DIR}"
)

target_include_directories(scissor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(scissor PUBLIC Threads::Threads)

if(SCISSOR_LIBURING_FROM_SYSTEM)
    target_link_libraries(scissor PUBLIC PkgConfig::PC_liburing)
else()
    add_dependencies(scissor liburing_ep)
    target_include_directories(scissor PUBLIC
        $<BUILD_INTERFACE:${LIBURING_INSTALL_DIR}/include>
    )
    target_link_libraries(scissor PUBLIC
        $<BUILD_INTERFACE:${LIBURING_INSTALL_DIR}/lib/liburing.a>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/liburing.a>
    )
endif()

# ======================================================
# Tests
# ======================================================
if(SCISSOR_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ======================================================
# Tools (scissor-profile CLI)
# ======================================================
if(SCISSOR_ENABLE_TOOLS)
    add_subdirectory(tools)
endif()

# ======================================================
# Install: library, headers, default models.conf
# ======================================================
install(TARGETS scissor EXPORT scissorTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES cmake/models.conf
    DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/scissor)

if(NOT SCISSOR_LIBURING_FROM_SYSTEM)
    install(FILES ${LIBURING_INSTALL_DIR}/lib/liburing.a
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY ${LIBURING_INSTALL_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# ======================================================
# Install: CMake export and config files
# ======================================================
install(EXPORT scissorTargets
    FILE        scissorTargets.cmake
    NAMESPACE   scissor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scissor)

export(EXPORT scissorTargets
    NAMESPACE scissor::
    FILE scissorTargets.cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/scissorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/scissorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scissor
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/scissorConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/scissorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/scissorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scissor)
