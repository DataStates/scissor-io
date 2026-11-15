cmake_minimum_required(VERSION 3.14)

# ======================================================
# scissor-profile: CLI tool for measuring I/O parameters
# and emitting a tuned models.conf.
# ======================================================

add_executable(scissor-profile scissor_profile.cpp)
target_link_libraries(scissor-profile PRIVATE scissor)
set_target_properties(scissor-profile PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)

install(TARGETS scissor-profile
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(PROGRAMS scissor_fit.py
    DESTINATION ${CMAKE_INSTALL_BINDIR})
