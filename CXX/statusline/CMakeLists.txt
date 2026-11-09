add_executable(cc-statusline main.cpp)
target_link_libraries(cc-statusline PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
	target_compile_options(cc-statusline PRIVATE /W4 /utf-8)
else()
	target_compile_options(cc-statusline PRIVATE -Wall -Wextra -Wpedantic)
endif()

# MinGW (MSYS2 UCRT64/MINGW64): link the GCC runtime statically so the .exe
# runs from Claude Code without libstdc++-6.dll etc. on PATH.
if(MINGW)
	target_link_options(cc-statusline PRIVATE -static)
endif()

install(TARGETS cc-statusline RUNTIME DESTINATION bin)
