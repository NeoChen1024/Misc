cmake_minimum_required(VERSION 3.25)
project(MiscCXX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 預設安裝到 ~/.local（執行檔進 ~/.local/bin）
# Windows 用 %USERPROFILE%：MSYS2 的 $HOME 是 /home/<user>，不是 Windows 的家目錄
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	if(WIN32 AND DEFINED ENV{USERPROFILE})
		file(TO_CMAKE_PATH "$ENV{USERPROFILE}/.local" _prefix)
	else()
		set(_prefix "$ENV{HOME}/.local")
	endif()
	set(CMAKE_INSTALL_PREFIX "${_prefix}" CACHE PATH "Install prefix" FORCE)
endif()

# nlohmann-json：只要 header-only target，不要測試和它自己的 install 規則
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(3rdparty/nlohmann-json EXCLUDE_FROM_ALL)

add_subdirectory(statusline)
