#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>
#include <c10/util/Excepton.h>
#endif

namespace c10 {
#if defined(_WIN32)
    inline std::wstring u8u16(const std::string& str) {
        if (str.empty()) {
            return std::wstring();
        }
        int size_needed = MultiByteToWideChar(
                CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), NULL, 0);
        TORCH_CHECK(size_needed > 0, "Error converting the content to Unicode");
        std::wstring wstr(size_needed, 0);
        MultiByteToWideChar(
                CP_UTF8,
                0,
                str.c_str(),
                static_cast<int>(str.size()),
                &wstr[0],
                size_needed);
        return wstr;
    }
#endif
}
