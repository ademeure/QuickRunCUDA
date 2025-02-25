#pragma once

#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

class IPCHelper {
public:
    static const std::string COMMAND_PIPE_NAME;
    static const std::string RESPONSE_PIPE_NAME;

    IPCHelper() {
        // Create named pipes if they don't exist
        if (mkfifo(COMMAND_PIPE_NAME.c_str(), 0666) == -1) {
            if (errno != EEXIST) {
                throw std::runtime_error("Failed to create command pipe: " + std::string(strerror(errno)));
            }
        }
        if (mkfifo(RESPONSE_PIPE_NAME.c_str(), 0666) == -1) {
            if (errno != EEXIST) {
                throw std::runtime_error("Failed to create response pipe: " + std::string(strerror(errno)));
            }
        }
    }

    ~IPCHelper() {
        // Clean up pipes
        unlink(COMMAND_PIPE_NAME.c_str());
        unlink(RESPONSE_PIPE_NAME.c_str());
    }

    bool waitForCommand(std::string& cmd) {
        int fd = open(COMMAND_PIPE_NAME.c_str(), O_RDONLY);
        if (fd == -1) return false;

        char buffer[4096];
        ssize_t bytes = read(fd, buffer, sizeof(buffer)-1);
        close(fd);

        if (bytes <= 0) return false;

        buffer[bytes] = '\0';
        cmd = std::string(buffer);
        return true;
    }

    bool sendResponse(const std::string& response) {
        int fd = open(RESPONSE_PIPE_NAME.c_str(), O_WRONLY);
        if (fd == -1) return false;

        write(fd, response.c_str(), response.length());
        close(fd);
        return true;
    }
};

const std::string IPCHelper::COMMAND_PIPE_NAME = "/tmp/quickruncuda_cmd";
const std::string IPCHelper::RESPONSE_PIPE_NAME = "/tmp/quickruncuda_resp";