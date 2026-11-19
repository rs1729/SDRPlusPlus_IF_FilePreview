#pragma once

#pragma once
#include <stdint.h>
#include <string.h>
#include <fstream>
#include <mutex>
#include <atomic>

inline float wavU8ToFloat(uint8_t x) {
    return ((float)x - 128.0f) / 128.0f; // B
    // return ((float)x - 127.5f) / 127.5f; // A
    // return ((float)x - 127.4f) / 128.0f; // C
}

#define WAV_SIGNATURE       "RIFF"
#define WAV_TYPE            "WAVE"
#define WAV_FORMAT_MARK     "fmt "
#define WAV_DATA_MARK       "data"
#define WAV_SAMPLE_TYPE_PCM 1

class WavReader {
public:
    WavReader(std::string path) {
        file = std::ifstream(path.c_str(), std::ios::binary);
        file.read((char*)&hdr, sizeof(WavHeader_t));
        valid = false;
        if (memcmp(hdr.signature, "RIFF", 4) != 0) { return; }
        if (memcmp(hdr.fileType, "WAVE", 4) != 0) { return; }
        valid = true;
    }

    uint16_t getBitDepth() {
        return hdr.bitDepth;
    }

    uint16_t getChannelCount() {
        return hdr.channelCount;
    }

    uint32_t getSampleRate() {
        return hdr.sampleRate;
    }

    bool isValid() {
        return valid;
    }

    void readSamples(void* data, size_t size) {
        std::lock_guard<std::mutex> lck(fileMtx);
        char* _data = (char*)data;
        file.read(_data, size);
        int read = file.gcount();
        if (read < size) {
            file.clear();
            file.seekg(sizeof(WavHeader_t));
            file.read(&_data[read], size - read);
            dataPos = size - read;
        }
        else {
            dataPos += size;
        }
        bytesRead += size;
    }

    void rewind() {
        std::lock_guard<std::mutex> lck(fileMtx);
        file.clear();
        file.seekg(sizeof(WavHeader_t));
        dataPos = 0;
    }

    // File Preview
    void seekData(uint64_t offset) {
        std::lock_guard<std::mutex> lck(fileMtx);
        file.clear();
        file.seekg(sizeof(WavHeader_t) + offset);
        dataPos = offset;
    }

    // File Preview
    uint64_t getDataPos() {
        return dataPos;
    }

    void close() {
        file.close();
    }

private:
    struct WavHeader_t {
        char signature[4];           // "RIFF"
        uint32_t fileSize;           // data bytes + sizeof(WavHeader_t) - 8
        char fileType[4];            // "WAVE"
        char formatMarker[4];        // "fmt "
        uint32_t formatHeaderLength; // Always 16
        uint16_t sampleType;         // PCM (1)
        uint16_t channelCount;
        uint32_t sampleRate;
        uint32_t bytesPerSecond;
        uint16_t bytesPerSample;
        uint16_t bitDepth;
        char dataMarker[4]; // "data"
        uint32_t dataSize;
    };

    bool valid = false;
    std::ifstream file;
    size_t bytesRead = 0;
    std::mutex fileMtx;
    std::atomic<uint64_t> dataPos{ 0 };
    WavHeader_t hdr;
};
