#pragma once
#include <imgui.h>
#include <fftw3.h>
#include <gui/style.h>
#include <wavreader.h>
#include <gui/gui.h>
#include <utils/opengl_include_code.h>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdint>

class FilePreview {
public:
    static constexpr int FFT_N = 4096;
    static constexpr int COLS = 256;
    static constexpr float MAX_BIAS_DB = 2.3f;
    static constexpr int MAX_ROWS = 1024;
    static constexpr int BLOCKS_PER_ROW = 8;
    static constexpr int HEADER_SIZE = 44;

    static double narrowCarrierGainDb() {
        return 10.0 * log10((double)FFT_N * 0.25 / 0.375) - MAX_BIAS_DB;
    }

    FilePreview() {
        fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * FFT_N);
        fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * FFT_N);

        plan = fftwf_plan_dft_1d(FFT_N, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);

        win.resize(FFT_N);
        winSumSq = 0.0f;
        for (int i = 0; i < FFT_N; i++) {
            win[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979f * (float)i / (float)FFT_N);
            winSumSq += win[i] * win[i];
        }
    }

    ~FilePreview() {
        stop();
        if (texture) { glDeleteTextures(1, &texture); }
        fftwf_destroy_plan(plan);
        fftwf_free(fftIn);
        fftwf_free(fftOut);
    }

    enum Format { FORMAT_S16 = 0, FORMAT_F32 = 1, FORMAT_U8 = 2 };

    void start(const std::string& filePath, int format, double sampleRate, bool dcRemove) {
        stop();
        path = filePath;
        sampleFormat = format;
        removeDC = dcRemove;
        scanKeyPath = filePath;
        scanKeyFormat = format;
        scanKeyDC = dcRemove;
        sr = std::max(sampleRate, 1.0);
        bytesPerFrame = (format == FORMAT_F32) ? 8 : ((format == FORMAT_U8) ? 2 : 4);
        totalFrames = 0;
        rows = 0;
        rowsDone = 0;
        lastTexRows = -1;
        error.clear();

        std::error_code ec;
        uint64_t fileSize = std::filesystem::file_size(path, ec);
        if (ec || fileSize <= HEADER_SIZE) {
            error = "Cannot read file";
            return;
        }
        totalFrames = (fileSize - HEADER_SIZE) / bytesPerFrame;
        if (totalFrames < FFT_N) {
            error = "File too short";
            totalFrames = 0;
            return;
        }
        rows = (int)std::min<uint64_t>(MAX_ROWS, totalFrames / FFT_N);
        db.assign((size_t)rows * COLS, -200.0f);
        cancel = false;
        worker = std::thread(&FilePreview::scan, this);
    }

    void stop() {
        cancel = true;
        if (worker.joinable()) { worker.join(); }
        scanKeyPath.clear();
    }

    bool isScanning(const std::string& filePath, int format, bool dcRemove) const {
        return scanKeyPath == filePath && scanKeyFormat == format && scanKeyDC == dcRemove;
    }

    bool hasData() const { return rows > 0 && totalFrames > 0; }
    uint64_t getTotalFrames() const { return totalFrames; }
    int getBytesPerFrame() const { return bytesPerFrame; }
    double getDuration() const { return (double)totalFrames / sr; }
    const std::string& getError() const { return error; }

    bool draw(double posFrac, double centerFreq, double refOffsetDb, double narrowBoostDb, bool& seekRequested, double& seekFrac) {
        seekRequested = false;
        if (!hasData()) {
            ImGui::TextWrapped("%s", error.empty() ? "Preview: no file." : error.c_str());
            return true;
        }

        updateTexture(refOffsetDb, narrowBoostDb);

        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();

        double dur = getDuration();
        static const double steps[] = { 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 21600, 43200, 86400 };
        double minPx = 36.0 * style::uiScale;
        double step = steps[(sizeof(steps) / sizeof(steps[0])) - 1];
        for (double s : steps) {
            if ((s / dur) * std::max<float>(avail.y, 10.0f) >= minPx) {
                step = s;
                break;
            }
        }
        const bool decimals = step < 1.0;
        const float tickLen = 3.0f * style::uiScale;
        const float labelW = ImGui::CalcTextSize(formatTime(std::floor(dur / step) * step, decimals).c_str()).x + tickLen + 1.0f;
        ImVec2 imgMin = ImVec2(origin.x + labelW, origin.y);
        ImVec2 imgSize = ImVec2(std::max<float>(avail.x - labelW, 10.0f), std::max<float>(avail.y, 10.0f));
        ImVec2 imgMax = ImVec2(imgMin.x + imgSize.x, imgMin.y + imgSize.y);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddImage((void*)(intptr_t)texture, imgMin, imgMax);

        ImGui::SetCursorScreenPos(imgMin);
        ImGui::InvisibleButton("##file_preview_img", imgSize);
        bool active = ImGui::IsItemActive();
        bool hovered = ImGui::IsItemHovered();
        ImVec2 mouse = ImGui::GetMousePos();
        double mouseFrac = std::clamp<double>((mouse.y - imgMin.y) / imgSize.y, 0.0, 1.0);
        if (active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            seekRequested = true;
            seekFrac = mouseFrac;
            posFrac = mouseFrac;
        }

        ImU32 tickCol = IM_COL32(200, 200, 200, 255);
        for (double t = 0.0; t <= dur; t += step) {
            float y = imgMin.y + (float)(t / dur) * imgSize.y;
            dl->AddLine(ImVec2(imgMin.x - tickLen, y), ImVec2(imgMin.x, y), tickCol);
            std::string txt = formatTime(t, decimals);
            float ty = std::min<float>(y, imgMax.y - ImGui::GetTextLineHeight());
            dl->AddText(ImVec2(imgMin.x - tickLen - 1.0f - ImGui::CalcTextSize(txt.c_str()).x, ty), tickCol, txt.c_str()); // right-aligned
        }

        // Current position: red horizontal line
        float py = imgMin.y + (float)std::clamp<double>(posFrac, 0.0, 1.0) * imgSize.y;
        dl->AddLine(ImVec2(imgMin.x - tickLen, py), ImVec2(imgMax.x, py), IM_COL32(255, 0, 0, 255), 2.0f);

        if (hovered || active) {
            double xFrac = std::clamp<double>((mouse.x - imgMin.x) / imgSize.x, 0.0, 0.999999);
            int col = std::clamp<int>((int)(xFrac * COLS), 0, COLS - 1);
            int row = std::clamp<int>((int)(mouseFrac * rows), 0, rows - 1);
            double freq = centerFreq + (xFrac - 0.5) * sr;
            std::string tt = formatTime(mouseFrac * dur, true) + "   " + formatFreq(freq);
            if (row < rowsDone.load(std::memory_order_acquire)) {
                char lv[32];
                snprintf(lv, sizeof(lv), "   %.0f dB", db[(size_t)row * COLS + col] + (float)refOffsetDb);
                tt += lv;
            }
            ImGui::SetTooltip("%s", tt.c_str());
        }

        // Scan progress
        int done = rowsDone.load(std::memory_order_acquire);
        if (done < rows) {
            char pr[32];
            snprintf(pr, sizeof(pr), "Scanning %d%%", (int)(100.0 * done / rows));
            dl->AddText(ImVec2(imgMin.x + 4.0f * style::uiScale, imgMax.y - ImGui::GetTextLineHeight() - 2.0f), IM_COL32(255, 255, 255, 255), pr);
        }
        return true;
    }

private:
    static std::string formatTime(double t, bool decimals) {
        int64_t whole = (int64_t)t;
        int h = (int)(whole / 3600);
        int m = (int)((whole / 60) % 60);
        double s = t - (double)(whole / 60) * 60.0;
        char buf[48];
        if (decimals) {
            if (h > 0) { snprintf(buf, sizeof(buf), "%d:%02d:%04.1f", h, m, s); }
            else { snprintf(buf, sizeof(buf), "%d:%04.1f", m, s); }
        }
        else {
            if (h > 0) { snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, (int)s); }
            else { snprintf(buf, sizeof(buf), "%d:%02d", m, (int)s); }
        }
        return buf;
    }

    static std::string formatFreq(double f) {
        char buf[48];
        if (fabs(f) >= 1e6) { snprintf(buf, sizeof(buf), "%.4f MHz", f / 1e6); }
        else if (fabs(f) >= 1e3) { snprintf(buf, sizeof(buf), "%.3f kHz", f / 1e3); }
        else { snprintf(buf, sizeof(buf), "%.0f Hz", f); }
        return buf;
    }

    static uint32_t colorize(float level, float lo, float hi) {
        float t = (hi > lo) ? std::clamp<float>((level - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
        // black -> blue -> cyan -> yellow -> red -> white
        static const float stops[6][3] = { { 0, 0, 0 }, { 0, 0, 0.7f }, { 0, 0.9f, 0.9f }, { 1, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 } };
        float x = t * 5.0f;
        int i = std::min<int>((int)x, 4);
        float f = x - (float)i;
        uint8_t r = (uint8_t)(255.0f * (stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f));
        uint8_t g = (uint8_t)(255.0f * (stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f));
        uint8_t b = (uint8_t)(255.0f * (stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f));
        return (0xFFu << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r; // RGBA bytes in memory (little endian)
    }

    void updateNoiseFloor(int done) {
        floorDb.assign(COLS, -200.0f);
        if (done < 1) { return; }
        std::vector<float> col(done);
        for (int c = 0; c < COLS; c++) {
            for (int r = 0; r < done; r++) { col[r] = db[(size_t)r * COLS + c]; }
            std::nth_element(col.begin(), col.begin() + done / 2, col.end());
            floorDb[c] = col[done / 2];
        }
    }

    void updateTexture(double refOffsetDb, double narrowBoostDb) {
        int done = rowsDone.load(std::memory_order_acquire);
        float lo = gui::waterfall.getWaterfallMin();
        float hi = gui::waterfall.getWaterfallMax();
        float off = (float)refOffsetDb;
        float boost = (float)narrowBoostDb;
        if (texture && done == lastTexRows && lo == lastLo && hi == lastHi && off == lastOff && boost == lastBoost) { return; }
        if (done != lastTexRows || floorDb.empty()) { updateNoiseFloor(done); }

        if (!texture) {
            glGenTextures(1, &texture);
        }
        pixels.assign((size_t)rows * COLS, 0xFF000000u);
        for (int r = 0; r < done; r++) {
            for (int c = 0; c < COLS; c++) {
                float v = db[(size_t)r * COLS + c];
                float ramp = std::clamp<float>((v - floorDb[c] - 1.5f) / 1.5f, 0.0f, 1.0f);
                pixels[(size_t)r * COLS + c] = colorize(v + off + boost * ramp, lo, hi);
            }
        }
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, COLS, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        lastTexRows = done;
        lastLo = lo;
        lastHi = hi;
        lastOff = off;
        lastBoost = boost;
    }

    void scan() {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { return; }
        std::vector<uint8_t> raw((size_t)FFT_N * bytesPerFrame);
        std::vector<float> acc(FFT_N);
        const float norm = 1.0f / winSumSq;

        for (int r = 0; r < rows && !cancel; r++) {
            uint64_t rowStart = totalFrames * (uint64_t)r / (uint64_t)rows;
            uint64_t rowSpan = std::max<uint64_t>(totalFrames / (uint64_t)rows, FFT_N);
            int blocks = (int)std::clamp<uint64_t>(rowSpan / FFT_N, 1, BLOCKS_PER_ROW);
            uint64_t slot = rowSpan / (uint64_t)blocks;
            std::fill(acc.begin(), acc.end(), 0.0f);
            int used = 0;

            for (int k = 0; k < blocks; k++) {
                uint64_t off = rowStart + (uint64_t)k * slot + ((slot > FFT_N) ? (slot - FFT_N) / 2 : 0);
                if (off + FFT_N > totalFrames) { off = totalFrames - FFT_N; }
                file.clear();
                file.seekg((std::streamoff)(HEADER_SIZE + off * bytesPerFrame));
                file.read((char*)raw.data(), raw.size());
                if ((size_t)file.gcount() < raw.size()) { continue; }

                if (sampleFormat == FORMAT_F32) {
                    const float* f = (const float*)raw.data();
                    for (int i = 0; i < FFT_N; i++) {
                        fftIn[i][0] = f[2 * i];
                        fftIn[i][1] = f[2 * i + 1];
                    }
                }
                else if (sampleFormat == FORMAT_U8) {
                    const uint8_t* u = raw.data();
                    for (int i = 0; i < FFT_N; i++) {
                        fftIn[i][0] = wavU8ToFloat(u[2 * i]);
                        fftIn[i][1] = wavU8ToFloat(u[2 * i + 1]);
                    }
                }
                else {
                    const int16_t* s = (const int16_t*)raw.data();
                    for (int i = 0; i < FFT_N; i++) {
                        fftIn[i][0] = (float)s[2 * i] / 32768.0f;
                        fftIn[i][1] = (float)s[2 * i + 1] / 32768.0f;
                    }
                }

                if (removeDC) {
                    double meanI = 0.0, meanQ = 0.0;
                    for (int i = 0; i < FFT_N; i++) {
                        meanI += fftIn[i][0];
                        meanQ += fftIn[i][1];
                    }
                    meanI /= FFT_N;
                    meanQ /= FFT_N;
                    for (int i = 0; i < FFT_N; i++) {
                        fftIn[i][0] -= (float)meanI;
                        fftIn[i][1] -= (float)meanQ;
                    }
                }

                for (int i = 0; i < FFT_N; i++) {
                    fftIn[i][0] *= win[i];
                    fftIn[i][1] *= win[i];
                }
                fftwf_execute(plan);
                for (int i = 0; i < FFT_N; i++) {
                    acc[i] += fftOut[i][0] * fftOut[i][0] + fftOut[i][1] * fftOut[i][1];
                }
                used++;
            }

            const int group = FFT_N / COLS;
            for (int c = 0; c < COLS; c++) {
                float pmax = 0.0f;
                for (int g = 0; g < group; g++) {
                    pmax = std::max<float>(pmax, acc[(c * group + g + FFT_N / 2) % FFT_N]);
                }
                float p = (used > 0) ? (pmax / (float)used) * norm : 0.0f;
                db[(size_t)r * COLS + c] = (p > 1e-20f) ? (10.0f * log10f(p) - MAX_BIAS_DB) : -200.0f;
            }
            rowsDone.store(r + 1, std::memory_order_release);
        }
    }

    std::string path;
    int sampleFormat = FORMAT_S16;
    bool removeDC = false;
    double sr = 1.0;
    int bytesPerFrame = 4;
    uint64_t totalFrames = 0;
    int rows = 0;
    std::string error;

    std::vector<float> db;
    std::atomic<int> rowsDone{ 0 };
    std::atomic<bool> cancel{ false };
    std::thread worker;

    fftwf_complex* fftIn = NULL;
    fftwf_complex* fftOut = NULL;
    fftwf_plan plan;
    std::vector<float> win;
    float winSumSq = 1.0f;

    std::string scanKeyPath;
    int scanKeyFormat = FORMAT_S16;
    bool scanKeyDC = false;

    GLuint texture = 0;
    std::vector<uint32_t> pixels;
    int lastTexRows = -1;
    float lastLo = 0.0f, lastHi = 0.0f, lastOff = 0.0f, lastBoost = 0.0f;
    std::vector<float> floorDb;
};
