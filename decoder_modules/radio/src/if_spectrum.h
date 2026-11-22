#pragma once
#include <imgui.h>
#include <fftw3.h>
#include <dsp/types.h>
#include <dsp/math/constants.h>
#include <gui/gui.h>
#include <gui/style.h>
#include <signal_path/signal_path.h>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

class IFSampleRing {
public:
    IFSampleRing() : ring(RING_SIZE) {}

    void push(const dsp::complex_t* data, int count) {
        std::lock_guard<std::mutex> lck(mtx);
        const dsp::complex_t* src = data;
        int n = count;
        if (n > RING_SIZE) {
            src += (n - RING_SIZE);
            n = RING_SIZE;
        }
        int first = std::min<int>(n, RING_SIZE - writePos);
        memcpy(ring.data() + writePos, src, first * sizeof(dsp::complex_t));
        if (first < n) {
            memcpy(ring.data(), src + first, (n - first) * sizeof(dsp::complex_t));
        }
        writePos = (writePos + n) % RING_SIZE;
        filled = std::min<int>(filled + n, RING_SIZE);
    }

    // Copy the latest 'count' samples (oldest first). Returns false if not enough data yet.
    bool snapshot(dsp::complex_t* dst, int count) {
        std::lock_guard<std::mutex> lck(mtx);
        if (count > RING_SIZE || filled < count) { return false; }
        int start = (writePos - count + RING_SIZE) % RING_SIZE;
        int first = std::min<int>(count, RING_SIZE - start);
        memcpy(dst, ring.data() + start, first * sizeof(dsp::complex_t));
        if (first < count) {
            memcpy(dst + first, ring.data(), (count - first) * sizeof(dsp::complex_t));
        }
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lck(mtx);
        writePos = 0;
        filled = 0;
    }

private:
    static constexpr int RING_SIZE = 65536;
    std::vector<dsp::complex_t> ring;
    int writePos = 0;
    int filled = 0;
    std::mutex mtx;
};

class IFSpectrum {
public:
    IFSpectrum() {
        allocate(FFT_SIZES[sizeIdx]);
    }

    ~IFSpectrum() {
        release();
    }

    // True : "Wide" view (extra downconverter, selectable span).
    // False: "IF" view (tap inside the radio VFO, no extra VFO).
    bool wideView() const {
        return viewIdx == 1;
    }

    double requestedSpan(double bandwidth) const {
        if (spanIdx > 0) { return SPAN_VALUES[spanIdx - 1]; }
        double target = std::max<double>(bandwidth * 2.0, 24000.0);
        for (int i = 0; i < SPAN_COUNT; i++) {
            if (SPAN_VALUES[i] >= target) { return SPAN_VALUES[i]; }
        }
        return SPAN_VALUES[SPAN_COUNT - 1];
    }

    // Returns the bandwidth requested by dragging a box edge with the mouse, or 0 if none.
    double drawPanel(IFSampleRing& ring, double sampleRate, bool active, double bandwidth, int vfoRef, bool bwLocked) {
        // Controls
        ImGui::Checkbox("Smooth##_ifspec", &smooth);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(75.0f * style::uiScale);
        if (ImGui::Combo("FFT##_ifspec", &sizeIdx, FFT_SIZE_NAMES, FFT_SIZE_COUNT)) {
            allocate(FFT_SIZES[sizeIdx]);
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(150.0f * style::uiScale);
        ImGui::Combo("View##_ifspec", &viewIdx, VIEW_NAMES, VIEW_COUNT);
        if (wideView()) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(70.0f * style::uiScale);
            ImGui::Combo("Span##_ifspec", &spanIdx, SPAN_NAMES, SPAN_COUNT + 1);
        }

        if (sampleRate != lastSampleRate) {
            haveSmooth = false;
            lastSampleRate = sampleRate;
        }

        // Noise floor offset
        if (sampleRate > 0.0) {
            double mainDb = 10.0 * std::log10(sigpath::iqFrontEnd.getEffectiveSamplerate() * sigpath::iqFrontEnd.getNzFFTSize())
                          - 20.0 * std::log10((double)sigpath::iqFrontEnd.getFFTSize());
            levelOffset = (float)(mainDb - 10.0 * std::log10(sampleRate / fftSize));
        }

        // Compute the spectrum
        bool valid = active && (sampleRate > 0.0) && compute(ring);

        // Plot area
        ImVec2 avail = ImGui::GetContentRegionAvail();
        avail.x = std::max<float>(avail.x, 120.0f);
        avail.y = std::max<float>(avail.y, 60.0f);
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##_ifspec_plot", avail);
        bool hovered = ImGui::IsItemHovered();

        const float leftMargin = 44.0f;
        const float bottomMargin = 20.0f;
        ImVec2 pMin(origin.x + leftMargin, origin.y + 2.0f);
        ImVec2 pMax(origin.x + avail.x - 6.0f, origin.y + avail.y - bottomMargin);
        float pw = pMax.x - pMin.x;
        float ph = pMax.y - pMin.y;
        if (pw < 10.0f || ph < 10.0f) { return 0.0; }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImU32 colText = ImGui::GetColorU32(ImGuiCol_Text);
        const ImU32 colGrid = IM_COL32(128, 128, 128, 60);
        const ImU32 colFrame = IM_COL32(128, 128, 128, 160);

        dl->AddRectFilled(pMin, pMax, IM_COL32(0, 0, 0, 255));

        float dbMin = gui::waterfall.getFFTMin();
        float dbMax = gui::waterfall.getFFTMax();
        if (dbMax - dbMin < 1.0f) { dbMax = dbMin + 1.0f; }
        float dbScale = ph / (dbMax - dbMin);

        // Horizontal grid (dB)
        float dbStep = ((dbMax - dbMin) > 80.0f) ? 20.0f : 10.0f;
        for (float db = std::ceil(dbMin / dbStep) * dbStep; db <= dbMax; db += dbStep) {
            float y = pMax.y - ((db - dbMin) * dbScale);
            dl->AddLine(ImVec2(pMin.x, y), ImVec2(pMax.x, y), colGrid);
            char buf[16];
            snprintf(buf, sizeof(buf), "%.0f", db);
            ImVec2 ts = ImGui::CalcTextSize(buf);
            dl->AddText(ImVec2(pMin.x - ts.x - 4.0f, y - (ts.y / 2.0f)), colText, buf);
        }

        // Vertical grid (frequency offset)
        double half = sampleRate / 2.0;
        double hzPerPx = (sampleRate > 0.0) ? (sampleRate / pw) : 1.0;
        if (sampleRate > 0.0) {
            double step = niceStep(sampleRate / 6.0);
            for (double f = std::ceil(-half / step) * step; f <= half; f += step) {
                float x = pMin.x + (float)((f + half) / hzPerPx);
                dl->AddLine(ImVec2(x, pMin.y), ImVec2(x, pMax.y), colGrid);
                char buf[24];
                if (std::fabs(f) < 1e-6) {
                    snprintf(buf, sizeof(buf), "0");
                }
                else if (std::fabs(f) >= 1e6) {
                    snprintf(buf, sizeof(buf), "%+.3fM", f / 1e6);
                }
                else if (std::fabs(f) >= 1e3) {
                    snprintf(buf, sizeof(buf), "%+.1fk", f / 1e3);
                }
                else {
                    snprintf(buf, sizeof(buf), "%+.0f", f);
                }
                ImVec2 ts = ImGui::CalcTextSize(buf);
                dl->AddText(ImVec2(x - (ts.x / 2.0f), pMax.y + 3.0f), colText, buf);
            }
        }

        dl->PushClipRect(pMin, pMax, true);

        if (valid) {
            const std::vector<float>& s = smooth ? smoothSpec : spec;
            int cols = (int)pw;
            float prevY = pMax.y;
            for (int x = 0; x < cols; x++) {
                int b0 = (int)(((int64_t)x * fftSize) / cols);
                int b1 = std::max<int>(b0 + 1, (int)(((int64_t)(x + 1) * fftSize) / cols));
                b1 = std::min<int>(b1, fftSize);
                float v = s[b0];
                for (int b = b0 + 1; b < b1; b++) { v = std::max<float>(v, s[b]); }
                float y = pMax.y - ((v - dbMin) * dbScale);
                y = std::clamp<float>(y, pMin.y, pMax.y);
                float px = pMin.x + x;
                dl->AddLine(ImVec2(px, y), ImVec2(px, pMax.y), IM_COL32(0, 255, 255, 50));
                if (x > 0) {
                    dl->AddLine(ImVec2(px - 1.0f, prevY), ImVec2(px, y), IM_COL32(0, 255, 255, 255));
                }
                prevY = y;
            }
        }
        else {
            const char* msg = "Waiting for data...";
            ImVec2 ts = ImGui::CalcTextSize(msg);
            dl->AddText(ImVec2(pMin.x + (pw - ts.x) / 2.0f, pMin.y + (ph - ts.y) / 2.0f), colText, msg);
        }

        double newBw = 0.0;
        if (sampleRate > 0.0 && bandwidth > 0.0) {
            float xl = pMin.x + (float)((-bandwidth / 2.0 + half) / hzPerPx);
            float xr = pMin.x + (float)((bandwidth / 2.0 + half) / hzPerPx);
            dl->AddRectFilled(ImVec2(xl, pMin.y), ImVec2(xr, pMax.y), IM_COL32(255, 255, 255, 30));
            dl->AddLine(ImVec2(xl, pMin.y), ImVec2(xl, pMax.y), IM_COL32(255, 255, 255, 140));
            dl->AddLine(ImVec2(xr, pMin.y), ImVec2(xr, pMax.y), IM_COL32(255, 255, 255, 140));

            if (!bwLocked) {
                ImVec2 m = ImGui::GetMousePos();
                bool onEdge = hovered && (std::fabs(m.x - xl) < 5.0f * style::uiScale || std::fabs(m.x - xr) < 5.0f * style::uiScale);
                if (onEdge && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { dragBw = true; }
                if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) { dragBw = false; }
                if (onEdge || dragBw) { ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW); }
                if (dragBw) { newBw = 2.0 * std::fabs(((m.x - pMin.x) * hzPerPx) - half); }
            }

            // Tuned frequency marker (depends on the VFO reference)
            double tuned = 0.0;
            if (vfoRef == ImGui::WaterfallVFO::REF_LOWER) { tuned = -bandwidth / 2.0; }
            else if (vfoRef == ImGui::WaterfallVFO::REF_UPPER) { tuned = bandwidth / 2.0; }
            float xt = pMin.x + (float)((tuned + half) / hzPerPx);
            dl->AddLine(ImVec2(xt, pMin.y), ImVec2(xt, pMax.y), IM_COL32(255, 0, 0, 220));

            char buf[48];
            if (bandwidth >= 1e3) { snprintf(buf, sizeof(buf), "BW %.2f kHz   Span %.0f kHz", bandwidth / 1e3, sampleRate / 1e3); }
            else { snprintf(buf, sizeof(buf), "BW %.0f Hz   Span %.0f kHz", bandwidth, sampleRate / 1e3); }
            dl->AddText(ImVec2(pMin.x + 6.0f, pMin.y + 2.0f), colText, buf);
        }

        if (hovered && valid) {
            ImVec2 m = ImGui::GetMousePos();
            if (m.x >= pMin.x && m.x <= pMax.x && m.y >= pMin.y && m.y <= pMax.y) {
                double f = (m.x - pMin.x) * hzPerPx - half;
                int bin = std::clamp<int>((int)(((m.x - pMin.x) / pw) * fftSize), 0, fftSize - 1);
                const std::vector<float>& s = smooth ? smoothSpec : spec;
                dl->AddLine(ImVec2(m.x, pMin.y), ImVec2(m.x, pMax.y), IM_COL32(255, 255, 255, 100));
                char buf[48];
                snprintf(buf, sizeof(buf), "%+.2f kHz  %.1f dB", f / 1e3, s[bin]);
                ImVec2 ts = ImGui::CalcTextSize(buf);
                float tx = std::min<float>(m.x + 8.0f, pMax.x - ts.x - 2.0f);
                dl->AddText(ImVec2(tx, pMin.y + 18.0f), colText, buf);
            }
        }

        dl->PopClipRect();
        dl->AddRect(pMin, pMax, colFrame);
        return newBw;
    }

private:
    static constexpr int FFT_SIZE_COUNT = 5;
    inline static const int FFT_SIZES[FFT_SIZE_COUNT] = { 1024, 2048, 4096, 8192, 16384 };
    inline static const char* const FFT_SIZE_NAMES[FFT_SIZE_COUNT] = { "1024", "2048", "4096", "8192", "16384" };

    static constexpr int VIEW_COUNT = 2;
    inline static const char* const VIEW_NAMES[VIEW_COUNT] = { "IF (no extra VFO)", "Wide (extra VFO)" };

    static constexpr int SPAN_COUNT = 7;
    inline static const double SPAN_VALUES[SPAN_COUNT] = { 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 384000.0, 768000.0 };
    inline static const char* const SPAN_NAMES[SPAN_COUNT + 1] = { "Auto", "12k", "24k", "48k", "96k", "192k", "384k", "768k" };

    static double niceStep(double target) {
        if (target <= 0.0) { return 1.0; }
        double mag = std::pow(10.0, std::floor(std::log10(target)));
        double r = target / mag;
        double m = (r <= 1.0) ? 1.0 : (r <= 2.0) ? 2.0 : (r <= 5.0) ? 5.0 : 10.0;
        return m * mag;
    }

    void allocate(int size) {
        release();
        fftSize = size;
        fftIn = (fftwf_complex*)fftwf_malloc(fftSize * sizeof(fftwf_complex));
        fftOut = (fftwf_complex*)fftwf_malloc(fftSize * sizeof(fftwf_complex));
        plan = fftwf_plan_dft_1d(fftSize, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);

        // Blackman window
        window.assign(fftSize, 0.0f);
        for (int i = 0; i < fftSize; i++) {
            double a = (2.0 * DB_M_PI * i) / (fftSize - 1);
            double w = 0.42 - (0.5 * std::cos(a)) + (0.08 * std::cos(2.0 * a));
            window[i] = (float)w * ((i % 2) ? -1.0f : 1.0f);
        }

        samples.assign(fftSize, dsp::complex_t{ 0.0f, 0.0f });
        spec.assign(fftSize, -200.0f);
        smoothSpec.assign(fftSize, -200.0f);
        haveSmooth = false;
    }

    void release() {
        if (plan) { fftwf_destroy_plan(plan); plan = NULL; }
        if (fftIn) { fftwf_free(fftIn); fftIn = NULL; }
        if (fftOut) { fftwf_free(fftOut); fftOut = NULL; }
    }

    bool compute(IFSampleRing& ring) {
        if (!ring.snapshot(samples.data(), fftSize)) { return false; }

        volk_32fc_32f_multiply_32fc((lv_32fc_t*)fftIn, (lv_32fc_t*)samples.data(), window.data(), fftSize);
        fftwf_execute(plan);
        volk_32fc_s32f_power_spectrum_32f(spec.data(), (lv_32fc_t*)fftOut, (float)fftSize, fftSize);
        if (levelOffset != 0.0f) {
            for (int i = 0; i < fftSize; i++) { spec[i] += levelOffset; }
        }

        if (!haveSmooth) {
            smoothSpec = spec;
            haveSmooth = true;
        }
        else {
            for (int i = 0; i < fftSize; i++) {
                smoothSpec[i] += 0.3f * (spec[i] - smoothSpec[i]);
            }
        }
        return true;
    }

    int sizeIdx = 2;
    int spanIdx = 0;
    int viewIdx = 0; // 0 = IF (default), 1 = Wide
    int fftSize = 0;
    bool smooth = true;
    bool haveSmooth = false;
    double lastSampleRate = 0.0;
    float levelOffset = 0.0f;
    bool dragBw = false;

    fftwf_complex* fftIn = NULL;
    fftwf_complex* fftOut = NULL;
    fftwf_plan plan = NULL;

    std::vector<float> window;
    std::vector<dsp::complex_t> samples;
    std::vector<float> spec;
    std::vector<float> smoothSpec;
};
