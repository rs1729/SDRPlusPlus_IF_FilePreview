#define NOMINMAX
#include <imgui.h>
#include <utils/flog.h>
#include <module.h>
#include <gui/gui.h>
#include <signal_path/signal_path.h>
#include <wavreader.h>
#include <preview.h>
#include <dsp/window/blackman.h>
#include <dsp/window/nuttall.h>
#include <cmath>
#include <core.h>
#include <gui/widgets/file_select.h>
#include <gui/style.h>
#include <filesystem>
#include <regex>
#include <gui/tuner.h>
#include <algorithm>
#include <stdexcept>

#define CONCAT(a, b) ((std::string(a) + b).c_str())

SDRPP_MOD_INFO{
    /* Name:            */ "file_source",
    /* Description:     */ "Wav file source module for SDR++",
    /* Author:          */ "Ryzerth",
    /* Version:         */ 0, 1, 1,
    /* Max instances    */ 1
};

ConfigManager config;

class FileSourceModule : public ModuleManager::Instance {
public:
    FileSourceModule(std::string name) : fileSelect("", { "Wav IQ Files (*.wav)", "*.wav", "All Files", "*" }) {
        this->name = name;

        if (core::args["server"].b()) { return; }

        config.acquire();
        fileSelect.setPath(config.conf["path"], true);
        if (config.conf.contains("preview")) { previewEnabled = config.conf["preview"]; }
        if (config.conf.contains("previewOffset")) { previewOffsetDb = config.conf["previewOffset"]; }
        config.release();

        handler.ctx = this;
        handler.selectHandler = menuSelected;
        handler.deselectHandler = menuDeselected;
        handler.menuHandler = menuHandler;
        handler.startHandler = start;
        handler.stopHandler = stop;
        handler.tuneHandler = tune;
        handler.stream = &stream;
        sigpath::sourceManager.registerSource("File", &handler);

        // Preview panel
        extraDrawHandler.ctx = this;
        extraDrawHandler.handler = extraDraw;
        previewDrawHandler.ctx = this;
        previewDrawHandler.handler = previewDraw;
        gui::mainWindow.onExtraWindowsDraw.bindHandler(&extraDrawHandler);
        gui::mainWindow.onDrawPreviewPanel.bindHandler(&previewDrawHandler);
    }

    ~FileSourceModule() {
        if (!core::args["server"].b()) {
            gui::mainWindow.onDrawPreviewPanel.unbindHandler(&previewDrawHandler);
            gui::mainWindow.onExtraWindowsDraw.unbindHandler(&extraDrawHandler);
            gui::mainWindow.previewPanelActive = false;
        }
        stop(this);
        sigpath::sourceManager.unregisterSource("File");
    }

    void postInit() {}

    void enable() {
        enabled = true;
    }

    void disable() {
        enabled = false;
    }

    bool isEnabled() {
        return enabled;
    }

private:
    static void menuSelected(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        _this->selected = true;
        core::setInputSampleRate(_this->sampleRate);
        tuner::tune(tuner::TUNER_MODE_IQ_ONLY, "", _this->centerFreq);
        sigpath::iqFrontEnd.setBuffering(false);
        gui::waterfall.centerFrequencyLocked = true;
        //gui::freqSelect.minFreq = _this->centerFreq - (_this->sampleRate/2);
        //gui::freqSelect.maxFreq = _this->centerFreq + (_this->sampleRate/2);
        //gui::freqSelect.limitFreq = true;
        flog::info("FileSourceModule '{0}': Menu Select!", _this->name);
    }

    static void menuDeselected(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        _this->selected = false;
        _this->preview.stop();
        gui::mainWindow.previewPanelActive = false;
        sigpath::iqFrontEnd.setBuffering(true);
        //gui::freqSelect.limitFreq = false;
        gui::waterfall.centerFrequencyLocked = false;
        flog::info("FileSourceModule '{0}': Menu Deselect!", _this->name);
    }

    static void start(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        if (_this->running) { return; }
        if (_this->reader == NULL) { return; }
        _this->running = true;
        if (_this->float32Mode) { _this->workerThread = std::thread(floatWorker, _this); }
        else if (_this->reader->getBitDepth() == 8) { _this->workerThread = std::thread(worker8, _this); }
        else { _this->workerThread = std::thread(worker, _this); }
        flog::info("FileSourceModule '{0}': Start!", _this->name);
    }

    static void stop(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        if (!_this->running) { return; }
        if (_this->reader == NULL) { return; }
        _this->stream.stopWriter();
        _this->workerThread.join();
        _this->stream.clearWriteStop();
        _this->running = false;
        _this->reader->rewind();
        flog::info("FileSourceModule '{0}': Stop!", _this->name);
    }

    static void tune(double freq, void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        flog::info("FileSourceModule '{0}': Tune: {1}!", _this->name, freq);
    }

    static void menuHandler(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;

        if (_this->fileSelect.render("##file_source_" + _this->name)) {
            if (_this->fileSelect.pathIsValid()) {
                bool wasRunning = _this->running;
                if (wasRunning) { stop(_this); }
                if (_this->reader != NULL) {
                    _this->reader->close();
                    delete _this->reader;
                    _this->reader = NULL;
                }
                try {
                    _this->reader = new WavReader(_this->fileSelect.path);
                    if (_this->reader->getSampleRate() == 0) {
                        _this->reader->close();
                        delete _this->reader;
                        _this->reader = NULL;
                        throw std::runtime_error("Sample rate may not be zero");
                    }
                    _this->sampleRate = _this->reader->getSampleRate();
                    core::setInputSampleRate(_this->sampleRate);
                    std::string filename = std::filesystem::path(_this->fileSelect.path).filename().string();
                    _this->centerFreq = _this->getFrequency(filename);
                    tuner::tune(tuner::TUNER_MODE_IQ_ONLY, "", _this->centerFreq);
                    //gui::freqSelect.minFreq = _this->centerFreq - (_this->sampleRate/2);
                    //gui::freqSelect.maxFreq = _this->centerFreq + (_this->sampleRate/2);
                    //gui::freqSelect.limitFreq = true;
                }
                catch (const std::exception& e) {
                    flog::error("Error: {}", e.what());
                }
                if (wasRunning && _this->reader != NULL) { start(_this); }
                config.acquire();
                config.conf["path"] = _this->fileSelect.path;
                config.release(true);
            }
        }

        ImGui::Checkbox("Float32 Mode##_file_source", &_this->float32Mode);

        if (ImGui::Checkbox("Preview##_file_source", &_this->previewEnabled)) {
            config.acquire();
            config.conf["preview"] = _this->previewEnabled;
            config.release(true);
        }

        if (_this->previewEnabled) {
            ImGui::LeftLabel("Preview-Offset (dB)");
            ImGui::FillWidth();
            ImGui::SliderFloat("##_file_source_preview_offset", &_this->previewOffsetDb, -20.0f, 20.0f, "%.0f dB");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                config.acquire();
                config.conf["previewOffset"] = _this->previewOffsetDb;
                config.release(true);
            }
        }
    }

    static void extraDraw(bool, void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        bool want = _this->enabled && _this->selected && _this->previewEnabled && _this->reader != NULL;
        if (want) {
            std::string path = _this->fileSelect.path;
            int format = _this->float32Mode ? FilePreview::FORMAT_F32 : ((_this->reader->getBitDepth() == 8) ? FilePreview::FORMAT_U8 : FilePreview::FORMAT_S16);
            // Same switch as the main FFT: "IQ Correction" removes DC offset
            bool dcRemove = false;
            core::configManager.acquire();
            if (core::configManager.conf.contains("iqCorrection")) { dcRemove = core::configManager.conf["iqCorrection"]; }
            core::configManager.release();
            if (!_this->preview.isScanning(path, format, dcRemove)) {
                _this->preview.start(path, format, _this->sampleRate, dcRemove);
            }
        }
        else {
            _this->preview.stop();
        }
        gui::mainWindow.previewPanelActive = want;
    }

    static void mainFFTReference(double& offsetDb, double& narrowBoostDb) {
        offsetDb = 0.0;
        narrowBoostDb = 0.0;
        int fftSize = 65536;
        double fftRate = 20.0;
        int window = 2;
        core::configManager.acquire();
        if (core::configManager.conf.contains("fftSize")) { fftSize = core::configManager.conf["fftSize"]; }
        if (core::configManager.conf.contains("fftRate")) { fftRate = core::configManager.conf["fftRate"]; }
        if (core::configManager.conf.contains("fftWindow")) { window = core::configManager.conf["fftWindow"]; }
        core::configManager.release();

        double sr = sigpath::iqFrontEnd.getEffectiveSamplerate();
        if (sr <= 0.0 || fftRate <= 0.0 || fftSize <= 0) { return; }
        int nz = std::min<int>((int)round(sr / fftRate), fftSize);
        if (nz < 1) { nz = 1; }

        static double mean[3] = { 1.0, -1.0, -1.0 };
        static double meanSq[3] = { 1.0, -1.0, -1.0 };
        int wi = std::clamp<int>(window, 0, 2);
        if (meanSq[wi] < 0.0) {
            const int n = 4096;
            double acc = 0.0, acc2 = 0.0;
            for (int i = 0; i < n; i++) {
                double w = (wi == 1) ? dsp::window::blackman(i, n) : dsp::window::nuttall(i, n);
                acc += w;
                acc2 += w * w;
            }
            mean[wi] = acc / n;
            meanSq[wi] = acc2 / n;
        }
        offsetDb = 10.0 * log10((double)nz * meanSq[wi]) - 20.0 * log10((double)fftSize);

        double mainGainDb = 10.0 * log10((double)nz * mean[wi] * mean[wi] / meanSq[wi]);
        narrowBoostDb = std::max<double>(0.0, mainGainDb - FilePreview::narrowCarrierGainDb());
    }

    static void previewDraw(bool* drawn, void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        if (!_this->selected || _this->reader == NULL) { return; }
        double posFrac = 0.0;
        if (_this->preview.hasData()) {
            double total = (double)_this->preview.getTotalFrames() * _this->preview.getBytesPerFrame();
            posFrac = std::clamp<double>((double)_this->reader->getDataPos() / total, 0.0, 1.0);
        }
        bool seek = false;
        double seekFrac = 0.0;
        double refOffset, boost;
        mainFFTReference(refOffset, boost);
        *drawn = _this->preview.draw(posFrac, _this->centerFreq, refOffset + _this->previewOffsetDb, boost, seek, seekFrac);
        if (seek && _this->preview.hasData()) {
            uint64_t frame = (uint64_t)(seekFrac * (double)_this->preview.getTotalFrames());
            frame = std::min<uint64_t>(frame, _this->preview.getTotalFrames() - 1);
            _this->reader->seekData(frame * (uint64_t)_this->preview.getBytesPerFrame());
        }
    }

    static void worker(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        double sampleRate = std::max(_this->reader->getSampleRate(), (uint32_t)1);
        int blockSize = std::min((int)(sampleRate / 200.0f), (int)STREAM_BUFFER_SIZE);
        int16_t* inBuf = new int16_t[blockSize * 2];

        while (true) {
            _this->reader->readSamples(inBuf, blockSize * 2 * sizeof(int16_t));
            volk_16i_s32f_convert_32f((float*)_this->stream.writeBuf, inBuf, 32768.0f, blockSize * 2);
            if (!_this->stream.swap(blockSize)) { break; };
        }

        delete[] inBuf;
    }

    // 8-bit unsigned WAV files; see wavreader.h:wavU8ToFloat()
    static void worker8(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        double sampleRate = std::max(_this->reader->getSampleRate(), (uint32_t)1);
        int blockSize = std::min((int)(sampleRate / 200.0f), (int)STREAM_BUFFER_SIZE);
        uint8_t* inBuf = new uint8_t[blockSize * 2];

        while (true) {
            _this->reader->readSamples(inBuf, blockSize * 2);
            float* out = (float*)_this->stream.writeBuf;
            for (int i = 0; i < blockSize * 2; i++) {
                out[i] = wavU8ToFloat(inBuf[i]);
            }
            if (!_this->stream.swap(blockSize)) { break; };
        }

        delete[] inBuf;
    }

    static void floatWorker(void* ctx) {
        FileSourceModule* _this = (FileSourceModule*)ctx;
        double sampleRate = std::max(_this->reader->getSampleRate(), (uint32_t)1);
        int blockSize = std::min((int)(sampleRate / 200.0f), (int)STREAM_BUFFER_SIZE);
        dsp::complex_t* inBuf = new dsp::complex_t[blockSize];

        while (true) {
            _this->reader->readSamples(_this->stream.writeBuf, blockSize * sizeof(dsp::complex_t));
            if (!_this->stream.swap(blockSize)) { break; };
        }

        delete[] inBuf;
    }

    double getFrequency(std::string filename) {
        std::regex expr("[0-9]+Hz");
        std::smatch matches;
        std::regex_search(filename, matches, expr);
        if (matches.empty()) { return 0; }
        std::string freqStr = matches[0].str();
        return std::atof(freqStr.substr(0, freqStr.size() - 2).c_str());
    }

    FileSelect fileSelect;
    std::string name;
    dsp::stream<dsp::complex_t> stream;
    SourceManager::SourceHandler handler;
    WavReader* reader = NULL;
    bool running = false;
    bool enabled = true;
    float sampleRate = 1000000;
    std::thread workerThread;

    double centerFreq = 100000000;

    bool float32Mode = false;

    bool previewEnabled = false;
    float previewOffsetDb = 0.0f;
    bool selected = false;
    FilePreview preview;
    EventHandler<bool> extraDrawHandler;
    EventHandler<bool*> previewDrawHandler;
};

MOD_EXPORT void _INIT_() {
    json def = json({});
    def["path"] = "";
    def["preview"] = false;
    def["previewOffset"] = 0.0;
    config.setPath(core::args["root"].s() + "/file_source_config.json");
    config.load(def);
    config.enableAutoSave();
}

MOD_EXPORT void* _CREATE_INSTANCE_(std::string name) {
    return new FileSourceModule(name);
}

MOD_EXPORT void _DELETE_INSTANCE_(void* instance) {
    delete (FileSourceModule*)instance;
}

MOD_EXPORT void _END_() {
    config.disableAutoSave();
    config.save();
}
