#pragma once

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <map>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

class TimingProfiler {
public:
    struct Statistics {
        size_t calls = 0;
        double total_ms = 0.0;
    };

    class Scope {
    public:
        Scope(TimingProfiler* profiler, std::string_view name) {
            if (profiler && profiler->enabled()) {
                statistics_ = &profiler->statistics_[std::string(name)];
                start_ = Clock::now();
            }
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { finish(); }

        void finish() {
            if (!statistics_) return;
            const auto end = Clock::now();
            statistics_->total_ms += std::chrono::duration<double, std::milli>(end - start_).count();
            ++statistics_->calls;
            statistics_ = nullptr;
        }

    private:
        using Clock = std::chrono::steady_clock;
        Statistics* statistics_ = nullptr;
        Clock::time_point start_;
    };

    void enable(const std::string& path) {
        output_.open(path, std::ios::app);
        if (!output_) throw std::runtime_error("Cannot open profiling CSV: " + path);
        output_.seekp(0, std::ios::end);
        if (output_.tellp() == 0)
            output_ << "run_id,phase,epoch,batches,samples,stage,kind,calls,total_ms,mean_ms,percent_batch\n";
        run_id_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        output_ << std::fixed << std::setprecision(6);
        output_.flush();
    }

    bool enabled() const { return output_.is_open(); }
    void reset() { statistics_.clear(); }
    const auto& statistics() const { return statistics_; }

    template<typename Function>
    decltype(auto) measure(std::string_view name, Function&& function) {
        Scope scope(this, name);
        return std::forward<Function>(function)();
    }

    void report(std::string_view phase, size_t epoch, size_t batches, size_t samples, bool console=true) {
        if (!enabled()) return;
        const auto total = statistics_.find("batch.total");
        const double batch_ms = total == statistics_.end() ? 0.0 : total->second.total_ms;
        if (console) {
            std::println("Profiling {} / epoch {}: {} batches, {} samples (kumulativ)", phase, epoch, batches, samples);
            std::println("{:<38} {:>8} {:>12} {:>12} {:>9}", "Stage (* = inklusive Summe)", "Aufrufe", "gesamt ms", "ms/Aufruf", "% Batch");
        }
        for (const auto& [name, stats] : statistics_) {
            const bool inclusive = name.ends_with(".total");
            const double mean = stats.calls ? stats.total_ms / stats.calls : 0.0;
            const double percent = batch_ms > 0.0 ? 100.0 * stats.total_ms / batch_ms : 0.0;
            output_ << run_id_ << ',' << phase << ',' << epoch << ',' << batches << ',' << samples << ','
                    << name << ',' << (inclusive ? "inclusive" : "stage") << ',' << stats.calls << ','
                    << stats.total_ms << ',' << mean << ',';
            if (batch_ms > 0.0) output_ << percent;
            output_ << '\n';
            if (console) {
                const auto label = name + (inclusive ? " *" : "");
                if (batch_ms > 0.0)
                    std::println("{:<38} {:>8} {:>12.3f} {:>12.3f} {:>8.2f}%", label, stats.calls, stats.total_ms, mean, percent);
                else
                    std::println("{:<38} {:>8} {:>12.3f} {:>12.3f} {:>9}", label, stats.calls, stats.total_ms, mean, "-");
            }
        }
        output_.flush();
        if (!output_) throw std::runtime_error("Cannot write profiling CSV");
    }

private:
    std::ofstream output_;
    std::string run_id_;
    std::map<std::string, Statistics> statistics_;
};
