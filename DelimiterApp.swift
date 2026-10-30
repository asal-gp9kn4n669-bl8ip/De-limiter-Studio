import AppKit
import AVFoundation
import Foundation
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Domain Models & Enums

/// WAV and FLAC are lossless 24-bit PCM at 44.1 kHz; AAC (.m4a) uses macOS afconvert constrained VBR;
/// Opus (.opus) uses libopus VBR at 48 kHz (resampled via SoXR VHQ).
enum OutputFormat: String, CaseIterable, Identifiable {
    case wav = "WAV"
    case flac = "FLAC"
    case aac = "AAC"
    case opus = "Opus"

    var id: String { rawValue }

    var fileExtension: String {
        switch self {
        case .wav: return "wav"
        case .flac: return "flac"
        case .aac: return "m4a"
        case .opus: return "opus"
        }
    }
}

enum LoudnessMode: String, CaseIterable, Identifiable {
    case albumRelative = "Album / Batch Relative Rebalance"
    case perTrackInference = "Per-Track (inference.py --save_output_loudnorm)"
    case rawModelOutput = "Raw Model Output (No Output Loudnorm)"

    var id: String { rawValue }
}

enum ComputeDevice: String, CaseIterable, Identifiable {
    case cpu = "CPU (Default)"
    case mps = "Apple Silicon GPU (MPS)"

    var id: String { rawValue }
}

enum TrackStatus: Equatable {
    case pending
    case preparing
    case delimiting
    case measuring
    case writing
    case done
    case cancelled
    case failed(String)

    var label: String {
        switch self {
        case .pending: return "Queued"
        case .preparing: return "Preparing WAV…"
        case .delimiting: return "De-limiting…"
        case .measuring: return "Measuring LUFS…"
        case .writing: return "Writing…"
        case .done: return "Done"
        case .cancelled: return "Cancelled"
        case .failed(let msg): return "Error: \(msg)"
        }
    }

    var isProcessing: Bool {
        switch self {
        case .preparing, .delimiting, .measuring, .writing:
            return true
        default:
            return false
        }
    }
}

struct TrackItem: Identifiable, Equatable {
    let id: UUID
    let sourceURL: URL
    let relativeAlbumFolder: String
    var status: TrackStatus = .pending

    // Measured Audio Metrics & Waveform Envelopes
    var origLUFS: Double?
    var origCoreLUFS: Double?
    var origPeak: Double?
    var origPeaks: [Float] = []
    var rawOutLUFS: Double?
    var rawOutPeak: Double?
    var finalLUFS: Double?
    var finalCoreLUFS: Double?
    var finalPeak: Double?
    var finalPeaks: [Float] = []

    // Preview & Output Paths
    var previewOrigWAV: URL?
    var outputURL: URL?

    var displayName: String {
        sourceURL.deletingPathExtension().lastPathComponent
    }

    var sourceExt: String {
        sourceURL.pathExtension.uppercased()
    }

    /// Albums are grouped by their actual folder, so same-named folders in different places stay separate.
    var albumKey: String {
        sourceURL.deletingLastPathComponent().path
    }

    /// Change in integrated loudness of the written file versus the original.
    var loudnessDelta: Double? {
        guard let orig = origLUFS, let out = finalLUFS else { return nil }
        return out - orig
    }

    /// Restored peak-to-core headroom (dB) of the de-limited track relative to the original.
    var restoredHeadroomDB: Double? {
        guard let oPeak = origPeak, let fPeak = finalPeak,
              let oRef = origCoreLUFS ?? origLUFS,
              let fRef = finalCoreLUFS ?? finalLUFS else { return nil }
        return (fPeak - fRef) - (oPeak - oRef)
    }

    /// The same queue entry with every result from a previous run cleared.
    func resetForNewRun() -> TrackItem {
        TrackItem(id: id, sourceURL: sourceURL, relativeAlbumFolder: relativeAlbumFolder)
    }
}

enum ABSource: String, CaseIterable, Identifiable {
    case original = "A: Original"
    case delimited = "B: De-limited"

    var id: String { rawValue }
}

struct LogLine: Identifiable {
    let id: Int
    let text: String
}

/// Loudness and waveform envelope of one file as reported by delimiter_tool.py (`null` when not measurable).
private struct AudioMetrics: Decodable {
    var lufs: Double?
    var core_lufs: Double?
    var peak: Double?
    var peaks: [Float]?
}

// MARK: - Audio A/B Player Controller

/// Every loaded version (A and B) plays in sync from the same start time; switching sources only
/// changes volumes, so comparisons are instant and gapless.
@MainActor
@Observable
final class ABPlayerController {
    private(set) var isPlaying: Bool = false
    private(set) var activeSource: ABSource = .delimited
    private(set) var currentTime: TimeInterval = 0
    private(set) var duration: TimeInterval = 0
    var loudnessMatchAB: Bool = true {
        didSet { applyVolumes() }
    }
    private(set) var loadedTrackID: UUID?
    private(set) var hasDelimitedAudio: Bool = false

    @ObservationIgnored private var players: [ABSource: AVAudioPlayer] = [:]
    @ObservationIgnored private var sourceLUFS: [ABSource: Double] = [:]
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var isScrubbing = false
    @ObservationIgnored private var playStartedAt: TimeInterval = 0
    /// The source the user last picked; `activeSource` falls back from it while it's unavailable.
    @ObservationIgnored private var preferredSource: ABSource = .delimited

    func load(track: TrackItem?) {
        stop()
        players = [:]
        sourceLUFS = [:]
        loadedTrackID = track?.id
        currentTime = 0

        if let track {
            addPlayer(.original, url: track.previewOrigWAV ?? track.sourceURL, lufs: track.origCoreLUFS ?? track.origLUFS)
            addPlayer(.delimited, url: track.outputURL, lufs: track.finalCoreLUFS ?? track.finalLUFS)
        }
        hasDelimitedAudio = players[.delimited] != nil

        if players[preferredSource] != nil {
            activeSource = preferredSource
        } else {
            activeSource = hasDelimitedAudio ? .delimited : .original
        }

        duration = players.values.map(\.duration).max() ?? 0
        applyVolumes()
    }

    func updateCoreLoudness(for track: TrackItem) {
        guard loadedTrackID == track.id else { return }
        if let origCore = track.origCoreLUFS ?? track.origLUFS {
            sourceLUFS[.original] = origCore
        }
        if let finalCore = track.finalCoreLUFS ?? track.finalLUFS {
            sourceLUFS[.delimited] = finalCore
        }
        applyVolumes()
    }

    private func addPlayer(_ source: ABSource, url: URL?, lufs: Double?) {
        guard let url, FileManager.default.fileExists(atPath: url.path),
              let player = try? AVAudioPlayer(contentsOf: url) else { return }
        player.prepareToPlay()
        players[source] = player
        sourceLUFS[source] = lufs
    }

    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    /// Starts every loaded version at the same device time so they stay aligned.
    private func play() {
        guard let reference = players.values.first else { return }
        if currentTime >= duration - 0.05 {
            currentTime = 0
        }
        let startAt = reference.deviceCurrentTime + 0.05
        for player in players.values {
            player.currentTime = min(currentTime, player.duration)
            player.play(atTime: startAt)
        }
        isPlaying = true
        playStartedAt = ProcessInfo.processInfo.systemUptime
        startTimer()
    }

    private func pause() {
        if let active = players[activeSource] {
            currentTime = active.currentTime
        }
        players.values.forEach { $0.pause() }
        isPlaying = false
        stopTimer()
    }

    func switchSource(to newSource: ABSource) {
        guard newSource != activeSource, players[newSource] != nil else { return }
        activeSource = newSource
        preferredSource = newSource
        applyVolumes(fade: 0.015)   // a short fade avoids a click at the switch point
    }

    func seek(to time: TimeInterval) {
        currentTime = min(max(0, time), duration)
        if isPlaying {
            players.values.forEach { $0.pause() }
            if currentTime >= duration - 0.05 {
                // Seeking to the end finishes playback, just like reaching it naturally
                stop()
                currentTime = 0
            } else {
                play()   // restarts all versions together from the new position
            }
        } else {
            players.values.forEach { $0.currentTime = currentTime }
        }
    }

    func skip(by delta: TimeInterval) {
        seek(to: currentTime + delta)
    }

    /// Slider drags only move the playhead; the actual seek happens when the drag ends.
    func scrub(to time: TimeInterval) {
        currentTime = min(max(0, time), duration)
        if !isScrubbing {
            seek(to: currentTime)
        }
    }

    func setScrubbing(_ scrubbing: Bool) {
        isScrubbing = scrubbing
        if !scrubbing {
            seek(to: currentTime)
        }
    }

    func stop() {
        players.values.forEach { $0.stop() }
        isPlaying = false
        stopTimer()
    }

    /// Mutes every source except the active one. With loudness matching on, the active source is
    /// attenuated to the quietest version's LUFS so the louder version doesn't win the comparison.
    private func applyVolumes(fade: TimeInterval = 0) {
        let reference = loudnessMatchAB ? sourceLUFS.values.min() : nil
        for (source, player) in players {
            var volume: Float = 0
            if source == activeSource {
                if let reference, let lufs = sourceLUFS[source] {
                    volume = Float(max(pow(10, (reference - lufs) / 20), 0.05))
                } else {
                    volume = 1
                }
            }
            player.setVolume(volume, fadeDuration: fade)
        }
    }

    private func startTimer() {
        stopTimer()
        let timer = Timer(timeInterval: 0.08, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.tick()
            }
        }
        // .common mode keeps the playhead moving while menus are open or sliders are dragged
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    private func tick() {
        guard isPlaying, let active = players[activeSource] else { return }
        if active.isPlaying {
            if !isScrubbing {
                currentTime = active.currentTime
            }
        } else if ProcessInfo.processInfo.systemUptime - playStartedAt > 0.3 {
            // Reached the end of the track (the grace period covers the scheduled start delay)
            stop()
            currentTime = 0
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

// MARK: - Main ViewModel (De-limiter Pipeline & Process Runner)

@MainActor
@Observable
final class DelimiterViewModel {
    // Queue & Selection
    private(set) var tracks: [TrackItem] = []
    var selectedTrackID: UUID?
    private(set) var isRunning: Bool = false
    private(set) var statusBanner: String = "Ready"

    // inference.py Core Parameters
    private(set) var availableTargets: [String] = ["all"]
    var targetModel: String = "all"
    var computeDevice: ComputeDevice = .cpu
    var useSingleTrackSet: Bool = false
    var avoidClipping: Bool = true
    var enableInputLoudnorm: Bool = false
    var inputLoudnormLUFS: Double = -14.0

    // Loudness & Ceiling Strategy
    var loudnessMode: LoudnessMode = .albumRelative
    var targetLoudnessLUFS: Double = -14.0
    var peakCeilingDBFS: Double = -1.0

    // Output Format
    var outputFormat: OutputFormat = .flac
    var aacBitrateKbps: Double = 256.0
    var opusBitrateKbps: Double = 192.0
    var copyMetadataTags: Bool = true
    var customOutputFolder: URL?

    // Logs & UI State
    private(set) var logLines: [LogLine] = []
    var showLogPanel: Bool = true
    var isDropTargeted: Bool = false

    // Environment Paths
    let delimiterDir: URL
    let pythonURL: URL
    /// delimiter_tool.py decodes, measures and writes audio with the same Python as inference.py (no ffmpeg needed).
    let audioToolURL: URL?
    let hasInferenceScript: Bool
    let hasPython: Bool
    let sessionPreviewDir: URL
    /// True when running the Python runtime shipped inside the app (see package_app.sh).
    let usesBundledPython: Bool
    /// Writable per-user cache for matplotlib/numba (the app bundle itself is never written to).
    let cacheDir: URL

    @ObservationIgnored private var activeProcess: Process?
    @ObservationIgnored private var isCancelled: Bool = false
    @ObservationIgnored private var nextLogID = 0
    private static let maxLogLines = 5_000

    /// - Parameter resourcesURL: where the app keeps `delimiter_tool.py` and, in a self-contained build
    ///   (package_app.sh), the `De-limiter` and `python` folders. Defaults to the app bundle's Resources.
    init(resourcesURL: URL? = Bundle.main.resourceURL) {
        let fm = FileManager.default

        // Locate De-limiter repo: $DELIMITER_DIR, inside the app (self-contained build), next to the app
        // bundle, the working dir, or ~/De-limiter
        let bundleParent = Bundle.main.bundleURL.deletingLastPathComponent()
        var candidates: [URL] = []
        if let envDir = ProcessInfo.processInfo.environment["DELIMITER_DIR"], !envDir.isEmpty {
            candidates.append(URL(fileURLWithPath: envDir))
        }
        if let resourcesURL {
            candidates.append(resourcesURL.appendingPathComponent("De-limiter"))
        }
        candidates += [
            bundleParent.appendingPathComponent("De-limiter"),
            bundleParent,
            URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("De-limiter"),
            fm.homeDirectoryForCurrentUser.appendingPathComponent("De-limiter")
        ]
        let foundRepo = candidates.first {
            fm.fileExists(atPath: $0.appendingPathComponent("inference.py").path)
        }
        self.delimiterDir = (foundRepo ?? candidates[0]).standardizedFileURL
        self.hasInferenceScript = foundRepo != nil

        // Python: the runtime inside the app, then the repo's .venv, then whatever python3 exists
        let bundledPython = resourcesURL?.appendingPathComponent("python/bin/python3")
        let venvPy = delimiterDir.appendingPathComponent(".venv/bin/python")
        if let bundledPython, fm.isExecutableFile(atPath: bundledPython.path) {
            self.pythonURL = bundledPython
            self.usesBundledPython = true
            self.hasPython = true
        } else if fm.isExecutableFile(atPath: venvPy.path) {
            self.pythonURL = venvPy
            self.usesBundledPython = false
            self.hasPython = true
        } else {
            self.pythonURL = Self.findBinary(named: "python3") ?? URL(fileURLWithPath: "/usr/bin/python3")
            self.usesBundledPython = false
            self.hasPython = false
        }

        // delimiter_tool.py: inside the app (build_app.sh copies it there), else next to the app or in the working dir
        let toolCandidates = [resourcesURL, bundleParent, URL(fileURLWithPath: fm.currentDirectoryPath)]
            .compactMap { $0?.appendingPathComponent("delimiter_tool.py") }
        self.audioToolURL = toolCandidates.first { fm.fileExists(atPath: $0.path) }

        let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first ?? fm.temporaryDirectory
        self.cacheDir = caches.appendingPathComponent(Bundle.main.bundleIdentifier ?? "DelimiterStudio")
        for sub in ["matplotlib", "numba"] {
            try? fm.createDirectory(at: cacheDir.appendingPathComponent(sub), withIntermediateDirectories: true)
        }

        let previewBase = fm.temporaryDirectory.appendingPathComponent("DelimiterGUI_Preview_\(UUID().uuidString)")
        try? fm.createDirectory(at: previewBase, withIntermediateDirectories: true)
        self.sessionPreviewDir = previewBase

        discoverModels()
        appendLog("De-limiter Studio initialized.")
        appendLog("Repo: \(delimiterDir.path)")
        if !hasInferenceScript {
            appendLog("WARNING: inference.py not found. Put the De-limiter repo next to the app or set DELIMITER_DIR.")
        }
        appendLog("Python: \(pythonURL.path)\(usesBundledPython ? " (bundled)" : "")")
        if !hasPython {
            appendLog("WARNING: no bundled Python or repo .venv found; this Python may not have torch installed.")
        }
        if let audioToolURL {
            appendLog("Audio tool: \(audioToolURL.path)")
        } else {
            appendLog("WARNING: delimiter_tool.py not found. Keep it next to the app (build_app.sh copies it inside).")
        }

        // Don't leave inference running or preview WAVs behind when the app quits
        _ = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.activeProcess?.terminate()
                if let dir = self?.sessionPreviewDir {
                    try? FileManager.default.removeItem(at: dir)
                }
            }
        }
    }

    private static func findBinary(named name: String) -> URL? {
        let searchDirs = [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/opt/local/bin",
            "/usr/bin",
            "/bin"
        ]
        for dir in searchDirs {
            let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
            if FileManager.default.isExecutableFile(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    func discoverModels() {
        let weightDir = delimiterDir.appendingPathComponent("weight")
        guard let items = try? FileManager.default.contentsOfDirectory(at: weightDir, includingPropertiesForKeys: nil) else {
            return
        }
        // inference.py needs both <name>.pth (weights) and <name>.json (model config)
        let fileNames = Set(items.map(\.lastPathComponent))
        let models = items
            .filter { $0.pathExtension == "pth" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .filter { fileNames.contains("\($0).json") }
            .sorted()
        if !models.isEmpty {
            availableTargets = models
            if !models.contains(targetModel) {
                targetModel = models[0]
            }
        }
    }

    func appendLog(_ line: String) {
        let timestamp = Self.timeFormatter.string(from: Date())
        logLines.append(LogLine(id: nextLogID, text: "[\(timestamp)] \(line)"))
        nextLogID += 1
        // Capped so a long session can't make the log view slow
        if logLines.count > Self.maxLogLines {
            logLines.removeFirst(logLines.count - Self.maxLogLines)
        }
    }

    func clearLog() {
        logLines.removeAll()
    }

    var logText: String {
        logLines.map(\.text).joined(separator: "\n")
    }

    private static let timeFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss"
        return df
    }()

    // MARK: - File & Folder Management

    nonisolated static let supportedExtensions: Set<String> = [
        "wav", "mp3", "flac", "m4a", "aiff", "aif", "ogg", "opus"
    ]

    func pickFiles() {
        let panel = NSOpenPanel()
        panel.title = "Select Audio Files to De-limit"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.audio]
        if panel.runModal() == .OK {
            addURLs(panel.urls)
        }
    }

    func pickFolder() {
        let panel = NSOpenPanel()
        panel.title = "Select Album or Library Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            addURLs(panel.urls)
        }
    }

    func pickOutputFolder() {
        let panel = NSOpenPanel()
        panel.title = "Choose Output Directory"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK {
            customOutputFolder = panel.url
        }
    }

    func addURLs(_ urls: [URL]) {
        guard !isRunning, !urls.isEmpty else { return }
        Task {
            // Enumerate off the main thread: big libraries or network drives would otherwise beachball
            let found = await Task.detached(priority: .userInitiated) {
                Self.collectAudioFiles(from: urls)
            }.value

            // The queue must not change under a running batch (it indexes into `tracks`)
            guard !isRunning else {
                appendLog("Ignored \(found.count) file(s) added while a batch was running.")
                return
            }

            var known = Set(tracks.map(\.sourceURL))
            var addedCount = 0
            for file in found {
                guard known.insert(file.url).inserted else { continue }
                tracks.append(TrackItem(id: UUID(), sourceURL: file.url, relativeAlbumFolder: file.albumFolder))
                addedCount += 1
            }

            if selectedTrackID == nil, let first = tracks.first {
                selectedTrackID = first.id
                preloadTrackMetricsIfNeeded(for: first.id)
            }
            statusBanner = addedCount > 0
                ? "Added \(addedCount) track(s) (\(tracks.count) total in queue)"
                : "No new supported audio files found"
        }
    }

    /// Expands folders recursively and returns supported audio files in Finder order, each with its
    /// album folder path relative to the parent of the added folder (single files use their folder's name).
    nonisolated private static func collectAudioFiles(from urls: [URL]) -> [(url: URL, albumFolder: String)] {
        let fm = FileManager.default
        let finderOrder: (URL, URL) -> Bool = {
            $0.path.localizedStandardCompare($1.path) == .orderedAscending
        }
        var result: [(url: URL, albumFolder: String)] = []

        for rawURL in urls.sorted(by: finderOrder) {
            let url = rawURL.standardizedFileURL
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }

            guard isDir.boolValue else {
                if supportedExtensions.contains(url.pathExtension.lowercased()) {
                    result.append((url, url.deletingLastPathComponent().lastPathComponent))
                }
                continue
            }

            // Enumerate audio files recursively inside directory
            guard let enumerator = fm.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else { continue }

            var foundURLs: [URL] = []
            for case let fileURL as URL in enumerator {
                guard supportedExtensions.contains(fileURL.pathExtension.lowercased()),
                      (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
                else { continue }
                foundURLs.append(fileURL.standardizedFileURL)
            }

            let rootParent = url.deletingLastPathComponent().path
            for fileURL in foundURLs.sorted(by: finderOrder) {
                let parentFolder = fileURL.deletingLastPathComponent().path
                let relFolder = parentFolder.hasPrefix(rootParent + "/")
                    ? String(parentFolder.dropFirst(rootParent.count + 1))
                    : fileURL.deletingLastPathComponent().lastPathComponent
                result.append((fileURL, relFolder))
            }
        }
        return result
    }

    func clearQueue() {
        guard !isRunning else { return }
        tracks.forEach(deletePreview)
        tracks.removeAll()
        selectedTrackID = nil
        statusBanner = "Queue cleared"
    }

    func removeTrack(_ id: UUID) {
        guard !isRunning else { return }
        tracks.filter { $0.id == id }.forEach(deletePreview)
        tracks.removeAll { $0.id == id }
        if selectedTrackID == id {
            selectedTrackID = tracks.first?.id
        }
    }

    private func deletePreview(_ track: TrackItem) {
        if let url = track.previewOrigWAV {
            try? FileManager.default.removeItem(at: url)
        }
    }

    var selectedTrack: TrackItem? {
        guard let id = selectedTrackID else { return nil }
        return tracks.first { $0.id == id }
    }

    // MARK: - Processing Pipeline

    /// Preloads original loudness and waveform peaks when the user selects an unprocessed track in the queue.
    func preloadTrackMetricsIfNeeded(for id: UUID?, player: ABPlayerController? = nil) {
        guard !isRunning, let id,
              let idx = tracks.firstIndex(where: { $0.id == id }),
              tracks[idx].origPeaks.isEmpty else { return }
        let url = tracks[idx].sourceURL
        Task {
            let metrics = await measureLoudness(url)
            guard !isRunning,
                  let currentIdx = tracks.firstIndex(where: { $0.id == id }),
                  tracks[currentIdx].origPeaks.isEmpty else { return }
            tracks[currentIdx].origLUFS = metrics.lufs
            tracks[currentIdx].origCoreLUFS = metrics.core_lufs
            tracks[currentIdx].origPeak = metrics.peak
            tracks[currentIdx].origPeaks = metrics.peaks ?? []
            player?.updateCoreLoudness(for: tracks[currentIdx])
        }
    }

    func startProcessing(player: ABPlayerController) {
        guard !isRunning, !tracks.isEmpty else { return }
        isRunning = true
        isCancelled = false
        statusBanner = "Starting De-limiter pipeline…"

        // Clear results from any previous run so stale metrics or files can't outlive a failure,
        // and reload the player so it doesn't hold output files that are about to be rewritten
        tracks = tracks.map { $0.resetForNewRun() }
        player.load(track: selectedTrack)

        Task {
            await runPipeline(player: player)
        }
    }

    func cancelProcessing() {
        guard isRunning else { return }
        isCancelled = true
        activeProcess?.terminate()
        appendLog("Cancellation requested by user.")
        statusBanner = "Cancelling…"
    }

    /// Single exit point for a run: anything still in flight is marked cancelled (or interrupted),
    /// so no row is left with a spinner.
    private func finishRun(_ banner: String) {
        for i in tracks.indices where tracks[i].status.isProcessing {
            tracks[i].status = isCancelled ? .cancelled : .failed("Interrupted")
        }
        isRunning = false
        statusBanner = banner
    }

    private func runPipeline(player: ABPlayerController) async {
        let fm = FileManager.default
        let batchTemp = fm.temporaryDirectory.appendingPathComponent("DelimiterBatch_\(UUID().uuidString)")
        let inDir = batchTemp.appendingPathComponent("in")
        let outDir = batchTemp.appendingPathComponent("out")

        defer {
            try? fm.removeItem(at: batchTemp)
        }

        do {
            try fm.createDirectory(at: inDir, withIntermediateDirectories: true)
            try fm.createDirectory(at: outDir, withIntermediateDirectories: true)
        } catch {
            appendLog("Failed creating temporary staging directory: \(error.localizedDescription)")
            finishRun("Failed creating temp directory")
            return
        }

        guard audioToolURL != nil else {
            appendLog("ERROR: delimiter_tool.py not found; it should be next to the app or inside it.")
            for i in tracks.indices {
                tracks[i].status = .failed("delimiter_tool.py not found")
            }
            finishRun("delimiter_tool.py not found")
            return
        }

        appendLog("=== Starting Batch (\(tracks.count) track(s)) ===")

        // Step 1: Prepare 44.1kHz stereo WAVs for inference.py & measure original loudness
        var staged: [(index: Int, token: String)] = []

        for idx in tracks.indices {
            if isCancelled { break }
            let track = tracks[idx]
            let token = String(format: "%03d", idx + 1)
            let stagedWAV = inDir.appendingPathComponent("\(token).wav")

            tracks[idx].status = .preparing
            statusBanner = "[\(idx + 1)/\(tracks.count)] Preparing 44.1kHz WAV: \(track.displayName)"
            appendLog("Converting \(track.sourceURL.lastPathComponent) -> 44.1kHz stereo 32-bit float WAV…")

            // The model is stereo-only and inference.py runs the whole batch in one process, so a
            // mono or multichannel file has to be converted here or it would fail every track.
            // The same call measures loudness & true peak of the staged audio: exactly what the model
            // hears and A plays.
            let metrics = await stageInput(track.sourceURL, to: stagedWAV)
            if isCancelled { break }

            guard let metrics, fm.fileExists(atPath: stagedWAV.path) else {
                tracks[idx].status = .failed("Couldn't read this file")
                continue
            }
            tracks[idx].origLUFS = metrics.lufs
            tracks[idx].origCoreLUFS = metrics.core_lufs
            tracks[idx].origPeak = metrics.peak
            tracks[idx].origPeaks = metrics.peaks ?? []
            if let l = metrics.lufs, let p = metrics.peak {
                appendLog(String(format: "  Original loudness: %.2f LUFS, True Peak: %.2f dBFS", l, p))
            }

            // Copy to session preview cache for fast A/B playback
            let previewOrig = sessionPreviewDir.appendingPathComponent("orig_\(track.id.uuidString).wav")
            try? fm.removeItem(at: previewOrig)
            if (try? fm.copyItem(at: stagedWAV, to: previewOrig)) != nil {
                tracks[idx].previewOrigWAV = previewOrig
            }

            staged.append((index: idx, token: token))
        }

        if isCancelled || staged.isEmpty {
            finishRun(isCancelled ? "Cancelled" : "No valid tracks to process")
            return
        }

        // Step 2: Run De-limiter inference.py over staged directory
        for item in staged {
            tracks[item.index].status = .delimiting
        }
        statusBanner = "Running De-limiter model inference (\(targetModel))…"

        let pyArgs = inferenceArguments(dataRoot: inDir, outputDirectory: outDir)
        appendLog("$ \(pythonURL.path) " + pyArgs.joined(separator: " "))

        let inferRes = await runProcess(
            executable: pythonURL,
            arguments: pyArgs,
            currentDirectory: delimiterDir,
            streamToLog: true
        )

        if isCancelled {
            finishRun("Cancelled")
            return
        }

        if inferRes.exitCode != 0 {
            // inference.py writes each output as soon as it's done, so tracks finished before the
            // failure are still usable; Step 3 fails only the tracks that have no output file.
            appendLog("WARNING: inference.py exited with code \(inferRes.exitCode); keeping the outputs it already wrote")
        }

        // Step 3: Locate raw delimited WAV outputs and measure loudness
        var rawOutWAVs: [Int: URL] = [:]
        for item in staged {
            if isCancelled { break }
            let idx = item.index
            let rawWAV = outDir.appendingPathComponent("\(item.token).wav")
            guard fm.fileExists(atPath: rawWAV.path) else {
                tracks[idx].status = .failed(inferRes.exitCode == 0 ? "Missing output WAV" : "inference.py failed")
                continue
            }

            tracks[idx].status = .measuring
            statusBanner = "Measuring output loudness: \(tracks[idx].displayName)…"
            let metrics = await measureLoudness(rawWAV)
            tracks[idx].rawOutLUFS = metrics.lufs
            tracks[idx].rawOutPeak = metrics.peak
            rawOutWAVs[idx] = rawWAV
        }

        if isCancelled {
            finishRun("Cancelled")
            return
        }

        // Step 4: Compute per-track gain if Album Relative Rebalance mode is enabled (grouped by album folder)
        var computedGainsDB: [Int: Double] = [:]
        if loudnessMode == .albumRelative && !rawOutWAVs.isEmpty {
            let albums = Dictionary(grouping: staged.map { $0.index }, by: { tracks[$0].albumKey })
            for albumKey in albums.keys.sorted() {
                let indices = albums[albumKey] ?? []
                // The loudest original lands at the target and every other track keeps its offset from
                // it; then the whole album shifts down if any track would exceed the true-peak ceiling.
                let loudestOrig = indices.compactMap { tracks[$0].origLUFS }.max() ?? targetLoudnessLUFS
                var shift: Double = 0.0
                for idx in indices {
                    guard let lOrig = tracks[idx].origLUFS,
                          let lOut = tracks[idx].rawOutLUFS,
                          let pOut = tracks[idx].rawOutPeak else {
                        continue   // unmeasured tracks get 0 dB and don't affect the album
                    }
                    let gain = targetLoudnessLUFS + (lOrig - loudestOrig) - lOut
                    computedGainsDB[idx] = gain
                    shift = min(shift, peakCeilingDBFS - pOut - gain)
                }
                for idx in indices {
                    computedGainsDB[idx]? += shift
                }
                appendLog(String(format: "Album '%@': loudest orig = %.2f LUFS, ceiling shift = %.2f dB",
                                 (albumKey as NSString).abbreviatingWithTildeInPath, loudestOrig, shift))
            }
        }

        // Step 5: Write the final file (gain applied in the same pass) and update metrics
        let multipleAlbums = Set(tracks.map(\.albumKey)).count > 1
        var usedOutputPaths: Set<String> = []
        var completedCount = 0

        for item in staged {
            if isCancelled { break }
            let idx = item.index
            guard let rawWAV = rawOutWAVs[idx] else { continue }

            let track = tracks[idx]
            tracks[idx].status = .writing

            let destDir = resolveOutputDirectory(for: track, multipleAlbums: multipleAlbums)
            do {
                try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
            } catch {
                tracks[idx].status = .failed("Can't create output folder")
                appendLog("ERROR creating \(destDir.path): \(error.localizedDescription)")
                continue
            }

            let baseName = uniqueBaseName(for: track.displayName, in: destDir, used: &usedOutputPaths)
            let destURL = destDir.appendingPathComponent("\(baseName).\(outputFormat.fileExtension)")
            let gainDB = computedGainsDB[idx] ?? 0.0
            statusBanner = "[\(completedCount + 1)/\(rawOutWAVs.count)] Writing \(destURL.lastPathComponent)…"

            let written = await writeOutput(
                source: rawWAV,
                gainDB: gainDB,
                tagsFrom: copyMetadataTags ? track.sourceURL : nil,
                destination: destURL
            )
            if isCancelled { break }
            guard let written else {
                tracks[idx].status = .failed("Writing output failed")
                continue
            }
            tracks[idx].outputURL = destURL
            tracks[idx].finalLUFS = written.lufs
            tracks[idx].finalCoreLUFS = written.core_lufs
            tracks[idx].finalPeak = written.peak
            tracks[idx].finalPeaks = written.peaks ?? []

            tracks[idx].status = .done
            completedCount += 1
            appendLog("Finished: \(destURL.path)")
        }

        finishRun(isCancelled ? "Cancelled" : "Completed \(completedCount)/\(staged.count) track(s)")
        appendLog("=== Batch \(isCancelled ? "Cancelled" : "Complete"): \(completedCount)/\(staged.count) succeeded ===")

        // Reload selected track in A/B player so B: De-limited is immediately playable
        if let sel = selectedTrack {
            player.load(track: sel)
        }
    }

    private func inferenceArguments(dataRoot: URL, outputDirectory: URL) -> [String] {
        func flag(_ name: String, _ enabled: Bool) -> String {
            "--\(name)=\(enabled ? "True" : "False")"
        }

        let outputLoudnorm: String
        switch loudnessMode {
        case .perTrackInference:
            outputLoudnorm = String(format: "%.2f", targetLoudnessLUFS)
        case .albumRelative:
            // Let inference.py output default -14 LUFS and rebalance accurately per album/batch afterwards
            outputLoudnorm = "-14.0"
        case .rawModelOutput:
            outputLoudnorm = "None"
        }

        return [
            "-m", "inference",
            "--target=\(targetModel)",
            "--data_root=\(dataRoot.path)",
            "--output_directory=\(outputDirectory.path)",
            flag("use_gpu", false),
            flag("use_mps", computeDevice == .mps),
            flag("avoid_clipping", avoidClipping),
            flag("use_singletrackset", useSingleTrackSet),
            flag("save_histogram", false),
            flag("save_16k_mono", false),
            flag("save_sgi_params", false),
            flag("save_name_as_target", false),
            "--loudnorm_input_lufs=" + (enableInputLoudnorm ? String(format: "%.2f", inputLoudnormLUFS) : "None"),
            "--save_output_loudnorm=" + outputLoudnorm,
            "--save_mixed_output=None"
        ]
    }

    private func resolveOutputDirectory(for track: TrackItem, multipleAlbums: Bool) -> URL {
        if let custom = customOutputFolder {
            if multipleAlbums && !track.relativeAlbumFolder.isEmpty {
                return custom.appendingPathComponent(track.relativeAlbumFolder)
            }
            return custom
        }
        let parent = track.sourceURL.deletingLastPathComponent()
        return URL(fileURLWithPath: parent.path + "_delimited")
    }

    /// Returns a base name no other track in this batch has used in `dir`, so two sources with the same
    /// name (e.g. from same-named album folders) can't overwrite each other.
    private func uniqueBaseName(for name: String, in dir: URL, used: inout Set<String>) -> String {
        var candidate = name
        var counter = 2
        // APFS is case-insensitive by default, so compare lowercased paths
        while used.contains(dir.appendingPathComponent(candidate).path.lowercased()) {
            candidate = "\(name) (\(counter))"
            counter += 1
        }
        used.insert(dir.appendingPathComponent(candidate).path.lowercased())
        return candidate
    }

    /// Renders a raw inference.py WAV as the chosen format with `gainDB` applied, optionally
    /// copying basic tags from `tagsFrom`. Returns the loudness of the audio as written, or nil on failure.
    private func writeOutput(source: URL, gainDB: Double, tagsFrom: URL?, destination: URL) async -> AudioMetrics? {
        let bitrateKbps = outputFormat == .opus ? opusBitrateKbps : aacBitrateKbps
        var args = [
            "write", source.path, destination.path,
            "--format", outputFormat.fileExtension,
            "--bitrate", String(Int(bitrateKbps.rounded())),
            String(format: "--gain=%.4f", gainDB)   // "=" form, so a negative gain isn't read as an option
        ]
        if let tagsFrom {
            args += ["--tags-from", tagsFrom.path]
        }
        guard let metrics: AudioMetrics = await runAudioTool(args) else {
            try? FileManager.default.removeItem(at: destination)   // never leave a partial file behind
            return nil
        }
        return metrics
    }

    /// Decodes `source` to the 44.1 kHz stereo float WAV inference.py expects, and measures it.
    private func stageInput(_ source: URL, to destination: URL) async -> AudioMetrics? {
        await runAudioTool(["stage", source.path, destination.path])
    }

    /// Integrated loudness (LUFS) and true peak (dBTP); both are nil if the file couldn't be measured.
    private func measureLoudness(_ file: URL) async -> AudioMetrics {
        let results: [AudioMetrics]? = await runAudioTool(["measure", file.path])
        return results?.first ?? AudioMetrics()
    }

    /// Runs one delimiter_tool.py command with the same Python as inference.py and decodes the JSON on its
    /// final `RESULT <json>` line. Returns nil (after logging the tool's output) if the command failed.
    private func runAudioTool<Value: Decodable>(_ arguments: [String]) async -> Value? {
        guard let audioToolURL else { return nil }
        let res = await runProcess(executable: pythonURL, arguments: [audioToolURL.path] + arguments)
        let resultLine = res.output.split(whereSeparator: \.isNewline).last { $0.hasPrefix("RESULT ") }
        if res.exitCode == 0, let json = resultLine?.dropFirst("RESULT ".count),
           let value = try? JSONDecoder().decode(Value.self, from: Data(json.utf8)) {
            return value
        }
        if !isCancelled {
            let details = res.output.trimmingCharacters(in: .whitespacesAndNewlines)
            appendLog("ERROR (delimiter_tool.py \(arguments.first ?? "")): "
                      + (details.isEmpty ? "exit code \(res.exitCode)" : details))
        }
        return nil
    }

    // MARK: - Safe Subprocess Execution (No Shell Interpolation)

    private func runProcess(
        executable: URL,
        arguments: [String],
        currentDirectory: URL? = nil,
        streamToLog: Bool = false
    ) async -> (exitCode: Int32, output: String) {
        await withCheckedContinuation { continuation in
            let proc = Process()
            proc.executableURL = executable
            proc.arguments = arguments
            if let cwd = currentDirectory {
                proc.currentDirectoryURL = cwd
            }

            var env = ProcessInfo.processInfo.environment
            let existingPath = env["PATH"] ?? "/usr/bin:/bin"
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + existingPath
            env["MPLCONFIGDIR"] = cacheDir.appendingPathComponent("matplotlib").path
            env["MPLBACKEND"] = "Agg"   // render plots off-screen; no Dock icon for the Python process
            env["NUMBA_CACHE_DIR"] = cacheDir.appendingPathComponent("numba").path
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   // run any op MPS lacks on the CPU instead of crashing
            if usesBundledPython {
                // Isolate the bundled runtime from whatever Python setup this Mac has, and never write
                // into the app bundle (bytecode is precompiled by package_app.sh)
                env["PYTHONNOUSERSITE"] = "1"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env.removeValue(forKey: "PYTHONPATH")
                env.removeValue(forKey: "PYTHONHOME")
            }
            proc.environment = env

            let pipe = Pipe()
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = pipe
            proc.standardError = pipe

            do {
                try proc.run()
            } catch {
                appendLog("ERROR launching \(executable.lastPathComponent): \(error.localizedDescription)")
                continuation.resume(returning: (-1, error.localizedDescription))
                return
            }
            self.activeProcess = proc

            // Always drain the pipe on a background queue while the process runs
            // to prevent 64 KB OS pipe buffer deadlocks.
            // The view model is main-actor isolated (so Sendable) and lives as long as the app,
            // so the transient closures below can capture it strongly.
            let readHandle = pipe.fileHandleForReading
            DispatchQueue.global(qos: .userInitiated).async {
                var collectedData = Data()
                var pendingLine = Data()
                while true {
                    let chunk = readHandle.availableData
                    if chunk.isEmpty {
                        break
                    }
                    collectedData.append(chunk)
                    if streamToLog {
                        // Emit only complete lines, so lines or multi-byte characters split across
                        // chunks aren't dropped or garbled
                        pendingLine.append(chunk)
                        let lines = Self.extractCompleteLines(from: &pendingLine)
                        if !lines.isEmpty {
                            DispatchQueue.main.async {
                                MainActor.assumeIsolated {
                                    lines.forEach { self.appendLog($0) }
                                }
                            }
                        }
                    }
                }

                proc.waitUntilExit()
                let status = proc.terminationStatus
                // Lossy decoding: one invalid byte must not blank the whole output
                let outputStr = String(decoding: collectedData, as: UTF8.self)
                let lastLine = String(decoding: pendingLine, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                // The main queue is FIFO, so streamed lines are logged before the result is delivered
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        if !lastLine.isEmpty {
                            self.appendLog(lastLine)
                        }
                        if self.activeProcess === proc {
                            self.activeProcess = nil
                        }
                        continuation.resume(returning: (status, outputStr))
                    }
                }
            }
        }
    }

    /// Splits complete `\n`/`\r`-terminated lines off the front of `buffer`, leaving any partial line.
    /// (`\r` covers progress-bar updates.)
    nonisolated private static func extractCompleteLines(from buffer: inout Data) -> [String] {
        var lines: [String] = []
        while let end = buffer.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
            let line = String(decoding: buffer[buffer.startIndex..<end], as: UTF8.self)
                .trimmingCharacters(in: .whitespaces)
            if !line.isEmpty {
                lines.append(line)
            }
            buffer.removeSubrange(buffer.startIndex...end)
        }
        return lines
    }
}

// MARK: - SwiftUI Views

struct ContentView: View {
    let vm: DelimiterViewModel
    let player: ABPlayerController

    var body: some View {
        HSplitView {
            // Left Inspector Sidebar
            SidebarControlsView(vm: vm)
                .frame(minWidth: 320, idealWidth: 340, maxWidth: 380)

            // Right Main Workspace
            VStack(spacing: 0) {
                TopHeaderBar(vm: vm, player: player)
                Divider()

                // Queue & Loudness Metrics Table
                TrackQueueTableView(vm: vm, player: player)

                Divider()

                // Synchronized A/B Audio Player & Core-Matched Waveform Bar
                ABPlayerBarView(vm: vm, player: player)

                // Collapsible Live Console Log
                if vm.showLogPanel {
                    Divider()
                    ConsoleLogPanelView(vm: vm)
                        .frame(height: 150)
                }
            }
            .frame(minWidth: 680)
        }
        .frame(minWidth: 1020, minHeight: 680)
        .onChange(of: vm.selectedTrackID) {
            vm.preloadTrackMetricsIfNeeded(for: vm.selectedTrackID, player: player)
            // The row ▶︎ button loads its own track; reloading here would stop that playback
            guard player.loadedTrackID != vm.selectedTrackID else { return }
            player.load(track: vm.selectedTrack)
        }
    }
}

// MARK: - Sidebar Controls

struct SidebarControlsView: View {
    @Bindable var vm: DelimiterViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // 1. De-limiter Model & inference.py Settings
                GroupBox(label: Label("De-limiter Model (inference.py)", systemImage: "waveform.path.ecg")) {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Device:")
                            Spacer()
                            Picker("", selection: $vm.computeDevice) {
                                ForEach(ComputeDevice.allCases) { dev in
                                    Text(dev.rawValue).tag(dev)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 175)
                        }

                        Toggle("Chunk long tracks (--use_singletrackset)", isOn: $vm.useSingleTrackSet)
                            .help("Recommended for long tracks or lower memory usage")

                        Toggle("Prevent 0 dBFS clipping (--avoid_clipping)", isOn: $vm.avoidClipping)
                            .help("Forcefully normalizes output if any sample exceeds 0 dBFS")

                        Divider()

                        Toggle("Normalize input loudness (--loudnorm_input_lufs)", isOn: $vm.enableInputLoudnorm)
                        if vm.enableInputLoudnorm {
                            HStack {
                                Text("Input Target:")
                                Slider(value: $vm.inputLoudnormLUFS, in: -24.0...(-6.0), step: 0.5)
                                Text(String(format: "%.1f LUFS", vm.inputLoudnormLUFS))
                                    .font(.system(.caption, design: .monospaced))
                                    .frame(width: 68, alignment: .trailing)
                            }
                        }
                    }
                    .padding(8)
                }

                // 2. Loudness & Ceiling Strategy
                GroupBox(label: Label("Loudness & Peak Ceiling", systemImage: "slider.vertical.3")) {
                    VStack(alignment: .leading, spacing: 10) {
                        Picker("Mode:", selection: $vm.loudnessMode) {
                            ForEach(LoudnessMode.allCases) { mode in
                                Text(mode.rawValue).tag(mode)
                            }
                        }
                        .pickerStyle(.menu)

                        if vm.loudnessMode != .rawModelOutput {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(vm.loudnessMode == .albumRelative ? "Album Loudest Track:" : "Output Loudnorm:")
                                    Spacer()
                                    Text(String(format: "%.1f LUFS", vm.targetLoudnessLUFS))
                                        .font(.system(.caption, design: .monospaced))
                                        .bold()
                                }
                                Slider(value: $vm.targetLoudnessLUFS, in: -24.0...(-6.0), step: 0.5)
                            }
                        }

                        if vm.loudnessMode == .albumRelative {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("True Peak Ceiling:")
                                    Spacer()
                                    Text(String(format: "%.1f dBFS", vm.peakCeilingDBFS))
                                        .font(.system(.caption, design: .monospaced))
                                        .bold()
                                }
                                Slider(value: $vm.peakCeilingDBFS, in: -6.0...0.0, step: 0.1)
                                Text("Preserves track-to-track dynamic differences per album folder and shifts the album down if any track peaks above ceiling.")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(8)
                }

                // 3. Output Format (WAV, FLAC, AAC, Opus)
                GroupBox(label: Label("Output Format", systemImage: "opticaldisc")) {
                    VStack(alignment: .leading, spacing: 10) {
                        Picker("Format:", selection: $vm.outputFormat) {
                            ForEach(OutputFormat.allCases) { fmt in
                                Text(fmt.rawValue).tag(fmt)
                            }
                        }
                        .pickerStyle(.segmented)

                        if vm.outputFormat == .aac {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("Bitrate (Constrained VBR):")
                                    Spacer()
                                    Text("\(Int(vm.aacBitrateKbps.rounded())) kbps")
                                        .font(.system(.caption, design: .monospaced))
                                        .bold()
                                }
                                Slider(value: $vm.aacBitrateKbps, in: 96...320, step: 16)
                                Text("Apple AAC-LC (-q 127 -s 2) via macOS afconvert (.m4a)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        } else if vm.outputFormat == .opus {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("Bitrate (VBR):")
                                    Spacer()
                                    Text("\(Int(vm.opusBitrateKbps.rounded())) kbps")
                                        .font(.system(.caption, design: .monospaced))
                                        .bold()
                                }
                                Slider(value: $vm.opusBitrateKbps, in: 64...320, step: 16)
                                Text("Opus VBR at 48 kHz via SoXR VHQ + libopus (.opus)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        } else {
                            Text("Lossless 24-bit, 44.1 kHz (the model's native rate)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }

                        Toggle("Copy Metadata Tags (Title, Artist, Album…)", isOn: $vm.copyMetadataTags)
                            .help("Copies basic tags into WAV (RIFF INFO), FLAC (Vorbis), M4A (iTunes atoms), or Opus (OpusTags)")

                        Divider()

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Output Directory:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(vm.customOutputFolder?.path ?? "Auto (<Input Folder>_delimited)")
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(2)
                                .truncationMode(.middle)

                            HStack {
                                Button("Choose…") {
                                    vm.pickOutputFolder()
                                }
                                if vm.customOutputFolder != nil {
                                    Button("Reset to Auto") {
                                        vm.customOutputFolder = nil
                                    }
                                    Button("Reveal") {
                                        if let folder = vm.customOutputFolder {
                                            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folder.path)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding(8)
                }
            }
            .padding(14)
        }
        .disabled(vm.isRunning)
    }
}

// MARK: - Top Header Bar

struct TopHeaderBar: View {
    let vm: DelimiterViewModel
    let player: ABPlayerController

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Image(systemName: "waveform.badge.magnifyingglass")
                        .font(.title2)
                        .foregroundColor(.accentColor)
                    Text("De-limiter Studio")
                        .font(.headline)
                }
                Text(vm.statusBanner)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 8) {
                Button {
                    vm.pickFiles()
                } label: {
                    Label("Add Files…", systemImage: "doc.badge.plus")
                }
                .disabled(vm.isRunning)

                Button {
                    vm.pickFolder()
                } label: {
                    Label("Add Folder…", systemImage: "folder.badge.plus")
                }
                .disabled(vm.isRunning)

                Button {
                    player.load(track: nil)
                    vm.clearQueue()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .disabled(vm.isRunning || vm.tracks.isEmpty)

                Button {
                    vm.showLogPanel.toggle()
                } label: {
                    Label("Log", systemImage: "terminal")
                }
                .help("Toggle Live Console Log")

                if vm.isRunning {
                    Button(role: .destructive) {
                        vm.cancelProcessing()
                    } label: {
                        Label("Stop", systemImage: "xmark.octagon.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Button {
                        vm.startProcessing(player: player)
                    } label: {
                        Label("Run De-limiter", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.tracks.isEmpty)
                }
            }
            .fixedSize(horizontal: true, vertical: false)
            .layoutPriority(1)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

// MARK: - Track Queue & Metrics Table

struct TrackQueueTableView: View {
    @Bindable var vm: DelimiterViewModel
    let player: ABPlayerController

    var body: some View {
        ZStack {
            if vm.tracks.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "waveform.and.magnifyingglass")
                        .font(.system(size: 44))
                        .foregroundColor(.secondary)
                    Text("Drop WAV, MP3, FLAC, or Album Folders Here")
                        .font(.title3)
                        .fontWeight(.medium)
                    Text("Or click “Add Files…” / “Add Folder…” above to queue tracks for De-limiter.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    // Table Column Header
                    HStack(spacing: 8) {
                        Text("Status").frame(width: 115, alignment: .leading)
                        Text("Track").frame(maxWidth: .infinity, alignment: .leading)
                        Text("Orig LUFS").frame(width: 78, alignment: .trailing)
                        Text("Orig Peak").frame(width: 78, alignment: .trailing)
                        Text("Out LUFS").frame(width: 78, alignment: .trailing)
                        Text("Out Peak").frame(width: 78, alignment: .trailing)
                        Text("Δ LUFS").frame(width: 72, alignment: .trailing)
                    }
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    .background(Color(NSColor.windowBackgroundColor))

                    Divider()

                    List(selection: $vm.selectedTrackID) {
                        ForEach(vm.tracks) { track in
                            TrackRowView(track: track, vm: vm, player: player)
                                .tag(track.id)
                        }
                    }
                    .listStyle(.inset(alternatesRowBackgrounds: true))
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(vm.isDropTargeted ? Color.accentColor : Color.clear, lineWidth: 3)
                .padding(4)
        )
        .onDrop(of: [.fileURL], isTargeted: $vm.isDropTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        // Collect every dropped item first, so a multi-file drop is added once and in Finder order
        guard !vm.isRunning, !providers.isEmpty else { return false }
        let collector = DropCollector(expected: providers.count) { [vm] urls in
            vm.addURLs(urls)
        }
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                let url = (item as? Data).flatMap { URL(dataRepresentation: $0, relativeTo: nil) }
                Task { @MainActor in
                    collector.receive(url)
                }
            }
        }
        return true
    }
}

/// Gathers the URLs of a multi-item drop (each item loads asynchronously) and delivers them as one batch.
@MainActor
private final class DropCollector {
    private var remaining: Int
    private var urls: [URL] = []
    private let completion: ([URL]) -> Void

    init(expected: Int, completion: @escaping ([URL]) -> Void) {
        self.remaining = expected
        self.completion = completion
    }

    func receive(_ url: URL?) {
        if let url {
            urls.append(url)
        }
        remaining -= 1
        if remaining == 0 {
            completion(urls)
        }
    }
}

struct TrackRowView: View {
    let track: TrackItem
    let vm: DelimiterViewModel
    let player: ABPlayerController

    var body: some View {
        HStack(spacing: 8) {
            // Status column
            HStack(spacing: 6) {
                if track.status.isProcessing {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    statusIcon(for: track.status)
                }
                Text(track.status.label)
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1)
            }
            .frame(width: 115, alignment: .leading)
            .help(track.status.label)

            // Track Name + Album subtitle
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(track.displayName)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                    Text(track.sourceExt)
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.secondary.opacity(0.15))
                        .cornerRadius(3)
                }
                Text(track.relativeAlbumFolder)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Metrics
            Text(formatMetric(track.origLUFS))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 78, alignment: .trailing)

            Text(formatMetric(track.origPeak))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 78, alignment: .trailing)

            Text(formatMetric(track.finalLUFS ?? track.rawOutLUFS))
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(track.finalLUFS != nil ? .primary : .secondary)
                .frame(width: 78, alignment: .trailing)

            Text(formatMetric(track.finalPeak ?? track.rawOutPeak))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 78, alignment: .trailing)

            Text(formatDelta(track.loudnessDelta))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(deltaColor(track.loudnessDelta))
                .frame(width: 72, alignment: .trailing)
        }
        .padding(.vertical, 3)
    }

    @ViewBuilder
    private func statusIcon(for status: TrackStatus) -> some View {
        switch status {
        case .pending:
            Image(systemName: "clock").foregroundColor(.secondary)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
        case .cancelled:
            Image(systemName: "xmark.circle").foregroundColor(.secondary)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.red)
        default:
            Image(systemName: "circle.dotted").foregroundColor(.accentColor)
        }
    }

    private func formatMetric(_ val: Double?) -> String {
        guard let v = val else { return "—" }
        return String(format: "%.1f", v)
    }

    private func formatDelta(_ val: Double?) -> String {
        guard let v = val else { return "—" }
        return String(format: "%+.2f dB", v)
    }

    private func deltaColor(_ val: Double?) -> Color {
        guard let v = val else { return .secondary }
        if abs(v) < 0.05 { return .secondary }
        return v < 0 ? .orange : .blue
    }
}

// MARK: - Before / After (A/B) Audio Player & Core-Matched Waveform Bar

struct ABPlayerBarView: View {
    let vm: DelimiterViewModel
    @Bindable var player: ABPlayerController

    var body: some View {
        VStack(spacing: 10) {
            if let track = vm.selectedTrack, (!track.origPeaks.isEmpty || !track.finalPeaks.isEmpty) {
                WaveformComparisonView(track: track, player: player)
            }

            HStack(spacing: 14) {
                // Playback Transport Controls
                HStack(spacing: 8) {
                    Button {
                        player.skip(by: -5)
                    } label: {
                        Image(systemName: "gobackward.5")
                    }
                    .disabled(vm.selectedTrack == nil)

                    Button {
                        player.togglePlayPause()
                    } label: {
                        Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                            .frame(width: 18, height: 18)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.selectedTrack == nil)

                    Button {
                        player.skip(by: 5)
                    } label: {
                        Image(systemName: "goforward.5")
                    }
                    .disabled(vm.selectedTrack == nil)
                }

                // Selected Track Info & Time Readout
                HStack(spacing: 10) {
                    Text(vm.selectedTrack?.displayName ?? "Select a track to A/B preview")
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.tail)

                    if vm.selectedTrack != nil {
                        PlaybackTimeLabel(player: player)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                // A/B Selector & Relative Volume Controls
                HStack(spacing: 14) {
                    Picker("", selection: Binding(
                        get: { player.activeSource },
                        set: { player.switchSource(to: $0) }
                    )) {
                        ForEach(ABSource.allCases) { source in
                            Text(source.rawValue).tag(source)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(width: 220)
                    .padding(.horizontal, 6)
                    .disabled(!player.hasDelimitedAudio)
                    .help("Both versions play in sync, so switching is instant and gapless")

                    Toggle("Relative Volume Match", isOn: $player.loudnessMatchAB)
                        .font(.system(size: 11))
                        .help("Attenuates the louder source using sustained Core Loudness (same metric as the waveform overlay) so you compare pure transient restoration without volume bias")
                }
                .fixedSize(horizontal: true, vertical: false)
                .layoutPriority(1)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

/// Isolated observer for `player.currentTime` so 80 ms playback ticks never re-evaluate the A/B segmented control.
private struct PlaybackTimeLabel: View {
    let player: ABPlayerController

    var body: some View {
        Text("\(Self.formatTime(player.currentTime)) / \(Self.formatTime(player.duration))")
            .font(.system(size: 11, design: .monospaced))
            .monospacedDigit()
            .foregroundColor(.secondary)
            .fixedSize()
    }

    private static func formatTime(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite && seconds >= 0 else { return "0:00" }
        let total = Int(seconds.rounded(.down))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

/// Overlays the Core-Volume-Matched Original waveform (lower opacity) on top of the De-limited waveform
/// so restored peaks and transients rising above the original brickwalled ceiling are immediately visible.
struct WaveformComparisonView: View {
    let track: TrackItem
    let player: ABPlayerController

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 10) {
                Label("Core-Volume Matched Waveform Overlay", systemImage: "waveform")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)

                if let headroom = track.restoredHeadroomDB {
                    Text(String(format: "%+.1f dB Peak Headroom Restored", headroom))
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(RoundedRectangle(cornerRadius: 4).fill(Color.cyan.opacity(0.16)))
                        .foregroundColor(.cyan)
                }

                Spacer()

                HStack(spacing: 12) {
                    if !track.finalPeaks.isEmpty {
                        HStack(spacing: 4) {
                            Circle().fill(Color.cyan).frame(width: 7, height: 7)
                            Text("B: De-limited (Restored Peaks)")
                                .font(.system(size: 10, weight: player.activeSource == .delimited ? .bold : .regular))
                        }
                    }
                    if !track.origPeaks.isEmpty {
                        HStack(spacing: 4) {
                            Circle().fill(Color.orange.opacity(0.75)).frame(width: 7, height: 7)
                            Text(track.finalPeaks.isEmpty ? "A: Original Waveform" : "A: Original (Core-Matched Overlay)")
                                .font(.system(size: 10, weight: player.activeSource == .original ? .bold : .regular))
                        }
                    }
                }
                .foregroundColor(.secondary)
            }

            GeometryReader { geo in
                ZStack {
                    RoundedRectangle(cornerRadius: 7)
                        .fill(Color(NSColor.underPageBackgroundColor).opacity(0.9))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
                        )

                    Canvas { context, size in
                        let midY = size.height * 0.5
                        let halfHeight = size.height * 0.45

                        // Reference grid lines (-3 dB, -6 dB, -12 dB)
                        for linear in [CGFloat(0.7079), CGFloat(0.5012), CGFloat(0.2512)] {
                            let yTop = midY - linear * halfHeight
                            let yBot = midY + linear * halfHeight
                            var grid = Path()
                            grid.move(to: CGPoint(x: 0, y: yTop))
                            grid.addLine(to: CGPoint(x: size.width, y: yTop))
                            grid.move(to: CGPoint(x: 0, y: yBot))
                            grid.addLine(to: CGPoint(x: size.width, y: yBot))
                            context.stroke(
                                grid,
                                with: .color(.secondary.opacity(0.14)),
                                style: StrokeStyle(lineWidth: 0.5, dash: [4, 4])
                            )
                        }

                        // Center zero line
                        var center = Path()
                        center.move(to: CGPoint(x: 0, y: midY))
                        center.addLine(to: CGPoint(x: size.width, y: midY))
                        context.stroke(center, with: .color(.secondary.opacity(0.28)), lineWidth: 0.75)

                        let origPeaks = track.origPeaks
                        let finalPeaks = track.finalPeaks

                        if !origPeaks.isEmpty && !finalPeaks.isEmpty {
                            // Scale Original by Core Loudness difference so sustained body matches De-limited
                            let origRef = track.origCoreLUFS ?? track.origLUFS ?? -14.0
                            let finalRef = track.finalCoreLUFS ?? track.finalLUFS ?? -14.0
                            let origScale = Float(pow(10.0, (finalRef - origRef) / 20.0))

                            let maxOrig = (origPeaks.max() ?? 0) * origScale
                            let maxFinal = finalPeaks.max() ?? 0
                            let globalMax = max(maxOrig, maxFinal, 0.01)

                            let normFinal = CGFloat(1.0 / globalMax)
                            let normOrig = CGFloat(origScale / globalMax)

                            // 1. Base Layer: De-limited waveform (cyan — taller restored peaks/transients)
                            let fCount = finalPeaks.count
                            let fStepX = size.width / CGFloat(fCount)
                            var finalPath = Path()
                            for i in 0..<fCount {
                                let x = (CGFloat(i) + 0.5) * fStepX
                                let h = min(1.0, CGFloat(finalPeaks[i]) * normFinal) * halfHeight
                                finalPath.move(to: CGPoint(x: x, y: midY - h))
                                finalPath.addLine(to: CGPoint(x: x, y: midY + h))
                            }
                            context.stroke(
                                finalPath,
                                with: .color(Color.cyan.opacity(player.activeSource == .delimited ? 0.92 : 0.72)),
                                lineWidth: max(1.0, fStepX * 0.82)
                            )

                            // 2. Overlay Layer: Core-Matched Original waveform (lower opacity orange)
                            let oCount = origPeaks.count
                            let oStepX = size.width / CGFloat(oCount)
                            var origPath = Path()
                            for i in 0..<oCount {
                                let x = (CGFloat(i) + 0.5) * oStepX
                                let h = min(1.0, CGFloat(origPeaks[i]) * normOrig) * halfHeight
                                origPath.move(to: CGPoint(x: x, y: midY - h))
                                origPath.addLine(to: CGPoint(x: x, y: midY + h))
                            }
                            context.stroke(
                                origPath,
                                with: .color(Color.orange.opacity(player.activeSource == .original ? 0.62 : 0.44)),
                                lineWidth: max(1.0, oStepX * 0.82)
                            )

                            // Dashed limiter ceiling lines showing where the brickwalled Original peaked
                            let ceilH = min(1.0, CGFloat(maxOrig / globalMax)) * halfHeight
                            var ceilLine = Path()
                            ceilLine.move(to: CGPoint(x: 0, y: midY - ceilH))
                            ceilLine.addLine(to: CGPoint(x: size.width, y: midY - ceilH))
                            ceilLine.move(to: CGPoint(x: 0, y: midY + ceilH))
                            ceilLine.addLine(to: CGPoint(x: size.width, y: midY + ceilH))
                            context.stroke(
                                ceilLine,
                                with: .color(Color.orange.opacity(0.65)),
                                style: StrokeStyle(lineWidth: 0.85, dash: [4, 3])
                            )
                        } else if !origPeaks.isEmpty {
                            let maxOrig = max(origPeaks.max() ?? 0, 0.01)
                            let normOrig = CGFloat(1.0 / maxOrig)
                            let oCount = origPeaks.count
                            let oStepX = size.width / CGFloat(oCount)
                            var origPath = Path()
                            for i in 0..<oCount {
                                let x = (CGFloat(i) + 0.5) * oStepX
                                let h = min(1.0, CGFloat(origPeaks[i]) * normOrig) * halfHeight
                                origPath.move(to: CGPoint(x: x, y: midY - h))
                                origPath.addLine(to: CGPoint(x: x, y: midY + h))
                            }
                            context.stroke(
                                origPath,
                                with: .color(Color.orange.opacity(0.75)),
                                lineWidth: max(1.0, oStepX * 0.82)
                            )
                        }

                        // Playhead cursor
                        if player.duration > 0 {
                            let progress = min(max(player.currentTime / player.duration, 0.0), 1.0)
                            let playX = CGFloat(progress) * size.width
                            var playhead = Path()
                            playhead.move(to: CGPoint(x: playX, y: 0))
                            playhead.addLine(to: CGPoint(x: playX, y: size.height))
                            context.stroke(playhead, with: .color(.white), lineWidth: 1.6)
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                guard player.duration > 0 else { return }
                                let progress = min(max(0.0, Double(value.location.x / max(1.0, geo.size.width))), 1.0)
                                player.seek(to: progress * player.duration)
                            }
                    )
                }
            }
            .frame(height: 92)
        }
    }
}

// MARK: - Live Console Log View

struct ConsoleLogPanelView: View {
    let vm: DelimiterViewModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Live Process Log", systemImage: "terminal.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(vm.logText, forType: .string)
                }
                .controlSize(.small)
                Button("Clear") {
                    vm.clearLog()
                }
                .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    // Lazy rows: appending a line stays cheap however long the session gets
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(vm.logLines) { line in
                            Text(line.text)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(line.id)
                        }
                    }
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
                }
                .background(Color(NSColor.textBackgroundColor))
                .onChange(of: vm.logLines.last?.id) {
                    if let lastID = vm.logLines.last?.id {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
            }
        }
    }
}

// MARK: - App Entry Point

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct DelimiterStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    // Owned by the app (App.init runs once per launch) and shared with the window
    private let vm = DelimiterViewModel()
    private let player = ABPlayerController()

    var body: some Scene {
        // A single window: a second one would mean a second, independent pipeline and player
        Window("De-limiter Studio", id: "main") {
            ContentView(vm: vm, player: player)
        }
        .windowStyle(.titleBar)
    }
}
