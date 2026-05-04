"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import type { TranscriptSegment } from "@/entity/Transcription";
import EmailGate from "./components/EmailGate";
import RecorderPanel from "./components/RecorderPanel";
import FileUpload from "./components/FileUpload";
import SummaryBlock from "./components/SummaryBlock";
import TranscriptPanel from "./components/TranscriptPanel";
import ChatPanel from "./components/ChatPanel";
import SessionHistory from "./components/SessionHistory";

const EMAIL_KEY = "transcriber_email";

type Status = "idle" | "recording" | "processing" | "done" | "error";

type HistoryRecord = {
  id: string;
  title: string | null;
  summary: string | null;
  duration: number;
  segments: TranscriptSegment[];
  createdAt: string;
};

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [currentTranscriptId, setCurrentTranscriptId] = useState<string | null>(null);
  const [currentTitle, setCurrentTitle] = useState<string | null>(null);
  const [currentSummary, setCurrentSummary] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [showEmailModal, setShowEmailModal] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(EMAIL_KEY);
    if (stored) setUserEmail(stored);
    else setShowEmailModal(true);
  }, []);

  const loadHistory = useCallback(async (email: string) => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`/api/transcriptions?email=${encodeURIComponent(email)}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setHistory(await res.json());
      setHistoryLoaded(true);
    } catch {
      toast.error("Couldn't load your transcripts. Try refreshing.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (userEmail) loadHistory(userEmail);
  }, [userEmail, loadHistory]);

  const handleEmailSubmit = (email: string) => {
    localStorage.setItem(EMAIL_KEY, email);
    setUserEmail(email);
    setShowEmailModal(false);
    setSegments([]);
    setStatus("idle");
    setError(null);
    setHistory([]);
    setHistoryLoaded(false);
  };

  const startRecording = async () => {
    setError(null);
    setSegments([]);
    setCurrentTranscriptId(null);
    setCurrentTitle(null);
    setCurrentSummary(null);
    setElapsed(0);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const isDenied =
        err instanceof DOMException &&
        (err.name === "NotAllowedError" || err.name === "PermissionDeniedError");
      setError(
        isDenied
          ? "Microphone access blocked. Click the lock icon in your browser's address bar → Site settings → Microphone → Allow, then refresh."
          : "Could not access microphone. Make sure a microphone is connected and try again."
      );
      setStatus("error");
      return;
    }

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(timerRef.current!);
      setStatus("processing");

      const blob = new Blob(chunksRef.current, {
        type: chunksRef.current[0]?.type ?? "audio/webm",
      });
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");
      formData.append("email", userEmail ?? "");

      try {
        const res = await fetch("/api/transcribe", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) { setError(data.error ?? "Transcription failed"); setStatus("error"); return; }
        setSegments(data.segments);
        setCurrentTranscriptId(data.id);
        setCurrentSummary(data.summary ?? null);
        setStatus("done");
        if (userEmail) loadHistory(userEmail);
      } catch {
        setError("Network error. Please try again.");
        setStatus("error");
      }
    };

    mediaRecorder.start(250);
    setStatus("recording");
    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
  };

  const stopRecording = () => mediaRecorderRef.current?.stop();

  const handleFileUpload = async (file: File) => {
    setError(null);
    setSegments([]);
    setCurrentTranscriptId(null);
    setCurrentTitle(null);
    setCurrentSummary(null);
    setStatus("processing");

    const formData = new FormData();
    formData.append("audio", file, file.name);
    formData.append("email", userEmail ?? "");

    try {
      const res = await fetch("/api/transcribe", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) { setError(data.error ?? "Transcription failed"); setStatus("error"); return; }
      setSegments(data.segments);
      setCurrentTranscriptId(data.id);
      setCurrentSummary(data.summary ?? null);
      setStatus("done");
      if (userEmail) loadHistory(userEmail);
    } catch {
      setError("Network error. Please try again.");
      setStatus("error");
    }
  };

  const showTranscript = status === "done" || status === "processing";

  return (
    <>
      <EmailGate
        open={showEmailModal}
        userEmail={userEmail}
        onSubmit={handleEmailSubmit}
        onClose={() => setShowEmailModal(false)}
      />

      <div className="min-h-screen bg-[#f1f4f9] lg:flex">

        {/* ── SIDEBAR (desktop only) ── */}
        <aside className="hidden lg:flex w-72 flex-shrink-0 flex-col bg-white border-r border-[#e2e8f2] sticky top-0 h-screen overflow-auto">

          {/* Logo */}
          <div className="px-6 py-5 border-b border-[#e2e8f2]">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-[#5b21b6] to-[#4f46e5] flex items-center justify-center shadow-sm">
                <div className="w-2.5 h-2.5 rounded-full bg-white" />
              </div>
              <span className="font-bold text-[#1e1b4b] text-base tracking-tight">Audia</span>
            </div>
            <p className="text-[10px] text-[#94a3b8] mt-2 leading-relaxed">
              Record conversations — speakers identified automatically.
            </p>
          </div>

          {/* Recorder + upload — centered in remaining space */}
          <div className="flex-1 flex flex-col items-center justify-center px-6 py-8 gap-6">
            <RecorderPanel
              status={status}
              elapsed={elapsed}
              error={error}
              onStart={startRecording}
              onStop={stopRecording}
            />

            {/* Divider */}
            <div className="flex items-center gap-3 w-full">
              <div className="flex-1 h-px bg-[#e2e8f2]" />
              <span className="text-[10px] text-[#94a3b8] uppercase tracking-widest">or</span>
              <div className="flex-1 h-px bg-[#e2e8f2]" />
            </div>

            <FileUpload
              onUpload={handleFileUpload}
              disabled={status === "recording" || status === "processing"}
            />
          </div>

          {/* Account */}
          <div className="px-6 py-5 border-t border-[#e2e8f2]">
            {userEmail ? (
              <div>
                <p className="text-[10px] text-[#94a3b8] uppercase tracking-widest font-bold mb-1">
                  Signed in as
                </p>
                <p className="text-sm font-medium text-[#1e1b4b] truncate">{userEmail}</p>
                <button
                  onClick={() => setShowEmailModal(true)}
                  className="text-xs text-[#6d28d9] hover:text-[#5b21b6] mt-2 font-medium transition-colors"
                >
                  Switch account
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowEmailModal(true)}
                className="text-sm text-[#6d28d9] font-medium hover:text-[#5b21b6] transition-colors"
              >
                Sign in →
              </button>
            )}
          </div>
        </aside>

        {/* ── MAIN SCROLLABLE AREA ── */}
        <div className="flex-1 min-w-0">

          {/* Mobile top bar */}
          <header className="lg:hidden bg-white border-b border-[#e2e8f2] px-4 py-3 flex items-center justify-between sticky top-0 z-10 shadow-sm">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-[#5b21b6] to-[#4f46e5] flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-white" />
              </div>
              <span className="font-bold text-[#1e1b4b] text-sm">Audia</span>
            </div>
            {userEmail && (
              <button
                onClick={() => setShowEmailModal(true)}
                className="text-xs text-[#64748b] border border-[#e2e8f2] rounded-lg px-2.5 py-1 hover:border-[#c4b5fd] hover:text-[#6d28d9] transition-colors"
              >
                Switch
              </button>
            )}
          </header>

          {/* Mobile recorder + upload */}
          <div className="lg:hidden px-4 pt-5 pb-2 space-y-3">
            <div className="bg-white rounded-2xl border border-[#e2e8f2] shadow-sm p-6">
              <RecorderPanel
                status={status}
                elapsed={elapsed}
                error={error}
                onStart={startRecording}
                onStop={stopRecording}
              />
            </div>
            <div className="bg-white rounded-2xl border border-[#e2e8f2] shadow-sm px-4 py-4">
              <p className="text-[10px] font-bold tracking-[0.2em] text-[#94a3b8] uppercase mb-3">
                Upload File
              </p>
              <FileUpload
                onUpload={handleFileUpload}
                disabled={status === "recording" || status === "processing"}
              />
            </div>
          </div>

          {/* Content */}
          <main className="max-w-2xl mx-auto px-4 lg:px-10 py-6 lg:py-10 space-y-8">

            {/* Live session */}
            {showTranscript && (
              <section className="space-y-3 animate-fade-in">
                <div className="flex items-center gap-3">
                  <h2 className="text-[10px] font-bold tracking-[0.25em] text-[#94a3b8] uppercase flex-shrink-0">
                    Live Session
                  </h2>
                  <div className="flex-1 h-px bg-[#e2e8f2]" />
                  {status === "done" && (
                    <span className="text-[10px] text-[#6d28d9] font-semibold flex-shrink-0">
                      ✓ Complete
                    </span>
                  )}
                </div>

                {status === "done" && currentTranscriptId && (
                  <SummaryBlock
                    transcriptionId={currentTranscriptId}
                    initialSummary={currentSummary}
                    onSaved={setCurrentSummary}
                  />
                )}

                <TranscriptPanel
                  segments={segments}
                  transcriptionId={currentTranscriptId ?? undefined}
                  currentTitle={currentTitle}
                  onSegmentsUpdate={setSegments}
                  onTitleSave={setCurrentTitle}
                  isProcessing={status === "processing"}
                />

                {status === "done" && segments.length > 0 && (
                  <ChatPanel
                    segments={segments}
                    transcriptionId={currentTranscriptId ?? undefined}
                    currentTitle={currentTitle}
                    onTitleSave={setCurrentTitle}
                  />
                )}
              </section>
            )}

            {/* Archive */}
            <SessionHistory
              history={history}
              loading={historyLoading}
              loaded={historyLoaded}
              onDelete={(id) => setHistory((prev) => prev.filter((r) => r.id !== id))}
              onSegmentsUpdate={(id, updated) =>
                setHistory((prev) =>
                  prev.map((r) => (r.id === id ? { ...r, segments: updated } : r))
                )
              }
              onTitleUpdate={(id, title) =>
                setHistory((prev) =>
                  prev.map((r) => (r.id === id ? { ...r, title } : r))
                )
              }
              onSummaryUpdate={(id, summary) =>
                setHistory((prev) =>
                  prev.map((r) => (r.id === id ? { ...r, summary } : r))
                )
              }
            />

          </main>
        </div>
      </div>
    </>
  );
}
