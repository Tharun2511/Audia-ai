"use client";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import type { TranscriptSegment } from "@/entity/Transcription";
import { buildColorMap, formatDuration } from "./utils";
import TitleInput from "./TitleInput";

interface Props {
  segments: TranscriptSegment[];
  transcriptionId?: string;
  currentTitle?: string | null;
  onSegmentsUpdate?: (segments: TranscriptSegment[]) => void;
  onTitleSave?: (title: string | null) => void;
  isProcessing?: boolean;
}

export default function TranscriptPanel({
  segments,
  transcriptionId,
  currentTitle,
  onSegmentsUpdate,
  onTitleSave,
  isProcessing,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [showRenamer, setShowRenamer] = useState(false);
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  const uniqueSpeakers = [...new Set(segments.map((s) => s.speaker))];
  const colorMap = buildColorMap(uniqueSpeakers);

  useEffect(() => {
    setSpeakerNames(Object.fromEntries(uniqueSpeakers.map((s) => [s, s])));
    setShowRenamer(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments]);

  const handleSaveRenames = async () => {
    if (!transcriptionId || !onSegmentsUpdate) return;
    const speakerMap = Object.fromEntries(
      Object.entries(speakerNames).filter(([orig, renamed]) => orig !== renamed)
    );
    if (Object.keys(speakerMap).length === 0) { setShowRenamer(false); return; }
    setSaving(true);
    try {
      const res = await fetch(`/api/transcriptions/${transcriptionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speakerMap }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const { segments: updated } = await res.json();
      onSegmentsUpdate(updated);
      setShowRenamer(false);
    } catch {
      toast.error("Couldn't rename speakers. Try again.");
    } finally {
      setSaving(false);
    }
  };

  const handleCopy = async () => {
    const text = segments
      .map((s) => `${s.speaker} (${formatDuration(s.start)}): ${s.text}`)
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  return (
    <div className="bg-white rounded-xl border border-[#e2e8f2] overflow-hidden shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-[#e2e8f2]">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-[10px] font-bold tracking-[0.2em] text-[#94a3b8] uppercase flex-shrink-0">
            Transcript
          </span>
          {transcriptionId && onTitleSave && (
            <TitleInput
              transcriptionId={transcriptionId}
              initialTitle={currentTitle ?? null}
              fallback="Untitled — click to name"
              onSaved={onTitleSave}
            />
          )}
        </div>
        {!isProcessing && segments.length > 0 && (
          <button
            onClick={handleCopy}
            className="text-xs px-3 py-1.5 border border-[#e2e8f2] rounded-lg text-[#64748b] hover:text-[#1e1b4b] hover:border-[#c4b5fd] transition-colors flex-shrink-0 ml-2"
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        )}
      </div>

      <div className="p-4">
        {isProcessing ? (
          <div className="flex items-center gap-3 py-6 justify-center">
            <div className="flex items-end gap-[3px] h-6">
              {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                <div
                  key={i}
                  className="w-1.5 rounded-sm"
                  style={{
                    height: "100%",
                    background: "linear-gradient(to top, #5b21b6, #a78bfa)",
                    animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                  }}
                />
              ))}
            </div>
            <span className="text-sm text-[#64748b]">Processing audio…</span>
          </div>
        ) : segments.length === 0 ? (
          <p className="text-[#94a3b8] italic text-sm py-6 text-center">No speech detected.</p>
        ) : (
          <>
            {/* Speaker legend + rename */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex gap-3 flex-wrap">
                {uniqueSpeakers.map((sp) => (
                  <div key={sp} className="flex items-center gap-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ background: colorMap[sp] }}
                    />
                    <span className="text-xs text-[#64748b] font-medium">{sp}</span>
                  </div>
                ))}
              </div>
              {transcriptionId && onSegmentsUpdate && (
                <button
                  onClick={() => setShowRenamer((v) => !v)}
                  className={`text-xs px-3 py-1 rounded-lg border transition-colors flex-shrink-0 ml-2 ${
                    showRenamer
                      ? "border-[#6d28d9] text-[#6d28d9] bg-[#f5f3ff]"
                      : "border-[#e2e8f2] text-[#64748b] hover:border-[#c4b5fd] hover:text-[#6d28d9]"
                  }`}
                >
                  Rename Users
                </button>
              )}
            </div>

            {/* Renamer panel */}
            {showRenamer && (
              <div className="bg-[#f5f3ff] border border-[#ddd6fe] rounded-xl p-4 mb-4 space-y-3">
                <p className="text-[10px] font-bold tracking-[0.2em] text-[#6d28d9] uppercase">
                  Rename Speakers
                </p>
                {uniqueSpeakers.map((sp) => (
                  <div key={sp} className="flex items-center gap-3">
                    <div className="flex items-center gap-2 flex-shrink-0 min-w-[80px]">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ background: colorMap[sp] }}
                      />
                      <span className="text-xs font-medium text-[#475569]">{sp}</span>
                    </div>
                    <span className="text-[#94a3b8] text-xs">→</span>
                    <input
                      value={speakerNames[sp] ?? sp}
                      onChange={(e) =>
                        setSpeakerNames((prev) => ({ ...prev, [sp]: e.target.value }))
                      }
                      className="border border-[#ddd6fe] focus:border-[#6d28d9] focus:ring-1 focus:ring-[#6d28d9]/10 text-[#1e1b4b] text-sm rounded-lg px-3 py-1.5 outline-none flex-1 max-w-[180px] bg-white transition-all"
                    />
                  </div>
                ))}
                <div className="flex gap-2 pt-1">
                  <button
                    onClick={handleSaveRenames}
                    disabled={saving}
                    className="text-xs px-4 py-1.5 bg-[#5b21b6] hover:bg-[#4c1d95] text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    {saving ? "Saving…" : "Save"}
                  </button>
                  <button
                    onClick={() => setShowRenamer(false)}
                    className="text-xs px-3 py-1.5 text-[#64748b] hover:text-[#1e1b4b] transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Conversation bubbles — left-bordered by speaker color */}
            <div className="space-y-3">
              {segments.map((seg, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[#e2e8f2] overflow-hidden"
                  style={{
                    borderLeftColor: colorMap[seg.speaker] ?? "#94a3b8",
                    borderLeftWidth: "3px",
                  }}
                >
                  <div className="px-4 py-3">
                    <div className="flex items-center gap-2.5 mb-1.5">
                      <span
                        className="text-xs font-bold"
                        style={{ color: colorMap[seg.speaker] ?? "#64748b" }}
                      >
                        {seg.speaker}
                      </span>
                      <span className="text-[10px] text-[#94a3b8] font-mono bg-[#f1f4f9] px-1.5 py-0.5 rounded">
                        {formatDuration(seg.start)}
                      </span>
                    </div>
                    <p className="text-sm text-[#334155] leading-relaxed">{seg.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
