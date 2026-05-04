"use client";
import { useState } from "react";
import { toast } from "sonner";
import type { TranscriptSegment } from "@/entity/Transcription";
import { buildColorMap, formatDuration } from "./utils";
import TitleInput from "./TitleInput";
import SummaryBlock from "./SummaryBlock";
import TranscriptPanel from "./TranscriptPanel";
import ChatPanel from "./ChatPanel";

type HistoryRecord = {
  id: string;
  title: string | null;
  summary: string | null;
  duration: number;
  segments: TranscriptSegment[];
  createdAt: string;
};

interface Props {
  record: HistoryRecord;
  onDelete: (id: string) => void;
  onSegmentsUpdate: (id: string, segments: TranscriptSegment[]) => void;
  onTitleUpdate: (id: string, title: string | null) => void;
  onSummaryUpdate: (id: string, summary: string | null) => void;
}

export default function HistoryEntry({
  record,
  onDelete,
  onSegmentsUpdate,
  onTitleUpdate,
  onSummaryUpdate,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const speakers = [...new Set(record.segments.map((s) => s.speaker))];
  const colorMap = buildColorMap(speakers);

  const date = new Date(record.createdAt);
  const relativeTime = (() => {
    const diff = Date.now() - date.getTime();
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    if (mins < 60) return `${mins}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  })();

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleting(true);
    try {
      const res = await fetch(`/api/transcriptions/${record.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      onDelete(record.id);
    } catch {
      toast.error("Couldn't delete this transcript. Try again.");
      setDeleting(false);
    }
  };

  return (
    <div
      className={`bg-white rounded-xl border overflow-hidden transition-shadow ${expanded
        ? "border-[#ddd6fe] shadow-md"
        : "border-[#e2e8f2] shadow-sm hover:shadow-md hover:border-[#ddd6fe]"
        }`}
    >
      {/* Accent line at top — shows only when expanded */}
      {expanded && (
        <div className="h-[2px] bg-gradient-to-r from-[#5b21b6] to-[#4f46e5]" />
      )}

      {/* Header row */}
      <div
        className="flex items-center justify-between px-4 py-3.5 cursor-pointer"
        onClick={() => setExpanded((v) => !v)}
      >
        {/* Left: title + meta */}
        <div className="flex-1 min-w-0 mr-4">
          <div onClick={(e) => e.stopPropagation()}>
            <TitleInput
              transcriptionId={record.id}
              initialTitle={record.title}
              fallback={`${date.toLocaleString()} (Rename title)`}
              onSaved={(t) => onTitleUpdate(record.id, t)}
            />
          </div>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <span className="text-[10px] text-[#94a3b8]">{relativeTime}</span>
            <span className="text-[#e2e8f2] text-[10px]">·</span>
            <span className="text-[10px] font-mono text-[#94a3b8]">
              {formatDuration(record.duration)}
            </span>
            {speakers.length > 0 && (
              <>
                <span className="text-[#e2e8f2] text-[10px]">·</span>
                <div className="flex items-center gap-1.5">
                  {speakers.map((sp) => (
                    <span
                      key={sp}
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: colorMap[sp] }}
                      title={sp}
                    />
                  ))}
                  <span className="text-[10px] text-[#94a3b8]">
                    {speakers.join(", ")}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Right: delete + chevron */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="text-[10px] text-[#94a3b8] hover:text-red-500 transition-colors disabled:opacity-40 font-medium"
          >
            {deleting ? "Deleting…" : "Delete"}
          </button>
          <span
            className="text-[#94a3b8] text-sm transition-transform duration-200 select-none"
            style={{ display: "inline-block", transform: expanded ? "rotate(180deg)" : "rotate(0deg)" }}
          >
            ▾
          </span>
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-[#e2e8f2] px-4 pb-4 pt-4 space-y-3 animate-fade-in">
          <SummaryBlock
            transcriptionId={record.id}
            initialSummary={record.summary}
            onSaved={(s) => onSummaryUpdate(record.id, s)}
          />
          <TranscriptPanel
            segments={record.segments}
            transcriptionId={record.id}
            onSegmentsUpdate={(updated) => onSegmentsUpdate(record.id, updated)}
          />
          <ChatPanel segments={record.segments} />
        </div>
      )}
    </div>
  );
}
