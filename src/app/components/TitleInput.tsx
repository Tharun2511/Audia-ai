"use client";
import { useState } from "react";
import { toast } from "sonner";

interface Props {
  transcriptionId: string;
  initialTitle: string | null;
  fallback: string;
  onSaved: (title: string | null) => void;
}

export default function TitleInput({ transcriptionId, initialTitle, fallback, onSaved }: Props) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(initialTitle ?? "");
  const [saving, setSaving] = useState(false);

  const save = async () => {
    const trimmed = value.trim();
    setSaving(true);
    try {
      const res = await fetch(`/api/transcriptions/${transcriptionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: trimmed || null }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      onSaved(trimmed || null);
      setEditing(false);
    } catch {
      toast.error("Couldn't save the title.");
    } finally {
      setSaving(false);
    }
  };

  const cancel = () => {
    setValue(initialTitle ?? "");
    setEditing(false);
  };

  if (editing) {
    return (
      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") save();
            if (e.key === "Escape") cancel();
          }}
          placeholder="Name this session…"
          className="border border-[#e2e8f2] focus:border-[#6d28d9] focus:ring-2 focus:ring-[#6d28d9]/10 text-[#1e1b4b] text-sm rounded-lg px-3 py-1.5 outline-none w-44 bg-white transition-all"
        />
        <button
          onClick={save}
          disabled={saving}
          className="text-xs px-3 py-1.5 bg-[#5b21b6] hover:bg-[#4c1d95] text-white rounded-lg transition-colors disabled:opacity-50"
        >
          {saving ? "…" : "Save"}
        </button>
        <button
          onClick={cancel}
          className="text-xs px-2 py-1.5 text-[#94a3b8] hover:text-[#64748b] transition-colors"
        >
          ✕
        </button>
      </div>
    );
  }

  return (
    <button
      className="group flex items-center gap-1.5 text-left"
      onClick={(e) => { e.stopPropagation(); setEditing(true); }}
    >
      <span
        className={`text-sm font-semibold ${
          initialTitle ? "text-[#1e1b4b]" : "text-[#94a3b8] italic"
        }`}
      >
        {initialTitle ?? fallback}
      </span>
      <span className="text-[10px] text-transparent group-hover:text-[#c4b5fd] transition-colors">
        ✏
      </span>
    </button>
  );
}
