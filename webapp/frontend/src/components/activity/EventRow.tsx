// Activity 페이지와 Dashboard cross-run feed 가 공유하는 한 이벤트 카드
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Activity as ActivityIcon, Edit3, MessageSquare, PlayCircle, Plus, RotateCcw, Sliders, Star, XCircle } from "lucide-react";
import type { ActivityEvent, ActivityKind } from "@/api/client";
import { stripEndOfPrompt } from "@/lib/ttsInstruct";

interface Props {
  event: ActivityEvent;
  // cross-run feed 에서는 run_id 칩이 link 로 노출 — 단일 run 페이지에서는 false
  showRunLink?: boolean;
}

export function EventRow({ event, showRunLink = false }: Props) {
  const meta = kindMeta(event.kind);
  return (
    <li className="relative pb-5 last:pb-0">
      <span
        className={`absolute -left-[34px] flex h-7 w-7 items-center justify-center rounded-full border-2 border-background ${meta.bg} ${meta.fg}`}
        aria-hidden
      >
        <meta.Icon className="h-3.5 w-3.5" />
      </span>
      <div className="rounded-[1.5rem] border border-border-hairline bg-surface-container-lowest p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-display text-body-sm-strong text-primary">{meta.label}</span>
          {showRunLink && event.run_id ? (
            <Link to={`/runs/${event.run_id}/activity`} className="rounded-full bg-primary/10 px-2.5 py-0.5 font-mono text-caption-sm text-primary hover:underline">
              {event.run_id}
            </Link>
          ) : null}
          {event.chunk_id ? <Tag>{event.chunk_id}</Tag> : null}
          {event.step ? <Tag>{event.step}</Tag> : null}
          {event.status ? <Tag>status · {event.status}</Tag> : null}
          <span className="ml-auto font-mono text-caption-sm text-mute">{formatTime(event.ts)}</span>
        </div>
        <Detail event={event} />
      </div>
    </li>
  );
}

function Detail({ event }: { event: ActivityEvent }) {
  if (event.kind === "chunk_instruction_edit") {
    // 종료 토큰은 UI 어디서도 안 보이게 — diff 도 strip
    return (
      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
        <DiffBox label="before" value={stripEndOfPrompt(asString(event.before))} tone="mute" />
        <DiffBox label="after" value={stripEndOfPrompt(asString(event.after))} tone="primary" />
      </div>
    );
  }
  if (event.kind === "chunk_text_edit") {
    return (
      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
        <DiffBox label="before" value={asString(event.before)} tone="mute" />
        <DiffBox label="after" value={asString(event.after)} tone="primary" />
      </div>
    );
  }
  if (event.kind === "chunk_emotion_edit") {
    return (
      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
        <DiffBox label="before" value={emotionSummary(event.before)} tone="mute" />
        <DiffBox label="after" value={emotionSummary(event.after)} tone="primary" />
      </div>
    );
  }
  if (event.kind === "chunk_speaker_edit" || event.kind === "chunk_reference_edit") {
    return (
      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
        <DiffBox label="before" value={asString(event.before)} tone="mute" />
        <DiffBox label="after" value={asString(event.after)} tone="primary" />
      </div>
    );
  }
  if (event.kind === "status_change") {
    const before = asString(event.before);
    const after = asString(event.after);
    return (
      <p className="mt-2 font-mono text-code-sm text-secondary">
        {before || "—"} <span className="text-mute">→</span> <span className="text-primary">{after || "—"}</span>
        {event.note ? <span className="ml-2 text-term-red">{event.note}</span> : null}
      </p>
    );
  }
  if (event.note) {
    return <p className="mt-2 text-body-sm text-secondary">{event.note}</p>;
  }
  return null;
}

function DiffBox({ label, value, tone }: { label: string; value: string; tone: "mute" | "primary" }) {
  return (
    <div className="rounded-[1rem] bg-surface-soft p-3">
      <p className="font-mono text-data-label uppercase text-data-label">{label}</p>
      <pre className={`mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words font-mono text-code-sm ${tone === "primary" ? "text-primary" : "text-mute"}`}>{value || "—"}</pre>
    </div>
  );
}

function Tag({ children }: { children: ReactNode }) {
  return <span className="rounded-full bg-surface-container px-2.5 py-0.5 font-mono text-caption-sm text-secondary">{children}</span>;
}

export function kindMeta(kind: ActivityKind) {
  switch (kind) {
    case "run_created": return { label: "Run 생성", Icon: Plus, bg: "bg-primary/10", fg: "text-primary" };
    case "status_change": return { label: "상태 변경", Icon: ActivityIcon, bg: "bg-surface-container", fg: "text-secondary" };
    case "run_canceled": return { label: "Run 취소", Icon: XCircle, bg: "bg-status-failed/10", fg: "text-status-failed" };
    case "run_resumed": return { label: "Run 재시작", Icon: RotateCcw, bg: "bg-term-yellow/15", fg: "text-term-yellow" };
    case "chunk_text_edit": return { label: "번역 텍스트 편집", Icon: Edit3, bg: "bg-primary/10", fg: "text-primary" };
    case "chunk_instruction_edit": return { label: "TTS 프롬프트 편집", Icon: MessageSquare, bg: "bg-primary/10", fg: "text-primary" };
    case "chunk_emotion_edit": return { label: "감정 벡터 편집", Icon: Sliders, bg: "bg-primary/10", fg: "text-primary" };
    case "chunk_speaker_edit": return { label: "Speaker edit", Icon: Edit3, bg: "bg-primary/10", fg: "text-primary" };
    case "chunk_reference_edit": return { label: "Reference edit", Icon: MessageSquare, bg: "bg-primary/10", fg: "text-primary" };
    case "chunk_redub": return { label: "Chunk Redub", Icon: RotateCcw, bg: "bg-term-yellow/15", fg: "text-term-yellow" };
    case "step_rerun": return { label: "Step 재실행", Icon: PlayCircle, bg: "bg-term-yellow/15", fg: "text-term-yellow" };
    case "mos_scored": return { label: "MOS 추천 채점", Icon: Star, bg: "bg-primary/10", fg: "text-primary" };
  }
}

export function asString(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

export function emotionSummary(value: unknown): string {
  if (!value || typeof value !== "object") return asString(value);
  const obj = value as { label?: string; scores?: Record<string, number> };
  const top = obj.scores ? Object.entries(obj.scores).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
  const lines = [`label: ${obj.label ?? "—"}`];
  for (const [k, v] of top) lines.push(`${k}: ${(v * 100).toFixed(0)}%`);
  return lines.join("\n");
}

export function formatTime(ts: number): string {
  const ms = ts > 10_000_000_000 ? ts : ts * 1000;
  return new Date(ms).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatDay(ts: number): string {
  const ms = ts > 10_000_000_000 ? ts : ts * 1000;
  return new Date(ms).toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit", weekday: "short" });
}

export function groupByDay(events: ActivityEvent[]): [string, ActivityEvent[]][] {
  const buckets = new Map<string, ActivityEvent[]>();
  for (const ev of events) {
    const day = formatDay(ev.ts);
    const arr = buckets.get(day);
    if (arr) arr.push(ev);
    else buckets.set(day, [ev]);
  }
  return [...buckets.entries()];
}
