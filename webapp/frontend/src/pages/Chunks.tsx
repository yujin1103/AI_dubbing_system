// 실제 chunk 산출물이 준비된 run에서만 청크 인스펙터를 여는 화면
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, ChevronDown, ChevronUp, Database, Loader2, RefreshCcw, Sliders, X } from "lucide-react";
import { api, isOptionalEndpointMissing, type ChunkRow, type PatchChunkPayload, type ReferenceCandidate, type RunRecord } from "@/api/client";
import { ChunkInspectorTable } from "@/components/chunks/ChunkInspectorTable";
import { EmotionEqualizer } from "@/components/chunks/EmotionEqualizer";
import { LogTail } from "@/components/LogTail";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PillInput } from "@/components/ui/pill-input";
import { useRunSocket } from "@/hooks/useRunSocket";
import { canCompareOutput, canInspectChunks, canPlayChunkAudio, stepRecord } from "@/lib/runReadiness";
import { toStaticUrl } from "@/lib/staticUrl";
import { ensureEndOfPrompt, hasNonLatinScript, stripEndOfPrompt, stripTokenOnly } from "@/lib/ttsInstruct";

type ToastKind = "redub-done" | "final-done" | null;

export function Chunks() {
  const { id = "" } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<string>();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set<string>());
  const [bulkEditOpen, setBulkEditOpen] = useState(false);
  const [speakerFilter, setSpeakerFilter] = useState("all");
  const [emotionFilter, setEmotionFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [problemOnly, setProblemOnly] = useState(false);
  const [redubbingId, setRedubbingId] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastKind>(null);
  const [pendingAction, setPendingAction] = useState<"redub" | "final" | null>(null);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);

  const runQuery = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
    enabled: Boolean(id),
    retry: 1,
    staleTime: 10_000,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "running" || status === "queued" ? 5000 : false;
    },
  });
  const ready = canInspectChunks(runQuery.data);
  const writesDisabled = runQuery.data?.status === "queued" || runQuery.data?.status === "running";
  const chunksQuery = useQuery({
    queryKey: ["chunks", id],
    queryFn: () => api.listChunks(id),
    enabled: ready,
    retry: 1,
    // 페이지 재진입 시 즉시 캐시 노출 + 백그라운드 갱신 — "세션 만료"처럼 느껴지지 않도록
    staleTime: 30_000,
  });
  const referenceBankQuery = useQuery({
    queryKey: ["speaker-reference-bank", id],
    queryFn: () => api.getSpeakerReferenceBank(id),
    enabled: ready,
    retry: 1,
    staleTime: 30_000,
  });
  const rows = chunksQuery.data ?? [];
  const referenceBank = referenceBankQuery.data ?? {};
  const filteredRows = useMemo(() => filterRows(rows, speakerFilter, emotionFilter, search, problemOnly), [rows, speakerFilter, emotionFilter, search, problemOnly]);
  const selectedRows = useMemo(() => rows.filter((row) => selectedIds.has(row.chunk_id)), [rows, selectedIds]);
  const selected = filteredRows.find((row) => row.chunk_id === selectedId) ?? filteredRows[0];
  const selectedRedubBlocked = Boolean(selected?.translation_blocked) || selected?.status === "blocked";
  const speakers = unique(rows.map((row) => row.speaker).filter(Boolean) as string[]);
  const emotions = unique(rows.map((row) => row.emotion).filter(Boolean) as string[]);
  const endpointMissing = chunksQuery.isError && isOptionalEndpointMissing(chunksQuery.error);
  const repairCounts = useMemo(() => {
    const stale = rows.filter((row) => row.dub_stale || row.status === "stale").length;
    const failed = rows.filter((row) => row.error || row.status === "error").length;
    const blocked = rows.filter((row) => row.translation_blocked || row.status === "blocked").length;
    const missingReference = rows.filter((row) => row.reference_mode !== "self" && !row.reference_chunk_id).length;
    return { stale, failed, blocked, missingReference, total: stale + failed + blocked + missingReference };
  }, [rows]);

  const muxStep = stepRecord(runQuery.data, "mux");
  const muxPending = muxStep?.state === "pending";
  const muxDone = muxStep?.state === "done";
  // status=success면 항상 재mux 가능 — 청크 편집/redub 후 stale 한 mux 결과 갱신용
  const finalRunnable = runQuery.data?.status === "success";
  // 라벨용 — 청크가 stale 이거나 output_video 가 없거나 mux pending 이면 "필요" 상태로 표시
  const finalNeeded = muxPending || !runQuery.data?.output_video || rows.some((r) => r.dub_stale || r.status === "stale");

  // run_done 토스트 — 진행 중인 작업 종류에 따라 다른 메시지
  const handleRunDone = (status: RunRecord["status"]) => {
    // run 상태가 바뀌면 chunks/metrics 도 stale → 재조회
    queryClient.invalidateQueries({ queryKey: ["chunks", id] });
    queryClient.invalidateQueries({ queryKey: ["metrics", id] });
    queryClient.invalidateQueries({ queryKey: ["log", id] });
    if (status !== "success") {
      setRedubbingId(null);
      setPendingAction(null);
      return;
    }
    if (pendingAction === "final") setToast("final-done");
    else if (pendingAction === "redub") setToast("redub-done");
    setRedubbingId(null);
    setPendingAction(null);
  };

  const { logLines, connected } = useRunSocket(id, { enabled: writesDisabled, onRunDone: handleRunDone });
  const historyLogQuery = useQuery({
    queryKey: ["log", id],
    queryFn: () => api.getLog(id),
    enabled: Boolean(id) && (writesDisabled || terminalOpen),
    retry: 1,
  });
  const combinedLogLines = useMemo(
    () => [...(historyLogQuery.data?.lines ?? []), ...logLines],
    [historyLogQuery.data?.lines, logLines],
  );

  // 작업 시작되면 자동으로 터미널 펼침
  useEffect(() => {
    if (writesDisabled) setTerminalOpen(true);
  }, [writesDisabled]);

  useEffect(() => {
    if (!selected) setMobileDetailOpen(false);
  }, [selected]);

  useEffect(() => {
    const available = new Set(rows.map((row) => row.chunk_id));
    setSelectedIds((prev) => {
      const next = new Set([...prev].filter((chunkId) => available.has(chunkId)));
      return next.size === prev.size ? prev : next;
    });
  }, [rows]);

  // 5초 후 자동 dismiss
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 8000);
    return () => clearTimeout(t);
  }, [toast]);

  const toggleSelect = (row: ChunkRow) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(row.chunk_id)) next.delete(row.chunk_id);
      else next.add(row.chunk_id);
      return next;
    });
  };

  const toggleAllVisible = () => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      const allVisibleSelected = filteredRows.length > 0 && filteredRows.every((row) => next.has(row.chunk_id));
      for (const row of filteredRows) {
        if (allVisibleSelected) next.delete(row.chunk_id);
        else next.add(row.chunk_id);
      }
      return next;
    });
  };

  const updateMutation = useMutation({
    mutationFn: ({ chunkId, payload }: { chunkId: string; payload: Parameters<typeof api.patchChunk>[2] }) => api.patchChunk(id, chunkId, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chunks", id] });
      queryClient.invalidateQueries({ queryKey: ["speaker-reference-bank", id] });
      queryClient.invalidateQueries({ queryKey: ["run", id] });
      queryClient.invalidateQueries({ queryKey: ["metrics", id] });
    },
  });
  const redubMutation = useMutation({
    mutationFn: (chunkId: string) => api.redubChunk(id, chunkId),
    onMutate: (chunkId) => {
      setRedubbingId(chunkId);
      setPendingAction("redub");
      setToast(null);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chunks", id] });
      queryClient.invalidateQueries({ queryKey: ["speaker-reference-bank", id] });
      queryClient.invalidateQueries({ queryKey: ["run", id] });
      queryClient.invalidateQueries({ queryKey: ["metrics", id] });
      queryClient.invalidateQueries({ queryKey: ["log", id] });
    },
    onError: () => {
      setRedubbingId(null);
      setPendingAction(null);
    },
  });
  const bulkUpdateMutation = useMutation({
    mutationFn: ({ chunkIds, payload }: { chunkIds: string[]; payload: PatchChunkPayload }) =>
      api.patchChunks(id, { updates: chunkIds.map((chunkId) => ({ chunk_id: chunkId, ...payload })) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chunks", id] });
      queryClient.invalidateQueries({ queryKey: ["speaker-reference-bank", id] });
      queryClient.invalidateQueries({ queryKey: ["run", id] });
      queryClient.invalidateQueries({ queryKey: ["metrics", id] });
      setBulkEditOpen(false);
    },
  });
  const bulkRedubMutation = useMutation({
    mutationFn: (chunkIds: string[]) => api.redubChunks(id, { chunk_ids: chunkIds }),
    onMutate: (chunkIds) => {
      setRedubbingId(`${chunkIds.length} chunks`);
      setPendingAction("redub");
      setToast(null);
      setTerminalOpen(true);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chunks", id] });
      queryClient.invalidateQueries({ queryKey: ["run", id] });
      queryClient.invalidateQueries({ queryKey: ["metrics", id] });
      queryClient.invalidateQueries({ queryKey: ["log", id] });
    },
    onError: () => {
      setRedubbingId(null);
      setPendingAction(null);
    },
  });
  const finalMutation = useMutation({
    mutationFn: () => api.rerunStep(id, "compose_audio"),
    onMutate: () => {
      setPendingAction("final");
      setToast(null);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run", id] });
    },
    onError: () => {
      setPendingAction(null);
    },
  });

  const chunkActionPending = updateMutation.isPending || redubMutation.isPending || bulkUpdateMutation.isPending || bulkRedubMutation.isPending;
  const chunkWritesDisabled = writesDisabled || chunkActionPending;

  if (runQuery.isLoading) return <PageFrame><EmptyState title="run을 불러오는 중입니다" body="Chunk 산출물 사용 가능 여부를 확인하고 있습니다." runId={id} /></PageFrame>;
  if (!runQuery.data) return <PageFrame><EmptyState title="run을 찾을 수 없습니다" body="Run History에서 존재하는 run을 선택하세요." runId={id} /></PageFrame>;
  if (!ready) {
    return (
      <PageFrame>
        <EmptyState
          title="아직 chunk 산출물이 없습니다"
          body="Chunks 화면은 translate 또는 build_timeline 단계가 끝난 뒤 실제 청크 데이터를 볼 때 사용합니다. 지금은 Run Detail에서 진행 상태와 로그를 먼저 확인하세요."
          runId={id}
        />
      </PageFrame>
    );
  }
  if (endpointMissing || (chunksQuery.isSuccess && rows.length === 0)) {
    const title = endpointMissing ? "Chunk API가 아직 연결되지 않았습니다" : "표시할 chunk가 없습니다";
    const body = endpointMissing
      ? "이 run은 chunk 화면을 열 수 있는 단계까지 진행됐지만 현재 백엔드가 /api/runs/{id}/chunks endpoint를 제공하지 않습니다."
      : "백엔드가 빈 chunk 목록을 반환했습니다. 산출물이 갱신되는 중일 수 있으니 Refresh 또는 Run Detail에서 진행 상태를 확인하세요.";
    return (
      <PageFrame>
        <EmptyState
          title={title}
          body={body}
          runId={id}
          action={<Button onClick={() => chunksQuery.refetch()} disabled={chunksQuery.isFetching}>{chunksQuery.isFetching ? "갱신 중" : "Refresh"}</Button>}
        />
      </PageFrame>
    );
  }

  return (
    <section
      className="grid h-[calc(100vh-56px)] grid-cols-1 overflow-hidden bg-background lg:grid-cols-[minmax(720px,1fr)_360px]"
      style={{
        gridTemplateRows: `minmax(0,1fr) ${terminalOpen ? "260px" : "44px"}`,
        // 쫀들한 spring-like easing — back-out 커브로 살짝 튀는 느낌
        transition: "grid-template-rows 320ms cubic-bezier(0.34, 1.32, 0.64, 1)",
      }}
    >
      <main className="flex min-w-0 flex-col overflow-hidden bg-surface-container-lowest lg:border-r lg:border-border-hairline">
        <header className="shrink-0 border-b border-border-hairline p-5">
          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-5">
            <div>
              <p className="font-mono text-data-label uppercase text-data-label">Chunk Inspector</p>
              <h1 className="mt-1 font-display text-heading-lg text-primary">청크별 번역과 더빙 상태</h1>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              {redubbingId ? (
                <span className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-caption-strong text-primary">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  redubbing {redubbingId}
                </span>
              ) : null}
              {pendingAction === "final" ? (
                <span className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-caption-strong text-primary">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  composing & muxing
                </span>
              ) : null}
              <span className="rounded-full border border-border-hairline bg-surface-soft px-4 py-2 font-mono text-code-sm text-secondary">{id}</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {repairCounts.total > 0 ? (
              <div className="flex w-full flex-wrap items-center gap-2 rounded-[1.25rem] border border-border-hairline bg-surface-soft p-2">
                <span className="px-2 font-mono text-data-label uppercase text-data-label">Review queue</span>
                <RepairChip label="Stale" count={repairCounts.stale} active={problemOnly} onClick={() => setProblemOnly(true)} />
                <RepairChip label="Error" count={repairCounts.failed} active={problemOnly} onClick={() => setProblemOnly(true)} />
                <RepairChip label="Blocked" count={repairCounts.blocked} active={problemOnly} onClick={() => setProblemOnly(true)} />
                <RepairChip label="Needs ref" count={repairCounts.missingReference} active={problemOnly} onClick={() => setProblemOnly(true)} />
                {problemOnly ? <button type="button" onClick={() => setProblemOnly(false)} className="ml-auto rounded-full px-3 py-1 text-caption-strong text-secondary hover:bg-white">Show all</button> : null}
              </div>
            ) : null}
            <Select value={speakerFilter} onChange={setSpeakerFilter} options={["all", ...speakers]} label="speaker" />
            <Select value={emotionFilter} onChange={setEmotionFilter} options={["all", ...emotions]} label="emotion" />
            <PillInput name="chunk-search" aria-label="Search chunk text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search text" className="min-w-0 flex-1 sm:w-64 sm:flex-none" />
            <button type="button" onClick={() => setProblemOnly(!problemOnly)} className={`h-10 rounded-full px-4 text-body-sm-strong ${problemOnly ? "bg-primary text-white" : "bg-surface-soft text-secondary hover:bg-surface-container"}`}>Show Stale/Error Only</button>
            <div className="flex w-full flex-wrap items-center gap-2 sm:ml-auto sm:w-auto">
              <Button
                variant="secondary"
                onClick={() => selected && redubMutation.mutate(selected.chunk_id)}
                disabled={!selected || selectedRedubBlocked || chunkWritesDisabled}
                className="h-10"
                title={selectedRedubBlocked ? "번역 차단 chunk는 먼저 번역문을 입력해야 합니다" : "현재 선택한 chunk를 다시 더빙"}
              >
                <RefreshCcw className="mr-2 h-4 w-4" />Redub Chunk
              </Button>
              <Link to={`/runs/${id}/activity`} className="inline-flex h-10 items-center rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm-strong text-primary hover:bg-surface-container">Activity</Link>
              {muxDone && canCompareOutput(runQuery.data) ? (
                <Link to={`/runs/${id}/compare`} className="inline-flex h-10 items-center rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm-strong text-primary hover:bg-surface-container">Compare</Link>
              ) : null}
              <Button
                onClick={() => finalMutation.mutate()}
                disabled={!finalRunnable || finalMutation.isPending || writesDisabled}
                className="h-10"
                title={finalRunnable ? "compose_audio + mux 실행 — 현재 청크 dub로 최종 영상 만들기" : "파이프라인이 아직 끝나지 않아 비활성화"}
              >
                {finalMutation.isPending
                  ? "Starting…"
                  : finalNeeded
                    ? "Run Final Output (mux)"
                    : "Re-run Final Output"}
              </Button>
            </div>
          </div>
        </header>
        <div className="relative min-h-0 flex-1 overflow-hidden">
          <div className="hidden h-full lg:block">
            <ChunkInspectorTable
              rows={filteredRows}
              selectedId={selected?.chunk_id}
              selectedIds={selectedIds}
              writesDisabled={chunkWritesDisabled}
              audioReady={canPlayChunkAudio(runQuery.data)}
              onSelect={(row) => setSelectedId(row.chunk_id)}
              onToggleSelect={toggleSelect}
              onToggleAllVisible={toggleAllVisible}
              onUpdateText={(row, text) => updateMutation.mutate({ chunkId: row.chunk_id, payload: { translated_text: text } })}
            />
          </div>
          <ChunkCardList
            rows={filteredRows}
            selectedId={selected?.chunk_id}
            selectedIds={selectedIds}
            writesDisabled={chunkWritesDisabled}
            audioReady={canPlayChunkAudio(runQuery.data)}
            onSelect={(row) => setSelectedId(row.chunk_id)}
            onOpenDetails={(row) => {
              setSelectedId(row.chunk_id);
              setMobileDetailOpen(true);
            }}
            onToggleSelect={toggleSelect}
            onUpdateText={(row, text) => updateMutation.mutate({ chunkId: row.chunk_id, payload: { translated_text: text } })}
          />
          {selectedRows.length > 0 ? (
            <div className="pointer-events-none absolute inset-x-3 bottom-3 z-20 lg:inset-x-5">
              <div className="pointer-events-auto mx-auto max-w-5xl">
                {bulkEditOpen ? (
                  <BulkEditPanel
                    rows={selectedRows}
                    pending={chunkWritesDisabled}
                    onCancel={() => setBulkEditOpen(false)}
                    onApply={(payload) => bulkUpdateMutation.mutate({ chunkIds: selectedRows.map((row) => row.chunk_id), payload })}
                  />
                ) : null}
                <BulkActionBar
                  selectedCount={selectedRows.length}
                  pending={chunkWritesDisabled}
                  onClear={() => {
                    setSelectedIds(new Set<string>());
                    setBulkEditOpen(false);
                  }}
                  onEdit={() => setBulkEditOpen((open) => !open)}
                  onRedub={() => bulkRedubMutation.mutate(selectedRows.map((row) => row.chunk_id))}
                />
              </div>
            </div>
          ) : null}
        </div>
      </main>

      <aside className="hidden overflow-y-auto bg-background p-5 lg:block">
        {selected ? (
          <ChunkDetail
            runId={id}
            row={selected}
            pending={chunkWritesDisabled}
            speakers={speakers}
            referenceBank={referenceBank}
            onSaveInstruction={(text) => updateMutation.mutate({ chunkId: selected.chunk_id, payload: { tts_instruct_text: text } })}
            onSaveEmotion={(label, scores) => updateMutation.mutate({ chunkId: selected.chunk_id, payload: { emotion_label: label, emotion_scores: scores } })}
            onSaveSpeakerReference={(payload) => updateMutation.mutate({ chunkId: selected.chunk_id, payload })}
          />
        ) : (
          <div className="text-body-sm text-mute">선택된 chunk가 없습니다.</div>
        )}
      </aside>

      {mobileDetailOpen && selected ? (
        <div className="fixed inset-0 z-50 bg-black/30 lg:hidden" role="dialog" aria-modal="true" aria-label={`${selected.chunk_id} details`}>
          <div className="ml-auto flex h-full w-full max-w-[440px] flex-col bg-background shadow-running-ring">
            <div className="flex h-14 shrink-0 items-center justify-between border-b border-border-hairline px-5">
              <div>
                <p className="font-mono text-data-label uppercase text-data-label">Chunk Details</p>
                <p className="font-display text-heading-sm text-primary">{selected.chunk_id}</p>
              </div>
              <button type="button" aria-label="Close chunk details" onClick={() => setMobileDetailOpen(false)} className="flex h-9 w-9 items-center justify-center rounded-full bg-surface-soft text-primary">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-5">
              <ChunkDetail
                runId={id}
                row={selected}
                pending={chunkWritesDisabled}
                speakers={speakers}
                referenceBank={referenceBank}
                onSaveInstruction={(text) => updateMutation.mutate({ chunkId: selected.chunk_id, payload: { tts_instruct_text: text } })}
                onSaveEmotion={(label, scores) => updateMutation.mutate({ chunkId: selected.chunk_id, payload: { emotion_label: label, emotion_scores: scores } })}
                onSaveSpeakerReference={(payload) => updateMutation.mutate({ chunkId: selected.chunk_id, payload })}
              />
            </div>
          </div>
        </div>
      ) : null}

      <TerminalPanel
        open={terminalOpen}
        toggle={() => setTerminalOpen((v) => !v)}
        connected={connected}
        active={writesDisabled || Boolean(pendingAction)}
        lines={combinedLogLines}
      />

      {toast ? (
        <ToastBanner
          kind={toast}
          onClose={() => setToast(null)}
          onCompare={() => navigate(`/runs/${id}/compare`)}
          onBackToRun={() => navigate(`/runs/${id}`)}
          onRunFinal={() => {
            setToast(null);
            finalMutation.mutate();
          }}
        />
      ) : null}
    </section>
  );
}

function PageFrame({ children }: { children: ReactNode }) {
  return <section className="min-h-[calc(100vh-56px)] bg-background px-8 py-7">{children}</section>;
}

function EmptyState({ title, body, runId, action }: { title: string; body: string; runId: string; action?: ReactNode }) {
  return (
    <div className="mx-auto max-w-[760px] rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-8 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-surface-soft text-primary"><Database className="h-5 w-5" /></div>
      <p className="mt-5 font-mono text-data-label uppercase text-data-label">Chunk Inspector · {runId}</p>
      <h1 className="mt-2 font-display text-heading-lg text-primary">{title}</h1>
      <p className="mx-auto mt-3 max-w-[560px] text-body-sm text-secondary">{body}</p>
      <div className="mt-6 flex justify-center gap-3">
        <Link to={`/runs/${runId}`} className="inline-flex h-10 items-center gap-2 rounded-full border border-border-hairline bg-surface-soft px-5 text-body-sm-strong text-primary hover:bg-surface-container"><ArrowLeft className="h-4 w-4" />Run Detail</Link>
        {action}
      </div>
    </div>
  );
}

function ChunkCardList({
  rows,
  selectedId,
  selectedIds,
  writesDisabled,
  audioReady,
  onSelect,
  onOpenDetails,
  onToggleSelect,
  onUpdateText,
}: {
  rows: ChunkRow[];
  selectedId?: string;
  selectedIds: Set<string>;
  writesDisabled: boolean;
  audioReady: boolean;
  onSelect: (row: ChunkRow) => void;
  onOpenDetails: (row: ChunkRow) => void;
  onToggleSelect: (row: ChunkRow) => void;
  onUpdateText: (row: ChunkRow, text: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const startEdit = (row: ChunkRow) => {
    if (writesDisabled) return;
    onSelect(row);
    setEditingId(row.chunk_id);
    setDraft(row.translated_text ?? "");
  };

  const save = (row: ChunkRow) => {
    setEditingId(null);
    if (draft !== (row.translated_text ?? "")) onUpdateText(row, draft);
  };

  if (!rows.length) {
    return (
      <div className="h-full overflow-y-auto p-5 lg:hidden">
        <div className="rounded-[1.5rem] border border-border-hairline bg-white p-5 text-body-sm text-secondary">
          No chunks match the current filters.
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 lg:hidden">
      <div className="space-y-3">
        {rows.map((row) => {
          const selected = row.chunk_id === selectedId;
          const checked = selectedIds.has(row.chunk_id);
          const status = getChunkStatus(row);
          const originalSrc = audioReady ? toStaticUrl(row.original_audio) : null;
          const dubbedSrc = audioReady ? toStaticUrl(row.dubbed_audio) : null;
          return (
            <article
              key={row.chunk_id}
              className={`rounded-[1.25rem] border bg-white p-4 ${selected ? "border-primary shadow-running-ring" : "border-border-hairline"}`}
              onClick={() => onSelect(row)}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="shrink-0 whitespace-nowrap font-mono text-code-sm text-primary">{row.chunk_id}</span>
                    <span className="shrink-0 whitespace-nowrap rounded-full border border-border-hairline bg-surface-soft px-2 py-0.5 font-mono text-data-label uppercase text-data-label">SPK {row.speaker ?? "--"}</span>
                    <span className={`shrink-0 whitespace-nowrap rounded-full px-2 py-0.5 font-mono text-data-label uppercase ${status.className}`}>{status.label}</span>
                  </div>
                  <p className="mt-1 font-mono text-caption-sm text-mute">{formatRange(row.start, row.end)} · {row.emotion ?? "neutral"}</p>
                </div>
                <input
                  type="checkbox"
                  name="chunk-selection"
                  checked={checked}
                  onClick={(event) => event.stopPropagation()}
                  onChange={() => onToggleSelect(row)}
                  aria-label={`${row.chunk_id} select`}
                  className="mt-1 h-4 w-4 accent-black"
                />
              </div>

              <div className="mt-4 grid gap-3">
                <div>
                  <p className="font-mono text-data-label uppercase text-data-label">Source</p>
                  <p className="mt-1 text-body-sm text-secondary">{row.source_text ?? "No source text"}</p>
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <p className="font-mono text-data-label uppercase text-data-label">Translated</p>
                    {editingId !== row.chunk_id ? (
                      <button type="button" onClick={(event) => { event.stopPropagation(); startEdit(row); }} disabled={writesDisabled} className="text-caption-strong text-primary disabled:text-mute">Edit</button>
                    ) : null}
                  </div>
                  {editingId === row.chunk_id ? (
                    <div className="rounded-[1rem] border border-primary bg-white p-2">
                      <textarea
                        name={`mobile-translation-${row.chunk_id}`}
                        aria-label={`${row.chunk_id} translated text`}
                        value={draft}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => setDraft(event.target.value)}
                        rows={3}
                        className="w-full resize-none bg-transparent text-body-sm text-primary focus:outline-none"
                      />
                      <div className="mt-2 flex justify-end gap-2">
                        <Button variant="secondary" size="sm" onClick={(event) => { event.stopPropagation(); setEditingId(null); }}>Cancel</Button>
                        <Button size="sm" onClick={(event) => { event.stopPropagation(); save(row); }}>Save</Button>
                      </div>
                    </div>
                  ) : (
                    <p className="rounded-[1rem] bg-surface-soft p-3 text-body-sm text-primary">{row.translated_text ?? "No translation yet"}</p>
                  )}
                </div>
              </div>

              <div className="mt-4 grid gap-2">
                <MobileAudioPreview label="A" src={originalSrc} />
                <MobileAudioPreview label="B" src={dubbedSrc} />
              </div>

              <div className="mt-4 flex items-center justify-between gap-3">
                <span className="font-mono text-caption-sm text-mute">
                  {formatDuration(row.duration_original)} / {formatDuration(row.duration_dub)}
                </span>
                <button
                  type="button"
                  onClick={(event) => { event.stopPropagation(); onOpenDetails(row); }}
                  className="inline-flex h-9 items-center rounded-full bg-primary px-4 text-body-sm-strong text-white"
                >
                  Details
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function MobileAudioPreview({ label, src }: { label: string; src: string | null }) {
  return (
    <div className={`flex items-center gap-2 rounded-full bg-surface-soft px-2 py-1 ${src ? "" : "opacity-45"}`}>
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary font-mono text-data-label text-white">{label}</span>
      <audio src={src ?? undefined} controls preload="none" aria-label={`${label} audio preview`} className="h-8 min-w-0 flex-1" />
    </div>
  );
}

function getChunkStatus(row: ChunkRow): { label: string; className: string } {
  if (row.translation_blocked || row.status === "blocked") return { label: "Blocked", className: "bg-term-yellow/15 text-term-yellow" };
  if (row.error || row.status === "error") return { label: "Error", className: "bg-error-container text-on-error-container" };
  if (row.dub_stale || row.status === "stale") return { label: "Stale", className: "border border-term-yellow bg-term-yellow/10 text-term-yellow" };
  if (row.status === "done") return { label: "Done", className: "bg-status-done/10 text-status-done" };
  return { label: row.status ?? "Queued", className: "bg-surface-container text-secondary" };
}

function RepairChip({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={count === 0}
      className={`inline-flex h-8 items-center gap-2 rounded-full px-3 font-mono text-caption-strong disabled:opacity-40 ${active ? "bg-primary text-white" : "bg-white text-primary hover:bg-surface-container"}`}
    >
      <span>{label}</span>
      <span>{count}</span>
    </button>
  );
}

function Select({ value, onChange, options, label }: { value: string; onChange: (next: string) => void; options: string[]; label: string }) {
  const fieldName = `chunk-${label.toLowerCase().replace(/\s+/g, "-")}-filter`;
  return (
    <label className="inline-flex h-10 items-center gap-2 rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm text-secondary">
      <span className="font-mono text-data-label uppercase text-data-label">{label}</span>
      <select name={fieldName} aria-label={`${label} filter`} value={value} onChange={(e) => onChange(e.target.value)} className="bg-transparent text-primary focus:outline-none">
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function BulkActionBar({ selectedCount, pending, onClear, onEdit, onRedub }: { selectedCount: number; pending: boolean; onClear: () => void; onEdit: () => void; onRedub: () => void }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-[1.25rem] border border-border-hairline bg-white/95 px-3 py-3 shadow-running-ring backdrop-blur sm:px-4">
      <div className="min-w-0">
        <p className="font-mono text-data-label uppercase text-data-label">Bulk selection</p>
        <p className="text-body-sm-strong text-primary">{selectedCount}개 chunk 선택됨</p>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <Button variant="secondary" size="sm" onClick={onClear} disabled={pending}>선택 해제</Button>
        <Button variant="secondary" size="sm" onClick={onEdit} disabled={pending}>선택 항목 편집</Button>
        <Button size="sm" onClick={onRedub} disabled={pending}><RefreshCcw className="mr-2 h-4 w-4" />선택 항목 Redub</Button>
      </div>
    </div>
  );
}

function BulkEditPanel({ rows, pending, onCancel, onApply }: { rows: ChunkRow[]; pending: boolean; onCancel: () => void; onApply: (payload: PatchChunkPayload) => void }) {
  const first = rows[0];
  const [applyText, setApplyText] = useState(false);
  const [translatedText, setTranslatedText] = useState(first?.translated_text ?? "");
  const [applyInstruction, setApplyInstruction] = useState(false);
  const [instructionText, setInstructionText] = useState(stripEndOfPrompt(first?.tts_instruct_text));
  const [applyEmotion, setApplyEmotion] = useState(false);
  const [emotionDraft, setEmotionDraft] = useState<{ label: string; scores: Record<string, number> } | null>(null);
  const canApply = applyText || applyInstruction || applyEmotion;

  useEffect(() => {
    setTranslatedText(first?.translated_text ?? "");
    setInstructionText(stripEndOfPrompt(first?.tts_instruct_text));
    setEmotionDraft(null);
  }, [first?.chunk_id, first?.translated_text, first?.tts_instruct_text]);

  const submit = () => {
    const payload: PatchChunkPayload = {};
    if (applyText) payload.translated_text = translatedText;
    if (applyInstruction) payload.tts_instruct_text = ensureEndOfPrompt(instructionText);
    if (applyEmotion) {
      const fallbackLabel = first?.emotion ?? "neutral";
      const fallbackScores = first?.emotion_scores ?? { [fallbackLabel]: 1 };
      const emotion = emotionDraft ?? { label: fallbackLabel, scores: fallbackScores };
      payload.emotion_label = emotion.label;
      payload.emotion_scores = emotion.scores;
    }
    onApply(payload);
  };

  return (
    <Card className="mb-3 max-h-[min(70vh,560px)] overflow-y-auto rounded-[1.5rem] border border-primary/20 bg-primary/5 p-4 shadow-running-ring">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-data-label uppercase text-primary">Bulk edit</p>
          <h2 className="mt-1 font-display text-heading-sm text-primary">선택한 {rows.length}개 chunk에 같은 값을 적용</h2>
          <p className="mt-1 text-caption-sm text-secondary">켜 둔 필드만 저장합니다. 번역문은 같은 문장이 모든 선택 chunk에 들어갑니다.</p>
        </div>
        <button type="button" onClick={onCancel} className="rounded-full p-1 text-mute hover:text-primary"><X className="h-4 w-4" /></button>
      </div>
      <div className="grid gap-3 lg:grid-cols-3">
        <label className="rounded-[1.25rem] bg-surface-container-lowest p-3">
          <label className="flex items-center gap-2 font-mono text-data-label uppercase text-data-label"><input type="checkbox" name="bulk-apply-translated-text" checked={applyText} onChange={(e) => setApplyText(e.target.checked)} className="h-4 w-4 accent-black" />translated_text</label>
          <textarea value={translatedText} onChange={(e) => setTranslatedText(e.target.value)} disabled={!applyText || pending} rows={5} className="mt-3 w-full rounded-[1rem] border border-border-hairline bg-white p-3 text-body-sm text-primary focus:outline-none" />
          {applyText ? <p className="mt-2 text-caption-sm text-term-yellow">여러 청크에 같은 번역문을 적용합니다.</p> : null}
        </label>
        <label className="rounded-[1.25rem] bg-surface-container-lowest p-3">
          <label className="flex items-center gap-2 font-mono text-data-label uppercase text-data-label"><input type="checkbox" name="bulk-apply-tts-instruction" checked={applyInstruction} onChange={(e) => setApplyInstruction(e.target.checked)} className="h-4 w-4 accent-black" />tts_instruct_text</label>
          <textarea value={instructionText} onChange={(e) => setInstructionText(stripTokenOnly(e.target.value))} disabled={!applyInstruction || pending} rows={5} placeholder="Please say it ..." className="mt-3 w-full rounded-[1rem] border border-border-hairline bg-white p-3 font-mono text-code-sm text-primary focus:outline-none" />
          {applyInstruction && hasNonLatinScript(instructionText) ? <p className="mt-2 text-caption-sm text-term-yellow">한국어/한자가 포함돼 있습니다. 가능하면 영어 directive로 바꿔 주세요.</p> : null}
        </label>
        <div className="rounded-[1.25rem] bg-surface-container-lowest p-3">
          <label className="flex items-center gap-2 font-mono text-data-label uppercase text-data-label"><input type="checkbox" name="bulk-apply-emotion" checked={applyEmotion} onChange={(e) => setApplyEmotion(e.target.checked)} className="h-4 w-4 accent-black" />emotion vector</label>
          <div className={applyEmotion ? "mt-3" : "pointer-events-none mt-3 opacity-45"}>
            <EmotionEqualizer scores={first?.emotion_scores} label={first?.emotion} pending={!applyEmotion || pending} onCancel={() => setApplyEmotion(false)} onChange={setEmotionDraft} onSave={setEmotionDraft} />
          </div>
        </div>
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <Button variant="secondary" size="sm" onClick={onCancel} disabled={pending}>Cancel</Button>
        <Button size="sm" onClick={submit} disabled={!canApply || pending}>{pending ? "Saving" : "Apply to selected"}</Button>
      </div>
    </Card>
  );
}

function ChunkDetail({
  runId,
  row,
  pending,
  speakers,
  referenceBank,
  onSaveInstruction,
  onSaveEmotion,
  onSaveSpeakerReference,
}: {
  runId: string;
  row: ChunkRow;
  pending: boolean;
  speakers: string[];
  referenceBank: Record<string, ReferenceCandidate[]>;
  onSaveInstruction: (text: string) => void;
  onSaveEmotion: (label: string, scores: Record<string, number>) => void;
  onSaveSpeakerReference: (payload: PatchChunkPayload) => void;
}) {
  const stale = row.dub_stale || row.status === "stale";
  const blocked = Boolean(row.translation_blocked) || row.status === "blocked";
  const [eqEdit, setEqEdit] = useState(false);
  const [instructionEdit, setInstructionEdit] = useState(false);
  // <|endofprompt|> 는 저장 직전에 자동 부여 — 사용자에게는 노출/편집 모두 차단
  const visibleInstruction = stripEndOfPrompt(row.tts_instruct_text);
  const [draftInstruction, setDraftInstruction] = useState(visibleInstruction);
  // EQ 미저장 draft — Preview 가 이 값으로 LLM 호출 가능
  const [eqDraft, setEqDraft] = useState<{ label: string; scores: Record<string, number> } | null>(null);
  // Preview 결과 — 사용자가 "Use as manual" 누르기 전까지 비저장 상태
  const [preview, setPreview] = useState<{ instruction: string; source: "llm" | "fallback" } | null>(null);
  const [speakerDraft, setSpeakerDraft] = useState(row.speaker ?? "");
  const [referenceModeDraft, setReferenceModeDraft] = useState(normalizeReferenceMode(row.reference_override_mode));
  const [referenceChunkDraft, setReferenceChunkDraft] = useState(row.reference_override_chunk_id ?? "");

  const previewMutation = useMutation({
    mutationFn: (payload: { emotion_label?: string; emotion_scores?: Record<string, number> }) =>
      api.previewInstruction(runId, row.chunk_id, payload),
    onSuccess: (res) => setPreview({ instruction: res.instruction, source: res.source }),
  });

  // MOS 추천 채점 — controller 에서 MOS 모델로 화자별 레퍼런스 후보를 채점, 최고 MOS 에 추천 표시.
  // 기존 후보/선택은 불변(추천 배지만 추가). blocking ~30-60s.
  const refBankClient = useQueryClient();
  const scoreMosMutation = useMutation({
    mutationFn: () => api.scoreSpeakerReferenceBank(runId),
    onSuccess: (data) => refBankClient.setQueryData(["speaker-reference-bank", runId], data),
  });

  // 새 청크 선택 시 편집 상태 초기화
  useEffect(() => {
    setEqEdit(false);
    setInstructionEdit(false);
    setDraftInstruction(stripEndOfPrompt(row.tts_instruct_text));
    setEqDraft(null);
    setPreview(null);
    setSpeakerDraft(row.speaker ?? "");
    setReferenceModeDraft(normalizeReferenceMode(row.reference_override_mode));
    setReferenceChunkDraft(row.reference_override_chunk_id ?? "");
    previewMutation.reset();
  }, [row.chunk_id, row.tts_instruct_text, row.speaker, row.reference_override_mode, row.reference_override_chunk_id]);

  const triggerPreview = () => {
    setPreview(null);
    const payload = eqEdit && eqDraft
      ? { emotion_label: eqDraft.label, emotion_scores: eqDraft.scores }
      : {};
    previewMutation.mutate(payload);
  };

  const adoptPreviewAsManual = () => {
    if (!preview) return;
    setDraftInstruction(stripEndOfPrompt(preview.instruction));
    setInstructionEdit(true);
    setPreview(null);
  };
  const referenceCandidates = referenceBank[speakerDraft] ?? [];
  const acceptedCandidates = referenceCandidates.filter((candidate) => candidate.accepted);
  const selectedReferenceCandidate = referenceCandidates.find((candidate) => candidate.chunk_id === referenceChunkDraft);
  const speakerReferenceDirty =
    speakerDraft !== (row.speaker ?? "") ||
    referenceModeDraft !== normalizeReferenceMode(row.reference_override_mode) ||
    referenceChunkDraft !== (row.reference_override_chunk_id ?? "");
  const saveSpeakerReference = () => {
    const payload: PatchChunkPayload = { speaker: speakerDraft };
    payload.reference_mode = referenceModeDraft;
    payload.reference_chunk_id = referenceModeDraft === "self" ? null : referenceChunkDraft || null;
    onSaveSpeakerReference(payload);
  };

  return (
    <div className="space-y-5">
      <header>
        <p className="font-mono text-data-label uppercase text-data-label">Chunk Details</p>
        <h2 className="mt-1 font-display text-heading-md text-primary">{row.chunk_id}</h2>
      </header>
      {blocked ? (
        <Card className="rounded-[2rem] border border-term-yellow/40 bg-term-yellow/5 p-5">
          <div className="inline-flex rounded-full bg-term-yellow/15 px-3 py-1 text-caption-strong text-term-yellow">TRANSLATION BLOCKED</div>
          <p className="mt-3 text-body-sm text-secondary">
            이 청크의 번역이 LLM 콘텐츠 필터에 차단됐어요. 본문이 비어 있어 합성을 못 하니 한국어 번역을 직접 입력해주세요.
          </p>
          {row.translation_blocked_reason ? (
            <p className="mt-2 font-mono text-caption-sm text-mute">사유 · {row.translation_blocked_reason}</p>
          ) : null}
          <p className="mt-3 font-mono text-caption-sm text-secondary">
            왼쪽 표의 <span className="text-primary">Translated Text</span> 칸을 더블클릭해서 입력 → Resume from build_timeline 으로 이어 돌리면 됩니다.
          </p>
        </Card>
      ) : (
        <Card className="rounded-[2rem] p-5">
          <div className={`inline-flex rounded-full px-3 py-1 text-caption-strong ${stale ? "bg-term-yellow/10 text-term-yellow" : "bg-status-done/10 text-status-done"}`}>{stale ? "DUB STALE" : "DUB READY"}</div>
          <p className="mt-3 text-body-sm text-secondary">{stale ? "번역·감정·프롬프트 변경으로 chunk 단위 redub가 필요합니다." : "현재 chunk의 더빙 산출물이 최신 상태입니다."}</p>
        </Card>
      )}

      <Card className="rounded-[2rem] p-5">
        <div className="flex items-center justify-between">
          <p className="font-mono text-data-label uppercase text-data-label">TTS Instruction</p>
          <span className="font-mono text-caption-strong text-mute">source · {row.tts_instruct_source ?? "—"}</span>
        </div>
        {instructionEdit ? (
          <>
            <div className="mt-3 rounded-[1rem] border border-border-hairline bg-surface-soft p-3">
              <p className="font-mono text-caption-sm text-secondary">
                CosyVoice 는 <span className="font-mono text-primary">영어 directive</span> 로만 학습됐어요. 한국어로 쓰면 모델이 합성 텍스트로 오인해 그대로 읽어버립니다.
              </p>
              <p className="mt-1 font-mono text-caption-sm text-mute">
                예 · "Please say it close to the speaker's natural delivery, with a faint hint of curiosity."
              </p>
              <p className="mt-1 font-mono text-caption-sm text-mute">
                한국어로 의도를 쓰고 싶다면 <span className="font-mono text-primary">Preview LLM</span> 으로 영어 directive 를 자동 생성한 뒤 Use as manual 을 사용하세요.
              </p>
            </div>
            <textarea
              value={draftInstruction}
              onChange={(e) => setDraftInstruction(stripTokenOnly(e.target.value))}
              disabled={pending}
              rows={5}
              placeholder="Please say it ..."
              className="mt-3 w-full rounded-[1.5rem] border border-primary bg-white p-4 font-mono text-code-sm text-primary focus:outline-none"
            />
            {hasNonLatinScript(draftInstruction) ? (
              <p className="mt-2 rounded-[1rem] border border-term-yellow bg-term-yellow/10 p-3 font-mono text-caption-sm text-term-yellow">
                한국어/한자가 포함돼 있어요. CosyVoice 는 이걸 directive 로 못 알아듣고 그대로 합성해 버립니다. 영어로 바꿔 주세요.
              </p>
            ) : null}
            <p className="mt-2 font-mono text-caption-sm text-mute">CosyVoice 종료 토큰은 저장 시 자동으로 붙습니다.</p>
            <div className="mt-3 flex justify-end gap-2">
              <Button variant="secondary" size="sm" onClick={() => { setInstructionEdit(false); setDraftInstruction(stripEndOfPrompt(row.tts_instruct_text)); }} disabled={pending}>Cancel</Button>
              <Button size="sm" onClick={() => { onSaveInstruction(ensureEndOfPrompt(draftInstruction)); setInstructionEdit(false); }} disabled={pending}>{pending ? "Saving" : "Save"}</Button>
            </div>
          </>
        ) : (
          <>
            <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-[1.5rem] bg-surface-soft p-4 font-mono text-code-sm text-secondary">{visibleInstruction || "— 아직 생성된 프롬프트가 없습니다 —"}</pre>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={triggerPreview}
                disabled={pending || previewMutation.isPending}
                title={eqEdit ? "현재 EQ 미저장 값으로 LLM 미리보기" : "현재 emotion 으로 LLM 미리보기"}
              >
                {previewMutation.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                {previewMutation.isPending ? "Previewing" : "Preview LLM"}
              </Button>
              <Button variant="secondary" size="sm" onClick={() => setInstructionEdit(true)} disabled={pending}>Edit</Button>
            </div>
            {previewMutation.isError ? (
              <p className="mt-3 rounded-[1rem] bg-error-container p-3 text-caption-sm text-on-error-container">
                미리보기 실패 — {(previewMutation.error as Error).message}
              </p>
            ) : null}
            {preview ? (
              <div className="mt-3 rounded-[1.5rem] border border-primary/40 bg-primary/5 p-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-data-label uppercase text-primary">preview · {preview.source}</span>
                  <button type="button" onClick={() => setPreview(null)} className="rounded-full p-1 text-mute hover:text-primary"><X className="h-3.5 w-3.5" /></button>
                </div>
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-code-sm text-primary">
                  {stripEndOfPrompt(preview.instruction)}
                </pre>
                <p className="mt-2 font-mono text-caption-sm text-mute">
                  EQ {eqEdit ? "draft" : "현재 저장값"} 으로 LLM 이 만든 후보. 저장 안 됐어요.
                </p>
                <div className="mt-3 flex justify-end gap-2">
                  <Button variant="secondary" size="sm" onClick={() => setPreview(null)} disabled={pending}>닫기</Button>
                  <Button size="sm" onClick={adoptPreviewAsManual} disabled={pending}>Use as manual</Button>
                </div>
              </div>
            ) : null}
          </>
        )}
      </Card>

      <Card className="rounded-[2rem] p-5">
        <div className="flex items-center justify-between gap-3">
          <p className="font-mono text-data-label uppercase text-data-label">Speaker & Reference</p>
          {speakerReferenceDirty ? <span className="rounded-full bg-term-yellow/10 px-3 py-1 text-caption-strong text-term-yellow">UNSAVED</span> : null}
        </div>
        <div className="mt-4 grid gap-3">
          <label className="grid gap-2">
            <span className="font-mono text-data-label uppercase text-data-label">speaker</span>
            <select
              name="speaker-reference"
              aria-label="Speaker reference"
              value={speakerDraft}
              onChange={(event) => {
                const nextSpeaker = event.target.value;
                setSpeakerDraft(nextSpeaker);
                const nextCandidates = referenceBank[nextSpeaker] ?? [];
                if (referenceModeDraft !== "self" && !nextCandidates.some((candidate) => candidate.chunk_id === referenceChunkDraft)) {
                  setReferenceChunkDraft(nextCandidates.find((candidate) => candidate.accepted)?.chunk_id ?? nextCandidates[0]?.chunk_id ?? "");
                }
              }}
              disabled={pending}
              className="h-10 rounded-full border border-border-hairline bg-white px-4 font-mono text-code-sm text-primary focus:outline-none disabled:opacity-60"
            >
              {unique([speakerDraft, ...speakers].filter(Boolean)).map((speaker) => <option key={speaker} value={speaker}>{speaker}</option>)}
            </select>
          </label>
          <div>
            <p className="font-mono text-data-label uppercase text-data-label">reference source</p>
            <p className="mt-1 font-mono text-caption-sm text-mute">
              current · {row.reference_mode ?? "self"} {row.reference_chunk_id ? `· ${row.reference_chunk_id}` : ""}
            </p>
            <div className="mt-2 grid grid-cols-2 rounded-full border border-border-hairline bg-surface-soft p-1">
              {(["self", "speaker_bank"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => {
                    setReferenceModeDraft(mode);
                    if (mode === "self") setReferenceChunkDraft("");
                    else if (!referenceCandidates.some((candidate) => candidate.chunk_id === referenceChunkDraft)) {
                      setReferenceChunkDraft(acceptedCandidates[0]?.chunk_id ?? referenceCandidates[0]?.chunk_id ?? "");
                    }
                  }}
                  disabled={pending}
                  className={`h-8 rounded-full px-3 text-caption-strong ${referenceModeDraft === mode ? "bg-primary text-white" : "text-secondary hover:bg-surface-container"}`}
                >
                  {mode === "self" ? "Self" : "Speaker bank"}
                </button>
              ))}
            </div>
          </div>
          {referenceModeDraft === "speaker_bank" ? (
            <div className="rounded-[1.5rem] border border-border-hairline bg-surface-soft p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="font-mono text-data-label uppercase text-data-label">bank candidates</p>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-caption-sm text-mute">{acceptedCandidates.length}/{referenceCandidates.length} accepted</span>
                  <button
                    type="button"
                    onClick={() => scoreMosMutation.mutate()}
                    disabled={scoreMosMutation.isPending}
                    title="MOS 모델로 화자별 후보를 채점해 가장 깨끗한 음성을 추천합니다(기존 선택은 유지)"
                    className="rounded-full border border-border-hairline bg-white px-2 py-0.5 text-[10px] font-semibold text-primary hover:border-primary disabled:opacity-50"
                  >
                    {scoreMosMutation.isPending ? "MOS 채점 중…" : "★ MOS로 추천"}
                  </button>
                </div>
              </div>
              {referenceCandidates.length ? (
                <div className="space-y-2">
                  {referenceCandidates.slice(0, 8).map((candidate) => (
                    <ReferenceCandidateRow
                      key={candidate.chunk_id}
                      candidate={candidate}
                      selected={candidate.chunk_id === referenceChunkDraft}
                      pending={pending}
                      onSelect={() => setReferenceChunkDraft(candidate.chunk_id)}
                    />
                  ))}
                </div>
              ) : (
                <p className="rounded-[1rem] bg-white p-3 text-caption-sm text-secondary">이 화자의 reference 후보가 아직 없습니다.</p>
              )}
            </div>
          ) : (
            <p className="rounded-[1.5rem] bg-surface-soft p-3 text-caption-sm text-secondary">현재 청크 오디오를 prompt reference로 사용합니다.</p>
          )}
          {selectedReferenceCandidate ? (
            <p className="font-mono text-caption-sm text-mute">
              selected · {selectedReferenceCandidate.chunk_id} · {formatDuration(selectedReferenceCandidate.duration)}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setSpeakerDraft(row.speaker ?? "");
                setReferenceModeDraft(normalizeReferenceMode(row.reference_override_mode));
                setReferenceChunkDraft(row.reference_override_chunk_id ?? "");
              }}
              disabled={pending || !speakerReferenceDirty}
            >
              Reset
            </Button>
            <Button size="sm" onClick={saveSpeakerReference} disabled={pending || !speakerReferenceDirty || !speakerDraft}>
              Save Assignment
            </Button>
          </div>
        </div>
      </Card>

      <Card className="rounded-[2rem] p-5">
        <div className="flex items-center justify-between">
          <p className="font-mono text-data-label uppercase text-data-label">Emotion Vector</p>
          {!eqEdit ? (
            <Button variant="secondary" size="sm" onClick={() => setEqEdit(true)} disabled={pending}>
              <Sliders className="mr-2 h-4 w-4" />Adjust
            </Button>
          ) : null}
        </div>

        {eqEdit ? (
          <div className="mt-4">
            <EmotionEqualizer
              scores={row.emotion_scores}
              label={row.emotion}
              pending={pending}
              onCancel={() => { setEqEdit(false); setEqDraft(null); }}
              onChange={setEqDraft}
              onSave={({ label, scores }) => {
                onSaveEmotion(label, scores);
                setEqEdit(false);
                setEqDraft(null);
              }}
            />
          </div>
        ) : (
          <div className="mt-4 space-y-3">
            {Object.entries(row.emotion_scores ?? { [row.emotion ?? "neutral"]: 1 }).map(([label, value]) => (
              <div key={label}>
                <div className="mb-1 flex justify-between font-mono text-code-sm text-secondary"><span>{label}</span><span>{Math.round(value * 100)}%</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-surface-container"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.round(value * 100)}%` }} /></div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="rounded-[2rem] p-5">
        <p className="font-mono text-data-label uppercase text-data-label">Timing & Pace</p>
        <div className="mt-4 grid grid-cols-2 gap-3 font-mono text-code-sm">
          <div className="rounded-[1.5rem] bg-surface-soft p-4"><div className="text-mute">original</div><div className="mt-1 text-primary">{formatDuration(row.duration_original)}</div></div>
          <div className={`rounded-[1.5rem] p-4 ${stale ? "border border-term-yellow bg-term-yellow/10" : "bg-surface-soft"}`}><div className="text-mute">dub</div><div className="mt-1 text-primary">{formatDuration(row.duration_dub)}</div></div>
        </div>
      </Card>
    </div>
  );
}

function ReferenceCandidateRow({
  candidate,
  selected,
  pending,
  onSelect,
}: {
  candidate: ReferenceCandidate;
  selected: boolean;
  pending: boolean;
  onSelect: () => void;
}) {
  const src = toStaticUrl(candidate.wav);
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={pending}
      className={`w-full rounded-[1rem] border p-3 text-left ${selected ? "border-primary bg-white" : "border-border-hairline bg-white/70 hover:bg-white"}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-code-sm text-primary">{candidate.chunk_id}</span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${candidate.accepted ? "bg-status-done/10 text-status-done" : "bg-term-yellow/10 text-term-yellow"}`}>
              {candidate.accepted ? "OK" : "CHECK"}
            </span>
            {candidate.mos_recommended ? (
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">★ MOS 추천</span>
            ) : null}
          </div>
          <p className="mt-1 line-clamp-2 text-caption-sm text-secondary">{candidate.text_src || "No transcript"}</p>
          <p className="mt-1 font-mono text-caption-sm text-mute">
            {formatRange(candidate.start, candidate.end)} · {formatDuration(candidate.duration)}
            {candidate.score != null ? ` · score ${candidate.score.toFixed(2)}` : ""}
            {candidate.mos != null ? ` · MOS ${candidate.mos.toFixed(2)}` : ""}
          </p>
          {candidate.critical_flags.length ? (
            <p className="mt-1 font-mono text-caption-sm text-term-yellow">{candidate.critical_flags.join(", ")}</p>
          ) : null}
        </div>
        <span className={`mt-1 h-3 w-3 shrink-0 rounded-full border ${selected ? "border-primary bg-primary" : "border-border-hairline bg-white"}`} />
      </div>
      {src ? (
        <audio
          src={src}
          controls
          preload="none"
          onClick={(event) => event.stopPropagation()}
          className="mt-2 h-8 w-full"
        />
      ) : null}
    </button>
  );
}

function TerminalPanel({
  open,
  toggle,
  connected,
  active,
  lines,
}: {
  open: boolean;
  toggle: () => void;
  connected: boolean;
  active: boolean;
  lines: string[];
}) {
  return (
    <footer className="col-span-1 flex min-h-0 flex-col border-t border-white/10 bg-[#080808] text-white lg:col-span-2">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-white/10 bg-[#151515] px-5">
        <div className="flex items-center gap-3">
          <span className="font-mono text-data-label uppercase text-white/60">Pipeline Logs</span>
          <span className="inline-flex items-center gap-2 font-mono text-code-sm text-white/60">
            <span className={`h-2 w-2 rounded-full ${connected ? "bg-status-done" : active ? "bg-term-yellow" : "bg-mute"}`} />
            {connected ? "live" : active ? "connecting" : "idle"}
          </span>
          <span className="font-mono text-code-sm text-white/40">{lines.length} lines</span>
        </div>
        <button
          type="button"
          onClick={toggle}
          className="inline-flex h-7 items-center gap-1 rounded-full bg-white/5 px-3 font-mono text-caption-sm text-white/70 hover:bg-white/10"
        >
          {open ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />}
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {/* 마운트 유지 — overflow 로 잘림 처리해 토글 시 깜빡임 없이 슬라이딩 */}
      <LogTail
        lines={lines}
        className="min-h-0 flex-1 px-5 py-3 text-white/70"
        emptyLabel={active ? "WebSocket 로그를 기다리는 중입니다." : "저장된 로그가 아직 없습니다."}
      />
    </footer>
  );
}

function ToastBanner({
  kind,
  onClose,
  onCompare,
  onBackToRun,
  onRunFinal,
}: {
  kind: ToastKind;
  onClose: () => void;
  onCompare: () => void;
  onBackToRun: () => void;
  onRunFinal: () => void;
}) {
  const message = kind === "final-done" ? "Final output 완료" : "Redub 완료";
  return (
    <div className="pointer-events-none fixed right-6 top-[72px] z-30 w-[360px]">
      <div className="pointer-events-auto rounded-[1.5rem] border border-border-hairline bg-surface-container-lowest p-4 shadow-running-ring">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-display text-heading-sm text-primary">{message}</p>
            <p className="mt-1 text-caption-sm text-secondary">
              {kind === "final-done"
                ? "최종 영상이 만들어졌습니다. 결과를 확인하세요."
                : "청크가 새 dub 으로 갱신됐습니다. 더 검수하거나 최종 출력을 만드세요."}
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded-full p-1 text-mute hover:text-primary"><X className="h-4 w-4" /></button>
        </div>
        <div className="mt-4 flex gap-2">
          {kind === "final-done" ? (
            <>
              <Button size="sm" onClick={onCompare}>Compare 보기</Button>
              <Button variant="secondary" size="sm" onClick={onBackToRun}>Run으로 돌아가기</Button>
            </>
          ) : (
            <>
              <Button size="sm" onClick={onRunFinal}>Run Final Output</Button>
              <Button variant="secondary" size="sm" onClick={onBackToRun}>Run으로 돌아가기</Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function filterRows(rows: ChunkRow[], speaker: string, emotion: string, search: string, problemOnly: boolean): ChunkRow[] {
  const query = search.trim().toLowerCase();
  return rows.filter((row) => {
    if (speaker !== "all" && row.speaker !== speaker) return false;
    if (emotion !== "all" && row.emotion !== emotion) return false;
    if (problemOnly && !row.dub_stale && row.status !== "stale" && row.status !== "error" && !row.error && !row.translation_blocked && row.status !== "blocked" && !(row.reference_mode !== "self" && !row.reference_chunk_id)) return false;
    if (!query) return true;
    return `${row.source_text ?? ""} ${row.translated_text ?? ""} ${row.chunk_id}`.toLowerCase().includes(query);
  });
}

function unique(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function normalizeReferenceMode(value: string | null | undefined): "self" | "speaker_bank" {
  const normalized = (value ?? "").trim().toLowerCase();
  if (["speaker_bank", "speaker_best", "bank", "best"].includes(normalized)) return "speaker_bank";
  return "self";
}

function formatRange(start: number | null | undefined, end: number | null | undefined): string {
  if (start == null || end == null) return "--:-- → --:--";
  return `${formatClock(start)} → ${formatClock(end)}`;
}

function formatClock(value: number): string {
  const min = Math.floor(value / 60);
  const sec = Math.floor(value % 60).toString().padStart(2, "0");
  return `${min}:${sec}`;
}

function formatDuration(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(2)}s`;
}
