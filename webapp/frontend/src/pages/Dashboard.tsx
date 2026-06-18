// 시스템 전반 관제실 — 진행 중 run / KPI / GPU 헬스 / cross-run activity feed
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, FolderKanban, Plus } from "lucide-react";
import { api, PIPELINE_STEPS, type ActivityEvent, type ProjectSummary, type RunRecord, type StepRecord } from "@/api/client";
import { EventRow, groupByDay } from "@/components/activity/EventRow";
import { PipelineGraph } from "@/components/pipeline/PipelineGraph";
import { fileName } from "@/lib/staticUrl";
import { canCompareOutput, completedStepCount, progressPercent, runStageMessage } from "@/lib/runReadiness";

export function Dashboard() {
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 10000, retry: 1 });
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects, refetchInterval: 15000, retry: 1 });
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 10000, retry: 1 });
  const activityQuery = useQuery({ queryKey: ["activity-cross"], queryFn: () => api.getCrossActivity(50), refetchInterval: 15000, retry: 1 });

  const sortedRuns = [...(runsQuery.data ?? [])].sort((a, b) => b.created_at - a.created_at);
  const activeRun = sortedRuns.find((r) => r.status === "running" || r.status === "queued") ?? null;
  const heroRun = activeRun ?? sortedRuns[0] ?? null;
  const heroSteps = heroRun?.steps?.length ? heroRun.steps : pendingSteps();
  const heroDone = completedStepCount(heroSteps);
  const heroPct = progressPercent(heroSteps);

  const projects = projectsQuery.data ?? [];
  const kpi = computeKpi(sortedRuns, projects);
  const services = healthQuery.data?.services ?? {};
  const events = activityQuery.data ?? [];
  const grouped = groupByDay(events).slice(0, 2);

  return (
    <section className="min-h-[calc(100vh-56px)] bg-background">
      <div className="absolute inset-x-0 top-[56px] h-[480px] bg-dot-grid bg-[length:16px_16px] opacity-40" aria-hidden />
      <div className="relative mx-auto max-w-[1480px] px-8 py-7">
        <Hero run={heroRun} active={Boolean(activeRun)} steps={heroSteps} done={heroDone} pct={heroPct} />

        <KpiStrip kpi={kpi} />

        <div className="mt-7 grid grid-cols-1 gap-5 lg:grid-cols-[420px_minmax(0,1fr)]">
          <SystemHealth services={services} ok={healthQuery.data?.docker_cli ?? false} />
          <ActivityFeedPanel events={events} grouped={grouped} loading={activityQuery.isLoading} />
        </div>
      </div>

      <Link to="/runs/new" className="fixed bottom-8 right-8 z-20 inline-flex h-12 items-center gap-3 rounded-full bg-primary px-6 text-body-sm-strong text-white shadow-running-ring hover:bg-ink-deep">
        <Plus className="h-4 w-4" />
        Start New Dubbing
      </Link>
    </section>
  );
}

function Hero({ run, active, steps, done, pct }: { run: RunRecord | null; active: boolean; steps: StepRecord[]; done: number; pct: number }) {
  if (!run) {
    return (
      <div className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-8 text-center">
        <h1 className="font-display text-heading-lg text-primary">아직 실행한 더빙이 없습니다.</h1>
        <p className="mt-2 text-body-sm text-secondary">New Project 로 입력 영상을 업로드하면 첫 run 이 시작됩니다.</p>
        <div className="mt-5 flex justify-center gap-2">
          <Link to="/runs/new" className="inline-flex h-10 items-center gap-2 rounded-full bg-primary px-5 text-body-sm-strong text-white hover:bg-ink-deep">
            <Plus className="h-4 w-4" />
            New Project
          </Link>
          <Link to="/projects" className="inline-flex h-10 items-center gap-2 rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm-strong text-primary hover:bg-surface-container">
            <FolderKanban className="h-4 w-4" />
            Projects 보기
          </Link>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <span className={`rounded-full px-3 py-1 font-mono text-data-label uppercase ${active ? "bg-primary text-white" : "bg-surface-container text-data-label"}`}>
              {active ? "now running" : "latest"}
            </span>
            <h1 className="truncate font-display text-heading-lg text-primary">{fileName(run.input_video)}</h1>
            <span className="rounded-full bg-surface-container px-3 py-1 font-mono text-data-label uppercase text-data-label">{run.status}</span>
          </div>
          <p className="mt-2 font-mono text-code-sm text-mute">{run.run_id}</p>
          <p className="mt-2 text-body-sm text-secondary">{runStageMessage(run)}</p>
        </div>
        <div className="flex shrink-0 flex-col gap-2">
          <Link to={`/runs/${run.run_id}`} className="inline-flex h-9 items-center justify-center rounded-full bg-primary px-4 text-body-sm-strong text-white hover:bg-ink-deep">
            진행 상태 보기
          </Link>
          {canCompareOutput(run) ? (
            <Link to={`/runs/${run.run_id}/compare`} className="inline-flex h-9 items-center justify-center rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm-strong text-primary hover:bg-surface-container">
              결과 비교
            </Link>
          ) : null}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between text-caption-sm text-secondary">
          <span>Pipeline Progress</span>
          <span className="font-mono text-code-sm text-primary">{done}/{steps.length} · {pct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-surface-container">
          <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className="mt-5 rounded-[1.5rem] border border-border-hairline bg-surface-container-lowest/70 p-4">
        <PipelineGraph steps={steps} runId={run.run_id} />
      </div>
    </div>
  );
}

interface Kpi {
  projectCount: number;
  runCount: number;
  successRate: number; // 0~100
  avgDurationMin: number | null;
}

function computeKpi(runs: RunRecord[], projects: ProjectSummary[]): Kpi {
  const finished = runs.filter((r) => r.status === "success" || r.status === "failed");
  const success = finished.filter((r) => r.status === "success").length;
  const successRate = finished.length ? Math.round((success / finished.length) * 100) : 0;

  const durations: number[] = [];
  for (const run of runs) {
    if (run.status !== "success") continue;
    const starts = run.steps.map((s) => s.started_at).filter((v): v is number => typeof v === "number");
    const ends = run.steps.map((s) => s.ended_at).filter((v): v is number => typeof v === "number");
    if (!starts.length || !ends.length) continue;
    durations.push(Math.max(...ends) - Math.min(...starts));
  }
  const avgSec = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null;
  return {
    projectCount: projects.length,
    runCount: runs.length,
    successRate,
    avgDurationMin: avgSec == null ? null : Math.round(avgSec / 60),
  };
}

function KpiStrip({ kpi }: { kpi: Kpi }) {
  return (
    <div className="mt-7 grid grid-cols-2 gap-3 md:grid-cols-4">
      <KpiCard label="Projects" value={kpi.projectCount} hint="영상 단위 그룹" linkTo="/projects" />
      <KpiCard label="Total Runs" value={kpi.runCount} hint="모든 실행 시도" linkTo="/projects" />
      <KpiCard label="Success Rate" value={`${kpi.successRate}%`} hint="success / (success+failed)" />
      <KpiCard label="Avg Duration" value={kpi.avgDurationMin == null ? "—" : `${kpi.avgDurationMin}m`} hint="success run 평균" />
    </div>
  );
}

function KpiCard({ label, value, hint, linkTo }: { label: string; value: number | string; hint: string; linkTo?: string }) {
  const inner = (
    <>
      <p className="font-mono text-data-label uppercase text-data-label">{label}</p>
      <p className="mt-2 font-display text-heading-lg text-primary">{value}</p>
      <p className="mt-1 font-mono text-caption-sm text-mute">{hint}</p>
    </>
  );
  if (linkTo) {
    return (
      <Link to={linkTo} className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-5 transition hover:bg-surface-soft">
        {inner}
      </Link>
    );
  }
  return <div className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-5">{inner}</div>;
}

function SystemHealth({ services, ok }: { services: Record<string, string>; ok: boolean }) {
  // 모든 파이프라인 단계가 controller 한 컨테이너에서 실행됨(PIPELINE_ALL_IN_SERVICE=controller).
  // separator/diarizer/speaker/tts-cosyvoice 는 별도 컨테이너가 아니라 controller 내부 단계이므로,
  // 5개를 나열해 4개가 stopped(빨강)로 보이는 대신 단일 "더빙 파이프라인" 상태로 합쳐 표시한다.
  const engineState = services["controller"] ?? "unknown";
  const running = engineState === "running";
  const tone = running
    ? "bg-status-done/10 text-status-done"
    : engineState === "stopped"
      ? "bg-status-failed/10 text-status-failed"
      : "bg-surface-container text-mute";
  return (
    <div className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="font-mono text-data-label uppercase text-data-label">System Health</p>
          <h2 className="mt-1 font-display text-heading-sm text-primary">더빙 파이프라인</h2>
        </div>
        <span className={`rounded-full px-3 py-1 font-mono text-caption-strong ${ok ? "bg-status-done/10 text-status-done" : "bg-status-failed/10 text-status-failed"}`}>
          {ok ? "docker ok" : "docker offline"}
        </span>
      </div>
      <ul className="space-y-2">
        <li className="flex items-center justify-between rounded-[1rem] bg-surface-soft px-4 py-3">
          <span className="font-mono text-code-sm text-primary">pipeline</span>
          <span className={`rounded-full px-3 py-1 font-mono text-caption-strong ${tone}`}>{running ? "running" : engineState}</span>
        </li>
      </ul>
    </div>
  );
}

function ActivityFeedPanel({ events, grouped, loading }: { events: ActivityEvent[]; grouped: [string, ActivityEvent[]][]; loading: boolean }) {
  return (
    <div className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-5">
      <div className="mb-4 flex items-end justify-between">
        <div>
          <p className="font-mono text-data-label uppercase text-data-label">Activity Feed</p>
          <h2 className="mt-1 font-display text-heading-sm text-primary">전체 run 의 최근 변경</h2>
        </div>
        <Link to="/projects" className="inline-flex items-center gap-1 text-caption-strong text-primary hover:underline">
          전체 프로젝트
          <ArrowUpRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      {loading ? (
        <p className="text-body-sm text-mute">활동 이력을 불러오는 중입니다.</p>
      ) : events.length === 0 ? (
        <p className="text-body-sm text-secondary">아직 기록된 활동이 없습니다. run 을 시작하거나 청크를 편집하면 여기에 쌓입니다.</p>
      ) : (
        <div className="space-y-6">
          {grouped.map(([day, items]) => (
            <div key={day}>
              <p className="mb-2 font-mono text-data-label uppercase text-data-label">{day}</p>
              <ul className="relative border-l border-border-hairline pl-6">
                {items.slice(0, 8).map((event, idx) => (
                  <EventRow key={`${day}-${idx}`} event={event} showRunLink />
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function pendingSteps(): StepRecord[] {
  return PIPELINE_STEPS.map((name) => ({ name, state: "pending" }));
}
