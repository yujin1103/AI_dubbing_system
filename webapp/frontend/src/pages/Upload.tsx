// 영상 선택과 더빙 run 시작을 담당하는 참조 스타일 런처 화면
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";
import { api, type ConfigEntry, type InputFile, type RunOverrides } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PillInput } from "@/components/ui/pill-input";

// 백엔드가 임의 언어를 자동 지원(중국어 출력지시를 LLM 으로 자동 생성)하므로 목록은 빠른 선택용일 뿐,
// 검색창에 아무 언어나 입력하면 그대로 target_language 로 사용 가능(자유입력).
const COSYVOICE_LANGUAGES = [
  { label: "Korean", aliases: "ko korea 한국어" },
  { label: "Japanese", aliases: "ja jp japan 일본어" },
  { label: "English", aliases: "en 영어" },
  { label: "Chinese", aliases: "zh china 중국어 mandarin" },
  { label: "Cantonese", aliases: "yue 홍콩어 광둥어" },
  { label: "Spanish", aliases: "es 스페인어" },
  { label: "French", aliases: "fr 프랑스어" },
  { label: "German", aliases: "de 독일어" },
  { label: "Italian", aliases: "it 이탈리아어" },
  { label: "Portuguese", aliases: "pt 포르투갈어" },
  { label: "Russian", aliases: "ru 러시아어" },
  { label: "Vietnamese", aliases: "vi 베트남어" },
  { label: "Thai", aliases: "th 태국어" },
  { label: "Indonesian", aliases: "id 인도네시아어" },
  { label: "Arabic", aliases: "ar 아랍어" },
  { label: "Hindi", aliases: "hi 힌디어" },
] as const;

export function Upload() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const prefillInput = searchParams.get("input_path") ?? "";

  const inputsQuery = useQuery({ queryKey: ["inputs"], queryFn: api.listInputs });
  const configsQuery = useQuery({ queryKey: ["configs"], queryFn: api.listConfigs });
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30000 });

  const [selectedInput, setSelectedInput] = useState<string>(prefillInput);
  const [baseConfig, setBaseConfig] = useState<string>("configs/cosyvoice3-docker-draft.json");
  const [overrides, setOverrides] = useState<RunOverrides>({
    target_language: "Korean",
    fit_to_duration: true,
    duration_fit_max_tempo: 1.25,
    use_separator: true,
    skip_existing: false,
  });

  useEffect(() => {
    const inputs = inputsQuery.data;
    if (!inputs || inputs.length === 0) return;
    // prefill 이 실제 inputs 목록에 있으면 그대로 두고, 없거나 비어있으면 첫 번째로 폴백
    const exists = inputs.some((entry) => entry.input_path === selectedInput);
    if (!selectedInput || !exists) {
      setSelectedInput(inputs[0].input_path);
    }
  }, [inputsQuery.data, selectedInput]);

  useEffect(() => {
    const configs = configsQuery.data ?? [];
    if (configs.length > 0 && !configs.some((cfg) => cfg.path === baseConfig)) {
      setBaseConfig(configs[0].path);
    }
  }, [baseConfig, configsQuery.data]);

  const uploadMutation = useMutation({
    mutationFn: api.uploadVideo,
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["inputs"] });
      setSelectedInput(res.input_path);
    },
  });

  const runMutation = useMutation({
    mutationFn: api.createRun,
    onSuccess: (run) => {
      navigate(`/runs/${run.run_id}`);
    },
  });

  const onUploadFile = (file: File | undefined) => {
    if (!file) return;
    uploadMutation.mutate(file);
  };

  const onRun = () => {
    if (!selectedInput || !baseConfig) return;
    runMutation.mutate({ input_path: selectedInput, base_config: baseConfig, overrides });
  };

  return (
    <section className="min-h-[calc(100vh-56px)] bg-background px-8 py-7">
      <div className="mx-auto grid max-w-[1180px] grid-cols-[1fr_360px] gap-6">
        <div className="space-y-6">
          <header className="rounded-[2rem] border border-border-hairline bg-surface-container-lowest p-8">
            <p className="font-mono text-data-label uppercase text-data-label">Run Launcher</p>
            <h1 className="mt-3 font-display text-display-lg text-primary">새 더빙 파이프라인 시작</h1>
            <p className="mt-3 max-w-[640px] text-body-sm text-secondary">입력 영상과 config를 선택하면 Docker 내부 webapp-backend가 GPU/CPU 서비스에 파이프라인을 위임합니다.</p>
          </header>

          <Card className="space-y-5 rounded-[2rem] p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-mono text-data-label uppercase text-data-label">Input Video</p>
                <h2 className="mt-1 font-display text-heading-sm text-primary">영상 선택</h2>
              </div>
              <label className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-full border border-border-hairline bg-surface-soft px-4 text-body-sm-strong text-primary hover:bg-surface-container">
                <UploadCloud className="h-4 w-4" />
                Upload
                <input type="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm" onChange={(e) => onUploadFile(e.target.files?.[0])} className="sr-only" />
              </label>
            </div>
            <InputPicker inputs={inputsQuery.data ?? []} selected={selectedInput} onSelect={setSelectedInput} />
            {uploadMutation.isPending && <span className="text-caption-sm text-mute">업로드 중입니다.</span>}
          </Card>

          <Card className="space-y-5 rounded-[2rem] p-5">
            <div>
              <p className="font-mono text-data-label uppercase text-data-label">Config & Knobs</p>
              <h2 className="mt-1 font-display text-heading-sm text-primary">실행 설정</h2>
            </div>
            <ConfigPicker configs={configsQuery.data ?? []} value={baseConfig} loading={configsQuery.isLoading} error={configsQuery.isError} onChange={setBaseConfig} />
            <Knobs overrides={overrides} onChange={setOverrides} />
          </Card>
        </div>

        <aside className="space-y-6">
          <ServicesHealth services={healthQuery.data?.services} />
          <Card className="sticky top-[80px] space-y-5 rounded-[2rem] p-5">
            <div>
              <p className="font-mono text-data-label uppercase text-data-label">Ready State</p>
              <h2 className="mt-1 font-display text-heading-sm text-primary">Pipeline Queue</h2>
            </div>
            <div className="space-y-3 rounded-[1.5rem] bg-surface-soft p-4 font-mono text-code-sm text-secondary">
              <div className="flex justify-between gap-3"><span>input</span><span className="truncate text-primary">{selectedInput || "not selected"}</span></div>
              <div className="flex justify-between gap-3"><span>config</span><span className="truncate text-primary">{baseConfig}</span></div>
              <div className="flex justify-between gap-3"><span>target</span><span className="text-primary">{overrides.target_language}</span></div>
            </div>
            {runMutation.isError && <div className="rounded-[1.5rem] bg-error-container p-4 text-caption-sm text-on-error-container">{(runMutation.error as Error).message}</div>}
            <Button onClick={onRun} disabled={!selectedInput || !baseConfig || runMutation.isPending} className="h-12 w-full">
              {runMutation.isPending ? "시작 중" : "Start Dubbing"}
            </Button>
          </Card>
        </aside>
      </div>
    </section>
  );
}

function ServicesHealth({ services }: { services?: Record<string, string> }) {
  const labels: Record<string, string> = {
    controller: "controller",
    separator: "separator",
    diarizer: "diarizer",
    speaker: "speaker",
    "tts-cosyvoice": "tts",
  };
  return (
    <Card className="space-y-4 rounded-[2rem] p-5">
      <div>
        <p className="font-mono text-data-label uppercase text-data-label">Docker Services</p>
        <h2 className="mt-1 font-display text-heading-sm text-primary">서비스 상태</h2>
      </div>
      <div className="space-y-2 text-body-sm text-secondary">
        {Object.entries(labels).map(([key, label]) => {
          const status = services?.[key] ?? "unknown";
          const dot = status === "running" ? "bg-status-done" : "bg-status-failed";
          return (
            <div key={key} className="flex items-center justify-between rounded-full bg-surface-soft px-4 py-2">
              <span>{label}</span>
              <span className="inline-flex items-center gap-2 font-mono text-code-sm text-primary"><span className={`h-2 w-2 rounded-full ${dot}`} />{status}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function InputPicker({ inputs, selected, onSelect }: { inputs: InputFile[]; selected: string; onSelect: (path: string) => void }) {
  if (inputs.length === 0) {
    return <div className="rounded-[1.5rem] border border-dashed border-border-hairline bg-surface-soft p-5 text-body-sm text-mute">input 디렉터리에 영상이 없습니다. 우측 상단 Upload를 사용하세요.</div>;
  }
  return (
    <div className="grid gap-2">
      {inputs.map((entry) => {
        const isSelected = entry.input_path === selected;
        return (
          <button key={entry.input_path} type="button" onClick={() => onSelect(entry.input_path)} className={`flex w-full items-center justify-between rounded-full px-5 py-3 text-left transition-colors ${isSelected ? "bg-primary text-white" : "bg-surface-soft text-primary hover:bg-surface-container"}`}>
            <span className="truncate font-mono text-code-sm">{entry.name}</span>
            <span className={isSelected ? "text-white/70" : "text-mute"}>{(entry.size_bytes / (1024 * 1024)).toFixed(1)} MB</span>
          </button>
        );
      })}
    </div>
  );
}

function ConfigPicker({ configs, value, loading, error, onChange }: { configs: ConfigEntry[]; value: string; loading: boolean; error: boolean; onChange: (path: string) => void }) {
  if (loading) return <div className="rounded-[1.5rem] bg-surface-soft p-4 text-body-sm text-mute">베이스 config를 불러오는 중입니다.</div>;
  if (error) return <div className="rounded-[1.5rem] bg-error-container p-4 text-body-sm text-on-error-container">베이스 config 목록을 불러오지 못했습니다.</div>;
  if (configs.length === 0) return <div className="rounded-[1.5rem] border border-dashed border-border-hairline bg-surface-soft p-4 text-body-sm text-mute">configs 폴더에 선택 가능한 JSON config가 없습니다.</div>;
  return (
    <div className="space-y-3">
      <label className="text-body-sm-strong text-primary">베이스 config</label>
      <div className="grid gap-2">
        {configs.map((cfg) => {
          const selected = cfg.path === value;
          return (
            <button key={cfg.path} type="button" onClick={() => onChange(cfg.path)} className={`flex w-full items-center justify-between rounded-[1.25rem] px-4 py-3 text-left transition-colors ${selected ? "bg-primary text-white" : "bg-surface-soft text-primary hover:bg-surface-container"}`}>
              <span className="truncate font-mono text-code-sm">{cfg.name}</span>
              <span className={selected ? "text-white/70" : "text-mute"}>{selected ? "selected" : "select"}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Knobs({ overrides, onChange }: { overrides: RunOverrides; onChange: (next: RunOverrides) => void }) {
  const set = <K extends keyof RunOverrides>(key: K, value: RunOverrides[K]) => onChange({ ...overrides, [key]: value });
  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
      <LanguagePicker value={overrides.target_language ?? "Korean"} onChange={(value) => set("target_language", value)} />
      <TempoStepper value={overrides.duration_fit_max_tempo ?? 1.25} onChange={(value) => set("duration_fit_max_tempo", value)} />
      <div className="space-y-2">
        <label className="text-body-sm-strong text-primary">옵션</label>
        <div className="flex flex-wrap gap-2">
          <Toggle label="fit_to_duration" value={overrides.fit_to_duration ?? true} onChange={(v) => set("fit_to_duration", v)} />
          <Toggle label="use_separator" value={overrides.use_separator ?? true} onChange={(v) => set("use_separator", v)} />
          <Toggle label="skip_existing" value={overrides.skip_existing ?? false} onChange={(v) => set("skip_existing", v)} />
        </div>
      </div>
    </div>
  );
}

function LanguagePicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!needle) return COSYVOICE_LANGUAGES;
    return COSYVOICE_LANGUAGES.filter((lang) => `${lang.label} ${lang.aliases}`.toLowerCase().includes(needle));
  }, [needle]);
  const custom = query.trim();
  const hasExact = COSYVOICE_LANGUAGES.some((lang) => lang.label.toLowerCase() === needle);
  const isPreset = COSYVOICE_LANGUAGES.some((lang) => lang.label === value);

  return (
    <div className="space-y-2 md:col-span-2">
      <div className="flex items-end justify-between gap-3">
        <label className="text-body-sm-strong text-primary">target_language</label>
        <span className="font-mono text-code-sm text-mute">any language — auto</span>
      </div>
      <PillInput value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Korean, Japanese, Spanish, Vietnamese… or type any language" />
      <div className="flex flex-wrap gap-2">
        {filtered.map((lang) => {
          const selected = value === lang.label;
          return (
            <button key={lang.label} type="button" onClick={() => onChange(lang.label)} className={`inline-flex h-9 items-center gap-2 rounded-full px-4 text-button-md ${selected ? "bg-primary text-white" : "bg-surface-soft text-primary hover:bg-surface-container"}`}>
              {lang.label}
            </button>
          );
        })}
        {custom && !hasExact && (
          <button type="button" onClick={() => onChange(custom)} className={`inline-flex h-9 items-center gap-2 rounded-full px-4 text-button-md ${value === custom ? "bg-primary text-white" : "bg-surface-soft text-primary hover:bg-surface-container"}`}>
            Use “{custom}”
          </button>
        )}
      </div>
      {value && !isPreset && (
        <p className="text-caption-sm text-mute">선택됨: {value} — 백엔드가 언어 지시를 자동 생성합니다.</p>
      )}
    </div>
  );
}

function TempoStepper({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const step = (delta: number) => onChange(clampTempo(value + delta));
  return (
    <div className="space-y-2">
      <label className="text-body-sm-strong text-primary">duration_fit_max_tempo</label>
      <div className="flex items-center justify-between rounded-full bg-surface-soft px-3 py-2">
        <button type="button" onClick={() => step(-0.05)} className="h-8 w-8 rounded-full bg-surface-container text-heading-sm text-primary hover:bg-border-hairline">−</button>
        <span className="font-mono text-code-sm text-primary">{value.toFixed(2)}</span>
        <button type="button" onClick={() => step(0.05)} className="h-8 w-8 rounded-full bg-primary text-heading-sm text-white hover:bg-ink-deep">+</button>
      </div>
      <p className="text-caption-sm text-mute">0.80부터 1.50까지 0.05 단위로 조정합니다.</p>
    </div>
  );
}

function clampTempo(value: number): number {
  return Math.min(1.5, Math.max(0.8, Number(value.toFixed(2))));
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button type="button" onClick={() => onChange(!value)} className={`flex h-8 items-center gap-2 rounded-full px-3 text-body-sm-strong ${value ? "bg-primary text-white" : "bg-surface-soft text-secondary hover:bg-surface-container"}`}>
      <span className={`h-2 w-2 rounded-full ${value ? "bg-white" : "bg-mute"}`} />
      {label}
    </button>
  );
}
