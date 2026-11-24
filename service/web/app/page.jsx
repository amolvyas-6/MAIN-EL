"use client";

import { useEffect, useMemo, useState } from "react";
import MeshView from "./mesh-view";
import Measurements from "./measurements";

const api = process.env.NEXT_PUBLIC_NGP_API || "http://127.0.0.1:8000";
const stages = ["queued", "poses", "training", "exporting", "complete"];
const labels = {
  queued: "Queued", poses: "Solving cameras", training: "Training NeRF",
  exporting: "Extracting mesh", complete: "Ready", failed: "Failed",
};
const policyNames = { fifo: "First in, first out", sjf: "Shortest job first" };
const format = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
const artifact = (job, name) => `${api}/api/jobs/${job.id}/artifacts/${name}`;

async function request(path, options) {
  const response = await fetch(api + path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}

function CaptureForm({ onUpload }) {
  const [files, setFiles] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (files.length < 3 || files.length > 120) {
      setError("Choose between 3 and 120 overlapping photos.");
      return;
    }
    const form = event.currentTarget;
    const payload = new FormData(form);
    for (const file of files) payload.append("photos", file);
    setSubmitting(true);
    setError("");
    try {
      const job = await request("/api/jobs", { method: "POST", body: payload });
      onUpload(job);
      form.reset();
      setFiles([]);
    } catch (error) {
      setError(error.message);
    } finally {
      setSubmitting(false);
    }
  }

  return <form onSubmit={submit} className="capture-form">
    <h2>New capture</h2>
    <p>Choose overlapping photos taken around the same object.</p>
    <label className="file-zone">
      <strong>{files.length ? `${files.length} photos selected` : "Choose photos"}</strong>
      <small>JPEG, PNG or WebP</small>
      <input aria-label="Choose photos" type="file" multiple accept="image/jpeg,image/png,image/webp"
        onChange={(event) => setFiles(Array.from(event.target.files))} />
    </label>
    <label>Capture name<input name="label" placeholder="Untitled capture" maxLength={80} /></label>
    <div className="form-row">
      <label>Mesh grid<select name="resolution" defaultValue="256">
        <option value="128">128³</option><option value="256">256³</option><option value="512">512³</option>
      </select></label>
      <label>Training steps<input name="steps" type="number" min="100" max="35000" defaultValue="3500" required /></label>
    </div>
    <button type="submit" className="primary" disabled={submitting}>{submitting ? "Uploading…" : "Reconstruct"}</button>
    {error && <p role="alert" className="error">{error}</p>}
  </form>;
}

function Viewer({ job }) {
  const [mode, setMode] = useState("mesh");
  const [frame, setFrame] = useState(0);
  const ready = job?.status === "complete";
  const name = job?.frames?.[frame];
  const activeStage = stages.indexOf(job?.status);
  return <section className="viewer-panel">
    <div className="panel-heading">
      <div><h2>{job?.label || "Reconstruction viewer"}</h2>{job && <p>{labels[job.status]}</p>}</div>
      {ready && <a className="button" href={artifact(job, "mesh.glb")}>Download GLB</a>}
    </div>
    <div className="viewport">
      {ready ? mode === "render" ? (
        <img className="nerf-render" src={artifact(job, name)} alt={`NeRF render ${frame + 1}`} />
      ) : <MeshView url={artifact(job, "mesh.glb")} wireframe={mode === "wireframe"} /> : (
        <div className="viewport-empty">
          <div className="coordinate-mark" aria-hidden="true"><span /><span /><span /></div>
          <strong>{job ? labels[job.status] : "Your mesh will appear here"}</strong>
          <span>{job ? `${job.images} photos · ${job.resolution}³ mesh grid` : "Upload photos to start a reconstruction."}</span>
          {job?.error && <p>{job.error}</p>}
        </div>
      )}
    </div>
    {ready ? <div className="viewer-controls">
      <div className="tabs" aria-label="Preview mode">
        {[["mesh", "Shaded"], ["wireframe", "Wireframe"], ["render", "NeRF views"]].map(([id, label]) =>
          <button key={id} aria-pressed={mode === id} onClick={() => setMode(id)}>{label}</button>)}
      </div>
      {mode === "render" ? <label className="frames">View {frame + 1} of {job.frames.length}
        <input aria-label="NeRF viewpoint" type="range" min="0" max={job.frames.length - 1} value={frame}
          onChange={(event) => setFrame(Number(event.target.value))} /></label>
        : <span>{format(job.triangles)} triangles</span>}
    </div> : job?.status === "failed" ? <div className="viewer-controls"><a href={artifact(job, "worker.log")}>Open worker log</a></div>
      : job && <ol className="progress-stages" aria-label="Reconstruction stages">
        {stages.map((stage, index) => <li key={stage} className={index <= activeStage ? "reached" : ""}
          aria-current={index === activeStage ? "step" : undefined}>{labels[stage]}</li>)}
      </ol>}
  </section>;
}

function Queue({ state, chosen, setChosen, refresh }) {
  const [error, setError] = useState("");
  async function configure(policy, paused) {
    try {
      await request("/api/queue", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy, paused }) });
      setError("");
      refresh();
    } catch (error) {
      setError(error.message);
    }
  }
  return <section className="queue-panel">
    <div className="panel-heading"><h2>Capture queue</h2>
      <div className="queue-actions">
        <select aria-label="Scheduling policy" value={state.policy} onChange={(event) => configure(event.target.value, state.paused)}>
          {Object.entries(policyNames).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
        </select>
        <button onClick={() => configure(state.policy, !state.paused)}>{state.paused ? "Resume queue" : "Pause queue"}</button>
      </div>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {!state.jobs.length ? <p className="panel-note">Submitted captures appear here.</p> :
      <ul>{state.jobs.map((job) => <li key={job.id}>
        <button onClick={() => setChosen(job.id)} aria-pressed={chosen === job.id}>
          <span>{job.label}<small>{job.images} photos · {job.resolution}³ grid</small></span>
          <span className={`status ${job.status}`}>{labels[job.status]}</span>
        </button>
      </li>)}</ul>}
  </section>;
}

function Gpu({ telemetry }) {
  const sample = telemetry.samples.at(-1);
  return <section className="gpu-panel">
    <div className="panel-heading"><h2>GPU telemetry</h2><span>{sample && "Live · NVML"}</span></div>
    {sample ? <>
      <dl><div><dt>Utilization</dt><dd>{sample.utilization}%</dd></div>
        <div><dt>VRAM</dt><dd>{(sample.memory_mib / 1024).toFixed(2)} / {(sample.total_mib / 1024).toFixed(2)} GiB</dd></div>
        <div><dt>Power</dt><dd>{sample.power_w == null ? "Unavailable" : `${format(sample.power_w)} W`}</dd></div></dl>
      <div className="spark" aria-label="Recent GPU utilization history">
        {telemetry.samples.slice(-60).map((point) => <i key={point.time} title={`${point.utilization}%`}
          style={{ "--level": `${Math.max(2, point.utilization)}%` }} />)}
      </div>
    </> : <p className="panel-note">{telemetry.error || "Waiting for GPU data."}</p>}
  </section>;
}

export default function Studio() {
  const [state, setState] = useState({ jobs: [], policy: "fifo", paused: false });
  const [chosen, setChosen] = useState(null);
  const [telemetry, setTelemetry] = useState({ samples: [] });
  const [measurements, setMeasurements] = useState({ rows: [], status: "loading" });
  const [connected, setConnected] = useState(null);
  const [page, setPage] = useState("studio");

  async function refresh() {
    try {
      const [jobs, gpu] = await Promise.all([request("/api/jobs"), request("/api/gpu")]);
      setState(jobs);
      setTelemetry(gpu);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 1800);
    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    if (page !== "measurements") return;
    const controller = new AbortController();
    setMeasurements({ rows: [], status: "loading" });
    request("/api/benchmarks", { signal: controller.signal }).then((rows) => {
      setMeasurements({ rows, status: "ready" });
    }).catch(() => {
      if (!controller.signal.aborted) setMeasurements({ rows: [], status: "error" });
    });
    return () => controller.abort();
  }, [page]);
  const selected = useMemo(() => state.jobs.find((job) => job.id === chosen) || state.jobs[0], [state.jobs, chosen]);

  return <main>
    <header>
      <div className="brand"><span className="brand-mark" aria-hidden="true" />
        <div><h1>NeRF2Mesh Studio</h1><p>Photo reconstruction lab</p></div></div>
      <nav aria-label="Workspace">
        <button aria-pressed={page === "studio"} onClick={() => setPage("studio")}>Studio</button>
        <button aria-pressed={page === "measurements"} onClick={() => setPage("measurements")}>Measurements</button>
      </nav>
      <span className={`connection ${connected ? "online" : ""}`}>{connected == null ? "Connecting…" : connected ? "Worker connected" : "Worker offline"}</span>
    </header>
    {page === "studio" ? <div className="workspace">
      <CaptureForm onUpload={(job) => { setChosen(job.id); refresh(); }} />
      <Viewer job={selected} key={selected?.id} />
      <Queue state={state} chosen={selected?.id} setChosen={setChosen} refresh={refresh} />
      <Gpu telemetry={telemetry} />
    </div> : <Measurements {...measurements} />}
  </main>;
}
