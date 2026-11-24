"use client";

import { useState } from "react";

const sections = [
  { id: "ray", label: "Ray skipping", experiments: ["ray_render", "ray_quality", "ray_conditions"] },
  { id: "mesh", label: "Mesh extraction", experiments: ["marching_cubes", "mesh_quality"] },
  { id: "distance", label: "Distance field", experiments: ["distance_transform"] },
  { id: "queue", label: "Job scheduling", experiments: ["scheduling"] },
  { id: "capture", label: "Capture test", experiments: ["capture_pipeline"] },
];
const variantNames = {
  no_skip: "No skipping",
  exact_skip: "Distance-guided skipping", dense: "Dense", sparse: "Sparse", exact: "Exact distance transform",
  jfa: "Jump flooding", fifo: "First in, first out", sjf: "Shortest job first", studio: "Studio",
};
const powerNames = {
  battery: "On battery", external_power: "Plugged in", changed_power: "Power source changed during the run",
  unknown_power: "Power source unavailable",
};
const metricNames = {
  gpu_time: "GPU time", nvml_peak_gpu_memory_used: "Peak GPU memory",
  empty_advances_per_pixel_ray: "Empty-space advances per ray",
  no_skip_max_rgba_error: "Max pixel error vs no skipping", oom: "Out of GPU memory",
  turnaround: "Time to result", dispatch_order: "Dispatch position", chamfer: "Sampled surface distance",
};

const number = (value, digits = 2) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const words = (value) => value.replaceAll("_", " ");
const sceneName = (value) => value[0]?.toUpperCase() + value.slice(1).replaceAll("_", " ");
const frame = (value) => value.replace(/(\d)x(\d)/, "$1 × $2");
const rayVariants = ["no_skip", "exact_skip"];

function runScene(run) {
  return sceneName(run.scene) + (/_fused(?:_ablation|_no_skip)?\.csv$/.test(run.source) ? " (fused)" : "");
}

function runPower(run) {
  return run.rows.find((row) => row.experiment === "ray_conditions" && row.metric === "power_source")?.variant;
}

function runLabel(run, section) {
  const label = `${runScene(run)}, ${frame(run.resolution)}`;
  if (section !== "ray") return label;
  const power = runPower(run);
  const qualifier = run.source.endsWith("_ablation.csv") ? "Earlier run"
    : power === "battery" ? "Battery" : power === "external_power" ? "Plugged in" : "No-skip test";
  return `${label} · ${qualifier}`;
}

function makeRuns(rows, section) {
  const groups = new Map();
  for (const row of rows) {
    if (!section.experiments.includes(row.experiment)) continue;
    const scene = section.id === "queue" ? "Capture batch" : row.scene;
    const key = JSON.stringify([row.source, scene, row.resolution]);
    if (!groups.has(key)) {
      groups.set(key, { key, source: row.source, recorded: row.recorded_at, scene, resolution: row.resolution, rows: [] });
    }
    groups.get(key).rows.push(row);
  }
  return [...groups.values()]
    .filter((run) => section.id !== "ray" || rayVariants.every((variant) => find(run, "ray_render", variant, "gpu_time")))
    .sort((a, b) => b.recorded - a.recorded || a.scene.localeCompare(b.scene));
}

function preferredRun(section, runs) {
  if (section === "mesh") return runs.find((run) => run.resolution === "512³") || runs[0];
  if (section === "distance") return runs.find((run) => run.scene === "single_seed_128") || runs[0];
  return runs[0];
}

function find(run, experiment, variant, metric) {
  return run.rows.find((row) => row.experiment === experiment && row.variant === variant && row.metric === metric);
}

function relativeTime(reference, measured) {
  if (!(reference > 0 && measured > 0)) return "Timing unavailable";
  const ratio = reference / measured;
  if (Math.abs(ratio - 1) < 0.005) return "About the same";
  return ratio > 1 ? `${number(ratio)}× as fast` : `${number((1 / ratio - 1) * 100, 1)}% slower`;
}

function Bars({ title, unit, items }) {
  const maximum = Math.max(1e-9, ...items.map((item) => item.value ?? 0));
  return <section className="measure-bars">
    <h3>{title}</h3>
    <ul>
      {items.map(({ name, value, status, accent }) => <li key={name}>
        <div className="measure-bar-label"><span>{name}</span>
          <strong>{value == null ? status ?? "Timing unavailable" : `${number(value, 3)} ${unit}`}</strong></div>
        {value != null && <div className="measure-track" aria-hidden="true">
          <div className={accent ? "accent" : ""} style={{ width: `${value / maximum * 100}%` }} />
        </div>}
      </li>)}
    </ul>
  </section>;
}

function Findings({ items }) {
  if (!items.length) return null;
  return <dl className="measure-findings">
    {items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>;
}

function RayResults({ run }) {
  const time = (variant) => find(run, "ray_render", variant, "gpu_time")?.median;
  const empty = (variant) => find(run, "ray_quality", variant, "empty_advances_per_pixel_ray")?.median;
  const none = time("no_skip");
  const exact = time("exact_skip");
  const maxError = find(run, "ray_quality", "exact_skip", "no_skip_max_rgba_error")?.maximum;
  const noSkipPsnr = find(run, "ray_quality", "exact_skip", "no_skip_psnr")?.median;
  const power = runPower(run);
  const items = [["Distance-guided vs no skipping", relativeTime(none, exact)]];
  if (empty("no_skip") > 0 && empty("exact_skip") != null) {
    const change = 100 * (1 - empty("exact_skip") / empty("no_skip"));
    items.push(["Empty-space advances vs no skipping", `${number(Math.abs(change), 1)}% ${change < 0 ? "more" : "fewer"}`]);
  }
  if (maxError === 0) items.push(["Rendered images vs no skipping", "Pixel-for-pixel match"]);
  else if (noSkipPsnr != null) items.push(["Image similarity vs no skipping", `${number(noSkipPsnr, 1)} dB PSNR`]);

  return <>
    <p className="measure-context">{runScene(run)}, {frame(run.resolution)} pixels
      {power && ` · ${powerNames[power]}`}</p>
    <Bars title="Median time per frame" unit="ms" items={[
      { name: variantNames.no_skip, value: none },
      { name: variantNames.exact_skip, value: exact, accent: true },
    ]} />
    <Findings items={items} />
    <p className="measure-note">Both modes use the same scene, camera and occupancy grid.
      The no-skipping mode checks every sample position. Distance-guided skipping jumps across empty space.</p>
  </>;
}

function MeshResults({ run }) {
  const time = (variant) => find(run, "marching_cubes", variant, "gpu_time")?.median;
  const peak = (variant) => find(run, "marching_cubes", variant, "nvml_peak_gpu_memory_used")?.median;
  const dense = time("dense");
  const sparse = time("sparse");
  const chamfer = find(run, "mesh_quality", "sparse", "chamfer")?.median;
  const items = [];
  if (dense != null && sparse != null) items.push(["Sparse extraction speed", relativeTime(dense, sparse)]);
  if (peak("dense") > 0 && peak("sparse") != null) {
    const reduction = 100 * (1 - peak("sparse") / peak("dense"));
    items.push(["Peak GPU memory", `${number(Math.abs(reduction), 1)}% ${reduction < 0 ? "higher" : "lower"}`]);
  }
  if (chamfer === 0) items.push(["Sampled mesh surfaces", "Matched"]);

  return <>
    <p className="measure-context">{sceneName(run.scene)}, {run.resolution} grid</p>
    <Bars title="Median extraction time" unit="ms" items={["dense", "sparse"].map((variant) => ({
      name: variantNames[variant], value: time(variant), accent: variant === "sparse",
      status: find(run, "marching_cubes", variant, "oom") ? "Out of GPU memory" : undefined,
    }))} />
    <Findings items={items} />
    <p className="measure-note">Both methods use the same trained scene and density threshold. The sparse method
      avoids evaluating empty blocks.</p>
  </>;
}

function DistanceResults({ run }) {
  const time = (variant) => find(run, "distance_transform", variant, "gpu_time")?.median;
  const error = find(run, "distance_transform", "jfa", "max_overestimate")?.median;
  return <>
    <p className="measure-context">{run.scene === "single_seed_128" ? "One occupied voxel" : sceneName(run.scene)}, {run.resolution}³ grid</p>
    <Bars title="Median time to build the distance field" unit="ms"
      items={["exact", "jfa"].map((variant) => ({ name: variantNames[variant], value: time(variant), accent: variant === "exact" }))} />
    <Findings items={[
      ["Exact vs jump flooding", relativeTime(time("jfa"), time("exact"))],
      ...(error == null ? [] : [["Largest jump-flood overestimate", `${number(error, 4)} voxels`]]),
    ]} />
    <p className="measure-note">The renderer uses the exact distance field. An overestimate can send a ray past occupied space.</p>
  </>;
}

function QueueResults({ run }) {
  const average = (variant) => {
    const jobs = run.rows.filter((row) => row.variant === variant && row.metric === "turnaround");
    return jobs.length ? jobs.reduce((total, row) => total + row.mean, 0) / jobs.length : undefined;
  };
  const order = (variant) => run.rows.filter((row) => row.variant === variant && row.metric === "dispatch_order")
    .sort((a, b) => a.median - b.median).map((row) => row.scene).join(", ");
  const fifo = average("fifo");
  const sjf = average("sjf");
  const items = [["Shortest-job-first vs FIFO", relativeTime(fifo, sjf)]];
  for (const variant of ["fifo", "sjf"]) {
    const utilization = find(run, "scheduling", variant, "gpu_utilization")?.mean;
    if (utilization != null) items.push([`${variantNames[variant]} GPU utilization`, `${number(utilization, 1)}%`]);
  }

  return <>
    <Bars title="Average time from queue release to result" unit="s" items={[
      { name: variantNames.fifo, value: fifo }, { name: variantNames.sjf, value: sjf, accent: true },
    ]} />
    <Findings items={items} />
    <p className="measure-note">FIFO order: {order("fifo") || "Unavailable"}<br />
      Shortest-job-first order: {order("sjf") || "Unavailable"}</p>
  </>;
}

function CaptureResults({ run }) {
  return <Findings items={run.rows.map((row) => [sceneName(row.metric), `${number(row.median)} ${row.unit}`])} />;
}

const panels = { ray: RayResults, mesh: MeshResults, distance: DistanceResults, queue: QueueResults, capture: CaptureResults };

function SourceDetails({ run, section }) {
  const rows = section === "ray" ? run.rows.filter((row) =>
    rayVariants.includes(row.variant) && !/^(legacy|stable)_/.test(row.metric)) : run.rows;
  return <details className="measure-source">
    <summary>Measurement details <span>{run.source}</span></summary>
    <div className="table-wrap"><table>
      <thead><tr><th>Variant</th><th>Metric</th><th>Median</th><th>Observed range</th><th>Samples</th></tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>
        <td>{variantNames[row.variant] ?? words(row.variant)}{row.experiment === "scheduling" && row.scene !== "batch" && ` · ${row.scene}`}</td>
        <td>{metricNames[row.metric] ?? words(row.metric)}</td>
        <td>{number(row.median, 4)} {row.unit}</td>
        <td>{row.runs > 1 ? `${number(row.minimum, 4)}–${number(row.maximum, 4)}` : "—"}</td>
        <td>{row.runs}</td>
      </tr>)}</tbody>
    </table></div>
  </details>;
}

export default function Measurements({ rows, status }) {
  const [active, setActive] = useState("ray");
  const [chosen, setChosen] = useState({});
  const section = sections.find((item) => item.id === active);
  const runs = makeRuns(rows, section);
  const run = runs.find((item) => item.key === chosen[active]) ?? preferredRun(active, runs);
  const Panel = panels[active];

  return <section className="measurements">
    <div className="panel-heading"><h2>Measured comparisons</h2></div>
    <div className="measure-tabs tabs" aria-label="Measurement topic">
      {sections.map((item) => <button key={item.id} aria-pressed={active === item.id}
        onClick={() => setActive(item.id)}>{item.label}</button>)}
    </div>
    {status === "loading" ? <p className="panel-note" role="status">Loading measurements…</p>
      : status === "error" ? <p className="panel-note" role="alert">Measurements could not be loaded. Reopen this tab when the worker is connected.</p>
      : !run ? <p className="panel-note">No measurements available for this comparison.</p>
      : <>
        {runs.length > 1 && <label className="measure-select">Benchmark run
          <select value={run.key} onChange={(event) => setChosen({ ...chosen, [active]: event.target.value })}>
            {runs.map((item) => <option key={item.key} value={item.key}>
              {runLabel(item, active)}
            </option>)}
          </select>
        </label>}
        <Panel run={run} />
        <SourceDetails run={run} section={active} />
      </>}
  </section>;
}
