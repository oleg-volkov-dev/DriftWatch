#!/usr/bin/env python3
"""DriftWatch dashboard server — stdlib only, no extra deps."""

import json
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PORT = 8765
ROOT = Path(__file__).resolve().parent.parent.parent  # infra/dashboard/ -> infra/ -> project root
API  = "http://localhost:8000"

ALLOWED_COMMANDS = {
    "demo-drift-feature", "demo-black-friday",
    "gen-base", "gen-feature", "gen-blackfriday",
    "train", "promote-prod", "monitor", "control",
}

DEMO_COMMANDS = {"demo-drift-feature", "demo-black-friday"}

SERVICE_MATCHES = {
    "mlflow":     "mlflow",
    "api":        "-api-",
    "prometheus": "prometheus",
    "grafana":    "grafana",
}

# ──────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DriftWatch</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 40px 24px 80px;
    }

    /* ── Two-column layout ── */
    #layout { display: flex; gap: 20px; max-width: 1380px; margin: 0 auto; align-items: flex-start; }
    #main   { flex: 1; min-width: 0; max-width: 960px; }
    .spacer { height: 32px; }

    /* ── Header ── */
    header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin-bottom: 36px; }
    h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; color: #f8fafc; }
    h1 span { color: #6366f1; }
    header p { margin-top: 6px; color: #64748b; font-size: 0.88rem; line-height: 1.55; max-width: 460px; }

    /* ── Status badges ── */
    .status-row { display: flex; gap: 8px; flex-wrap: wrap; padding-top: 4px; }
    .badge { display: flex; align-items: center; gap: 6px; font-size: 0.73rem; font-weight: 500; color: #94a3b8; background: #1e2330; border: 1px solid #2d3348; border-radius: 20px; padding: 4px 10px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #334155; transition: background 0.4s; flex-shrink: 0; }
    .dot.up   { background: #22c55e; box-shadow: 0 0 6px #22c55e77; }
    .dot.down { background: #475569; }

    /* ── Section label ── */
    .section-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 12px; }

    /* ── Card base ── */
    .card { background: #1e2330; border: 1px solid #2d3348; border-radius: 10px; padding: 20px; }

    /* ── Services grid ── */
    .services-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
    .svc-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .icon { width: 30px; height: 30px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 15px; flex-shrink: 0; }
    .svc-title { font-size: 0.9rem; font-weight: 600; color: #f1f5f9; }
    .svc-desc  { font-size: 0.76rem; color: #64748b; line-height: 1.5; margin-bottom: 10px; }
    .links { display: flex; flex-wrap: wrap; gap: 5px; }
    .lnk { font-size: 0.71rem; font-weight: 500; padding: 3px 9px; border-radius: 5px; border: 1px solid; cursor: pointer; transition: background 0.12s, color 0.12s; text-decoration: none; }
    .lnk-p { color: #818cf8; border-color: #3730a3; background: #1e1b4b; }
    .lnk-p:hover { background: #312e81; color: #c7d2fe; }
    .lnk-s { color: #94a3b8; border-color: #2d3348; background: transparent; }
    .lnk-s:hover { background: #2d3348; color: #cbd5e1; }
    .bg-green  { background: #052e16; }
    .bg-blue   { background: #0c1a2e; }
    .bg-orange { background: #1c0a00; }
    .bg-indigo { background: #1e1b4b; }

    /* ── Predict section ── */
    .predict-card { background: #1e2330; border: 1px solid #2d3348; border-radius: 10px; padding: 24px; }
    .fields-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; margin-bottom: 16px; }
    .field label { display: block; font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .field input[type="number"] {
      width: 100%; background: #0f1117; border: 1px solid #2d3348; border-radius: 6px;
      padding: 8px 10px; color: #e2e8f0; font-size: 0.9rem;
      outline: none; transition: border-color 0.15s;
    }
    .field input[type="number"]:focus { border-color: #6366f1; }
    .predict-footer { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
    .check-label { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #94a3b8; cursor: pointer; }
    .check-label input { width: 16px; height: 16px; accent-color: #6366f1; cursor: pointer; }
    .btn-row { display: flex; gap: 8px; }
    .btn {
      font-size: 0.82rem; font-weight: 600; padding: 9px 18px; border-radius: 7px;
      border: 1px solid; cursor: pointer; transition: background 0.12s, color 0.12s;
    }
    .btn-predict { color: #818cf8; border-color: #3730a3; background: #1e1b4b; }
    .btn-predict:hover:not(:disabled) { background: #312e81; color: #c7d2fe; }
    .btn-reload  { color: #94a3b8; border-color: #2d3348; background: transparent; }
    .btn-reload:hover:not(:disabled)  { background: #2d3348; color: #cbd5e1; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ── Predict result ── */
    #predict-result { margin-top: 20px; padding-top: 20px; border-top: 1px solid #2d3348; display: none; }
    .result-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
    .result-badge {
      font-size: 1rem; font-weight: 800; letter-spacing: 0.05em;
      padding: 8px 20px; border-radius: 8px; border: 2px solid;
    }
    .result-badge.fraud { color: #f87171; border-color: #7f1d1d; background: #1c0505; }
    .result-badge.safe  { color: #4ade80; border-color: #14532d; background: #021308; }
    .result-prob { flex: 1; min-width: 160px; }
    .result-prob-label { font-size: 0.72rem; color: #64748b; margin-bottom: 5px; }
    .result-prob-val { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }
    .bar-wrap { height: 6px; background: #2d3348; border-radius: 3px; margin-top: 8px; overflow: hidden; }
    .bar { height: 100%; border-radius: 3px; transition: width 0.4s, background 0.4s; }
    .result-stage { font-size: 0.75rem; color: #475569; margin-top: 6px; }
    #predict-error { margin-top: 20px; padding-top: 20px; border-top: 1px solid #2d3348; font-size: 0.82rem; color: #f87171; display: none; }

    /* ── Pipeline steps ── */
    .pipeline { display: flex; flex-direction: column; gap: 0; }
    .pipe-step {
      display: flex; gap: 14px; align-items: flex-start;
      background: #1e2330; border: 1px solid #2d3348; border-radius: 10px;
      padding: 16px 20px;
    }
    .pipe-num {
      width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
      background: #1e1b4b; border: 1px solid #3730a3;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.72rem; font-weight: 700; color: #818cf8; margin-top: 1px;
    }
    .pipe-body { flex: 1; }
    .pipe-title { font-size: 0.88rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
    .pipe-desc  { font-size: 0.75rem; color: #64748b; line-height: 1.5; margin-bottom: 10px; }
    .pipe-btns  { display: flex; flex-wrap: wrap; gap: 6px; }
    .pipe-arrow { text-align: center; color: #2d3348; font-size: 1rem; line-height: 1; padding: 4px 0; }

    /* ── Demo cards ── */
    .demo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .demo-title { font-size: 0.9rem; font-weight: 600; color: #f1f5f9; margin-bottom: 6px; }
    .demo-desc  { font-size: 0.76rem; color: #64748b; line-height: 1.5; flex: 1; margin-bottom: 14px; }
    .demo-card  { display: flex; flex-direction: column; }
    .btn-run { color: #818cf8; border-color: #3730a3; background: #1e1b4b; width: fit-content; }
    .btn-run:hover:not(:disabled) { background: #312e81; color: #c7d2fe; }
    .btn-run.running { border-color: #ca8a04; background: #1c1400; color: #fbbf24; }
    .btn-run.ok { border-color: #166534; background: #042012; color: #4ade80; }

    /* ── Terminal panel ── */
    #terminal-panel {
      display: none; flex-direction: column;
      width: 400px; flex-shrink: 0;
      position: sticky; top: 24px;
      max-height: calc(100vh - 48px);
    }
    .term-header { display: flex; align-items: center; justify-content: space-between; background: #161b27; border: 1px solid #2d3348; border-bottom: none; border-radius: 8px 8px 0 0; padding: 8px 14px; }
    .term-title  { font-size: 0.73rem; font-weight: 600; color: #64748b; font-family: "SF Mono", monospace; }
    .term-close  { font-size: 0.73rem; color: #475569; background: none; border: none; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
    .term-close:hover { background: #2d3348; color: #94a3b8; }
    #terminal-output {
      background: #0a0d14; border: 1px solid #2d3348; border-radius: 0 0 8px 8px;
      padding: 14px 16px; font-family: "SF Mono", "Fira Code", monospace; font-size: 0.74rem;
      line-height: 1.6; color: #94a3b8; flex: 1; overflow-y: auto;
      white-space: pre-wrap; word-break: break-all;
    }
    #terminal-output .prompt  { color: #6366f1; font-weight: 700; }
    #terminal-output .success { color: #22c55e; font-weight: 600; }
    #terminal-output .err     { color: #f87171; font-weight: 600; }
  </style>
</head>
<body>
<div id="layout">
<div id="main">

  <!-- ── Header ── -->
  <header>
    <div>
      <h1>Drift<span>Watch</span></h1>
      <p>Autonomous MLOps platform for fraud detection with policy-driven drift response.</p>
    </div>
    <div class="status-row">
      <div class="badge" data-service="mlflow">     <span class="dot"></span>MLflow</div>
      <div class="badge" data-service="api">        <span class="dot"></span>API</div>
      <div class="badge" data-service="grafana">    <span class="dot"></span>Grafana</div>
      <div class="badge" data-service="prometheus"> <span class="dot"></span>Prometheus</div>
    </div>
  </header>

  <!-- ── Services ── -->
  <p class="section-label">Services</p>
  <div class="services-grid">

    <div class="card">
      <div class="svc-header"><div class="icon bg-green">🔍</div><div class="svc-title">Fraud API</div></div>
      <p class="svc-desc">FastAPI inference service. Use the predict form below to test it — use Reload model after running a demo.</p>
    </div>

    <div class="card">
      <div class="svc-header"><div class="icon bg-blue">🧪</div><div class="svc-title">MLflow</div></div>
      <p class="svc-desc">Experiment tracking and model registry. Check which model is in Production.</p>
      <div class="links">
        <a class="lnk lnk-p" href="http://localhost:5000"          target="_blank">Open</a>
        <a class="lnk lnk-s" href="http://localhost:5000/#/models" target="_blank">Models</a>
      </div>
    </div>

    <div class="card">
      <div class="svc-header"><div class="icon bg-orange">📊</div><div class="svc-title">Grafana</div></div>
      <p class="svc-desc">API latency, request rates, error counts. Populates after predict calls.</p>
      <div class="links">
        <a class="lnk lnk-p" href="http://localhost:3000"            target="_blank">Open</a>
        <a class="lnk lnk-s" href="http://localhost:3000/dashboards" target="_blank">Dashboards</a>
      </div>
    </div>

    <div class="card">
      <div class="svc-header"><div class="icon bg-indigo">🔥</div><div class="svc-title">Prometheus</div></div>
      <p class="svc-desc">Raw metrics scraping from the API. Query counters and histograms.</p>
      <div class="links">
        <a class="lnk lnk-p" href="http://localhost:9090"         target="_blank">Open</a>
        <a class="lnk lnk-s" href="http://localhost:9090/targets" target="_blank">Targets</a>
      </div>
    </div>

  </div>

  <!-- ── Predict ── -->
  <div class="spacer"></div>
  <p class="section-label">Try the fraud detection API</p>
  <div class="predict-card">
    <div class="fields-grid">
      <div class="field">
        <label>Amount ($)</label>
        <input type="number" id="f-amount" value="150" min="0" step="0.01" />
      </div>
      <div class="field">
        <label>Hour (0–23)</label>
        <input type="number" id="f-hour" value="2" min="0" max="23" />
      </div>
      <div class="field">
        <label>Customer Age</label>
        <input type="number" id="f-age" value="35" min="18" max="120" />
      </div>
      <div class="field">
        <label>Account Tenure (days)</label>
        <input type="number" id="f-tenure" value="180" min="0" />
      </div>
      <div class="field">
        <label>Merchant Risk (0–1)</label>
        <input type="number" id="f-risk" value="0.6" min="0" max="1" step="0.01" />
      </div>
      <div class="field">
        <label>Distance (km)</label>
        <input type="number" id="f-dist" value="500" min="0" />
      </div>
    </div>
    <div class="predict-footer">
      <label class="check-label">
        <input type="checkbox" id="f-intl" /> International transaction
      </label>
      <div class="btn-row">
        <button class="btn btn-predict" onclick="predict()">Predict</button>
        <button class="btn btn-reload"  onclick="reloadModel()">Reload model</button>
      </div>
    </div>
    <div id="predict-result">
      <div class="result-row">
        <div id="result-badge" class="result-badge">—</div>
        <div class="result-prob">
          <div class="result-prob-label">Fraud probability</div>
          <div id="result-pct" class="result-prob-val">—</div>
          <div class="bar-wrap"><div id="result-bar" class="bar" style="width:0%"></div></div>
          <div id="result-stage" class="result-stage"></div>
        </div>
      </div>
    </div>
    <div id="predict-error"></div>
  </div>

  <!-- ── Full demos ── -->
  <div class="spacer"></div>
  <p class="section-label">Full demos — one command runs the entire pipeline end-to-end</p>
  <div class="demo-grid">

    <div class="card demo-card">
      <div class="demo-title">Feature Drift</div>
      <p class="demo-desc">Gradual shift in transaction distributions — amounts, distances, and risk scores scale over time. Runs: gen-base → train → promote → gen-feature → monitor → control → reload.</p>
      <button class="btn btn-run" onclick="runCmd('demo-drift-feature', this)">▶ Run full demo</button>
    </div>

    <div class="card demo-card">
      <div class="demo-title">Black Friday Shock</div>
      <p class="demo-desc">Sudden spike event — late-night transaction surge with an elevated fraud multiplier. Simulates a one-off shock rather than gradual drift.</p>
      <button class="btn btn-run" onclick="runCmd('demo-black-friday', this)">▶ Run full demo</button>
    </div>

  </div>

  <!-- ── Pipeline steps ── -->
  <div class="spacer"></div>
  <p class="section-label">Pipeline — run steps individually</p>
  <div class="pipeline">

    <div class="pipe-step">
      <div class="pipe-num">1</div>
      <div class="pipe-body">
        <div class="pipe-title">Generate Reference Data</div>
        <div class="pipe-desc">Creates the baseline dataset that Evidently uses as the reference for drift detection.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('gen-base', this)">▶ gen-base</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">2</div>
      <div class="pipe-body">
        <div class="pipe-title">Generate Current Data</div>
        <div class="pipe-desc">Generates the "current" dataset compared against the reference. Pick a scenario.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('gen-feature', this)" title="Gradual shift in transaction amounts, distances and risk scores">▶ Feature drift</button>
          <button class="btn btn-run" onclick="runCmd('gen-blackfriday', this)" title="Sudden late-night spike with elevated fraud multiplier">▶ Black Friday</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">3</div>
      <div class="pipe-body">
        <div class="pipe-title">Train Model</div>
        <div class="pipe-desc">Trains a new model on the reference data and registers it to MLflow.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('train', this)">▶ train</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">4</div>
      <div class="pipe-body">
        <div class="pipe-title">Promote to Production</div>
        <div class="pipe-desc">Promotes the latest registered model version to the Production stage in MLflow.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('promote-prod', this)">▶ promote-prod</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">5</div>
      <div class="pipe-body">
        <div class="pipe-title">Run Drift Monitoring</div>
        <div class="pipe-desc">Runs Evidently against reference vs. current data and writes drift reports to shared/reports/.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('monitor', this)">▶ monitor</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">6</div>
      <div class="pipe-body">
        <div class="pipe-title">Run Control Plane</div>
        <div class="pipe-desc">Sentinel reads the drift reports → Planner decides action → Release retrains and promotes automatically.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" onclick="runCmd('control', this)">▶ control</button>
        </div>
      </div>
    </div>

    <div class="pipe-arrow">↓</div>

    <div class="pipe-step">
      <div class="pipe-num">7</div>
      <div class="pipe-body">
        <div class="pipe-title">Reload API Model</div>
        <div class="pipe-desc">Tells the running API to pick up the current Production model from MLflow without restarting.</div>
        <div class="pipe-btns">
          <button class="btn btn-run" id="pipe-reload-btn" onclick="pipeReload(this)">▶ reload-api</button>
        </div>
      </div>
    </div>

  </div>

</div><!-- /main -->

<!-- ── Terminal panel (right side) ── -->
<div id="terminal-panel">
  <div class="term-header">
    <span class="term-title" id="term-title">output</span>
    <button class="term-close" onclick="closeTerminal()">✕ close</button>
  </div>
  <div id="terminal-output"></div>
</div>

</div><!-- /layout -->

<script>
  // ── Status polling ────────────────────────────────────────────────────────
  function updateStatus() {
    fetch('/api/status').then(r => r.json()).then(status => {
      for (const [svc, up] of Object.entries(status)) {
        const b = document.querySelector(`[data-service="${svc}"] .dot`);
        if (b) b.className = 'dot ' + (up ? 'up' : 'down');
      }
    }).catch(() => {});
  }
  updateStatus();
  setInterval(updateStatus, 5000);

  // ── Predict ───────────────────────────────────────────────────────────────
  async function predict() {
    const btn = document.querySelector('.btn-predict');
    btn.disabled = true;
    btn.textContent = 'Predicting…';

    const body = {
      transaction_amount:   +document.getElementById('f-amount').value,
      transaction_hour:     +document.getElementById('f-hour').value,
      customer_age:         +document.getElementById('f-age').value,
      account_tenure_days:  +document.getElementById('f-tenure').value,
      merchant_risk_score:  +document.getElementById('f-risk').value,
      geo_distance_km:      +document.getElementById('f-dist').value,
      is_international:      document.getElementById('f-intl').checked,
    };

    document.getElementById('predict-error').style.display = 'none';
    document.getElementById('predict-result').style.display = 'none';

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await resp.json();

      if (!resp.ok) {
        showPredictError(data.detail || data.error || `Error ${resp.status}`);
        return;
      }

      const pct  = Math.round(data.fraud_probability * 100);
      const fraud = data.is_fraud;

      // Color: green → yellow → red
      const hue = Math.round((1 - data.fraud_probability) * 120); // 120=green, 0=red
      const barColor = `hsl(${hue}, 85%, 50%)`;

      document.getElementById('result-badge').textContent = fraud ? 'FRAUD' : 'SAFE';
      document.getElementById('result-badge').className   = 'result-badge ' + (fraud ? 'fraud' : 'safe');
      document.getElementById('result-pct').textContent   = pct + '%';
      document.getElementById('result-bar').style.width      = pct + '%';
      document.getElementById('result-bar').style.background = barColor;
      document.getElementById('result-stage').textContent = data.model_stage ? `Model stage: ${data.model_stage}` : '';
      document.getElementById('predict-result').style.display = 'block';
    } catch(e) {
      showPredictError('Could not reach API — is it running? (make up)');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Predict';
    }
  }

  function showPredictError(msg) {
    const el = document.getElementById('predict-error');
    el.textContent = '⚠ ' + msg;
    el.style.display = 'block';
  }

  // ── Pipeline reload step ──────────────────────────────────────────────────
  async function pipeReload(btn) {
    btn.disabled = true;
    btn.textContent = '⏳ reloading…';
    try {
      const resp = await fetch('/api/reload', {method: 'POST'});
      const data = await resp.json();
      if (data.ok) {
        btn.textContent = `✓ ${data.stage}`;
        btn.classList.add('ok');
        setTimeout(() => { btn.textContent = '▶ reload-api'; btn.classList.remove('ok'); btn.disabled = false; }, 3000);
      } else {
        btn.textContent = '✗ failed';
        setTimeout(() => { btn.textContent = '▶ reload-api'; btn.disabled = false; }, 2500);
      }
    } catch(e) {
      btn.textContent = '✗ unreachable';
      setTimeout(() => { btn.textContent = '▶ reload-api'; btn.disabled = false; }, 2500);
    }
  }

  // ── Reload model ──────────────────────────────────────────────────────────
  async function reloadModel() {
    const btn = document.querySelector('.btn-reload');
    btn.disabled = true;
    btn.textContent = 'Reloading…';
    try {
      const resp = await fetch('/api/reload', {method: 'POST'});
      const data = await resp.json();
      btn.textContent = data.ok ? `Reloaded (${data.stage})` : 'Reload failed';
      setTimeout(() => { btn.textContent = 'Reload model'; btn.disabled = false; }, 2500);
    } catch(e) {
      btn.textContent = 'Unreachable';
      setTimeout(() => { btn.textContent = 'Reload model'; btn.disabled = false; }, 2500);
    }
  }

  // ── Demo runner ───────────────────────────────────────────────────────────
  const DEMO_CMDS = new Set(['demo-drift-feature', 'demo-black-friday']);
  let activeBtn = null;

  function addLine(text, cls) {
    const out = document.getElementById('terminal-output');
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = text;
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
  }

  function closeTerminal() {
    const panel = document.getElementById('terminal-panel');
    panel.style.display = 'none';
    document.getElementById('terminal-output').innerHTML = '';
  }

  async function runCmd(cmd, btn) {
    const panel = document.getElementById('terminal-panel');
    const title = document.getElementById('term-title');

    document.getElementById('terminal-output').innerHTML = '';
    panel.style.display = 'flex';
    title.textContent  = '$ make ' + cmd;

    document.querySelectorAll('.btn-run').forEach(b => b.disabled = true);
    if (activeBtn) { activeBtn.textContent = activeBtn.dataset.label || '▶ Run'; activeBtn.classList.remove('running'); }
    activeBtn = btn;
    btn.dataset.label = btn.textContent;
    btn.textContent = '⏳ running…';
    btn.classList.add('running');

    addLine('$ make ' + cmd, 'prompt');

    let exitCode = null;
    try {
      const resp = await fetch('/api/run/' + cmd, {method: 'POST'});
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const events = buffer.split('\n\n');
        buffer = events.pop();
        for (const event of events) {
          for (const line of event.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if ('line' in data) addLine(data.line);
            if ('exit' in data) {
              exitCode = data.exit;
              const ok = exitCode === 0;
              addLine(ok ? '✓ Done' : `✗ Exited with code ${exitCode}`, ok ? 'success' : 'err');
            }
          }
        }
      }
    } catch(e) {
      addLine('Error: ' + e.message, 'err');
    } finally {
      document.querySelectorAll('.btn-run').forEach(b => b.disabled = false);
      btn.textContent = btn.dataset.label || '▶ Run';
      btn.classList.remove('running');
      activeBtn = null;
    }

    // After a successful demo, automatically reload the model
    if (DEMO_CMDS.has(cmd) && exitCode === 0) {
      addLine('', '');
      addLine('Reloading model in API…', 'prompt');
      try {
        const r = await fetch('/api/reload', {method: 'POST'});
        const d = await r.json();
        if (d.ok) {
          addLine(`Model reloaded — stage: ${d.stage}`, 'success');
          addLine('You can now use the Predict form above.', '');
        } else {
          addLine('Reload returned unexpected response', 'err');
        }
      } catch(e) {
        addLine('Could not reload model: ' + e.message, 'err');
      }
    }

    updateStatus();
  }
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", ""):
            self._html()
        elif path == "/api/status":
            self._status()
        else:
            self._respond(404, b"Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/run/"):
            self._stream(path[len("/api/run/"):])
        elif path == "/api/predict":
            self._proxy(f"{API}/predict")
        elif path == "/api/reload":
            self._proxy(f"{API}/reload")
        else:
            self._respond(404, b"Not found")

    # ── Routes ────────────────────────────────────────────────────────────────

    def _html(self):
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status(self):
        try:
            out = subprocess.check_output(
                ["docker", "ps", "--format", "{{.Names}}"],
                timeout=5, text=True,
            )
            running = [l for l in out.strip().split("\n") if l]
            status = {svc: any(m in c for c in running) for svc, m in SERVICE_MATCHES.items()}
        except Exception:
            status = {svc: False for svc in SERVICE_MATCHES}
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, url: str):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b""
        req    = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._respond(502, json.dumps({"error": str(e)}).encode())

    def _stream(self, cmd: str):
        if cmd not in ALLOWED_COMMANDS:
            self._respond(400, b"Command not allowed")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            proc = subprocess.Popen(
                ["make", "-C", str(ROOT), cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self._sse({"line": line.rstrip()})
            proc.wait()
            self._sse({"exit": proc.returncode})
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sse(self, data: dict):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _respond(self, code: int, body: bytes):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"DriftWatch dashboard → http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
