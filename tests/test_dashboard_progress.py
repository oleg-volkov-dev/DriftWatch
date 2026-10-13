"""Check progress summaries against streamed lifecycle messages using Node, if installed."""

import re
import shutil
import subprocess

import pytest

from infra.dashboard.server import HTML_PATH

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Dashboard JavaScript checks require Node")


@pytest.mark.parametrize("script", [
    """
    const p = new RunProgress('demo-card-testing');
    for (const line of [
      'Loading configuration config_path=/app/data/generator/config/base.yaml',
      'Dataset saved', 'Starting training pipeline', 'Training pipeline complete',
      'Promoting latest model to Production', 'Model promoted successfully stage=Production version=1',
      'Loading configuration config_path=/app/data/generator/config/shock_card_testing.yaml',
      'Dataset saved', 'Starting drift detection', 'Reports saved',
      'Control plane orchestrator starting', 'Phase 3: Executing retraining',
      'Starting training pipeline', "Created version '2' of model 'fraud_detector'.", 'Training pipeline complete',
      'Phase 4: Release evaluation',
      '\\x1b[33mQuality gates failed - promotion blocked\\x1b[0m',
      'Control plane cycle complete - promotion blocked',
      'Reloading model in API container...', 'Model reloaded: fraud_detector v1 (stage: Production)',
    ]) p.line(line);
    p.finish(0);
    assert.equal(p.steps.length, 7);
    assert(p.steps.every(s => s.state === 'done'));
    assert.equal(p.result, 'Candidate v2 did not pass checks. Baseline v1 remains in Production.');
    assert.equal(p.resultClass, 'notice');
    assert.equal(p.steps[5].detail, p.result);
    assert.equal(p.steps[2].detail, 'Baseline v1 promoted to Production.');
    assert.equal(p.steps[6].detail, 'API now serving baseline v1.');
    """,
    """
    for (const [message, expected] of [
      ['Control plane cycle complete - no action taken action=await_labels', 'Waiting for labels'],
      ['Control plane cycle complete - no action taken action=noop', 'No replacement requested'],
      ['Control plane cycle complete - model promoted stage=Staging', 'Production was not changed'],
      ['Control plane cycle complete - model promoted stage=Production', 'Candidate is now in Production'],
    ]) {
      const p = new RunProgress('control');
      p.line(message); p.finish(0);
      assert(p.result.includes(expected));
    }
    """,
    """
    const p = new RunProgress('demo-drift-feature');
    p.line('Starting training pipeline');
    p.finish(1);
    assert.equal(p.steps[1].state, 'failed');
    assert.equal(p.steps[2].state, 'pending');
    assert.equal(p.resultClass, 'err');
    const disconnected = new RunProgress('train');
    disconnected.disconnected();
    assert.equal(disconnected.steps[0].state, 'unknown');
    assert(disconnected.result.includes('may still be continuing'));
    """,
    """
    const p = new RunProgress('monitor');
    const before = JSON.stringify(p);
    p.line('#12 exporting layers');
    p.line('WARNING: library deprecation');
    assert.equal(JSON.stringify(p), before);
    p.line('Starting drift detection');
    p.line('Reports saved');
    p.finish(0);
    assert.equal(p.steps.length, 1);
    assert.equal(p.result, 'Finished.');
    """,
    """
    const p = new RunProgress('demo-drift-feature');
    p.line('Promoting latest model to Production');
    p.line('Model promoted successfully stage=Production version=11');
    p.line('Control plane orchestrator starting');
    p.line("Created version '12' of model 'fraud_detector'.");
    p.line('Model promoted successfully stage=Production version=12');
    p.line('Control plane cycle complete - model promoted stage=Production');
    p.line('Reloading model in API container...');
    p.line('Model reloaded: fraud_detector v12 (stage: Production)');
    p.finish(0);
    assert.equal(p.result, 'Candidate v12 is now in Production. It replaces baseline v11.');
    assert.equal(p.resultClass, 'success');
    assert.equal(p.steps[2].detail, 'Baseline v11 promoted to Production.');
    assert.equal(p.steps[6].detail, 'API now serving candidate v12.');
    """,
    """
    // A standalone control run must not invent a baseline or assume one exists.
    const p = new RunProgress('control');
    p.line("Created version '2' of model 'fraud_detector'.");
    p.line('Quality gates failed - promotion blocked');
    p.finish(0);
    assert.equal(p.result, 'Candidate v2 did not pass checks. Production was not changed.');
    const reload = new RunProgress('reload-api');
    reload.line('Model reloaded: fraud_detector');
    reload.finish(0);
    assert.equal(reload.steps[0].detail, 'API loaded the Production model.');
    assert.equal(reload.servingVersion, '');
    """,
])
def test_progress_matches_actual_run_outcome(script):
    source = re.search(
        r'<script id="progress-logic">(.*?)</script>', HTML_PATH.read_text(), re.S
    ).group(1)
    subprocess.run(
        [NODE, "-"], input="const assert = require('node:assert/strict');\n" + source + script,
        text=True, capture_output=True, check=True, timeout=10,
    )
