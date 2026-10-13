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
      'Promoting latest model to Production', 'Model promoted successfully',
      'Loading configuration config_path=/app/data/generator/config/shock_card_testing.yaml',
      'Dataset saved', 'Starting drift detection', 'Reports saved',
      'Control plane orchestrator starting', 'Phase 3: Executing retraining',
      'Starting training pipeline', 'Training pipeline complete',
      'Phase 4: Release evaluation',
      '\\x1b[33mQuality gates failed - promotion blocked\\x1b[0m',
      'Control plane cycle complete - promotion blocked',
      'Reloading model in API container...', 'Model reloaded: fraud_detector',
    ]) p.line(line);
    p.finish(0);
    assert.equal(p.steps.length, 7);
    assert(p.steps.every(s => s.state === 'done'));
    assert.equal(p.result, 'Candidate did not pass checks. Current model kept.');
    assert.equal(p.resultClass, 'notice');
    assert.equal(p.steps[5].detail, p.result);
    """,
    """
    for (const [message, expected] of [
      ['Control plane cycle complete - no action taken action=await_labels', 'Waiting for labels'],
      ['Control plane cycle complete - no action taken action=noop', 'No replacement requested'],
      ['Control plane cycle complete - model promoted stage=Staging', 'Production model unchanged'],
      ['Control plane cycle complete - model promoted stage=Production', 'Replacement model published'],
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
])
def test_progress_matches_actual_run_outcome(script):
    source = re.search(
        r'<script id="progress-logic">(.*?)</script>', HTML_PATH.read_text(), re.S
    ).group(1)
    subprocess.run(
        [NODE, "-"], input="const assert = require('node:assert/strict');\n" + source + script,
        text=True, capture_output=True, check=True, timeout=10,
    )
