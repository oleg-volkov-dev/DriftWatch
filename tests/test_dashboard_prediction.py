"""Exercise true probability display, six descriptions, and input feedback."""

import re
import shutil
import subprocess

import pytest

from infra.dashboard.server import HTML_PATH

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Dashboard JavaScript checks require Node")
def test_prediction_meter_and_field_feedback():
    source = re.search(
        r'<script id="prediction-logic">(.*?)</script>', HTML_PATH.read_text(), re.S
    ).group(1)
    script = """
    const assert = require('node:assert/strict');
    const nodes = {};
    const document = {getElementById(id) {
      return nodes[id] ??= {style: {}, validity: {stepMismatch: false}, setAttribute(k, v) { this[k] = v; }};
    }};
    """ + source + """
    function render(p, t = 0.03387) {
      renderPrediction({fraud_probability: p, decision_threshold: t, is_fraud: p >= t, model_stage: 'Production'});
      assert.equal(nodes['predict-result'].style.display, 'block');
      const width = parseFloat(nodes['result-bar'].style.width);
      assert.equal(width, p * 100);
      assert.equal(nodes['result-meter']['aria-valuenow'], width);
      assert.equal(nodes['result-badge'].textContent, p >= 0.5 ? 'Fraud' : 'Safe');
      return width;
    }
    assert.equal(render(0.00203), 0.203);
    assert.equal(nodes['result-pct'].textContent, '0.203%');
    assert.equal(render(0), 0);
    assert.equal(nodes['result-bar'].style.background, 'hsl(120, 85%, 50%)');
    assert.equal(render(0.5), 50);
    assert.equal(nodes['result-bar'].style.background, 'hsl(60, 85%, 50%)');
    assert.equal(render(1), 100);
    assert.equal(nodes['result-bar'].style.background, 'hsl(0, 85%, 50%)');
    // The probability and display classification must not be changed by API review thresholds.
    for (const threshold of [0, 0.001, 0.02, 0.5, 0.99, 1]) {
      for (let i = 0; i <= 1000; i++) render(i / 1000, threshold);
    }
    for (const [p, description] of [
      [0, 'Looks very safe'], [0.09999, 'Looks very safe'],
      [0.1, 'Safe'], [0.29999, 'Safe'],
      [0.3, 'A little suspicious'], [0.49999, 'A little suspicious'],
      [0.5, 'Might be fraud'], [0.69999, 'Might be fraud'],
      [0.7, 'Probably fraud'], [0.89999, 'Probably fraud'],
      [0.9, 'High chance of fraud'], [1, 'High chance of fraud'],
    ]) {
      render(p);
      assert.equal(nodes['result-description'].textContent, description);
      assert(nodes['result-meter']['aria-valuetext'].includes(description));
    }
    render(0.4999999);
    assert.equal(nodes['result-pct'].textContent, '<50%');
    render(0.0000001);
    assert.equal(nodes['result-pct'].textContent, '<0.001%');
    for (const bad of [-1, 2, NaN, Infinity, null, undefined]) {
      assert.throws(() => render(bad), /invalid/);
    }
    const defaults = {amount: 20, hour: 12, attempts: 1, tenure: 730, risk: 0.1, dist: 10};
    for (const [field, value] of Object.entries(defaults)) {
      document.getElementById('f-' + field).value = String(value);
      assert.equal(fieldWarning(field, value), '');
    }
    for (const [field, value, expected] of [
      ['dist', 200, 'capped'], ['dist', 1000000, 'capped'],
      ['amount', 150, 'capped'],
      ['attempts', 20, 'capped'], ['attempts', 1001, 'at most'],
      ['hour', 24, '0 to 23'],
      ['risk', 1.1, '0 to 1'],
      ['attempts', 1.5, 'whole number'], ['dist', -1, 'non-negative'],
    ]) assert(fieldWarning(field, value).includes(expected));
    for (const [field, value] of [['hour', 0], ['hour', 22], ['tenure', 1], ['tenure', 4000], ['amount', 3], ['risk', 0.9]]) {
      assert.equal(fieldWarning(field, value), '');
    }
    for (const [distance, capped] of [[0, false], [199, false], [200, true], [1000, true], [10, false]]) {
      nodes['f-dist'].value = String(distance);
      updateFieldHints();
      assert.equal(nodes['distance-hint'].hidden, !capped);
    }
    nodes['f-dist'].value = '';
    updateFieldHints();
    assert.equal(nodes['distance-hint'].hidden, false);
    assert(nodes['distance-hint'].textContent.includes('valid'));
    nodes['f-risk'].validity.stepMismatch = true;
    nodes['f-risk'].step = '0.01';
    updateFieldHints();
    assert(nodes['risk-hint'].textContent.includes('increments of 0.01'));
    """
    result = subprocess.run(
        [NODE, "-"], input=script, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
