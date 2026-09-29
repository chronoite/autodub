'use strict';
// Experiment review: list runs, play each candidate's artifacts, record pass/maybe/fail verdicts.
// Talks only to the local AutoDub API (/api/experiments...). Saves carry the review revision so two
// open tabs cannot silently overwrite each other.

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
const VIDEO = /\.(mp4|webm|mkv|mov)$/i;
const AUDIO = /\.(wav|mp3|flac|ogg|m4a)$/i;
const BLOCKERS = ['', 'timing', 'voice', 'mix', 'text'];

let runs = [];
let current = null;   // full public manifest of the open run
let draft = null;     // {verdicts, blockers, notes, status}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

// Deterministic per-run shuffle (FNV-1a seeded LCG) so blind letters stay stable across reloads.
function shuffled(items, seedText) {
  let seed = 2166136261;
  for (const ch of seedText) seed = Math.imul(seed ^ ch.charCodeAt(0), 16777619) >>> 0;
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const j = seed % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function artifactUrl(candidateId, path) {
  const query = new URLSearchParams({candidate: candidateId, path});
  return `/api/experiments/${encodeURIComponent(current.id)}/artifact?${query}`;
}

async function loadRuns() {
  const data = await api('/api/experiments');
  runs = data.experiments || [];
  renderRuns();
}

function renderRuns() {
  const node = $('#runs');
  if (!runs.length) {
    node.innerHTML = '<p class="muted pad">No experiment runs yet. Build one from a reviewed job in the main studio (Experiment mode).</p>';
    return;
  }
  node.innerHTML = runs.map(run => `
    <button class="run-item${current && current.id === run.id ? ' active' : ''}" data-run="${esc(run.id)}">
      <b>${esc(run.experiment)}</b>
      <span class="mono">${esc(run.id)}</span>
      <span>${esc(run.candidate_count)} candidates · ${esc(run.status)}</span>
      <span class="pill ${esc(run.review_status)}">${esc(run.review_status)}</span>
    </button>`).join('');
  node.querySelectorAll('.run-item').forEach(button => button.addEventListener('click', () => openRun(button.dataset.run)));
}

async function openRun(runId) {
  current = await api(`/api/experiments/${encodeURIComponent(runId)}`);
  const review = current.human_review || {};
  draft = {
    verdicts: {...(review.verdicts || {})},
    blockers: {...(review.blockers || {})},
    notes: (review.notes || []).join('\n'),
    status: review.status || 'pending',
  };
  renderRuns();
  renderRun();
}

function mediaFor(candidate) {
  const files = candidate.artifacts || [];
  const playable = files.find(file => VIDEO.test(file.name)) || files.find(file => AUDIO.test(file.name));
  let player = '<div class="empty">No playable artifact yet (the candidate may still be rendering or may have failed).</div>';
  if (playable) {
    const url = artifactUrl(candidate.id, playable.path);
    player = VIDEO.test(playable.name)
      ? `<video controls preload="metadata" src="${esc(url)}"></video>`
      : `<audio controls preload="metadata" src="${esc(url)}"></audio>`;
  }
  const others = files.filter(file => file !== playable)
    .map(file => `<a href="${esc(artifactUrl(candidate.id, file.path))}" target="_blank" rel="noopener">${esc(file.name)}</a>`);
  return player + (others.length ? `<div class="files">Other files: ${others.join(' · ')}</div>` : '');
}

function renderRun() {
  const blind = $('#blind').checked;
  const candidates = blind ? shuffled(current.candidates || [], current.id) : (current.candidates || []);
  const cards = candidates.map((candidate, index) => {
    const letter = String.fromCharCode(65 + index);
    const verdict = draft.verdicts[candidate.id] || '';
    const blocker = draft.blockers[candidate.id] || '';
    const failed = candidate.result && typeof candidate.result === 'object' && candidate.result.error;
    return `
      <div class="card" data-candidate="${esc(candidate.id)}">
        <div class="card-head">
          <div class="card-letter">${letter}</div>
          <div class="card-name">${blind ? 'hidden while blind' : esc(candidate.id)}${failed ? ' · <span class="pill">failed</span>' : ''}</div>
        </div>
        ${mediaFor(candidate)}
        <div class="verdicts">
          ${['pass', 'maybe', 'fail'].map(value => `<button class="${verdict === value ? 'on' : ''}" data-verdict="${value}">${value}</button>`).join('')}
        </div>
        <select data-blocker aria-label="Main problem">
          ${BLOCKERS.map(value => `<option value="${value}"${value === blocker ? ' selected' : ''}>${value ? `main problem: ${value}` : 'main problem: none'}</option>`).join('')}
        </select>
      </div>`;
  }).join('');
  $('#run').innerHTML = `
    <div class="run-head">
      <h2>${esc(current.experiment)}</h2>
      <p class="muted mono">${esc(current.id)} · job ${esc(current.job)} · ${esc(current.status)}</p>
      <div class="criteria">${(current.criteria || []).map(item => `<span class="pill">${esc(item)}</span>`).join('')}</div>
    </div>
    <div class="cards">${cards || '<p class="muted">This run has no candidates.</p>'}</div>
    <div class="review-foot">
      <textarea id="notes" placeholder="Notes (one per line)">${esc(draft.notes)}</textarea>
      <select id="review-status">
        ${['pending', 'in-progress', 'complete'].map(value => `<option value="${value}"${value === draft.status ? ' selected' : ''}>${value}</option>`).join('')}
      </select>
      <button id="save" class="primary">Save review</button>
      <span id="save-state"></span>
    </div>`;
  wireRun();
}

function wireRun() {
  document.querySelectorAll('.card').forEach(card => {
    const id = card.dataset.candidate;
    card.querySelectorAll('[data-verdict]').forEach(button => button.addEventListener('click', () => {
      draft.verdicts[id] = draft.verdicts[id] === button.dataset.verdict ? undefined : button.dataset.verdict;
      if (!draft.verdicts[id]) delete draft.verdicts[id];
      card.querySelectorAll('[data-verdict]').forEach(other => other.classList.toggle('on', other.dataset.verdict === draft.verdicts[id]));
    }));
    card.querySelector('[data-blocker]').addEventListener('change', event => {
      if (event.target.value) draft.blockers[id] = event.target.value; else delete draft.blockers[id];
    });
  });
  $('#notes').addEventListener('input', event => { draft.notes = event.target.value; });
  $('#review-status').addEventListener('change', event => { draft.status = event.target.value; });
  $('#save').addEventListener('click', save);
}

async function save() {
  const state = $('#save-state');
  state.textContent = 'Saving…';
  try {
    current = await api(`/api/experiments/${encodeURIComponent(current.id)}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        revision: (current.human_review || {}).revision ?? 0,
        verdicts: draft.verdicts,
        blockers: draft.blockers,
        notes: draft.notes.split('\n').map(line => line.trim()).filter(Boolean),
        status: draft.status,
      }),
    });
    state.textContent = 'Saved.';
    await loadRuns();
  } catch (error) {
    state.textContent = `Not saved: ${error.message}`;
  }
}

$('#blind').addEventListener('change', () => { if (current) renderRun(); });
$('#refresh').addEventListener('click', () => loadRuns().then(() => current && openRun(current.id)));
loadRuns().then(() => {
  const wanted = new URLSearchParams(location.search).get('run');
  if (wanted) return openRun(wanted);
  return null;
}).catch(error => { $('#runs').innerHTML = `<p class="muted pad">Could not load runs: ${esc(error.message)}</p>`; });
