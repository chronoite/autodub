const $ = (q) => document.querySelector(q);
const BASE = location.pathname.startsWith('/autodub') ? '/autodub' : '';
let current = null;
let poller = null;
let profiles = [];
let policies = {mix: [], timing: [], space: []};
let ttsCandidates = [];
let episodeQueue = {status: 'idle', items: []};
let songMarkMode = false;
let songMarkAnchor = null;

async function api(path, options = {}) {
  const response = await fetch(BASE + path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

// The CSP forbids inline style attributes, so bar widths travel as data-width and are applied
// through the CSSOM after rendering.
function applyWidths(root) {
  root.querySelectorAll('[data-width]').forEach(node => { node.style.width = `${node.dataset.width}%`; });
}

// ---- SIDEBAR -------------------------------------------------------------------------
// Library tab: flat show cards (sort: recent / A-Z / needs-review) with status-colored
// episode chips per season row. Activity tab: jobs + the render queue merged into
// RUNNING / WAITING ON YOU / QUEUED — a queued item and a finished job are the same
// object at different stages. Shows come from /api/projects; live status/progress joins
// in from /api/jobs client-side. A folder/category layer above shows would slot in here as
// extra group headers over the same card list, with no rework of the cards or Activity.
let jobsData = [];
let projectsData = [];
let sortMode = 'recent';

function jobStamp(id) { return Number(String(id || '').replace(/[^0-9]/g, '')) || 0; }

async function refreshJobs(selectNewest = false) {
  const [jobsRes, projectsRes] = await Promise.all([api('/api/jobs'), api('/api/projects')]);
  jobsData = jobsRes.jobs || [];
  projectsData = projectsRes.projects || [];
  renderLibrary();
  renderActivity();
  if (selectNewest && jobsData[0]) await loadJob(jobsData[0].id);
}

function projectStats(project) {
  const byId = Object.fromEntries(jobsData.map(job => [job.id, job]));
  const eps = project.episodes || [];
  return {
    byId,
    running: eps.filter(e => byId[e.job_id]?.status === 'running'),
    review: eps.filter(e => byId[e.job_id]?.status === 'review'),
    last: eps.reduce((max, e) => Math.max(max, jobStamp(e.job_id)), 0),
  };
}

function chipClass(job) {
  if (!job) return '';
  if (job.status === 'running') return 'run';
  if (job.status === 'review' || job.status === 'complete') return 'done';
  if (job.status === 'failed') return 'fail';
  return '';
}

function renderLibrary() {
  const list = [...projectsData];
  if (sortMode === 'az') list.sort((a, b) => a.series.localeCompare(b.series));
  else if (sortMode === 'review') list.sort((a, b) => projectStats(b).review.length - projectStats(a).review.length);
  else list.sort((a, b) => projectStats(b).last - projectStats(a).last);
  $('#projects').innerHTML = list.length ? list.map(project => {
    const stats = projectStats(project);
    const runningEp = stats.running[0];
    const runningJob = runningEp && stats.byId[runningEp.job_id];
    const seasons = (project.seasons || []).map(se => `
      <div class="chip-row"><span class="chip-season">${esc(se.season.replace('Season ', 'S'))}</span>${se.episodes.map(e => {
        const job = stats.byId[e.job_id];
        return `<button class="ep-pill ${chipClass(job)}" data-job="${esc(e.job_id)}" title="${esc(e.episode)} · ${esc(job?.status || '?')} · ${e.assigned}/${e.speakers} named">${esc(e.episode.replace(/^S\d{1,2}(?=E)/i, ''))}</button>`;
      }).join('')}</div>`).join('');
    return `<div class="show-card">
      <button class="show-head" data-slug="${esc(project.slug)}">
        <b>${esc(project.series)}</b>
        <small>${project.episodes.length} ep · ${stats.review.length} to review · ${project.bank_characters} named</small>
      </button>
      ${runningJob ? `<div class="show-progress"><div data-width="${Number(runningJob.progress) || 0}"></div></div>
        <small class="show-run">${esc(runningEp.episode)} · ${esc(runningJob.stage)} ${runningJob.progress}%</small>` : ''}
      <div class="chip-strip">${seasons}</div>
    </div>`;
  }).join('') : '<div class="empty-small">Nothing imported yet</div>';
  applyWidths($('#projects'));
  document.querySelectorAll('.show-head').forEach(button => button.onclick = () => openProject(button.dataset.slug));
  document.querySelectorAll('.ep-pill').forEach(button => button.onclick = () => loadJob(button.dataset.job));
}

function renderActivity() {
  const label = (id) => {
    for (const project of projectsData) for (const e of project.episodes || [])
      if (e.job_id === id) return `${project.series.length > 24 ? project.series.slice(0, 24) + '…' : project.series} · ${e.episode}`;
    return String(id || '').slice(-9);
  };
  const running = jobsData.filter(job => job.status === 'running');
  const review = jobsData.filter(job => job.status === 'review');
  const fresh = jobsData.filter(job => job.status === 'created');
  const finished = jobsData.filter(job => ['complete', 'failed', 'cancelled'].includes(job.status));
  const queued = (episodeQueue.items || []).filter(item => item.status === 'queued');
  const count = running.length + review.length + queued.length;
  $('#activity-count').textContent = count;
  $('#activity-count').classList.toggle('hidden', !count);
  const section = (title, cls, rows) => rows.length ? `<div class="activity-head ${cls}">${title}</div>${rows.join('')}` : '';
  $('#activity-list').innerHTML = (
    section('RUNNING', 'amber', running.map(job => `
      <button class="activity-row" data-job="${job.id}">
        <div class="activity-top"><b>${esc(label(job.id))}</b><span class="amber">${esc(job.stage)} ${job.progress}%</span></div>
        <div class="activity-bar"><div data-width="${Number(job.progress) || 0}"></div></div>
      </button>`)) +
    section('WAITING ON YOU', 'green', review.map(job => `
      <button class="activity-row" data-job="${job.id}">
        <div class="activity-top"><b>${esc(label(job.id))}</b><span class="green">REVIEW</span></div>
      </button>`)) +
    section(`RENDER QUEUE · ${queued.length}`, 'dim', queued.map(item => `
      <button class="activity-row dim" data-job="${esc(item.job)}">
        <div class="activity-top"><b>${esc(label(item.job))}</b><span>queued</span></div>
      </button>`)) +
    section('NOT ANALYZED', 'dim', fresh.map(job => `
      <button class="activity-row dim" data-job="${job.id}">
        <div class="activity-top"><b>${esc(label(job.id))}</b><span>imported</span></div>
      </button>`)) +
    section('FINISHED', 'dim', finished.map(job => `
      <button class="activity-row dim" data-job="${job.id}">
        <div class="activity-top"><b>${esc(label(job.id))}</b><span>${esc(job.status)}</span></div>
      </button>`))
  ) || '<div class="empty-small">Nothing running, nothing waiting.</div>';
  document.querySelectorAll('.activity-row[data-job]').forEach(button => button.onclick = () => loadJob(button.dataset.job));
  applyWidths($('#activity-list'));
}

function setSidebarTab(tab) {
  $('#library-pane').classList.toggle('hidden', tab !== 'library');
  $('#activity-pane').classList.toggle('hidden', tab !== 'activity');
  $('#tab-library').classList.toggle('active', tab === 'library');
  $('#tab-activity').classList.toggle('active', tab === 'activity');
}

async function deleteCurrentJob() {
  if (!current) return;
  const id = current.id;
  if (!await confirmModal(`Delete ${id} and its rendered outputs? The original video in input/ and every voice-bank character are kept. This cannot be undone.`, 'Delete job', 'Delete')) return;
  try {
    await api(`/api/jobs/${id}/delete`, {method: 'POST'});
    current = null;
    if (poller) clearInterval(poller);
    $('#job-view').classList.add('hidden');
    $('#welcome').classList.remove('hidden');
    await refreshJobs();
  } catch (error) { alert(error.message); }
}

async function loadProfiles() {
  const data = await api('/api/profiles');
  profiles = data.profiles;
  $('#quality-profile').innerHTML = profiles.map(item => `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join('');
}

async function loadPolicies() {
  policies = await api('/api/policies');
  $('#mix-policy').innerHTML = policies.mix.map(item => `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join('');
  $('#timing-policy').innerHTML = policies.timing.map(item => `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join('');
  $('#space-policy').innerHTML = policies.space.map(item => `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join('');
}

async function loadTtsCandidates() {
  const data = await api('/api/tts-candidates');
  ttsCandidates = data.candidates || [];
  $('#tts-candidates').innerHTML = ttsCandidates.map((item, index) => `
    <label class="tts-candidate">
      <input type="checkbox" value="${esc(item.id)}" ${item.implemented && (index < 3 || item.id === 'windows-sapi') ? 'checked' : ''} ${item.implemented ? '' : 'disabled'}>
      <div><b>${esc(item.id)}</b><span>${esc(item.device)} · ${item.requires_gpu ? 'shared GPU queue' : 'CPU'}${item.implemented ? '' : ' · not wired'}</span></div>
    </label>`).join('');
}

async function refreshEpisodeQueue() {
  episodeQueue = await api('/api/episode-queue');
  const items = episodeQueue.items || [];
  const thermal = episodeQueue.thermal_abort
    ? ` · THERMAL ABORT: ${episodeQueue.thermal_abort.reason}`
    : '';
  $('#episode-queue-state').textContent = (items.length
    ? `${episodeQueue.status} · ${items.filter(item => item.status === 'queued').length} pending`
    : 'No reviewed jobs queued.') + thermal;
  renderActivity();
  $('#queue-run').disabled = episodeQueue.status === 'running' || !items.some(item => item.status === 'queued');
  $('#queue-stop').disabled = episodeQueue.status !== 'running';
  $('#queue-clear').disabled = episodeQueue.status === 'running' || !items.some(item => ['complete', 'failed', 'cancelled'].includes(item.status));
}

async function refreshGpuStatus() {
  const state = await api('/api/gpu/status');
  const node = $('#gpu-state');
  node.textContent = state.safe ? 'READY - render queue idle' : `BLOCKED - ${(state.reasons || []).join('; ')}`;
  node.classList.toggle('ready', state.safe);
  node.classList.toggle('blocked', !state.safe);
  return state;
}

async function loadJob(id) {
  current = await api(`/api/jobs/${id}`);
  renderJob();
  await refreshJobs();
  if (poller) clearInterval(poller);
  poller = setInterval(async () => {
    if (!current) return;
    const fresh = await api(`/api/jobs/${current.id}`);
    const changed = fresh.updated_at !== current.updated_at;
    current = fresh;
    if (changed || current.status === 'running') renderJob();
    await refreshJobs().catch(() => {});
    await refreshEpisodeQueue().catch(() => {});
  }, 1500);
}

function renderJob() {
  $('#welcome').classList.add('hidden');
  $('#job-view').classList.remove('hidden');
  $('#job-title').textContent = current.id;
  $('#job-meta').textContent = `${formatBytes(current.source.bytes)} · SHA-256 ${current.source.sha256.slice(0, 12)}…`;
  $('#stage').textContent = current.stage.replaceAll('-', ' ');
  $('#percent').textContent = `${current.progress}%`;
  $('#bar').style.width = `${current.progress}%`;
  renderStageChecklist();
  $('#error').textContent = current.error || '';
  $('#error').classList.toggle('hidden', !current.error);
  const busy = current.status === 'running';
  $('#delete-job').disabled = busy;
  $('#analyze').disabled = busy;
  $('#render').disabled = busy || !current.segments.length;
  $('#realign').disabled = busy || !current.segments.length || !current.artifacts?.dialogue_stem;
  $('#adapt').disabled = busy || !current.segments.length || current.status !== 'review';
  $('#save').disabled = busy;
  $('#cancel-job').classList.toggle('hidden', !busy);
  $('#download').classList.toggle('hidden', current.status !== 'complete');
  $('#download').href = `${BASE}/api/jobs/${current.id}/output`;
  $('#srt').classList.toggle('hidden', !current.segments.length);
  $('#srt').href = `${BASE}/api/jobs/${current.id}/srt`;
  $('#source-language').value = current.settings.source_language || 'ja';
  const speakerCount = current.settings.speaker_count || {mode: 'automatic'};
  $('#speakers').value = speakerCount.mode === 'exact' ? speakerCount.count : 0;
  $('#mix-policy').value = current.settings.mix_policy || 'legacy-v1';
  $('#timing-policy').value = current.settings.timing_policy || 'segment-window-v1';
  $('#space-policy').value = current.settings.space_policy || 'dry-v1';
  $('#emotion-policy').value = current.settings.emotion_policy || 'source-energy-v1';
  $('#song-policy').value = current.settings.song_policy || 'dub-all-v1';
  $('#bed').value = current.settings.source_bed_gain;
  $('#bed-value').textContent = `${Math.round(current.settings.source_bed_gain * 100)}%`;
  $('#dialogue').value = current.settings.dialogue_gain ?? 1.0;
  $('#dialogue-value').textContent = `${Math.round((current.settings.dialogue_gain ?? 1.0) * 100)}%`;
  $('#tempo').value = current.settings.max_tempo;
  $('#tempo-value').textContent = `${Number(current.settings.max_tempo).toFixed(2)}×`;
  $('#line-count').textContent = `${current.segments.length} lines`;
  $('#glossary').value = Object.entries(current.glossary || {}).map(([key, value]) => `${key} = ${value}`).join('\n');
  const synth = current.synth_progress;
  $('#progress-detail').textContent = synth?.total
    ? `Line ${synth.done}/${synth.total} · average ${Number(synth.avg_secs || 0).toFixed(1)}s · ETA ${Math.round(synth.eta_secs || 0)}s`
    : '';
  const qc = current.qc_summary || {};
  const subtitles = current.subtitle_harvest || {};
  $('#qc-summary').textContent = qc.lines
    ? `${qc.flagged || 0}/${qc.lines} lines flagged · ${qc.overrun || 0} overruns · ${qc.tempo_limited || 0} tempo-limited · ${qc.overlap || 0} source overlaps · subtitles ${subtitles.mapped || 0}`
      + (qc.runaway ? ` · RUNAWAY TTS: ${qc.runaway} line(s) — check voice references` : '')
    : `Timing QC appears after line alignment. Embedded English subtitles mapped: ${subtitles.mapped || 0}.`;
  const adapt = current.adapt_summary;
  $('#adapt-summary').classList.toggle('hidden', !adapt);
  $('#adapt-summary').textContent = adapt
    ? `Adaptation ${adapt.policy}: ${adapt.changed}/${adapt.lines} lines rewritten · engines ${(adapt.engines || []).join('+')}`
      + ` · wins ${Object.entries(adapt.engine_wins || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}`
      + (adapt.counts?.residue ? ` · ${adapt.counts.residue} residue` : '')
    : '';
  const profileId = current.settings.quality_profile || 'prototype-cpu-v1';
  $('#quality-profile').value = profileId;
  const profile = profiles.find(item => item.id === profileId);
  $('#profile-note').textContent = profile ? `${profile.asr} · ${profile.diarization} · ${profile.tts}` : 'Legacy CPU job';
  $('#analyze').textContent = profile?.requires_gpu ? 'Analyze (arm GPU)' : 'Analyze locally';
  $('#render').textContent = profile?.requires_gpu ? 'Render (arm GPU)' : 'Render English dub';
  renderVoices(); renderSegments(); renderEvents();
  refreshGpuStatus().catch(() => { $('#gpu-state').textContent = 'BLOCKED - safety status unavailable'; });
}

function renderStageChecklist() {
  const steps = [
    ['Import', 0], ['Extract', 8], ['Speech', 24], ['Speakers', 47], ['Translate', 68],
    ['Review', 72], ['Voices', 78], ['Timing', 86], ['Mix', 92], ['Export', 97]
  ];
  const progress = Number(current.progress || 0);
  $('#stage-checklist').innerHTML = steps.map(([label, threshold], index) => {
    const next = steps[index + 1]?.[1] ?? 101;
    const done = progress >= next || current.status === 'complete';
    const active = !done && progress >= threshold;
    return `<span class="stage-step ${done ? 'done' : active ? 'active' : ''}" title="${esc(label)}">${done ? '✓ ' : active ? '• ' : ''}${esc(label)}</span>`;
  }).join('');
}

function renderVoices() {
  const speakers = [...new Set(current.segments.map(x => x.speaker))].sort();
  $('#voice-map').innerHTML = speakers.length ? speakers.map(speaker => `
    <div class="voice-row"><label>${esc(speaker)}</label><div class="voice-select-row"><select data-speaker="${esc(speaker)}">
      ${current.available_voices.map(voice => `<option value="${esc(voice)}" ${current.speaker_voices[speaker] === voice ? 'selected' : ''}>${esc(current.voice_labels?.[voice] || voice)}</option>`).join('')}
    </select><button class="mini voice-audition" data-audition="${esc(speaker)}" title="Play source reference">▶</button></div></div>`).join('') : '<p class="muted">Analyze first to discover speakers.</p>';
  document.querySelectorAll('[data-audition]').forEach(button => button.onclick = () => {
    const select = document.querySelector(`[data-speaker="${CSS.escape(button.dataset.audition)}"]`);
    playReview(`/api/jobs/${current.id}/voices/audition?voice=${encodeURIComponent(select.value)}`, `Voice reference · ${button.dataset.audition}`);
  });
}

function renderSegments() {
  // The badge must mirror the RENDER's policy gate: with Song skip toggled off these
  // lines WILL be dubbed, so showing "will NOT be dubbed" would mislead the reviewer.
  const songActive = (current.settings.song_policy || 'dub-all-v1') === 'skip-detected-v1';
  const songCount = songActive ? current.segments.filter(s => s.song_skip).length : 0;
  const songBanner = songCount
    ? `<div class="qc-flags song-banner"><span class="qc-flag song">SONG SKIP (EXPERIMENTAL): ${songCount} line(s) will NOT be dubbed - review them below</span></div>` : '';
  const songMark = s => songActive && s.song_skip;
  $('#segments').innerHTML = current.segments.length ? songBanner + current.segments.map(segment => `
    <div class="segment${songMark(segment) ? ' song-outline' : ''}" data-i="${segment.i}">
      <div class="time">${fmtTime(segment.start)}<br>${fmtTime(segment.end)}</div>
      <input class="speaker" value="${esc(segment.speaker)}" aria-label="Speaker">
      <textarea class="source" readonly aria-label="Source transcript">${esc(segment.text)}</textarea>
      <textarea class="translation" aria-label="English translation">${esc(segment.translation || '')}</textarea>
      <div class="line-tools">
        <button class="mini" data-source="${segment.i}">Source</button>
        <button class="mini" data-line="${segment.i}">Line</button>
        <button class="mini" data-preview="${segment.i}">Preview</button>
        <button class="mini" data-repair="${segment.i}">Repair</button>
      </div>
      ${songMark(segment) ? `<div class="qc-flags"><span class="qc-flag song" title="Detected song - will not be dubbed; original audio kept">SONG - skipped (${esc(segment.song_reason || 'detected')})</span></div>` : ''}
      ${segment.delivery ? `<div class="delivery-hint" title="Relative source-energy hint; human review required">${esc(segment.delivery.label)} · ${Math.round(Number(segment.delivery.confidence || 0) * 100)}%</div>` : ''}
      ${segment.adapt?.engine ? `<div class="delivery-hint" title="Dialogue adaptation: the pre-adaptation line is preserved; re-runs re-derive from it">adapted · ${esc(segment.adapt.engine)}</div>` : ''}
      ${(segment.qc?.flags || []).length ? `<div class="qc-flags">${segment.qc.flags.map(flag => `<span class="qc-flag">${esc(flag)}</span>`).join('')}</div>` : ''}
    </div>`).join('') : '<div class="empty-small">Analysis has not run.</div>';
  document.querySelectorAll('[data-source]').forEach(button => button.onclick = () => playReview(`/api/jobs/${current.id}/segments/${button.dataset.source}/source`, `Source line ${button.dataset.source}`));
  document.querySelectorAll('[data-line]').forEach(button => button.onclick = () => playReview(`/api/jobs/${current.id}/segments/${button.dataset.line}/line`, `Rendered line ${button.dataset.line}`));
  document.querySelectorAll('[data-preview]').forEach(button => button.onclick = () => lineAction(Number(button.dataset.preview), 'preview'));
  document.querySelectorAll('[data-repair]').forEach(button => button.onclick = () => lineAction(Number(button.dataset.repair), 'repair'));
  $('#segments').classList.toggle('song-marking', songMarkMode);
  if (songMarkMode) {
    document.querySelectorAll('.segment').forEach(row => {
      if (Number(row.dataset.i) === songMarkAnchor) row.classList.add('song-anchor');
      const cell = row.querySelector('.time');
      if (cell) cell.onclick = () => songMarkClick(Number(row.dataset.i));
    });
  }
}

function toggleSongMarkMode() {
  if (!current || !(current.segments || []).length) return;
  songMarkMode = !songMarkMode;
  songMarkAnchor = null;
  $('#song-mark-mode').textContent = songMarkMode ? 'Cancel marking' : 'Mark song range';
  renderSegments();
}

function songMarkClick(index) {
  if (!songMarkMode) return;
  if (songMarkAnchor === null) {
    songMarkAnchor = index;
    renderSegments();
    return;
  }
  const start = Math.min(songMarkAnchor, index);
  const end = Math.max(songMarkAnchor, index);
  const inRange = current.segments.filter(s => Number(s.i) >= start && Number(s.i) <= end);
  const mark = !inRange.every(s => s.song_skip);
  markSongRange(start, end, mark);
}

async function markSongRange(start, end, mark) {
  const verb = mark ? 'Mark' : 'Unmark';
  if (!await confirmModal(
    `${verb} lines ${start}-${end} as song? Marked lines keep the original audio and are never dubbed.`,
    'Song range', verb)) { songMarkAnchor = null; renderSegments(); return; }
  try {
    await save();
    const result = await api(`/api/jobs/${current.id}/mark-songs`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({start, end, mark}),
    });
    songMarkMode = false;
    songMarkAnchor = null;
    $('#song-mark-mode').textContent = 'Mark song range';
    if (!result.policy_active) {
      $('#error').textContent = 'Marks saved, but Song skip is OFF in Run settings — these lines will still be dubbed until you turn it on.';
      $('#error').classList.remove('hidden');
    }
    await loadJob(current.id);
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
    songMarkAnchor = null;
    renderSegments();
  }
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && songMarkMode) toggleSongMarkMode();
});

function renderEvents() {
  $('#events').innerHTML = [...current.events].reverse().map(item => `<div class="event"><span>${new Date(item.at).toLocaleString()}</span><b>${esc(item.stage)}</b><span>${esc(item.message)}</span></div>`).join('');
}

function collectEdits() {
  const segments = [...document.querySelectorAll('.segment')].map(row => ({
    i: Number(row.dataset.i), speaker: row.querySelector('.speaker').value.trim(), translation: row.querySelector('.translation').value.trim()
  }));
  const speaker_voices = {};
  document.querySelectorAll('[data-speaker]').forEach(select => speaker_voices[select.dataset.speaker] = select.value);
  return {
    settings: {
      quality_profile: $('#quality-profile').value,
      source_language: $('#source-language').value,
      speaker_count: Number($('#speakers').value) > 0
        ? {mode: 'exact', count: Number($('#speakers').value)}
        : (current.settings.speaker_count?.mode === 'min-max'
          ? current.settings.speaker_count
          : {mode: 'automatic'}),
      mix_policy: $('#mix-policy').value,
      timing_policy: $('#timing-policy').value,
      space_policy: $('#space-policy').value,
      emotion_policy: $('#emotion-policy').value,
      song_policy: $('#song-policy').value,
      source_bed_gain: Number($('#bed').value),
      dialogue_gain: Number($('#dialogue').value),
      max_tempo: Number($('#tempo').value)
    },
    glossary: parseGlossary($('#glossary').value),
    segments,
    speaker_voices
  };
}

async function save() {
  current = await api(`/api/jobs/${current.id}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify(collectEdits())});
  renderJob();
}

function parseGlossary(value) {
  const result = {};
  String(value || '').split(/\r?\n/).forEach(line => {
    const split = line.indexOf('=');
    if (split > 0) {
      const key = line.slice(0, split).trim();
      const replacement = line.slice(split + 1).trim();
      if (key) result[key] = replacement;
    }
  });
  return result;
}

function playReview(path, label) {
  const player = $('#player');
  player.src = BASE + path;
  $('#player-label').textContent = label;
  player.load();
  player.play().catch(() => {});
}

async function waitForIdle(index = null) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    current = await api(`/api/jobs/${current.id}`);
    renderJob();
    if (current.status !== 'running') {
      if (index !== null && current.status !== 'failed') {
        playReview(`/api/jobs/${current.id}/segments/${index}/line`, `Rendered line ${index}`);
      }
      return;
    }
  }
  throw new Error('The local action is still running; progress remains visible in this job.');
}

function voiceForSegment(index) {
  const segment = current.segments.find(item => Number(item.i) === Number(index));
  return current.speaker_voices?.[segment?.speaker] || '';
}

async function lineAction(index, kind) {
  try {
    await save();
    if (voiceForSegment(index).startsWith('qwen')) {
      if (!await confirmModal(`Arm the shared GPU to ${kind} only line ${index}?`, 'Arm shared GPU')) return;
      await api(`/api/jobs/${current.id}/arm-gpu`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({action:`${kind}-line`})
      });
    }
    await api(`/api/jobs/${current.id}/segments/${index}/${kind}`, {method:'POST'});
    current.status = 'running';
    current.stage = kind === 'preview' ? 'previewing' : 'repairing';
    renderJob();
    await waitForIdle(index);
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function cancelJob() {
  if (!current || !await confirmModal('Cancel at the next safe checkpoint? Completed artifacts and cached lines remain resumable.', 'Cancel job', 'Cancel safely')) return;
  try {
    await api(`/api/jobs/${current.id}/cancel`, {method:'POST'});
    $('#progress-detail').textContent = 'Cancellation requested; waiting for the current safe checkpoint.';
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function buildExperiment() {
  try {
    await save();
    $('#build-experiment').disabled = true;
    $('#experiment-state').textContent = 'Building six local comparison clips…';
    const run = await api(`/api/jobs/${current.id}/policy-experiment`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        start:Number($('#experiment-start').value),
        duration:Number($('#experiment-duration').value),
        line_set:$('#experiment-lines').value
      })
    });
    $('#experiment-state').textContent = `Comparison ${run.id} complete. Judge it in Experiment review (/experiments.html?run=${run.id}).`;
  } catch (error) {
    $('#experiment-state').textContent = error.message;
  } finally {
    $('#build-experiment').disabled = false;
  }
}

async function buildTtsExperiment() {
  try {
    await save();
    const selected = [...document.querySelectorAll('#tts-candidates input:checked')].map(node => node.value);
    if (!selected.length) throw new Error('Select at least one voice model.');
    const needsGpu = selected.some(id => ttsCandidates.find(item => item.id === id)?.requires_gpu);
    if (needsGpu) {
      if (!await confirmModal('Arm the shared GPU queue for the selected local voice comparison? It waits for any GPU work already running.', 'Arm TTS experiment', 'Arm and queue')) return;
      await api(`/api/jobs/${current.id}/arm-gpu`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'experiment'})
      });
    }
    $('#build-tts-experiment').disabled = true;
    $('#tts-experiment-state').textContent = 'Planning local comparison…';
    const run = await api(`/api/jobs/${current.id}/tts-experiment`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        start:Number($('#experiment-start').value),
        duration:Number($('#experiment-duration').value),
        candidates:selected
      })
    });
    $('#tts-experiment-state').textContent = `${run.id} queued. Candidates appear in Experiment review (/experiments.html?run=${run.id}) as they finish.`;
  } catch (error) {
    $('#tts-experiment-state').textContent = error.message;
  } finally {
    $('#build-tts-experiment').disabled = false;
  }
}

async function addCurrentToEpisodeQueue() {
  if (!current) return;
  try {
    await save();
    await api('/api/episode-queue/add', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({jobs:[current.id]})
    });
    await refreshEpisodeQueue();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function runEpisodeQueue() {
  try {
    await refreshEpisodeQueue();
    const needsGpu = episodeQueue.items.some(item => item.status === 'queued' && item.requires_gpu);
    if (needsGpu) {
      if (!await confirmModal('Arm the shared GPU once for this reviewed episode batch? Jobs run sequentially and acquire/release their own queue leases.', 'Arm episode queue', 'Arm and run')) return;
      await api('/api/episode-queue/arm-gpu', {method:'POST'});
    }
    await api('/api/episode-queue/run', {method:'POST'});
    await refreshEpisodeQueue();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function stopEpisodeQueue() {
  await api('/api/episode-queue/stop', {method:'POST'});
  await refreshEpisodeQueue();
}

async function clearEpisodeQueue() {
  await api('/api/episode-queue/clear-finished', {method:'POST'});
  await refreshEpisodeQueue();
}

async function applyGlossaryToLines() {
  if (!current || !await confirmModal('Apply the current glossary replacements to every reviewed English line? Save remains reviewable and no synthesis starts.', 'Apply glossary')) return;
  try {
    const update = collectEdits();
    update.apply_glossary = true;
    current = await api(`/api/jobs/${current.id}`, {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(update)
    });
    renderJob();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function action(name) {
  if (!current) return;
  try {
    await save();
    const profile = profiles.find(item => item.id === current.settings.quality_profile);
    if (profile?.requires_gpu) {
      if (!await confirmModal(`Arm the shared GPU for AutoDub ${name}? This is allowed only after image generation is idle.`, 'Arm shared GPU')) return;
      await api(`/api/jobs/${current.id}/arm-gpu`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:name})});
    }
    await api(`/api/jobs/${current.id}/${name}`, {method:'POST'});
    current.status = 'running'; current.stage = name === 'analyze' ? 'starting-analysis' : 'starting-render'; renderJob();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
    await refreshGpuStatus().catch(() => {});
  }
}

async function realignAndRemix() {
  if (!current) return;
  try {
    await save();
    if (!await confirmModal(
      'Use the pinned local forced aligner, reuse the existing English line audio, and build a new realigned review export? The original export is preserved.',
      'Realign source speech',
      'Arm and realign'
    )) return;
    await api(`/api/jobs/${current.id}/arm-gpu`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'realign'})
    });
    await api(`/api/jobs/${current.id}/realign`, {method:'POST'});
    current.status = 'running';
    current.stage = 'aligning-source';
    renderJob();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function adaptDialogue() {
  if (!current) return;
  try {
    await save();
    if (!await confirmModal(
      'Arm the shared GPU and rewrite English lines to fit their speech windows? Meaning is locked (names, numbers, negation, question-form, tone); the original lines are preserved and re-runs re-derive from them.',
      'Adapt dialogue',
      'Arm and adapt'
    )) return;
    await api(`/api/jobs/${current.id}/arm-gpu`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'adapt'})
    });
    await api(`/api/jobs/${current.id}/adapt`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({})
    });
    current.status = 'running';
    current.stage = 'adapting';
    renderJob();
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').classList.remove('hidden');
  }
}

async function importVoice() {
  if (!current) return;
  const file = $('#voice-file').files[0];
  const transcript = $('#voice-transcript').value.trim();
  if (!file || !transcript) { $('#voice-import-state').textContent = 'Choose a reference clip and enter its exact transcript.'; return; }
  const ext = file.name.includes('.') ? '.' + file.name.split('.').pop().toLowerCase() : '';
  const bytes = new TextEncoder().encode(transcript);
  let binary = ''; bytes.forEach(byte => binary += String.fromCharCode(byte));
  $('#voice-add').disabled = true; $('#voice-import-state').textContent = 'Importing locally…';
  try {
    current = await api(`/api/jobs/${current.id}/voice-profile?ext=${encodeURIComponent(ext)}`, {
      method: 'POST',
      headers: {'Content-Type':'application/octet-stream','X-AutoDub-Transcript':btoa(binary),'X-AutoDub-Language':$('#voice-language').value},
      body: file
    });
    $('#voice-file').value = ''; $('#voice-transcript').value = '';
    $('#voice-import-state').textContent = 'Local clone profile imported. Select it for any speaker.';
    renderJob();
  } catch (error) { $('#voice-import-state').textContent = error.message; }
  finally { $('#voice-add').disabled = false; }
}

$('#pick').onclick = () => $('#source').click();
$('#source').onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const ext = file.name.includes('.') ? '.' + file.name.split('.').pop().toLowerCase() : '';
  $('#upload-state').textContent = `Importing ${formatBytes(file.size)} locally…`;
  try {
    current = await api(`/api/jobs?ext=${encodeURIComponent(ext)}&name=${encodeURIComponent(file.name)}`, {method:'POST', headers:{'Content-Type':'application/octet-stream'}, body:file});
    $('#upload-state').textContent = 'Imported. Original filename was not stored.';
    await refreshJobs(); await loadJob(current.id);
  } catch (error) { $('#upload-state').textContent = error.message; }
  event.target.value = '';
};
$('#tab-library').onclick = () => setSidebarTab('library');
$('#tab-activity').onclick = () => setSidebarTab('activity');
$('#song-mark-mode').onclick = toggleSongMarkMode;
document.querySelectorAll('.sort-chip').forEach(chip => chip.onclick = () => {
  sortMode = chip.dataset.sort;
  document.querySelectorAll('.sort-chip').forEach(other => other.classList.toggle('active', other === chip));
  renderLibrary();
});
$('#delete-job').onclick = deleteCurrentJob;
$('#queue-add-current').onclick = addCurrentToEpisodeQueue;
$('#queue-run').onclick = runEpisodeQueue;
$('#queue-stop').onclick = stopEpisodeQueue;
$('#queue-clear').onclick = clearEpisodeQueue;
$('#save').onclick = save;
$('#analyze').onclick = () => action('analyze');
$('#render').onclick = () => action('render');
$('#realign').onclick = realignAndRemix;
$('#adapt').onclick = adaptDialogue;
$('#cancel-job').onclick = cancelJob;
$('#voice-add').onclick = importVoice;
$('#project-close').onclick = () => { $('#project-view').classList.add('hidden'); $('#welcome').classList.remove('hidden'); };
$('#projects-refresh').onclick = refreshProjects;
$('#characters-load').onclick = () => loadCharacters();
$('#characters-auto').onclick = autoApplyCharacters;
$('#characters-prewarm').onclick = () => { if (current) startJobPrewarm(current.id); };
$('#project-prewarm').onclick = () => { if (projectView) startPrewarm(projectView.slug); };
$('#build-experiment').onclick = buildExperiment;
$('#build-tts-experiment').onclick = buildTtsExperiment;
$('#apply-glossary').onclick = applyGlossaryToLines;
$('#quality-profile').onchange = () => {
  const profile = profiles.find(item => item.id === $('#quality-profile').value);
  $('#profile-note').textContent = profile ? `${profile.asr} · ${profile.diarization} · ${profile.tts}` : '';
};
$('#bed').oninput = () => $('#bed-value').textContent = `${Math.round($('#bed').value * 100)}%`;
$('#dialogue').oninput = () => $('#dialogue-value').textContent = `${Math.round($('#dialogue').value * 100)}%`;
$('#tempo').oninput = () => $('#tempo-value').textContent = `${Number($('#tempo').value).toFixed(2)}×`;
$('#mix-policy').onchange = () => {
  const policy = policies.mix.find(item => item.id === $('#mix-policy').value);
  if (!policy) return;
  $('#bed').value = policy.bed_gain;
  $('#dialogue').value = policy.dialogue_gain;
  $('#bed-value').textContent = `${Math.round(policy.bed_gain * 100)}%`;
  $('#dialogue-value').textContent = `${Math.round(policy.dialogue_gain * 100)}%`;
};
$('#timing-policy').onchange = () => {
  const policy = policies.timing.find(item => item.id === $('#timing-policy').value);
  if (!policy) return;
  $('#tempo').value = policy.max_tempo;
  $('#tempo-value').textContent = `${Number(policy.max_tempo).toFixed(2)}×`;
};
// ---- DUBPROJECTS: the whole show as one system -----------------------------------------
// Speakers merge across every analyzed
// episode; likely characters are grouped; one answer maps every member episode at once.
// The per-episode Characters panel stays - this is a grouped LAYER on top, not a swap.
let projectView = null;

async function refreshProjects() {
  // Sidebar redesign: the Library renderer owns the projects list now.
  try { await refreshJobs(); } catch { /* sidebar stays quiet */ }
}

// EVIDENCE PREWARM: first-click ffmpeg cuts made review slow, so the server cuts every
// card's media ahead of the reviewer on EVERY evidence
// surface — the DubProjects board and the per-episode Characters panel. Auto-fires when a
// surface opens; the visible button re-runs it on demand (cached files skip instantly).
const prewarmTimers = {};

function armPrewarm(url, elId, visible) {
  api(url, {method: 'POST'}).catch(() => {});
  if (prewarmTimers[elId]) clearInterval(prewarmTimers[elId]);
  const poll = async () => {
    const el = $(elId);
    if (!el || !visible()) {
      clearInterval(prewarmTimers[elId]); prewarmTimers[elId] = null; return;
    }
    try {
      const s = await api(url);
      if (s.state === 'running' || s.state === 'starting') {
        el.classList.remove('hidden');
        el.textContent = `Preparing clips ahead of you: ${s.done || 0}/${s.total || '…'} ready` +
          (s.errors ? ` (${s.errors} failed)` : '');
      } else {
        if (s.state === 'done') {
          el.textContent = 'All clips cut — plays are instant now.';
          setTimeout(() => el.classList.add('hidden'), 8000);
        } else { el.classList.add('hidden'); }
        clearInterval(prewarmTimers[elId]); prewarmTimers[elId] = null;
      }
    } catch { /* quiet — cosmetic surface */ }
  };
  poll();
  prewarmTimers[elId] = setInterval(poll, 4000);
}

function startPrewarm(slug) {
  armPrewarm(`/api/projects/${slug}/prewarm`, '#prewarm-state',
    () => !$('#project-view').classList.contains('hidden'));
}

function startJobPrewarm(jobId) {
  armPrewarm(`/api/jobs/${jobId}/prewarm-evidence`, '#characters-prewarm-state',
    () => !!(current && current.id === jobId));
}

async function openProject(slug) {
  $('#welcome').classList.add('hidden');
  $('#job-view').classList.add('hidden');
  $('#project-view').classList.remove('hidden');
  $('#project-groups').innerHTML = '<p class="muted">Merging speakers across episodes…</p>';
  try {
    const [tree, view] = await Promise.all([api('/api/projects'), api(`/api/projects/${slug}/characters`)]);
    projectView = view;
    const project = (tree.projects || []).find(p => p.slug === slug);
    $('#project-title').textContent = view.series;
    $('#project-tree').innerHTML = (project?.seasons || []).map(se => `
      <div class="season-block"><b>${esc(se.season)}</b>
        <div class="season-eps">${se.episodes.map(e =>
          `<span class="ep-chip ${e.analyzed ? 'done' : ''}" title="${esc(e.status)} · ${e.assigned}/${e.speakers} named">${esc(e.episode)}</span>`).join('')}</div>
      </div>`).join('');
    renderProjectGroups();
    startPrewarm(slug);
  } catch (error) {
    $('#project-groups').innerHTML = `<p class="muted">${esc(error.message)}</p>`;
  }
}

// CHARACTER ROSTER: one expandable tab per character, sorted by red/amber/green confidence so
// attention goes where it is needed; unknown clips can be reassigned or named. Sections:
// named cast (with pending also-them proposals), unnamed cast worst-first, minor voices
// collapsed. Per-clip verdicts: not-them (cannot-link, remembered), reassign, new.
function renderProjectGroups() {
  const view = projectView;
  const lvlOrder = {red: 0, amber: 1, green: 2, solo: 3};
  const chip = g => {
    const c = g.confidence || {};
    return c.level ? `<span class="conf conf-${c.level}" title="worst voice link: ${c.worst_link ?? 'n/a'} · thinnest solo audio: ${c.min_solo_s}s">● ${c.level}</span>` : '';
  };
  const groups0 = (view.groups || []);
  const face = g => {
    for (const m of g.members) {
      const card = (m.cards || [])[0];
      if (card && card.i != null) return `${BASE}/api/jobs/${m.job_id}/segments/${card.i}/evidence-frame`;
    }
    return null;
  };
  // Reassign menu: named characters AND every other candidate group, so a stray clip can
  // be wired to an unnamed character.
  const assignOpts = gi => `
    ${view.bank.length ? `<optgroup label="Named">${view.bank.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('')}</optgroup>` : ''}
    <optgroup label="Candidates">${groups0.map((g, i) => (i === gi || g.members.length < 2) ? '' :
      `<option value="g:${i}">Candidate ${g.group} (${g.episodes.length} ep)</option>`).join('')}</optgroup>
    <option value="__new__">a new character…</option>`;
  const bankOpts = view.bank.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  const memberCard = (g, gi, m, mi) => {
    const card = (m.cards || [])[0];
    const media = card && card.i != null ? `
        <div class="card-media" data-job="${esc(m.job_id)}" data-seg="${card.i}"><img loading="lazy" src="${BASE}/api/jobs/${m.job_id}/segments/${card.i}/evidence-frame" alt="frame"><span class="card-play">▶</span></div>
        <audio controls preload="none" src="${BASE}/api/jobs/${m.job_id}/segments/${card.i}/evidence-audio"></audio>` : '<p class="muted">no solo clip long enough</p>';
    return `<div class="evidence-card">
      ${media}
      <small><b>${esc(m.episode)}</b>${m.cohesion < 1 ? ` · link ${m.cohesion}` : ''} · ${esc((((card || {}).translation || (card || {}).text) || '').slice(0, 48))}</small>
      <div class="member-actions">
        <button class="mini" data-mlang title="Hear the official ENGLISH dub track for this clip (dual-audio sources) — you know these voices; analysis and cloning stay on the Japanese track">EN dub</button>
        ${g.members.length > 1 ? `<button class="mini ghost" data-mnot="${gi}:${mi}" title="Only if YOU disagree: split this clip out as a different person; the split is remembered">✗ different person?</button>` : ''}
        <select class="mini" data-massign="${gi}:${mi}" title="Reassign just this clip"><option value="">this clip is…</option>${assignOpts(gi)}</select>
      </div>
    </div>`;
  };
  const groupBody = (g, gi, isAttachment) => `
      ${g.members.length > 1 ? `<p class="muted roster-hint">The system thinks all ${g.members.length} clips below are the SAME person${
        (g.confidence || {}).level ? ` (confidence: ${g.confidence.level})` : ''}. Listen, then name the whole group — the buttons under each clip are only for correcting a clip that doesn't belong.</p>` : ''}
      <div class="evidence-cards">${g.members.map((m, mi) => memberCard(g, gi, m, mi)).join('')}</div>
      <div class="character-actions">
        ${isAttachment ? `<button class="secondary" data-gsame="${gi}">Yes — same person (all ${g.members.length})</button>
          <button class="secondary danger" data-reject="${gi}" title="Not that character — this proposal never comes back">Not ${esc(g.bank_candidate.name)}</button>` : ''}
        <select data-gpick="${gi}"><option value="">${isAttachment ? '…or who is it really?' : 'this is someone already named…'}</option>${bankOpts}</select>
        <input data-gname="${gi}" type="text" placeholder="new character's name">
        <button class="secondary" data-gnew="${gi}">Name → all episodes</button>
        <button class="secondary" data-gsong="${gi}" title="Opening/ending song — never dubbed; dropped from this board">Mark as intro/outro</button>
        <button class="secondary" data-gskip="${gi}">Skip</button>
      </div>`;
  // Cast-pack hint: "likely: <name>" from episode-signature matching
  const hintChip = o => {
    const h = ((o || {}).cast_hints || [])[0];
    if (!h) return '';
    const alt = (o.cast_hints || [])[1];
    const tip = `${esc(h.reason)}${h.gender_age ? ' · ' + esc(h.gender_age) : ''}${alt ? ' · or: ' + esc(alt.name) : ''}`;
    // A weak hint = "no character's episode pattern really fits; this name merely
    // APPEARS in these episodes" (usually the lead, who is in every episode). Weak hints
    // read like real claims, so they now say what they are.
    if (h.weak) return `<span class="cast-hint weak" title="No character's episode pattern fits this group well. Closest by mere presence: ${tip}">no strong name match</span>`;
    return `<span class="cast-hint" title="episode-pattern match (not voice, not image): ${tip}">likely: ${esc(h.name)}</span>`;
  };
  const acc = (g, gi, title, isAttachment, open) => `
    <details class="roster-acc" id="acc-${gi}"${open ? ' open' : ''}><summary>${chip(g)} <b>${title}</b> ${hintChip(g)}
      <span class="muted">${g.members.length} clip(s) · ${g.episodes.length} ep: ${g.episodes.slice(0, 6).map(esc).join(' ')}${g.episodes.length > 6 ? '…' : ''}</span></summary>
      ${groupBody(g, gi, isAttachment)}
    </details>`;

  const groups = (view.groups || []).map((g, gi) => ({g, gi}));
  const attached = groups.filter(x => x.g.bank_candidate && x.g.bank_candidate.id);
  const unnamed = groups.filter(x => !(x.g.bank_candidate && x.g.bank_candidate.id) && x.g.members.length > 1)
    .sort((a, b) => (lvlOrder[a.g.confidence.level] ?? 9) - (lvlOrder[b.g.confidence.level] ?? 9));
  const solos = groups.filter(x => !(x.g.bank_candidate && x.g.bank_candidate.id) && x.g.members.length === 1);
  const toSort = solos.filter(x => x.g.reviewer_split);      // clips the reviewer split out
  const minors = solos.filter(x => !x.g.reviewer_split);

  const namedSection = view.bank.length ? `<div class="roster-section">NAMED CAST</div>` +
    view.bank.map(c => {
      const pend = attached.filter(x => x.g.bank_candidate.id === c.id);
      return `<details class="roster-acc named" id="acc-char-${c.id}"${pend.length ? ' open' : ''}><summary><span class="conf conf-green">✓</span> <b>${esc(c.name)}</b>
          <span class="muted">${pend.length ? `${pend.length} “also them?” to answer` : 'no open questions'}</span></summary>
        ${pend.length ? pend.map(x => `<div class="attach-block"><p class="muted">Is this also ${esc(c.name)}? (match ${(x.g.bank_candidate.score ?? 0).toFixed(2)})</p>${groupBody(x.g, x.gi, true)}</div>`).join('')
                      : '<p class="muted">Nothing pending for this character.</p>'}
      </details>`;
    }).join('') : '';

  const unnamedSection = unnamed.length ? `<div class="roster-section">UNNAMED CAST — worst first, name the reds carefully, greens are safe one-clicks</div>` +
    unnamed.map((x, i) => acc(x.g, x.gi, `Character candidate ${x.g.group}`, false, i === 0)).join('') : '';

  // Near-miss suggestions among minors (real characters may be split into solos by the
  // 0.95 bar). Members shown here are hidden from the minors pile below.
  const sugs = view.minor_suggestions || [];
  const sugKey = m => `${m.job_id}::${m.speaker}`;
  const sugMembers = new Set(sugs.flatMap(s => s.members.map(sugKey)));
  const minorsShown = minors.filter(x => !sugMembers.has(sugKey(x.g.members[0])));
  const giOf = m => groups0.findIndex(g => g.members.length === 1 &&
    g.members[0].job_id === m.job_id && g.members[0].speaker === m.speaker);
  const sugSection = sugs.length ? `<div class="roster-section">LIKELY SAME PERSON — minor voices that nearly matched (0.90–0.95 band); merge or dismiss each</div>` +
    sugs.map((s, si) => `<details class="roster-acc" ${si === 0 ? 'open' : ''}><summary><span class="conf conf-amber">● ${s.score_min}–${s.score_max}</span>
        <b>Possible character across ${s.episodes.map(esc).join(' ')}</b> ${hintChip(s)}
        <span class="muted">${s.members.length} minor voices</span></summary>
      <div class="evidence-cards">${s.members.map(m => memberCard(groups0[giOf(m)] || {members: [m]}, giOf(m), m, 0)).join('')}</div>
      <div class="character-actions">
        <button class="secondary" data-sug-merge="${si}">Same person — merge ${s.members.length}</button>
        <button class="secondary" data-sug-not="${si}">Not the same — dismiss</button>
      </div>
    </details>`).join('') : '';

  const toSortSection = toSort.length ? `<div class="roster-section">TO SORT — clips you split out; give each one a home (reassign or name)</div>` +
    toSort.map(x => acc(x.g, x.gi, `${esc(x.g.members[0].episode)} — split by you`, false, true)).join('') : '';

  const minorSection = minorsShown.length ? `<details class="roster-acc minors"><summary><b>MINOR VOICES</b>
      <span class="muted">${minorsShown.length} one-scene speakers — name only the ones you care about</span></summary>
      ${minorsShown.map(x => acc(x.g, x.gi, `${esc(x.g.members[0].episode)} voice`, false, false)).join('')}
    </details>` : '';

  // CAST MAP: one face tile per named character + unnamed candidate; click jumps to the tab.
  const castStrip = (view.bank.length || unnamed.length) ? `<div class="cast-strip">
    ${view.bank.map(c => `<button class="cast-tile" data-jump="char-${c.id}" title="${esc(c.name)}">
      <span class="cast-face letter">${esc((c.name || '?')[0].toUpperCase())}</span>
      <small>✓ ${esc(c.name.slice(0, 10))}</small></button>`).join('')}
    ${unnamed.map(x => {
      const f = face(x.g);
      return `<button class="cast-tile" data-jump="${x.gi}" title="Candidate ${x.g.group} — ${x.g.members.length} clips, ${x.g.episodes.length} episodes">
        ${f ? `<img class="cast-face" loading="lazy" src="${f}" alt="">` : '<span class="cast-face letter">?</span>'}
        <small><span class="conf-dot ${(x.g.confidence || {}).level || ''}"></span>C${x.g.group}</small></button>`;
    }).join('')}
  </div>` : '';

  $('#project-groups').innerHTML = castStrip + namedSection + unnamedSection + sugSection + toSortSection + minorSection ||
    '<p class="muted">Nothing to review — analyze some episodes first.</p>';
  const jumpTo = id => {
    const target = document.querySelector(`#acc-${id}`);
    if (target) { target.open = true; target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
  };
  document.querySelectorAll('.cast-tile').forEach(t => t.onclick = () => {
    const img = t.querySelector('img');
    if (!img) { jumpTo(t.dataset.jump); return; }   // named letter tiles: nothing to zoom
    // Tiles are too small to recognize faces — click shows the FULL frame first
    const overlay = document.createElement('div');
    overlay.className = 'cast-zoom';
    overlay.innerHTML = `<img src="${img.src}" alt=""><div class="zoom-actions">
      <button class="accent zoom-open">Open this character</button>
      <button class="secondary zoom-close">Close</button></div>`;
    document.body.appendChild(overlay);
    const close = () => overlay.remove();
    overlay.onclick = e => { if (e.target === overlay) close(); };
    overlay.querySelector('.zoom-close').onclick = close;
    overlay.querySelector('.zoom-open').onclick = () => { close(); jumpTo(t.dataset.jump); };
  });
  const post = async (path, body) => {
    try {
      await api(`/api/projects/${projectView.slug}/characters/${path}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)});
      await openProject(projectView.slug);
      await refreshProjects();
    } catch (error) { alert(error.message); }
  };
  const answer = (gi, payload, only) => {
    const g = projectView.groups[gi];
    const members = (only || g.members).map(m => ({job_id: m.job_id, speaker: m.speaker}));
    return post('answer', {members, ...payload});
  };
  document.querySelectorAll('[data-mnot]').forEach(b => b.onclick = () => {
    const [gi, mi] = b.dataset.mnot.split(':').map(Number);
    const g = projectView.groups[gi];
    const m = g.members[mi];
    post('separate', {member: {job_id: m.job_id, speaker: m.speaker},
      others: g.members.filter((_, i) => i !== mi).map(o => ({job_id: o.job_id, speaker: o.speaker}))});
  });
  document.querySelectorAll('[data-massign]').forEach(sel => sel.onchange = () => {
    if (!sel.value) return;
    const [gi, mi] = sel.dataset.massign.split(':').map(Number);
    const g = projectView.groups[gi];
    const m = g.members[mi];
    if (sel.value === '__new__') {
      const name = prompt('Name for this new character:');
      if (!name || !name.trim()) { sel.value = ''; return; }
      answer(gi, {answer: 'different', new_name: name.trim()}, [m]);
    } else if (sel.value.startsWith('g:')) {
      // wire this clip into another unnamed candidate group (must-link, remembered)
      const tg = projectView.groups[+sel.value.slice(2)];
      const anchor = tg.members[0];
      post('unite', {member: {job_id: m.job_id, speaker: m.speaker},
                     target: {job_id: anchor.job_id, speaker: anchor.speaker}});
    } else {
      answer(gi, {answer: 'same', character_id: sel.value}, [m]);
    }
  });
  document.querySelectorAll('[data-reject]').forEach(b => b.onclick = () => {
    const g = projectView.groups[+b.dataset.reject];
    post('reject', {character_id: g.bank_candidate.id,
      members: g.members.map(m => ({job_id: m.job_id, speaker: m.speaker}))});
  });
  document.querySelectorAll('[data-sug-merge]').forEach(b => b.onclick = async () => {
    const s = projectView.minor_suggestions[+b.dataset.sugMerge];
    try {
      for (let i = 1; i < s.members.length; i++) {
        await api(`/api/projects/${projectView.slug}/characters/unite`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({member: {job_id: s.members[i].job_id, speaker: s.members[i].speaker},
                                target: {job_id: s.members[0].job_id, speaker: s.members[0].speaker}})});
      }
      await openProject(projectView.slug);
    } catch (error) { alert(error.message); }
  });
  document.querySelectorAll('[data-sug-not]').forEach(b => b.onclick = async () => {
    const s = projectView.minor_suggestions[+b.dataset.sugNot];
    try {
      for (let i = 0; i < s.members.length; i++) {
        await api(`/api/projects/${projectView.slug}/characters/separate`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({member: {job_id: s.members[i].job_id, speaker: s.members[i].speaker},
            others: s.members.filter((_, k) => k !== i).map(o => ({job_id: o.job_id, speaker: o.speaker}))})});
      }
      await openProject(projectView.slug);
    } catch (error) { alert(error.message); }
  });
  document.querySelectorAll('[data-gsame]').forEach(b => b.onclick = () =>
    answer(+b.dataset.gsame, {answer: 'same', character_id: projectView.groups[+b.dataset.gsame].bank_candidate.id}));
  document.querySelectorAll('[data-gpick]').forEach(sel => sel.onchange = () =>
    sel.value && answer(+sel.dataset.gpick, {answer: 'same', character_id: sel.value}));
  document.querySelectorAll('[data-gnew]').forEach(b => b.onclick = () => {
    const name = document.querySelector(`[data-gname="${b.dataset.gnew}"]`).value.trim();
    if (!name) { alert('Give the character a name first.'); return; }
    answer(+b.dataset.gnew, {answer: 'different', new_name: name});
  });
  document.querySelectorAll('[data-gskip]').forEach(b => b.onclick = () =>
    answer(+b.dataset.gskip, {answer: 'skip'}));
  document.querySelectorAll('[data-gsong]').forEach(b => b.onclick = async () => {
    const g = projectView.groups[+b.dataset.gsong];
    try {
      await api(`/api/projects/${projectView.slug}/characters/mark-song`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({members: g.members.map(m => ({job_id: m.job_id, speaker: m.speaker}))})});
      await openProject(projectView.slug);
      await refreshJobs().catch(() => {});
    } catch (error) { alert(error.message); }
  });
  // Per-clip EN-dub toggle: plays the source's official English track when present, which
  // can be easier to recognize; Japanese stays the analysis/cloning source.
  document.querySelectorAll('[data-mlang]').forEach(b => b.onclick = () => {
    const card = b.closest('.evidence-card');
    if (!card) return;
    const toEng = card.dataset.lang !== 'eng';
    card.dataset.lang = toEng ? 'eng' : '';
    b.textContent = toEng ? 'EN dub ✓' : 'EN dub';
    b.title = toEng ? 'Playing the official English dub track — click to go back to Japanese'
                    : b.title;
    for (const media of card.querySelectorAll('audio, video')) {
      const base = media.src.split('?')[0];
      media.src = toEng ? `${base}?lang=eng` : base;
    }
  });
  document.querySelectorAll('#project-groups .card-media').forEach(box => box.onclick = () => {
    const lang = (box.closest('.evidence-card') || {dataset: {}}).dataset.lang;
    const url = `${BASE}/api/jobs/${box.dataset.job}/segments/${box.dataset.seg}/evidence-video${lang ? `?lang=${lang}` : ''}`;
    box.outerHTML = `<div class="card-video"><video controls autoplay playsinline src="${url}"></video>
      <button class="card-close">✕</button><small class="muted card-note">first open cuts the clip…</small></div>`;
    const wrap = document.querySelector('.card-video video[src="' + url + '"]').parentNode;
    wrap.querySelector('video').oncanplay = () => { const n = wrap.querySelector('.card-note'); if (n) n.remove(); };
    wrap.querySelector('.card-close').onclick = () => openProject(projectView.slug);
  });
}

// ---- CHARACTERS: series voice bank + evidence cards ----------------------------------
// The reviewer answers one question per card: same person or not. Media comes from the
// lazy evidence endpoints; this page only hands over segment indices.
let charactersView = null;

async function loadCharacters() {
  if (!current) return;
  const body = $('#characters-body');
  body.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const series = $('#series-name').value.trim();
    charactersView = await api(`/api/jobs/${current.id}/characters${series ? `?series=${encodeURIComponent(series)}` : ''}`);
    $('#series-name').value = charactersView.series;
    renderCharacters();
    startJobPrewarm(current.id);
  } catch (error) {
    body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
  }
}

function renderCharacters() {
  const view = charactersView;
  const zoneLabel = {clear: 'match', ask: 'is this them?', new: 'new character?', quarantined: 'embedder changed — needs refresh'};
  const bankRow = view.bank.length
    ? `<div class="bank-row">In this series: ${view.bank.map(c => `<span class="bank-chip" title="${c.has_reference ? 'has a voice reference' : 'no reference yet'}">${esc(c.name)}</span>`).join(' ')}</div>`
    : '<div class="bank-row muted">Bank is empty — name this episode\'s speakers to seed it.</div>';
  $('#characters-body').innerHTML = bankRow + view.speakers.map(sp => {
    const m = sp.match;
    const assigned = sp.assigned_character
      ? `<span class="zone zone-clear">✓ ${esc(view.bank.find(c => c.id === sp.assigned_character)?.name || sp.assigned_character)}</span>`
      : m ? `<span class="zone zone-${m.zone}">${esc(zoneLabel[m.zone] || m.zone)}${m.character_name ? ` → ${esc(m.character_name)} (${(m.score ?? 0).toFixed(2)})` : ''}</span>`
          : '<span class="zone zone-new">no embedding evidence</span>';
    const cards = (sp.cards || []).map(card => `
      <div class="evidence-card">
        ${card.i != null ? `<div class="card-media" data-seg="${card.i}" title="Click for a short video clip with sound — see who is actually talking">
          <img loading="lazy" src="${BASE}/api/jobs/${current.id}/segments/${card.i}/evidence-frame" alt="frame">
          <span class="card-play">▶</span></div>
        <audio controls preload="none" src="${BASE}/api/jobs/${current.id}/segments/${card.i}/evidence-audio"></audio>` : '<p class="muted">no media</p>'}
        <small>${fmtTime(card.start)} · ${esc((card.text || '').slice(0, 60))}</small>
      </div>`).join('');
    const askButtons = sp.assigned_character ? '' : `
      <div class="character-actions">
        ${m && m.character_id ? `<button class="secondary" data-same="${esc(sp.speaker)}" data-char="${m.character_id}">Same — it's ${esc(m.character_name)}</button>` : ''}
        <select data-pick="${esc(sp.speaker)}"><option value="">…or pick who this is</option>
          ${view.bank.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('')}</select>
        <input data-name="${esc(sp.speaker)}" type="text" placeholder="new character's name">
        <button class="secondary" data-new="${esc(sp.speaker)}">Create character</button>
        <button class="secondary" data-skip="${esc(sp.speaker)}">Skip</button>
      </div>`;
    return `<div class="character-block"><div class="character-head"><b>${esc(sp.speaker)}</b> ${assigned}</div>
      <div class="evidence-cards">${cards || '<p class="muted">no solo clips long enough for a card</p>'}</div>${askButtons}</div>`;
  }).join('');
  const answer = async (payload) => {
    try {
      await api(`/api/jobs/${current.id}/characters/answer`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({series: charactersView.series, ...payload})});
      await loadCharacters();
    } catch (error) { alert(error.message); }
  };
  document.querySelectorAll('.card-media').forEach(box => box.onclick = () => {
    const seg = box.dataset.seg;
    const url = `${BASE}/api/jobs/${current.id}/segments/${seg}/evidence-video`;
    box.outerHTML = `<div class="card-video">
      <video controls autoplay playsinline src="${url}"></video>
      <button class="card-close" title="Back to the screenshot">✕</button>
      <small class="muted card-note">first open cuts the clip — a few seconds…</small></div>`;
    const wrap = document.querySelector('.card-video video[src="' + url + '"]').parentNode;
    wrap.querySelector('video').oncanplay = () => { const n = wrap.querySelector('.card-note'); if (n) n.remove(); };
    wrap.querySelector('.card-close').onclick = () => loadCharacters();
  });
  document.querySelectorAll('[data-same]').forEach(b => b.onclick = () =>
    answer({speaker: b.dataset.same, answer: 'same', character_id: b.dataset.char}));
  document.querySelectorAll('[data-pick]').forEach(sel => sel.onchange = () =>
    sel.value && answer({speaker: sel.dataset.pick, answer: 'same', character_id: sel.value}));
  document.querySelectorAll('[data-new]').forEach(b => b.onclick = () => {
    const name = document.querySelector(`[data-name="${b.dataset.new}"]`).value.trim();
    if (!name) { alert('Give the new character a name first.'); return; }
    answer({speaker: b.dataset.new, answer: 'different', new_name: name});
  });
  document.querySelectorAll('[data-skip]').forEach(b => b.onclick = () =>
    answer({speaker: b.dataset.skip, answer: 'skip'}));
}

async function autoApplyCharacters() {
  if (!current) return;
  try {
    const result = await api(`/api/jobs/${current.id}/characters/auto-apply`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({series: $('#series-name').value.trim()})});
    await loadCharacters();
    $('#characters-body').insertAdjacentHTML('afterbegin',
      `<p class="muted">${result.count ? `Applied ${result.count} clear match(es) — each is in the decision log.` : 'No clear-zone matches to apply.'}</p>`);
  } catch (error) { alert(error.message); }
}

// Themed confirm modal — replaces the native browser confirm() so dialogs match the app.
function confirmModal(message, title = 'Confirm', okLabel = 'Confirm') {
  return new Promise(resolve => {
    const overlay = $('#confirm-overlay'), ok = $('#confirm-ok'), cancel = $('#confirm-cancel');
    $('#confirm-title').textContent = title;
    $('#confirm-body').textContent = message;
    ok.textContent = okLabel;
    overlay.classList.remove('hidden');
    ok.focus();
    const done = (result) => {
      overlay.classList.add('hidden');
      ok.onclick = cancel.onclick = overlay.onclick = null;
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    const onKey = (e) => { if (e.key === 'Escape') done(false); else if (e.key === 'Enter') done(true); };
    ok.onclick = () => done(true);
    cancel.onclick = () => done(false);
    overlay.onclick = (e) => { if (e.target === overlay) done(false); };
    document.addEventListener('keydown', onKey);
  });
}
function formatBytes(bytes){if(!bytes)return'0 B';const units=['B','KB','MB','GB','TB'];const i=Math.min(Math.floor(Math.log(bytes)/Math.log(1024)),4);return`${(bytes/1024**i).toFixed(i?1:0)} ${units[i]}`}
function fmtTime(sec){const m=Math.floor(sec/60);return`${m}:${(sec-m*60).toFixed(2).padStart(5,'0')}`}
Promise.all([loadProfiles(), loadPolicies(), loadTtsCandidates(), refreshGpuStatus(), refreshEpisodeQueue()]).then(() => refreshJobs(true)).catch(error => $('#projects').innerHTML = `<div class="empty-small">${esc(error.message)}</div>`);
// Keep the sidebar live even when no job page is open — Activity is a cross-project
// watch surface, so running work from any show updates without a manual refresh.
setInterval(() => {
  if (document.hidden) return;
  refreshJobs().catch(() => {});
  refreshEpisodeQueue().catch(() => {});
}, 8000);
