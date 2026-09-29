/* Casting studio — its own page, staged workflow:
   step 1 clean groups (every clip, isolated audio, EN caption, reassign) ->
   step 2 merge & name (compare drawer, simultaneous-speech verdict) ->
   step 3 voices (demos) -> step 4 lock. Draft names/voices + checklist persist
   server-side via PATCH studio_progress; cast semantics are the shipped ones
   (apply-cast, merge, reject-merge, demos). */
'use strict';

const $ = sel => document.querySelector(sel);
const BASE = '';
const JOB_ID = new URLSearchParams(location.search).get('job') || '';

let job = null;             // public_job
let charView = null;        // /characters payload (cast_names, bank)
let cast = {};              // {label: {character, voice}} — the working draft
let cleanSet = new Set();   // step-1 verified groups (persisted server-side)
let dismissedPairs = new Set();
let step = 1;
let sortMode = 'clusters';
let expandedAll = new Set();   // labels showing every clip (default: first 12)
let compare = null;         // {key, side, clips:{a,b}, clipIndex:{a,b}}
let clipState = null;       // video popup {index, label, pad}
let playingIndex = null;    // segment index of the playing line audio (survives re-render)
let lockProblems = [];      // step-4 blockers, for the topbar button's feedback
let warnedSaveFailure = false;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g,
    ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
}

async function api(path, options) {
  const response = await fetch(BASE + path, options);
  if (!response.ok) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).error || detail; } catch (error) { /* raw status */ }
    throw new Error(detail);
  }
  return response.json();
}

function toast(message, isError = false) {
  const box = $('#toast');
  box.textContent = message;
  box.className = isError ? 'error' : '';
  clearTimeout(box._timer);
  box._timer = setTimeout(() => box.classList.add('hidden'), isError ? 6000 : 3000);
}

function confirmModal(title, text, okLabel) {
  return new Promise(resolve => {
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-ok').textContent = okLabel;
    $('#confirm-overlay').classList.remove('hidden');
    const done = value => { $('#confirm-overlay').classList.add('hidden'); resolve(value); };
    $('#confirm-ok').onclick = () => done(true);
    $('#confirm-cancel').onclick = () => done(false);
  });
}

/* ---------- data derivations ---------- */

function dialogueSegments(label) {
  return (job.segments || []).filter(s =>
    String(s.speaker || '') === label && !s.song_skip);
}

function songCount(label) {
  return (job.segments || []).filter(s =>
    String(s.speaker || '') === label && s.song_skip).length;
}

function labelCounts() {
  const counts = {};
  (job.segments || []).forEach(s => {
    if (s.song_skip) return;
    const label = String(s.speaker || 'speaker-01');
    counts[label] = (counts[label] || 0) + 1;
  });
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}

function tierOf(lines) { return lines >= 40 ? 'MAIN' : lines >= 10 ? 'MINOR' : 'BIT'; }

function isMuted(label) {
  return String(cast[label]?.voice || '').startsWith('mute');
}

function characterNameFor(label) {
  // precedence: locked cast name -> resolved bank ID -> '' (speaker_characters holds
  // bank character IDs by contract — never show or resubmit a raw ch_xxx ID)
  const locked = job.cast_lock?.cast?.[label]?.character;
  if (locked) return String(locked);
  const id = job.speaker_characters?.[label];
  if (id && charView) {
    const hit = (charView.bank || []).find(item => item.id === id);
    if (hit) return String(hit.name || '');
  }
  return '';
}

function seedCast() {
  const draft = (job.studio_progress || {}).draft_cast || {};
  labelCounts().forEach(([label]) => {
    if (cast[label]) return;
    cast[label] = {
      character: draft[label]?.character ?? characterNameFor(label),
      voice: draft[label]?.voice || job.speaker_voices?.[label] || `qwen-auto:${label}`,
    };
  });
}

function sortedLabels() {
  const labels = labelCounts();
  if (sortMode === 'lines') return labels;
  if (sortMode === 'label') return [...labels].sort((a, b) => a[0].localeCompare(b[0]));
  const parent = {};
  const find = x => parent[x] === x ? x : (parent[x] = find(parent[x]));
  labels.forEach(([label]) => { parent[label] = label; });
  (job.speaker_duplicate_suggestions || [])
    .filter(pair => Number(pair.cosine_similarity || 0) >= 0.88)
    .forEach(pair => {
      const a = String(pair.speaker_a), b = String(pair.speaker_b);
      if (parent[a] !== undefined && parent[b] !== undefined) parent[find(a)] = find(b);
    });
  const clusters = {};
  labels.forEach(entry => { const root = find(entry[0]); (clusters[root] = clusters[root] || []).push(entry); });
  return Object.values(clusters)
    .sort((x, y) => Math.max(...y.map(m => m[1])) - Math.max(...x.map(m => m[1])))
    .flat();
}

function rejectedPair(a, b) {
  const key = [a, b].sort();
  return (job.speaker_merge_rejections || []).some(r =>
    String(r[0]) === key[0] && String(r[1]) === key[1]);
}

function pairKeyOf(pair) { return [pair.speaker_a, pair.speaker_b].sort().join('|'); }

function openPairs() {
  const live = new Set(labelCounts().map(([label]) => label));
  return (job.speaker_duplicate_suggestions || [])
    .filter(pair => Number(pair.cosine_similarity || 0) >= 0.88)
    .filter(pair => live.has(String(pair.speaker_a)) && live.has(String(pair.speaker_b)))
    .filter(pair => !rejectedPair(String(pair.speaker_a), String(pair.speaker_b)))
    .filter(pair => !dismissedPairs.has(pairKeyOf(pair)));
}

function voiceOptions(label) {
  const reference = job.speaker_references?.[label];
  const quality = reference?.quality;
  const options = [];
  if (reference) {
    const clean = quality ? quality.clean !== false : false;
    options.push({
      voice: `qwen-auto:${label}`, label: 'Own voice (clone + transcript)',
      reason: clean ? `clean solo reference · conf ${(quality?.confidence ?? 1).toFixed(2)}`
                    : 'NO CLEAN WINDOW — audition carefully or pick another voice',
      amber: !clean, disabled: !(reference.text || '').trim(),
    });
    options.push({
      voice: `qwen-auto-xv:${label}`, label: 'Own voice (embedding only)',
      reason: 'less source-accent bleed, weaker identity', amber: !clean,
    });
  }
  (job.available_voices || []).forEach(voice => {
    if (voice.startsWith('bank:')) {
      options.push({voice, label: job.voice_labels?.[voice] || voice,
                    reason: 'series bank voice (consistent across episodes)'});
    }
  });
  options.push({voice: 'mute:', label: 'Mute (no dub)',
                reason: 'phantom or noise label — synthesize nothing'});
  return options;
}

function statusOf(label) {
  if (isMuted(label)) return {dot: 'faint', text: 'muted'};
  const named = (cast[label]?.character || '').trim();
  const inPair = openPairs().some(pair =>
    String(pair.speaker_a) === label || String(pair.speaker_b) === label);
  if (named && !inPair) return {dot: 'ok', text: 'cast'};
  if (inPair) return {dot: 'warn', text: 'reviewing'};
  return {dot: 'faint', text: 'unnamed'};
}

/* ---------- shared line audio ---------- */

function playLine(index, button) {
  const audio = $('#line-audio');
  if (playingIndex === index && !audio.paused) {
    audio.pause();
    button.classList.remove('playing');
    playingIndex = null;
    return;
  }
  document.querySelectorAll('.clip-audio.playing, .clip-play.playing')
    .forEach(node => node.classList.remove('playing'));
  playingIndex = index;
  button.classList.add('playing');
  audio.src = `${BASE}/api/jobs/${JOB_ID}/segments/${index}/source`;
  audio.onended = () => {
    document.querySelectorAll(`[data-line-audio="${index}"]`).forEach(n => n.classList.remove('playing'));
    if (playingIndex === index) playingIndex = null;
  };
  audio.play().catch(() => toast('Could not play line audio', true));
}

/* ---------- rendering ---------- */

const STEP_HINTS = {
  1: 'Step 1 — walk each group: play clips (the ▶ Audio pill is ONLY that line\'s voice), read the English caption, and send any clip that belongs to someone else to the right speaker. Mark each group clean when it\'s one person.',
  2: 'Step 2 — settle the purple pairs: click a chip to open the compare drawer (space flips A/B), then name each character. SAME merges; DIFFERENT is remembered.',
  3: 'Step 3 — pick voices. Render demos once, then audition calm/emphatic per candidate. Amber notes mean listen carefully.',
  4: 'Step 4 — review the final cast and lock it. Locking rewrites voices, purges stale audio, and enrolls characters in the series bank.',
};

function render() {
  const focused = document.activeElement;
  const focusName = focused?.dataset?.name;   // restore the caret if a rebuild lands mid-typing
  document.body.dataset.step = String(step);
  document.querySelectorAll('.step-tab').forEach(tab => {
    tab.classList.toggle('active', Number(tab.dataset.step) === step);
  });
  $('#step-hint').textContent = STEP_HINTS[step];
  $('#next-step').textContent = step < 4 ? 'Next step →' : 'Lock cast';
  seedCast();
  updateBadges();
  const labels = sortedLabels();
  if (step === 4) { renderLockStep(labels); return; }
  $('#rows').innerHTML = labels.map(([label, lines]) => groupCard(label, lines)).join('')
    || '<p class="muted pad">No speakers found.</p>';
  bindRows();
  if (playingIndex !== null) {
    document.querySelectorAll(`[data-line-audio="${playingIndex}"]`)
      .forEach(node => node.classList.add('playing'));
  }
  if (focusName) {
    const again = document.querySelector(`[data-name="${CSS.escape(focusName)}"]`);
    if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
  }
}

function updateBadges() {
  const labels = labelCounts();
  const total = labels.length;
  const clean = labels.filter(([label]) => cleanSet.has(label)).length;
  const pairs = openPairs().length;
  const named = labels.filter(([label]) => (cast[label]?.character || '').trim() || isMuted(label)).length;
  const withDemos = job.demo_manifest && !job.demo_manifest.error;
  $('#badge-1').textContent = `${clean}/${total}`;
  $('#badge-2').textContent = pairs ? `${pairs} open` : '✓';
  $('#badge-3').textContent = withDemos ? 'demos ✓' : '';
  $('#badge-4').textContent = `${named}/${total}`;
  document.querySelectorAll('.step-tab').forEach(tab => {
    const n = Number(tab.dataset.step);
    tab.classList.toggle('done',
      (n === 1 && clean === total && total > 0) ||
      (n === 2 && pairs === 0 && named === total && total > 0));
  });
}

function groupCard(label, lines) {
  const muted = isMuted(label);
  const reference = job.speaker_references?.[label];
  const noisy = reference?.quality && reference.quality.clean === false;
  const songs = songCount(label);
  const status = statusOf(label);
  const pairing = compare && compare.key.split('|').includes(label);
  const head = `
    <div class="group-head">
      <span class="speaker-id"><b>${esc(label)}</b>
        <span class="tier ${tierOf(lines) === 'MAIN' ? 'tier-main' : ''}">${tierOf(lines)}</span></span>
      <span class="speaker-sub">${lines} lines${songs ? ` · +${songs} song (skipped)` : ''}${noisy ? ' · <span class="warn">noisy ref</span>' : ''}${muted ? ' · MUTED' : ''}</span>
      <button class="clean-toggle only-1 ${cleanSet.has(label) ? 'on' : ''}" data-clean="${esc(label)}">${cleanSet.has(label) ? '✓ group is clean' : 'mark group clean'}</button>
      <span class="only-2 matches">${matchChips(label)}</span>
      <input class="name-input only-2 ${(cast[label].character || '').trim() ? 'named' : ''}" list="cast-names"
             data-name="${esc(label)}" value="${esc(cast[label].character)}" placeholder="Character name">
      <span class="only-3 fw-600">${esc((cast[label].character || '').trim() || '(unnamed)')}</span>
      <span class="spacer"></span>
      <span class="status-cell" data-status="${esc(label)}"><span class="status-dot dot-${status.dot}"></span>${status.text}</span>
    </div>
    <div class="voice-cell only-3">${voiceCell(label, muted)}</div>
    <div class="only-1">${clipStrip(label)}</div>`;
  return `<div class="group-card ${muted ? 'is-muted' : ''} ${pairing ? 'pairing' : ''}" data-label="${esc(label)}">${head}</div>`;
}

function matchChips(label) {
  return openPairs()
    .filter(pair => String(pair.speaker_a) === label || String(pair.speaker_b) === label)
    .map(pair => {
      const other = String(pair.speaker_a) === label ? pair.speaker_b : pair.speaker_a;
      const key = pairKeyOf(pair);
      const isOpen = compare && compare.key === key;
      return `<button class="match-chip ${isOpen ? 'open' : ''}" data-pair="${esc(key)}">${isOpen ? 'comparing ↓' : `${esc(other)} · ${Number(pair.cosine_similarity).toFixed(2).slice(1)}`}</button>`;
    }).join('') || '<span class="faint small">no matches</span>';
}

function voiceCell(label, muted) {
  const options = voiceOptions(label);
  const chosen = options.find(item => item.voice === cast[label].voice);
  const demoSet = job.demo_manifest?.[label]?.[cast[label].voice];
  return `
    <select class="voice-pick" data-voice="${esc(label)}">
      ${options.map(item => `<option value="${esc(item.voice)}" ${item.voice === cast[label].voice ? 'selected' : ''} ${item.disabled ? 'disabled' : ''}>${esc(item.label)}</option>`).join('')}
    </select>
    <span class="voice-reason ${chosen?.amber ? 'amber' : ''}">${esc(chosen?.reason || '')}</span>
    <span class="voice-demos">
      <button class="mini" data-audition="${esc(label)}" ${muted ? 'disabled' : ''}>Reference</button>
      <button class="mini" data-demo="calm" data-demo-label="${esc(label)}" ${demoSet?.calm ? '' : 'disabled'} ${demoSet?.calm ? '' : 'title="Render demos first (button top right)"'}>Demo · calm</button>
      <button class="mini" data-demo="emphatic" data-demo-label="${esc(label)}" ${demoSet?.emphatic ? '' : 'disabled'} ${demoSet?.emphatic ? '' : 'title="Render demos first (button top right)"'}>Demo · emphatic</button>
    </span>`;
}

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function clipStrip(label) {
  const lines = dialogueSegments(label);
  const showAll = expandedAll.has(label);
  const visible = showAll ? lines : lines.slice(0, 12);
  const others = labelCounts().map(([other]) => other).filter(other => other !== label);
  const cards = visible.map(segment => {
    const caption = String(segment.translation || '').trim() || '(no translation)';
    return `
    <div class="clip-card" data-clip="${segment.i}">
      <span class="clip-thumb-wrap">
        <span class="clip-thumb" data-video="${segment.i}" role="button" tabindex="0" title="Play video clip">
          <img loading="lazy" src="${BASE}/api/jobs/${JOB_ID}/segments/${segment.i}/evidence-frame" alt="segment ${segment.i}">
          <span class="clip-time">${fmtTime(segment.start)}</span>
        </span>
        <button class="clip-audio" data-line-audio="${segment.i}" title="Only this line's audio — isolate the voice">▶ Audio</button>
      </span>
      <div class="clip-caption" title="${esc(caption)}">${esc(caption)}</div>
      <div class="clip-foot">
        <select class="reassign" data-reassign="${segment.i}" title="This clip is someone else">
          <option value="">not this speaker →</option>
          ${others.map(other => `<option value="${esc(other)}">${esc(other)}${(cast[other]?.character || '').trim() ? ` (${esc(cast[other].character)})` : ''}</option>`).join('')}
        </select>
      </div>
    </div>`;
  }).join('');
  const more = lines.length > 12 && !showAll
    ? `<button class="secondary mini strip-more" data-more="${esc(label)}">show all ${lines.length} clips</button>` : '';
  return `<div class="clip-strip">${cards}</div>${more}`;
}

/* ---------- step 4: lock ---------- */

function renderLockStep(labels) {
  lockProblems = [];
  labels.forEach(([label]) => {
    if (isMuted(label)) return;
    if (!(cast[label].character || '').trim()) lockProblems.push(`${label} has no character name`);
    if (!(cast[label].voice || '').trim()) lockProblems.push(`${label} has no voice`);
  });
  const rows = labels.map(([label, lines]) => {
    const muted = isMuted(label);
    const options = voiceOptions(label);
    const chosen = options.find(item => item.voice === cast[label].voice);
    return `<tr>
      <td class="mono">${esc(label)}</td>
      <td>${muted ? '<span class="faint">muted</span>' : esc((cast[label].character || '').trim() || '—')}</td>
      <td>${muted ? '—' : esc(chosen?.label || cast[label].voice)}</td>
      <td class="mono">${lines}</td>
      <td>${job.demo_manifest?.[label]?.[cast[label].voice] ? 'demoed' : '<span class="faint">no demo</span>'}</td>
    </tr>`;
  }).join('');
  $('#rows').innerHTML = `<div id="lock-summary">
    <table class="summary-table">
      <tr><th>SPEAKER</th><th>CHARACTER</th><th>VOICE</th><th>LINES</th><th>DEMO</th></tr>${rows}
    </table>
    ${lockProblems.map(problem => `<p class="lock-problem">⚠ ${esc(problem)}</p>`).join('')}
    <div class="lock-actions">
      <button id="do-lock" class="primary" ${lockProblems.length ? 'disabled' : ''}>CAST LOCK — apply to this episode</button>
      <span class="muted">rewrites voices, purges stale audio, enrolls the cast in the series bank (second choice recorded automatically)</span>
    </div>
  </div>`;
  const lockButton = $('#do-lock');
  if (lockButton) lockButton.onclick = castLock;
}

async function castLock() {
  const labels = labelCounts();
  const series = $('#series').value.trim();
  const summary = labels.map(([label]) =>
    isMuted(label) ? `${label} → muted` : `${label} → ${cast[label].character} (${cast[label].voice})`).join('\n');
  const seriesNote = series ? `Series bank: ${series}.`
    : 'NO SERIES NAME — bank enrollment will be SKIPPED (voices will not carry to other episodes).';
  if (!await confirmModal('Lock the cast?',
    `${summary}\n\n${seriesNote}\nCached audio for changed voices is purged.`, 'Lock cast')) return;
  const payload = {};
  labels.forEach(([label]) => {
    if (isMuted(label)) { payload[label] = {voice: 'mute:'}; return; }
    const voice = cast[label].voice;
    const hasTranscript = Boolean((job.speaker_references?.[label]?.text || '').trim());
    // second choice: the other clone flavor — but never ICL when there is no
    // transcript (apply-cast hard-rejects it; the page only offered xv for a reason)
    const flip = voice.startsWith('qwen-auto-xv:') && hasTranscript
                   ? voice.replace('qwen-auto-xv:', 'qwen-auto:')
               : voice.startsWith('qwen-auto:') ? voice.replace('qwen-auto:', 'qwen-auto-xv:') : null;
    payload[label] = {character: cast[label].character.trim(), voice,
                     ...(flip ? {second_choice: flip} : {})};
  });
  try {
    const result = await api(`/api/jobs/${JOB_ID}/apply-cast`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cast: payload, series: series || undefined}),
    });
    toast(`Cast locked — ${result.enrolled ?? 0} character(s) enrolled${series ? '' : ' (no series: bank skipped)'}. Back to the job page for adapt + render.`);
    await reloadJob();
  } catch (error) { toast(`Lock failed: ${error.message}`, true); }
}

/* ---------- compare drawer ---------- */

function currentPair() {
  if (!compare) return null;
  const pair = openPairs().find(item => pairKeyOf(item) === compare.key);
  if (!pair) return null;
  return {a: String(pair.speaker_a), b: String(pair.speaker_b),
          sim: Number(pair.cosine_similarity), raw: pair};
}

function openCompare(key) {
  const pairs = openPairs();
  const hit = pairs.find(pair => pairKeyOf(pair) === key);
  if (!hit) { toast('That pair was already settled.'); render(); return; }
  compare = {key, side: 'a', clipIndex: {a: 0, b: 0}};
  $('#drawer').classList.remove('hidden');
  renderCompare(true);
  render();
}

function closeCompare() {
  $('#drawer').classList.add('hidden');
  $('#line-audio').pause();
  compare = null;
  render();
}

function nextPair() {
  if (!compare) return;
  const pairs = openPairs();
  if (!pairs.length) { closeCompare(); toast('All pairs settled ✓'); return; }
  const at = pairs.findIndex(pair => pairKeyOf(pair) === compare.key);
  const next = pairs[(at < 0 ? 0 : at + 1) % pairs.length];
  compare = {key: pairKeyOf(next), side: 'a', clipIndex: {a: 0, b: 0}};
  renderCompare(true);
  render();
}

function compareClips(label) {
  return dialogueSegments(label)
    .slice().sort((x, y) => (y.end - y.start) - (x.end - x.start)).slice(0, 6);
}

function renderCompare(autoplay) {
  const pair = currentPair();
  if (!pair) { closeCompare(); return; }
  compare.clips = {a: compareClips(pair.a), b: compareClips(pair.b)};
  const pairs = openPairs();
  const position = pairs.findIndex(item => pairKeyOf(item) === compare.key);
  $('#drawer-count').textContent = `SAME PERSON? · PAIR ${position + 1} OF ${pairs.length}`;
  $('#drawer-pair').innerHTML = `${esc(pair.a)} <span class="faint">vs</span> ${esc(pair.b)}`;
  $('#drawer-sim').textContent = `similarity ${pair.sim.toFixed(2)}`;
  ['a', 'b'].forEach(side => {
    const card = $(`#ab-${side}`);
    card.classList.toggle('active', compare.side === side);
    card.querySelector('.ab-label').textContent = pair[side];
    card.querySelector('.ab-state').textContent = 'ready';
    card.querySelector('.ab-fill').style.width = '0';
  });
  renderTimeline(pair);
  renderTranscripts(pair);
  if (autoplay) playSide(compare.side);
}

function playSide(side) {
  const pair = currentPair();
  if (!pair || !compare.clips) return;
  compare.side = side;
  const clips = compare.clips[side];
  if (!clips.length) { toast(`${pair[side]} has no dialogue lines`); return; }
  const clip = clips[compare.clipIndex[side] % clips.length];
  const audio = $('#line-audio');
  playingIndex = null;
  document.querySelectorAll('.clip-audio.playing').forEach(n => n.classList.remove('playing'));
  ['a', 'b'].forEach(other => $(`#ab-${other}`).classList.toggle('active', other === side));
  const card = $(`#ab-${side}`);
  card.querySelector('.ab-state').textContent = `playing · line ${clip.i}`;
  audio.src = `${BASE}/api/jobs/${JOB_ID}/segments/${clip.i}/source`;
  audio.ontimeupdate = () => {
    if (audio.duration) card.querySelector('.ab-fill').style.width = `${100 * audio.currentTime / audio.duration}%`;
  };
  audio.onended = () => { card.querySelector('.ab-state').textContent = 'done'; };
  audio.play().catch(() => {});
}

function renderTimeline(pair) {
  const duration = Math.max(...(job.segments || []).map(s => Number(s.end) || 0), 1);
  const lanes = ['a', 'b'].map((side, laneIndex) => {
    const segs = dialogueSegments(pair[side]).map(s =>
      `<span class="tl-seg" data-left="${100 * s.start / duration}" data-width="${Math.max(0.15, 100 * (s.end - s.start) / duration)}"></span>`).join('');
    return `<div class="tl-lane ${laneIndex === 0 ? 'first' : ''}"><span class="mono">${esc(pair[side])}</span><div class="tl-bar">${segs}</div></div>`;
  }).join('');
  $('#tl-lanes').innerHTML = lanes +
    `<div class="tl-scale"><span>0:00</span><span>${fmtTime(duration / 2)}</span><span>${fmtTime(duration)}</span></div>`;
  // The CSP forbids inline style attributes; geometry is applied through the CSSOM instead.
  document.querySelectorAll('#tl-lanes .tl-seg').forEach(seg => {
    seg.style.left = `${seg.dataset.left}%`;
    seg.style.width = `${seg.dataset.width}%`;
  });
  // Honest verdict: the segment timeline is EXCLUSIVE by
  // construction, so "segments never overlap" is true of every pair and proves
  // nothing. The raw diarizer's simultaneous-speech measurement rides each
  // suggestion — that is the evidence that can actually veto a merge.
  const verdict = $('#tl-verdict');
  const runMs = Number(pair.raw?.simultaneous_speech_longest_run_ms || 0);
  const totalMs = Number(pair.raw?.simultaneous_speech_total_ms || 0);
  if (runMs >= 450) {
    verdict.textContent = `speak over each other (${totalMs}ms total, longest ${runMs}ms) — very likely TWO people`;
    verdict.className = 'bad';
    $('#tl-note').textContent = 'The raw diarizer heard both voices at the same time — real conversations rarely do this for one person. Lean DIFFERENT unless your ears strongly disagree.';
  } else if (totalMs > 0) {
    verdict.textContent = `${totalMs}ms of crosstalk — inconclusive`;
    verdict.className = '';
    $('#tl-note').textContent = 'A little simultaneous speech can be noise or a real second voice. Let your ears decide.';
  } else {
    verdict.textContent = 'no simultaneous speech detected — merge is possible';
    verdict.className = 'ok';
    $('#tl-note').textContent = 'Interleaved turns in the same scenes are consistent with one person split by the diarizer. Let your ears decide.';
  }
}

function renderTranscripts(pair) {
  $('#drawer-transcripts').innerHTML = ['a', 'b'].map(side => {
    const clips = compare.clips[side].slice(0, 3);
    return `<div class="transcript-col"><h4>${esc(pair[side].toUpperCase())} SAYS</h4>
      ${clips.map(clip => `<blockquote>“${esc(String(clip.translation || '').trim() || '…')}”
        <button class="ghost mini from-line" data-play-line="${clip.i}" data-play-side="${side}">▶</button></blockquote>`).join('')
      || '<p class="faint">no dialogue</p>'}</div>`;
  }).join('');
  document.querySelectorAll('[data-play-line]').forEach(button => button.onclick = () => {
    const side = button.dataset.playSide;
    compare.clipIndex[side] = Math.max(0,
      compare.clips[side].findIndex(c => c.i === Number(button.dataset.playLine)));
    playSide(side);
  });
}

async function verdictSame() {
  const pair = currentPair();
  if (!pair) return;
  const counts = Object.fromEntries(labelCounts());
  const [source, target] = (counts[pair.a] || 0) <= (counts[pair.b] || 0)
    ? [pair.a, pair.b] : [pair.b, pair.a];
  const moved = (job.segments || []).filter(s => String(s.speaker) === source).length;
  if (!await confirmModal('Merge voices?',
    `Merge ${source} into ${target}? Its ${moved} line(s) (dialogue + song) become ${target}'s and its cached audio is purged. Re-analysis can undo this.`, 'Merge')) return;
  try {
    await api(`/api/jobs/${JOB_ID}/speakers/merge`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({from: source, to: target}),
    });
  } catch (error) { toast(`Merge failed: ${error.message}`, true); return; }
  delete cast[source];
  cleanSet.delete(source);
  saveProgress();
  try {
    await reloadJob();
    toast(`Merged ${source} → ${target}`);
    nextPair();
  } catch (error) {
    toast('Merged — but the refresh failed. Reload the page to continue.', true);
  }
}

async function verdictDifferent() {
  const pair = currentPair();
  if (!pair) return;
  try {
    await api(`/api/jobs/${JOB_ID}/speakers/reject-merge`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({a: pair.a, b: pair.b}),
    });
    job.speaker_merge_rejections = job.speaker_merge_rejections || [];
    job.speaker_merge_rejections.push([pair.a, pair.b].sort());
  } catch (error) {
    dismissedPairs.add(compare.key);
    toast(`DIFFERENT not saved (${error.message}) — it will re-ask next session`, true);
  }
  nextPair();
}

function verdictUnsure() {
  if (compare) dismissedPairs.add(compare.key);
  nextPair();
}

/* ---------- video popup ---------- */

function openClipModal(index, label, pad = 0) {
  clipState = {index, label, pad};
  $('#video-title').textContent = `${label} · segment ${index}${pad ? ` · ±${pad}s context` : ''}`;
  $('#video-longer').disabled = pad >= 16;
  $('#video-longer').textContent = pad ? `Longer clip (±${pad + 8}s)` : 'Longer clip';
  $('#video-episode').disabled = false;
  $('#video-note').textContent = 'first open cuts the clip — a few seconds…';
  $('#video-note').classList.remove('hidden');
  $('#video-overlay').classList.remove('hidden');
  const player = $('#video-player');
  player.oncanplay = () => $('#video-note').classList.add('hidden');
  player.src = `${BASE}/api/jobs/${JOB_ID}/segments/${index}/evidence-video${pad ? `?pad=${pad}` : ''}`;
}

function openEpisodeAtTime(at, title) {
  $('#video-title').textContent = `${title} · episode @ ${fmtTime(at)}`;
  $('#video-longer').disabled = true;
  $('#video-episode').disabled = true;
  $('#video-note').textContent = 'first open prepares the episode for the browser — up to a minute…';
  $('#video-note').classList.remove('hidden');
  $('#video-overlay').classList.remove('hidden');
  const player = $('#video-player');
  player.oncanplay = () => $('#video-note').classList.add('hidden');
  player.onloadedmetadata = () => { player.currentTime = Math.max(0, at); player.onloadedmetadata = null; };
  player.src = `${BASE}/api/jobs/${JOB_ID}/episode-video`;
}

function closeClipModal() {
  const player = $('#video-player');
  player.pause();
  player.removeAttribute('src');
  player.load();
  $('#video-overlay').classList.add('hidden');
}

/* ---------- persistence ---------- */

let patchSeq = 0;
async function patchJob(body) {
  const seq = ++patchSeq;
  const response = await api(`/api/jobs/${JOB_ID}`, {
    method: 'PATCH', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  // an out-of-order response (e.g. a slow saveProgress landing after a reassign)
  // must not roll the in-memory job back
  if (seq === patchSeq) job = response;
  return response;
}

let progressTimer = null;
function saveProgress() {
  clearTimeout(progressTimer);
  progressTimer = setTimeout(() => {
    const live = new Set(labelCounts().map(([label]) => label));
    const draft = {};
    Object.entries(cast).forEach(([label, entry]) => {
      if (live.has(label)) draft[label] = {character: entry.character, voice: entry.voice};
    });
    patchJob({studio_progress: {
      step,
      clean_labels: [...cleanSet].filter(label => live.has(label)),
      draft_cast: draft,
    }}).then(() => { warnedSaveFailure = false; })
      .catch(error => {
        if (!warnedSaveFailure) {
          warnedSaveFailure = true;
          toast(`Progress not saved (${error.message}) — will retry on your next change`, true);
        }
      });
  }, 600);
}

async function reloadJob() {
  job = await api(`/api/jobs/${JOB_ID}`);
  render();
}

/* ---------- demos ---------- */

async function renderDemos() {
  const candidates = {};
  labelCounts().forEach(([label]) => {
    if (isMuted(label)) return;
    const voices = voiceOptions(label)
      .filter(item => !item.disabled && !item.voice.startsWith('mute')).map(item => item.voice);
    if (voices.length) candidates[label] = voices;
  });
  if (!Object.keys(candidates).length) { toast('Nothing to demo.'); return; }
  if (!await confirmModal('Render casting demos?',
    'Arm the shared GPU once and render every candidate demo (two lines each)? The casting session then has zero waits.', 'Arm and render')) return;
  try {
    await api(`/api/jobs/${JOB_ID}/arm-gpu`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'demo'}),
    });
    await api(`/api/jobs/${JOB_ID}/demos/prerender`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({candidates, force: Boolean(job.demo_manifest)}),
    });
    toast('Rendering demos on the GPU — this page refreshes itself.');
    // completion = manifest PRESENT after the launch cleared it (a forced re-render
    // regenerates a byte-identical manifest, so identity-diffing never fires)
    const started = Date.now();
    let sawCleared = false;
    const poll = setInterval(async () => {
      let fresh;
      try { fresh = await api(`/api/jobs/${JOB_ID}`); } catch (error) { return; }
      const isTyping = document.activeElement?.classList?.contains('name-input');
      if (!fresh.demo_manifest) { sawCleared = true; job = fresh; return; }
      if (sawCleared || JSON.stringify(fresh.demo_manifest) !== JSON.stringify(job.demo_manifest)) {
        clearInterval(poll);
        job = fresh;
        if (job.demo_manifest.error) toast(`Demo render failed: ${job.demo_manifest.error}`, true);
        else toast('Demos ready ✓');
        if (!isTyping) render();
      } else if (Date.now() - started > 15 * 60 * 1000) {
        clearInterval(poll);
        toast('Demo render is taking longer than 15 minutes — check the job page.', true);
      }
    }, 5000);
  } catch (error) { toast(error.message, true); }
}

/* ---------- event wiring ---------- */

function bindRows() {
  document.querySelectorAll('[data-clean]').forEach(button => button.onclick = () => {
    const label = button.dataset.clean;
    if (cleanSet.has(label)) cleanSet.delete(label); else cleanSet.add(label);
    saveProgress();
    render();
  });
  document.querySelectorAll('[data-more]').forEach(button => button.onclick = () => {
    expandedAll.add(button.dataset.more);
    render();
  });
  document.querySelectorAll('[data-line-audio]').forEach(pill => pill.onclick = event => {
    event.stopPropagation();
    playLine(Number(pill.dataset.lineAudio), pill);
  });
  document.querySelectorAll('[data-video]').forEach(thumb => {
    const open = () => {
      const card = thumb.closest('.group-card');
      openClipModal(Number(thumb.dataset.video), card ? card.dataset.label : '');
    };
    thumb.onclick = open;
    thumb.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    };
  });
  document.querySelectorAll('[data-reassign]').forEach(select => select.onchange = async () => {
    const target = select.value;
    if (!target) return;
    const index = Number(select.dataset.reassign);
    try {
      await patchJob({segments: [{i: index, speaker: target}]});
      toast(`Line ${index} moved to ${target}`);
      if (compare) renderCompare(false);   // the drawer must not keep playing the moved line
      render();
    } catch (error) { toast(`Reassign failed: ${error.message}`, true); select.value = ''; }
  });
  document.querySelectorAll('[data-name]').forEach(input => input.oninput = () => {
    const label = input.dataset.name;
    cast[label].character = input.value;
    input.classList.toggle('named', Boolean(input.value.trim()));
    const status = statusOf(label);
    const cell = document.querySelector(`[data-status="${CSS.escape(label)}"]`);
    if (cell) cell.innerHTML = `<span class="status-dot dot-${status.dot}"></span>${status.text}`;
    updateBadges();
    saveProgress();
  });
  document.querySelectorAll('[data-pair]').forEach(chip => chip.onclick = () => {
    if (compare && compare.key === chip.dataset.pair) closeCompare();
    else openCompare(chip.dataset.pair);
  });
  document.querySelectorAll('[data-voice]').forEach(select => select.onchange = () => {
    cast[select.dataset.voice].voice = select.value;
    saveProgress();
    render();
  });
  document.querySelectorAll('[data-audition]').forEach(button => button.onclick = () => {
    const label = button.dataset.audition;
    // the xv variant shares the ICL reference clip — audition that (xv URLs 404)
    const voice = cast[label].voice.replace('qwen-auto-xv:', 'qwen-auto:');
    const audio = $('#line-audio');
    audio.src = `${BASE}/api/jobs/${JOB_ID}/voices/audition?voice=${encodeURIComponent(voice)}`;
    audio.play().catch(() => toast('Reference not available for that voice', true));
  });
  document.querySelectorAll('[data-demo]').forEach(button => button.onclick = () => {
    const label = button.dataset.demoLabel;
    const name = job.demo_manifest?.[label]?.[cast[label].voice]?.[button.dataset.demo];
    if (!name) return;
    const audio = $('#line-audio');
    audio.src = `${BASE}/api/jobs/${JOB_ID}/demos/${name}`;
    audio.play().catch(() => toast('Demo not available', true));
  });
}

document.querySelectorAll('.step-tab').forEach(tab => tab.onclick = () => {
  step = Number(tab.dataset.step);
  saveProgress();
  render();
});
$('#next-step').onclick = () => {
  if (step < 4) { step += 1; saveProgress(); render(); return; }
  const lock = $('#do-lock');
  if (lock && !lock.disabled) lock.click();
  else if (lockProblems.length) toast(lockProblems[0], true);   // never a silent dead button
};
$('#sort').onchange = event => { sortMode = event.target.value; render(); };
$('#render-demos').onclick = renderDemos;
$('#verdict-same').onclick = verdictSame;
$('#verdict-diff').onclick = verdictDifferent;
$('#verdict-unsure').onclick = verdictUnsure;
document.querySelectorAll('.ab-card').forEach(card => card.onclick = event => {
  if (event.target.classList.contains('ab-next')) return;
  playSide(card.id === 'ab-a' ? 'a' : 'b');
});
document.querySelectorAll('.ab-next').forEach(button => button.onclick = event => {
  event.stopPropagation();
  if (!compare) return;
  const side = button.closest('.ab-card').id === 'ab-a' ? 'a' : 'b';
  compare.clipIndex[side] += 1;
  playSide(side);
});
$('#video-longer').onclick = () => {
  if (clipState) openClipModal(clipState.index, clipState.label, Math.min(16, clipState.pad + 8));
};
$('#video-episode').onclick = () => {
  if (!clipState) return;
  const segment = (job.segments || []).find(s => Number(s.i) === Number(clipState.index));
  openEpisodeAtTime(Math.max(0, Number(segment?.start ?? 0) - 4), clipState.label);
};
$('#video-close').onclick = closeClipModal;
$('#video-overlay').onclick = event => { if (event.target.id === 'video-overlay') closeClipModal(); };

document.addEventListener('keydown', event => {
  // overlays swallow EVERYTHING except their own keys — drawer hotkeys must never
  // fire behind a video or a confirm dialog (a stray D during the merge confirm once
  // silently recorded a rejection)
  if (!$('#video-overlay').classList.contains('hidden')) {
    if (event.key === 'Escape') closeClipModal();
    return;
  }
  if (!$('#confirm-overlay').classList.contains('hidden')) {
    if (event.key === 'Escape') $('#confirm-cancel').click();
    else if (event.key === 'Enter') $('#confirm-ok').click();
    return;
  }
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
  if (!compare) return;
  if (event.key === 'Escape') closeCompare();
  else if (event.key === ' ') { event.preventDefault(); playSide(compare.side === 'a' ? 'b' : 'a'); }
  else if (event.key === 's' || event.key === 'S') verdictSame();
  else if (event.key === 'd' || event.key === 'D') verdictDifferent();
  else if (event.key === 'ArrowRight') nextPair();
});

/* ---------- boot ---------- */

async function boot() {
  if (!JOB_ID) { $('#rows').innerHTML = '<p class="muted pad">No job — open this page from a job\'s Casting studio button.</p>'; return; }
  try {
    job = await api(`/api/jobs/${JOB_ID}`);
  } catch (error) {
    $('#rows').innerHTML = `<p class="muted pad">Could not load job: ${esc(error.message)}</p>`;
    return;
  }
  const progress = job.studio_progress || {};
  step = [1, 2, 3, 4].includes(Number(progress.step)) ? Number(progress.step) : 1;
  cleanSet = new Set(Array.isArray(progress.clean_labels) ? progress.clean_labels.map(String) : []);
  try {
    // charView must load BEFORE the first render so ID->name resolution works
    charView = await api(`/api/jobs/${JOB_ID}/characters`);
    const names = new Set(charView.cast_names || []);
    (charView.bank || []).forEach(item => item.name && names.add(item.name));
    $('#cast-names').innerHTML = [...names].sort().map(name => `<option value="${esc(name)}"></option>`).join('');
    if (charView.series && !$('#series').value) $('#series').value = charView.series;
  } catch (error) { /* name suggestions are optional */ }
  render();
}

boot();
