// CASTING STUDIO (in-page v1) — one row per detected voice label:
// screenshot gallery + merge chips (never pre-checked) + voice casting with
// pre-rendered demos, then CAST LOCK -> apply-cast. Loads after app.js and reuses
// its globals (api, esc, $, confirmModal, playReview, loadJob, current).

let studioCast = {};
let studioDismissedPairs = new Set();
let studioJobId = null;
let studioSort = 'clusters';

// default sort: likely-same voices sit ADJACENT so comparing them
// is two neighboring rows, not a scroll hunt — union-find over the suggestion pairs
function studioSortedLabels() {
  const labels = studioLabels();                       // [[label, lines], ...] most-lines first
  if (studioSort === 'lines') return labels;
  if (studioSort === 'label') return [...labels].sort((a, b) => a[0].localeCompare(b[0]));
  const parent = {};
  const find = x => parent[x] === x ? x : (parent[x] = find(parent[x]));
  labels.forEach(([label]) => { parent[label] = label; });
  (current.speaker_duplicate_suggestions || [])
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

function studioLabels() {
  const counts = {};
  (current.segments || []).forEach(s => {
    const label = String(s.speaker || 'speaker-01');
    counts[label] = (counts[label] || 0) + 1;
  });
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}

function studioTier(lineCount) {
  return lineCount >= 40 ? 'MAIN' : lineCount >= 10 ? 'MINOR' : 'BIT';
}

function studioEvidence(label) {
  const speakers = current.speaker_evidence_manifest?.speakers || [];
  const entry = speakers.find(item => item.speaker === label);
  const segments = (entry?.segments || []).slice().sort(
    (a, b) => Number(a.borderline) - Number(b.borderline));
  return segments.slice(0, 3).map(item => item.segment_index);
}

function studioVoiceOptions(label) {
  const reference = current.speaker_references?.[label];
  const quality = reference?.quality;
  const options = [];
  if (reference) {
    const clean = quality ? quality.clean !== false : false;
    options.push({
      voice: `qwen-auto:${label}`,
      label: 'Own voice (clone + transcript)',
      reason: clean
        ? `clean solo reference · conf ${(quality?.confidence ?? 1).toFixed(2)}`
        : 'NO CLEAN WINDOW — audition carefully or pick another voice',
      amber: !clean,
      disabled: !(reference.text || '').trim(),
    });
    options.push({
      voice: `qwen-auto-xv:${label}`,
      label: 'Own voice (embedding only)',
      reason: 'less source-accent bleed, weaker identity',
      amber: !clean,
    });
  }
  (current.available_voices || []).forEach(voice => {
    if (voice.startsWith('bank:')) {
      options.push({voice, label: current.voice_labels?.[voice] || voice,
                    reason: 'series bank voice (consistent across episodes)'});
    }
  });
  options.push({voice: 'mute:', label: 'Mute (no dub)',
                reason: 'phantom or noise label — synthesize nothing'});
  return options;
}

function studioMergeChips(label) {
  const pairs = (current.speaker_duplicate_suggestions || [])
    .filter(pair => Number(pair.cosine_similarity || 0) >= 0.88)
    .filter(pair => pair.speaker_a === label || pair.speaker_b === label)
    .filter(pair => !studioDismissedPairs.has(`${pair.speaker_a}|${pair.speaker_b}`))
    .filter(pair => !(current.speaker_merge_rejections || []).some(r =>
      String(r[0]) === String([pair.speaker_a, pair.speaker_b].sort()[0])
      && String(r[1]) === String([pair.speaker_a, pair.speaker_b].sort()[1])));
  return pairs.map(pair => {
    const other = pair.speaker_a === label ? pair.speaker_b : pair.speaker_a;
    return `<div class="merge-chip" data-a="${esc(pair.speaker_a)}" data-b="${esc(pair.speaker_b)}">
      likely same as <b>${esc(other)}</b> · ${Number(pair.cosine_similarity).toFixed(2)}
      <button class="mini" data-merge-same>Same</button>
      <button class="mini" data-merge-diff>Different</button>
      <button class="mini" data-merge-unsure>Not sure</button>
    </div>`;
  }).join('');
}

function renderStudio() {
  const panel = $('#casting-studio');
  if (!current || panel.classList.contains('hidden')) return;
  if (studioJobId && current.id !== studioJobId) {
    panel.classList.add('hidden');   // job switched under the studio: stale casts must never post
    return;
  }
  const officialAudio = $('#studio-official-audio')?.checked;
  const labels = studioSortedLabels();
  labels.forEach(([label]) => {
    if (!studioCast[label]) {
      studioCast[label] = {
        character: current.speaker_characters?.[label] || '',
        voice: current.speaker_voices?.[label] || `qwen-auto:${label}`,
      };
    }
  });
  $('#studio-rows').innerHTML = labels.map(([label, count]) => {
    const cast = studioCast[label];
    const muted = (cast.voice || '').startsWith('mute');
    const tier = studioTier(count);
    const evidence = studioEvidence(label);
    const demoSet = current.demo_manifest?.[label]?.[cast.voice];
    const options = studioVoiceOptions(label);
    const chosen = options.find(item => item.voice === cast.voice);
    return `<div class="studio-row ${muted ? 'studio-muted' : ''}" data-label="${esc(label)}">
      <div class="studio-head"><b>${esc(label)}</b>
        <span class="tier tier-${tier.toLowerCase()}">${tier}</span>
        <span class="muted">${count} lines</span>
        ${muted ? '<span class="qc-flag">MUTED</span>' : ''}</div>
      <div class="studio-body">
        <div class="studio-gallery">
          ${evidence.map(index => `
            <div class="studio-card" data-video-seg="${index}" role="button" tabindex="0" title="Play speaking clip (segment ${index})">
              <img loading="lazy" src="${BASE}/api/jobs/${current.id}/segments/${index}/evidence-frame" alt="${esc(label)} at segment ${index}">
              <button class="mini card-audio" data-audio-seg="${index}" title="Only this line's audio — isolate the voice when several people talk in the clip">Audio</button>
            </div>`).join('') || '<p class="muted">No evidence cards.</p>'}
        </div>
        <div class="studio-identity">
          <label>This is
            <input class="studio-name" list="studio-names" value="${esc(cast.character)}" placeholder="Character name">
          </label>
          ${studioMergeChips(label)}
        </div>
        <div class="studio-voice">
          <select class="studio-voice-pick">
            ${options.map(item => `<option value="${esc(item.voice)}" ${item.voice === cast.voice ? 'selected' : ''} ${item.disabled ? 'disabled' : ''}>${esc(item.label)}</option>`).join('')}
          </select>
          <p class="muted studio-reason ${chosen?.amber ? 'studio-amber' : ''}">${esc(chosen?.reason || '')}</p>
          <div class="studio-demo-buttons">
            <button class="mini" data-studio-audition ${muted ? 'disabled' : ''}>Reference</button>
            <button class="mini" data-demo="calm" ${demoSet?.calm ? '' : 'disabled'}>Demo · calm</button>
            <button class="mini" data-demo="emphatic" ${demoSet?.emphatic ? '' : 'disabled'}>Demo · emphatic</button>
          </div>
        </div>
      </div>
    </div>`;
  }).join('');
  bindStudioRows();
  updateLockBar();
}

function bindStudioRows() {
  document.querySelectorAll('.studio-row').forEach(row => {
    const label = row.dataset.label;
    row.querySelectorAll('[data-video-seg]').forEach(card => {
      card.onclick = () => openClipModal(card.dataset.videoSeg, label);
      card.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openClipModal(card.dataset.videoSeg, label); } };
    });
    row.querySelectorAll('.card-audio').forEach(button => button.onclick = event => {
      event.stopPropagation();   // the card behind it opens the video popup
      playReview(`/api/jobs/${current.id}/segments/${button.dataset.audioSeg}/source`,
                 `${label} · line audio · segment ${button.dataset.audioSeg}`);
    });
    row.querySelector('.studio-name').oninput = event => {
      studioCast[label].character = event.target.value;
      updateLockBar();
    };
    row.querySelector('.studio-voice-pick').onchange = event => {
      studioCast[label].voice = event.target.value;
      renderStudio();
    };
    row.querySelector('[data-studio-audition]').onclick = () =>
      playReview(`/api/jobs/${current.id}/voices/audition?voice=${encodeURIComponent(studioCast[label].voice)}`,
                 `Reference · ${label}`);
    row.querySelectorAll('[data-demo]').forEach(button => button.onclick = () => {
      const name = current.demo_manifest?.[label]?.[studioCast[label].voice]?.[button.dataset.demo];
      if (name) playReview(`/api/jobs/${current.id}/demos/${name}`, `Demo · ${label} · ${button.dataset.demo}`);
    });
    row.querySelectorAll('.merge-chip').forEach(chip => {
      const a = chip.dataset.a, b = chip.dataset.b;
      chip.querySelector('[data-merge-same]').onclick = () => mergeLabels(a, b);
      chip.querySelector('[data-merge-diff]').onclick = async () => {
        studioDismissedPairs.add(`${a}|${b}`);
        try {   // DIFFERENT is a settled verdict — persist so it is never re-asked
          await api(`/api/jobs/${current.id}/speakers/reject-merge`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({a, b}),
          });
          current.speaker_merge_rejections = current.speaker_merge_rejections || [];
          current.speaker_merge_rejections.push([a, b].sort());
        } catch (error) { /* session-local dismissal still applies */ }
        renderStudio();
      };
      chip.querySelector('[data-merge-unsure]').onclick = () => {
        // NOT SURE keeps the split (never merge on hesitation)
        studioDismissedPairs.add(`${a}|${b}`); renderStudio();
      };
    });
  });
}

async function mergeLabels(a, b) {
  // fold the SMALLER label into the larger one
  const counts = Object.fromEntries(studioLabels());
  const [source, target] = (counts[a] || 0) <= (counts[b] || 0) ? [a, b] : [b, a];
  if (!await confirmModal(
    `Merge ${source} into ${target}? Its ${counts[source] || 0} line(s) become ${target}'s, and its cached audio is purged. Re-analysis can undo this.`,
    'Merge voices', 'Merge')) return;
  try {
    await api(`/api/jobs/${current.id}/speakers/merge`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({from: source, to: target}),
    });
    delete studioCast[source];
    await loadJob(current.id);
    renderStudio();
  } catch (error) { studioError(error.message); }
}

async function prerenderDemos() {
  const candidates = {};
  studioLabels().forEach(([label]) => {
    if ((studioCast[label]?.voice || '').startsWith('mute')) return;
    const voices = studioVoiceOptions(label)
      .filter(item => !item.disabled && !item.voice.startsWith('mute'))
      .map(item => item.voice);
    if (voices.length) candidates[label] = voices;
  });
  if (!Object.keys(candidates).length) { studioError('Nothing to demo.'); return; }
  if (!await confirmModal(
    'Arm the shared GPU once and render every voice demo (two lines per candidate)? The casting session then has zero waits.',
    'Render casting demos', 'Arm and render')) return;
  try {
    await api(`/api/jobs/${current.id}/arm-gpu`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'demo'}),
    });
    const hadManifest = JSON.stringify(current.demo_manifest || null);
    await api(`/api/jobs/${current.id}/demos/prerender`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({candidates, force: Boolean(current.demo_manifest)}),
    });
    $('#studio-lock-state').textContent = 'Rendering demos on the GPU — this page refreshes itself.';
    const started = Date.now();
    const poll = setInterval(async () => {
      await loadJob(current.id);
      const now = JSON.stringify(current.demo_manifest || null);
      if (current.demo_manifest && now !== hadManifest) {
        clearInterval(poll);
        if (current.demo_manifest.error) studioError(`Demo render failed: ${current.demo_manifest.error}`);
        renderStudio();
      } else if (Date.now() - started > 15 * 60 * 1000) {
        clearInterval(poll);
        studioError('Demo render is taking >15 min — check the job log.');
      }
    }, 5000);
  } catch (error) { studioError(error.message); }
}

function updateLockBar() {
  const pending = studioLabels().filter(([label]) => {
    const cast = studioCast[label] || {};
    if ((cast.voice || '').startsWith('mute')) return false;
    return !(cast.character || '').trim() || !(cast.voice || '').trim();
  });
  $('#studio-lock').disabled = pending.length > 0;
  $('#studio-lock-state').textContent = pending.length
    ? `${pending.length} voice(s) still need a name + pick: ${pending.map(([l]) => l).join(', ')}`
    : 'Every voice is cast — lock when ready.';
}

async function castLock() {
  const cast = {};
  studioLabels().forEach(([label, count]) => {
    const entry = studioCast[label];
    if ((entry.voice || '').startsWith('mute')) {
      cast[label] = {voice: 'mute:'};
      return;
    }
    // continuity insurance at no reviewer cost: the other clone mode is the
    // auto-recorded second choice
    const second = entry.voice.startsWith('qwen-auto:')
      ? entry.voice.replace('qwen-auto:', 'qwen-auto-xv:')
      : entry.voice.startsWith('qwen-auto-xv:')
        ? entry.voice.replace('qwen-auto-xv:', 'qwen-auto:') : null;
    cast[label] = {character: entry.character.trim(), voice: entry.voice,
                   second_choice: second, tier: studioTier(count).toLowerCase()};
  });
  const series = $('#studio-series').value.trim();
  if (!await confirmModal(
    `Lock this cast (${Object.keys(cast).length} character(s))${series ? ` and enroll it in the "${series}" bank` : ''}? Changed voices re-synthesize on the next render.`,
    'Cast lock', 'Lock')) return;
  try {
    const result = await api(`/api/jobs/${current.id}/apply-cast`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cast, series: series || null}),
    });
    $('#studio-lock-state').textContent =
      `Locked: ${result.cast} cast · ${result.changed.length} voice change(s) · ${result.purged_lines} cached line(s) purged · ${result.enrolled} enrolled.`;
    await loadJob(current.id);
  } catch (error) { studioError(error.message); }
}

function studioError(message) {
  $('#studio-lock-state').textContent = message;
}

// ---- Stage-4 attribution spot-check: flagged lines only, one-tap reassign ------------
async function loadSpotCheck() {
  const box = $('#studio-spotcheck');
  box.classList.remove('hidden');
  box.innerHTML = '<p class="muted">Loading flags…</p>';
  try {
    const data = await api(`/api/jobs/${current.id}/attribution-flags`);
    const merged = {};
    (data.flags || []).forEach(flag => {
      merged[flag.i] = {speaker: flag.speaker, reasons: [...flag.reasons]};
    });
    (data.embedding_disagreements || []).forEach(flag => {
      merged[flag.i] = merged[flag.i] || {speaker: flag.speaker, reasons: []};
      merged[flag.i].reasons.push(
        `voice-print suggests ${flag.suggests} (${flag.best_cosine} vs ${flag.own_cosine})`);
    });
    const labels = studioLabels().map(([label]) => label);
    const coverage = data.embedding_coverage || {};
    const rows = Object.entries(merged).sort((a, b) => Number(a[0]) - Number(b[0]));
    box.innerHTML = `<div class="panel-title"><div><h3>Attribution spot-check</h3>
      <p class="muted">${rows.length} flagged line(s) · voice-print check covers ${coverage.checked || 0}/${coverage.total || 0} lines (the rest need your eyes) · skipping is fine — unreviewed lines keep their label.</p></div></div>`
      + (rows.map(([index, flag]) => `
      <div class="spot-row" data-i="${index}">
        <img loading="lazy" src="${BASE}/api/jobs/${current.id}/segments/${index}/evidence-frame" alt="line ${index}">
        <div class="spot-meta"><b>line ${index}</b><span class="muted">${esc(flag.reasons.join(' · '))}</span></div>
        <button class="mini" data-spot-play>Source</button>
        <select class="spot-speaker">${labels.map(label =>
          `<option value="${esc(label)}" ${label === flag.speaker ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select>
        <button class="mini" data-spot-save disabled>Save</button>
      </div>`).join('') || '<p class="muted">Nothing flagged.</p>');
    box.querySelectorAll('.spot-row').forEach(row => {
      const index = Number(row.dataset.i);
      const select = row.querySelector('.spot-speaker');
      const save = row.querySelector('[data-spot-save]');
      row.querySelector('[data-spot-play]').onclick = () =>
        playReview(`/api/jobs/${current.id}/segments/${index}/source`, `Source line ${index}`);
      select.onchange = () => { save.disabled = false; };
      save.onclick = async () => {
        try {
          await api(`/api/jobs/${current.id}`, {
            method: 'PATCH', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({segments: [{i: index, speaker: select.value}]}),
          });
          save.textContent = 'Saved';
          save.disabled = true;
        } catch (error) { studioError(error.message); }
      };
    });
  } catch (error) {
    box.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
  }
}

async function openCastingStudio() {
  if (!current || !(current.segments || []).length) return;
  studioCast = {};
  studioDismissedPairs = new Set();
  studioJobId = current.id;
  $('#casting-studio').classList.remove('hidden');
  // the section lives at the bottom of the job view — without this the click "does nothing"
  $('#casting-studio').scrollIntoView({behavior: 'smooth', block: 'start'});
  try {
    const names = await api(`/api/jobs/${current.id}/characters`);
    const suggestions = new Set(names.cast_names || []);
    (names.bank || []).forEach(item => item.name && suggestions.add(item.name));
    $('#studio-names').innerHTML = [...suggestions].sort()
      .map(name => `<option value="${esc(name)}"></option>`).join('');
  } catch (error) { /* name suggestions are optional */ }
  renderStudio();
}

// Same-tab clip popup (no new tabs). Clearing src on close stops
// buffering AND the audio — hiding the overlay alone would keep the clip playing.
let clipState = null;   // {index, label, pad} of the open clip

function openClipModal(index, label, pad = 0) {
  clipState = {index, label, pad};
  const officialAudio = $('#studio-official-audio').checked;
  $('#video-title').textContent = `${label} · segment ${index}`
    + (pad ? ` · ±${pad}s context` : '') + (officialAudio ? ' · official EN audio' : '');
  $('#video-longer').disabled = pad >= 16;
  $('#video-longer').textContent = pad ? `Longer clip (±${pad + 8}s)` : 'Longer clip';
  $('#video-episode').disabled = false;
  $('#video-note').textContent = 'first open cuts the clip — a few seconds…';
  $('#video-note').classList.remove('hidden');
  $('#video-overlay').classList.remove('hidden');
  const player = $('#video-player');
  player.oncanplay = () => $('#video-note').classList.add('hidden');
  const query = [officialAudio ? 'lang=eng' : '', pad ? `pad=${pad}` : ''].filter(Boolean).join('&');
  player.src = `${BASE}/api/jobs/${current.id}/segments/${index}/evidence-video${query ? '?' + query : ''}`;
}

function openEpisodeAt() {
  // whole episode in the same popup, parked just before this line (short clips do not
  // always give the full picture) — source JP audio
  if (!clipState) return;
  const segment = (current.segments || []).find(s => Number(s.i) === Number(clipState.index));
  const at = Math.max(0, Number(segment?.start ?? 0) - 4);
  $('#video-title').textContent = `${clipState.label} · full episode @ ${Math.floor(at / 60)}:${String(Math.floor(at % 60)).padStart(2, '0')}`;
  $('#video-longer').disabled = true;
  $('#video-episode').disabled = true;
  $('#video-note').textContent = 'first open prepares the episode for the browser — up to a minute…';
  $('#video-note').classList.remove('hidden');
  const player = $('#video-player');
  player.oncanplay = () => $('#video-note').classList.add('hidden');
  player.onloadedmetadata = () => { player.currentTime = at; player.onloadedmetadata = null; };
  player.src = `${BASE}/api/jobs/${current.id}/episode-video`;
}

function closeClipModal() {
  const player = $('#video-player');
  player.pause();
  player.removeAttribute('src');
  player.load();
  $('#video-overlay').classList.add('hidden');
}

$('#video-longer').onclick = () => {
  if (clipState) openClipModal(clipState.index, clipState.label, Math.min(16, clipState.pad + 8));
};
$('#video-episode').onclick = openEpisodeAt;
$('#video-close').onclick = closeClipModal;
$('#video-overlay').onclick = event => { if (event.target.id === 'video-overlay') closeClipModal(); };
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('#video-overlay').classList.contains('hidden')) closeClipModal();
});

// The Casting studio button opens the dedicated staged page (studio.html). The in-page v1
// panel below is not reachable from the main UI; its clip popup is shared with v2.
$('#open-studio').onclick = () => {
  if (current) location.href = `/studio.html?job=${encodeURIComponent(current.id)}`;
};
$('#studio-close').onclick = () => $('#casting-studio').classList.add('hidden');
$('#studio-prerender').onclick = prerenderDemos;
$('#studio-lock').onclick = castLock;
$('#studio-official-audio').onchange = renderStudio;
$('#studio-sort').onchange = event => { studioSort = event.target.value; renderStudio(); };
$('#studio-spotcheck-open').onclick = loadSpotCheck;
