// Casting studio entry point on the main page: opens the staged studio (studio.html) for the
// current job. Loaded after app.js and uses its `current` job global.
$('#open-studio').onclick = () => {
  if (current) location.href = `/studio.html?job=${encodeURIComponent(current.id)}`;
};
