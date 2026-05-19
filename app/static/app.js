function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatScore(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "N/A";
  }
  return `${Math.round(number * 100)}%`;
}

function formatLatency(value) {
  if (value === null || value === undefined || value === "") {
    return "Latency unavailable";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "Latency unavailable";
  }
  return `${number.toFixed(0)} ms`;
}

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") {
    return "Unavailable";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "Unavailable";
  }
  return `$${number.toFixed(6)}`;
}

function formatInteger(value, fallback = "0") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.round(number).toString();
}

function formatDateTime(value) {
  if (!value) {
    return "Unknown time";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString();
}

function toggleVisibility(element, isVisible) {
  element.classList.toggle("hidden", !isVisible);
}

function setMetricText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return response.json();
}

function renderList(containerId, items, emptyMessage, formatter) {
  const container = document.getElementById(containerId);

  if (!Array.isArray(items) || items.length === 0) {
    container.innerHTML = `<div class="empty-state">${escapeHtml(emptyMessage)}</div>`;
    return;
  }

  container.innerHTML = items.map(formatter).join("");
}

function getTabButtons() {
  return Array.from(document.querySelectorAll(".mode-button"));
}

function activatePanel(targetId) {
  getTabButtons().forEach((button) => {
    const isActive = button.dataset.modeTarget === targetId;
    button.classList.toggle("active", isActive);
  });

  Array.from(document.querySelectorAll(".panel")).forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetId);
  });
}

function buildBadgeList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return `<div class="empty-state">No items.</div>`;
  }

  return `
    <ul class="inline-list">
      ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderPromptState(prompts) {
  const container = document.getElementById("phoenixPromptState");
  const current = prompts?.current;
  const candidate = prompts?.candidate;
  const previous = prompts?.previous;

  container.innerHTML = `
    <article class="stack-item">
      <p><strong>Current</strong></p>
      <p>${escapeHtml(current?.version_id || "Unavailable")}</p>
      <p>${escapeHtml(current?.activation_timestamp ? formatDateTime(current.activation_timestamp) : "Seeded state")}</p>
      <pre class="response-block compact-block">${escapeHtml(current?.prompt_text || "No active prompt.")}</pre>
    </article>
    <article class="stack-item">
      <p><strong>Candidate</strong></p>
      <p>${escapeHtml(candidate?.version_id || "No pending candidate")}</p>
      ${candidate ? buildBadgeList(candidate.patch_source || []) : `<div class="empty-state">No candidate queued.</div>`}
    </article>
    <article class="stack-item">
      <p><strong>Previous</strong></p>
      <p>${escapeHtml(previous?.version_id || "No rollback baseline")}</p>
      <p>${escapeHtml(previous?.rollback_reason || previous?.activation_reason || "No rollback events yet.")}</p>
    </article>
  `;

  const labPrompt = document.getElementById("labPrompt");
  if (labPrompt && (!labPrompt.value || labPrompt.dataset.autofilled !== "false")) {
    labPrompt.value = current?.prompt_text || labPrompt.value;
    labPrompt.dataset.autofilled = "true";
  }
}

function renderPhoenixTopOpportunities(items) {
  renderList(
    "phoenixTopIdeas",
    items,
    "No opportunity cards have been generated yet.",
    (item) => `
      <article class="trace-item">
        <h4>${escapeHtml(item.ticker || "Ticker")}</h4>
        <div class="trace-summary">
          <p><strong>Score:</strong> ${escapeHtml(formatInteger(item.score, "0"))}/100</p>
          <p><strong>Promotion:</strong> ${escapeHtml(item.promotion_decision || "observe_only")}</p>
          <p><strong>Data confidence:</strong> ${escapeHtml(formatInteger(item.data_confidence, "0"))}/100</p>
          <p><strong>Reasoning confidence:</strong> ${escapeHtml(formatInteger(item.reasoning_confidence, "0"))}/100</p>
          <p><strong>Trace:</strong> ${escapeHtml((item.trace_id || "no-trace").slice(0, 12))}</p>
        </div>
        ${Array.isArray(item.quality_flags) && item.quality_flags.length ? buildBadgeList(item.quality_flags) : ""}
        <p>${escapeHtml(item.thesis || "No thesis summary.")}</p>
      </article>
    `
  );
}

function renderPhoenixRuns(items) {
  renderList(
    "phoenixRecentRuns",
    items,
    "No market research runs yet.",
    (run) => `
      <article class="round-item">
        <h4>${escapeHtml(run.watchlist?.name || "Watchlist run")}</h4>
        <div class="round-meta">
          <p><strong>Run ID:</strong> ${escapeHtml(run.id || "-")}</p>
          <p><strong>Trace:</strong> ${escapeHtml((run.trace_id || "").slice(0, 12) || "no-trace")}</p>
          <p><strong>When:</strong> ${escapeHtml(formatDateTime(run.created_at))}</p>
          <p><strong>Latency:</strong> ${escapeHtml(formatLatency(run.latency_ms))}</p>
          <p><strong>Anomalies:</strong> ${escapeHtml(formatInteger(run.anomaly_count, "0"))}</p>
          <p><strong>Promotions:</strong> ${escapeHtml(formatInteger(run.promotion_candidates, "0"))}</p>
        </div>
      </article>
    `
  );
}

function renderLabQueue(items) {
  renderList(
    "phoenixLabQueue",
    items,
    "No continuous lab items yet.",
    (item) => `
      <article class="stack-item">
        <p><strong>${escapeHtml(item.status || "queued")}</strong></p>
        <p>${escapeHtml(item.run_id || "-")}</p>
        <p>Candidate cases: ${escapeHtml(formatInteger((item.candidate_case_ids || []).length, "0"))}</p>
        <p>Validation: ${escapeHtml(item.validation?.validation_status || "unknown")}</p>
        <p>Score delta: ${escapeHtml(item.validation?.score_delta ?? "N/A")}</p>
      </article>
    `
  );
}

function renderWatchlistOptions(watchlists) {
  const select = document.getElementById("watchlistSelect");
  if (!select) {
    return;
  }

  const previousValue = select.value;

  if (!Array.isArray(watchlists) || watchlists.length === 0) {
    select.innerHTML = `<option value="">No watchlists available</option>`;
    return;
  }

  select.innerHTML = watchlists
    .map(
      (watchlist) =>
        `<option value="${escapeHtml(watchlist.id)}">${escapeHtml(watchlist.name)} · ${escapeHtml(formatInteger((watchlist.tickers || []).length, "0"))} tickers</option>`
    )
    .join("");

  const hasPrevious = watchlists.some((watchlist) => watchlist.id === previousValue);
  if (hasPrevious) {
    select.value = previousValue;
    return;
  }

  const franceDefault = watchlists.find((watchlist) => watchlist.id === "watchlist_fr_sbf_120_core");
  if (franceDefault) {
    select.value = franceDefault.id;
  }
}

function renderExplorerRun(detail) {
  const run = detail?.run || {};
  const ideaCards = detail?.idea_cards || [];
  const introspection = detail?.introspection || {};
  const labQueueItem = detail?.lab_queue_item || {};

  toggleVisibility(document.getElementById("explorerResults"), true);
  setMetricText("explorerWatchlistMetric", run.watchlist?.name || "Unknown");
  setMetricText("explorerTraceMetric", (run.trace_id || "no-trace").slice(0, 12));
  setMetricText("explorerLatencyMetric", formatLatency(run.latency_ms));

  const bestIdea = ideaCards
    .slice()
    .sort((left, right) => Number(right.score || 0) - Number(left.score || 0))[0];
  setMetricText("explorerTopScoreMetric", bestIdea ? `${formatInteger(bestIdea.score, "0")}/100` : "No idea");
  setMetricText("explorerPromptMetric", run.prompt?.version_id || "Unknown prompt");
  setMetricText("explorerQueueMetric", labQueueItem.status || "No queue item");
  setMetricText("explorerQueueCaption", labQueueItem.validation?.validation_status || "No validation yet");

  renderList(
    "explorerIdeaCards",
    ideaCards,
    "No idea cards were generated for this run.",
    (card) => `
      <article class="trace-item">
        <h4>${escapeHtml(card.ticker || "Ticker")}</h4>
        <span class="badge badge-muted">${escapeHtml(card.promotion_decision || "observe_only")}</span>
        <div class="trace-summary">
          <p><strong>Score:</strong> ${escapeHtml(formatInteger(card.score, "0"))}/100</p>
          <p><strong>Data confidence:</strong> ${escapeHtml(formatInteger(card.data_confidence, "0"))}/100</p>
          <p><strong>Reasoning confidence:</strong> ${escapeHtml(formatInteger(card.reasoning_confidence, "0"))}/100</p>
          <p><strong>Conflicts:</strong> ${escapeHtml(formatInteger(card.conflict_count, "0"))}</p>
        </div>
        <p>${escapeHtml(card.thesis || "No thesis summary.")}</p>
        <p><strong>Positives</strong></p>
        ${buildBadgeList(card.positive_reasons || [])}
        <p><strong>Risks</strong></p>
        ${buildBadgeList(card.risks || [])}
        <p><strong>Missing information</strong></p>
        ${buildBadgeList(card.missing_information || [])}
        <p><strong>Promotion reasons</strong></p>
        ${buildBadgeList(card.promotion_reasons || [])}
      </article>
    `
  );

  const introspectionContainer = document.getElementById("explorerIntrospection");
  introspectionContainer.innerHTML = `
    <article class="stack-item">
      <p><strong>Failure summary</strong></p>
      <p>Warnings: ${escapeHtml(formatInteger(introspection.failure_summary?.warning_count, "0"))}</p>
      <p>Anomalies: ${escapeHtml(formatInteger(introspection.failure_summary?.critical_anomaly_count, "0"))}</p>
      <p>Conflicts: ${escapeHtml(formatInteger(introspection.failure_summary?.conflict_count, "0"))}</p>
      <p>Promoted ideas: ${escapeHtml(formatInteger(introspection.failure_summary?.promoted_ideas, "0"))}</p>
      <p>Average score: ${escapeHtml(introspection.failure_summary?.average_opportunity_score ?? "N/A")}</p>
      <p>Average confidence: ${escapeHtml(introspection.failure_summary?.average_data_confidence ?? "N/A")}</p>
    </article>
    <article class="stack-item">
      <p><strong>Prompt patch</strong></p>
      ${buildBadgeList(introspection.prompt_patch || [])}
      <p><strong>Tool policy patch</strong></p>
      ${buildBadgeList(introspection.tool_policy_patch || [])}
    </article>
    <article class="stack-item">
      <p><strong>Candidate prompt</strong></p>
      <pre class="response-block compact-block">${escapeHtml(introspection.recommended_prompt_candidate || "No candidate prompt.")}</pre>
    </article>
  `;

  renderList(
    "explorerToolTrace",
    run.tool_trace || [],
    "No tool trace available for this run.",
    (item) => `
      <article class="round-item">
        <h4>${escapeHtml(item.tool_name || "tool")}</h4>
        <div class="round-meta">
          <p><strong>Status:</strong> ${escapeHtml(item.status || "unknown")}</p>
          <p><strong>Latency:</strong> ${escapeHtml(formatLatency(item.latency_ms))}</p>
          <p><strong>Warnings:</strong> ${escapeHtml(formatInteger((item.warnings || []).length, "0"))}</p>
          <p><strong>Timestamp:</strong> ${escapeHtml(item.summary?.timestamp_utc || "n/a")}</p>
        </div>
        ${(item.warnings || []).length ? buildBadgeList(item.warnings) : ""}
      </article>
    `
  );

  const introspectButton = document.getElementById("rerunIntrospectionButton");
  introspectButton.dataset.runId = run.id || "";
}

function renderToolTrace(toolTrace) {
  renderList(
    "toolTrace",
    toolTrace,
    "No tool trace available.",
    (item) => {
      const summary = item.summary || {};
      const preview = summary.preview ? JSON.stringify(summary.preview, null, 2) : "";

      return `
        <article class="trace-item">
          <h4>${escapeHtml(item.tool_name || "tool")}</h4>
          <span class="badge badge-muted">${escapeHtml(item.status || "unknown")}</span>
          <div class="trace-summary">
            <p><strong>Latency:</strong> ${escapeHtml(formatLatency(item.latency_ms))}</p>
            <p><strong>Warnings:</strong> ${escapeHtml(formatInteger((item.warnings || []).length, "0"))}</p>
            <p><strong>Source:</strong> ${escapeHtml(summary.source || "yfinance")}</p>
            <p><strong>Timestamp:</strong> ${escapeHtml(summary.timestamp_utc || "n/a")}</p>
          </div>
          ${Array.isArray(item.warnings) && item.warnings.length > 0 ? buildBadgeList(item.warnings) : ""}
          ${preview ? `<pre class="response-block compact-block">${escapeHtml(preview)}</pre>` : ""}
        </article>
      `;
    }
  );
}

function renderLiveResult(data) {
  toggleVisibility(document.getElementById("liveResults"), true);

  setMetricText("liveTickersMetric", (data.tickers || []).join(", ") || "No ticker");
  setMetricText("liveModeMetric", data.analysis_mode || "Unknown mode");
  setMetricText("liveCostMetric", formatCurrency(data.cost_summary?.total_cost_usd));
  setMetricText(
    "liveTokenMetric",
    data.token_usage?.total_tokens != null
      ? `${data.token_usage.total_tokens} total tokens`
      : "Token usage unavailable"
  );
  setMetricText("liveTraceMetric", (data.trace_metadata?.trace_id || "no-trace").slice(0, 12));
  setMetricText("liveLatencyMetric", formatLatency(data.trace_metadata?.latency_ms));
  setMetricText("livePromptMetric", data.prompt_registry?.current_version_id || data.trace_metadata?.prompt_version || "Unknown prompt");
  setMetricText("livePromptCaption", data.prompt_registry?.template_id || data.trace_metadata?.prompt_template_id || "Phoenix live prompt");
  setMetricText("livePromotionMetric", data.promotion?.promoted_case_id ? "Promoted" : "Not promoted");
  setMetricText("livePromotionCaption", data.promotion?.promoted_case_id || "Awaiting benchmark promotion");

  document.getElementById("liveAnswer").textContent = data.answer || "No answer returned.";
  renderList(
    "liveWarnings",
    data.warnings || [],
    "No warnings.",
    (item) => `<div class="stack-item"><p>${escapeHtml(item)}</p></div>`
  );
  renderToolTrace(data.tool_trace || []);
}

function renderLiveInsights(insights) {
  const container = document.getElementById("liveInsights");
  const promotedCount = Number(insights?.promoted_case_count ?? 0);

  if (!promotedCount) {
    container.innerHTML = `<div class="empty-state">No promoted live traces yet.</div>`;
    return;
  }

  const categories = (insights.top_categories || [])
    .map(([name, count]) => `<li>${escapeHtml(name)}: ${escapeHtml(count)}</li>`)
    .join("");
  const tools = (insights.top_tools || [])
    .map(([name, count]) => `<li>${escapeHtml(name)}: ${escapeHtml(count)}</li>`)
    .join("");

  container.innerHTML = `
    <div class="stack-item">
      <p><strong>Promoted cases:</strong> ${escapeHtml(promotedCount)}</p>
      <p><strong>Top categories</strong></p>
      <ul class="inline-list">${categories || "<li>No categories yet.</li>"}</ul>
      <p><strong>Top tools</strong></p>
      <ul class="inline-list">${tools || "<li>No tools yet.</li>"}</ul>
    </div>
  `;
}

function renderRoundHistory(rounds) {
  renderList(
    "roundHistory",
    rounds,
    "No round history returned.",
    (round) => {
      const suite = round.suite_summary || {};
      const failedCases = suite.failed_case_summaries || [];
      const patchItems = round.prompt_patch || [];

      return `
        <article class="round-item">
          <h4>Round ${escapeHtml(round.round_number)}</h4>
          <div class="round-meta">
            <p><strong>Score:</strong> ${escapeHtml(formatScore(suite.overall_score))}</p>
            <p><strong>Cases:</strong> ${escapeHtml(suite.passed_cases ?? 0)}/${escapeHtml(suite.total_cases ?? 0)} passed</p>
            <p><strong>Avg latency:</strong> ${escapeHtml(formatLatency(suite.average_latency_ms))}</p>
            <p><strong>Avg cost:</strong> ${escapeHtml(formatCurrency(suite.average_cost_usd))}</p>
          </div>
          <p><strong>Prompt patch</strong></p>
          ${patchItems.length ? buildBadgeList(patchItems) : `<div class="empty-state">No patch for this round.</div>`}
          <p><strong>Failed cases</strong></p>
          ${
            failedCases.length
              ? `<ul class="inline-list">${failedCases
                  .map(
                    (item) =>
                      `<li>${escapeHtml(item.case_id)} - ${escapeHtml(item.title)} (${escapeHtml(formatScore(item.overall_score))})</li>`
                  )
                  .join("")}</ul>`
              : `<div class="empty-state">All cases passed.</div>`
          }
        </article>
      `;
    }
  );
}

function renderLabResult(data) {
  toggleVisibility(document.getElementById("labResults"), true);

  const summary = data.improvement_summary || {};
  setMetricText("baselineScoreMetric", formatScore(summary.baseline_score));
  setMetricText("baselineCostMetric", `Avg cost: ${formatCurrency(summary.baseline_average_cost_usd)}`);
  setMetricText("bestScoreMetric", formatScore(summary.best_score));
  setMetricText("bestRoundMetric", `Best round: ${summary.best_round || data.best_round || "-"}`);
  setMetricText("scoreDeltaMetric", formatScore(summary.score_delta));
  setMetricText("costDeltaMetric", `Cost delta: ${formatCurrency(summary.cost_delta_usd)}`);
  setMetricText("roundCountMetric", `${summary.round_count || 0} rounds`);
  setMetricText("stopReasonMetric", summary.stop_reason || "Unknown stop reason");

  document.getElementById("bestPrompt").textContent = data.best_prompt || "No prompt returned.";
  renderLiveInsights(data.live_trace_insights || {});
  renderRoundHistory(data.rounds || []);
}

async function loadPhoenixOverview() {
  const data = await fetchJson("/phoenix/overview");
  const headline = data.headline || {};
  setMetricText("overviewRunsMetric", formatInteger(headline.runs_today, "0"));
  setMetricText("overviewRunsCaption", `${formatInteger(headline.introspections_today, "0")} introspections today`);
  setMetricText("overviewWatchlistsMetric", formatInteger(headline.active_watchlists, "0"));
  setMetricText("overviewWatchlistsCaption", data.scheduler?.enabled ? "Scheduler enabled" : "Scheduler disabled");
  setMetricText("overviewCostMetric", formatCurrency(headline.total_cost_usd));
  setMetricText(
    "overviewCostCaption",
    headline.total_tokens != null ? `${headline.total_tokens} traced tokens` : "Token usage unavailable"
  );
  setMetricText("overviewLatencyMetric", formatLatency(headline.average_latency_ms));
  setMetricText("overviewLatencyCaption", `${formatInteger(headline.anomalies_today, "0")} anomaly flags today`);
  setMetricText("overviewPromptMetric", data.prompts?.current?.version_id || "Unknown prompt");
  setMetricText(
    "overviewPromptCaption",
    data.prompts?.candidate?.version_id ? `Candidate: ${data.prompts.candidate.version_id}` : "No candidate pending"
  );
  setMetricText("overviewPromotionMetric", formatInteger(headline.promotion_candidates_today, "0"));
  setMetricText("overviewPromotionCaption", `${formatInteger((data.lab_queue || []).length, "0")} lab queue items`);

  renderPromptState(data.prompts || {});
  renderPhoenixTopOpportunities(data.top_opportunities || []);
  renderPhoenixRuns(data.recent_runs || []);
  renderLabQueue(data.lab_queue || []);
  renderWatchlistOptions(data.watchlists || []);
}

async function loadWatchlists() {
  const data = await fetchJson("/explorer/watchlists");
  renderWatchlistOptions(data.watchlists || []);
}

async function loadBenchmarks() {
  const select = document.getElementById("benchmarkSelect");
  const data = await fetchJson("/lab/benchmarks");
  const benchmarks = data.benchmarks || [];

  if (!benchmarks.length) {
    select.innerHTML = `<option value="">No benchmarks available</option>`;
    return;
  }

  select.innerHTML = benchmarks
    .map(
      (item) =>
        `<option value="${escapeHtml(item.benchmark_id)}">${escapeHtml(item.title)} (${escapeHtml(item.case_count)})</option>`
    )
    .join("");

  const defaultOption = benchmarks.find((item) => item.benchmark_id === "all_equity_research");
  if (defaultOption) {
    select.value = defaultOption.benchmark_id;
  }
}

async function createNewWatchlist() {
  const button = document.getElementById("createWatchlistButton");
  button.disabled = true;

  try {
    const data = await fetchJson("/explorer/watchlists", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: document.getElementById("watchlistName").value,
        tickers: document
          .getElementById("watchlistTickers")
          .value.split(",")
          .map((item) => item.trim())
          .filter(Boolean),
        description: document.getElementById("watchlistDescription").value,
        schedule_enabled: document.getElementById("watchlistScheduleEnabled").checked,
      }),
    });

    document.getElementById("watchlistName").value = "";
    document.getElementById("watchlistTickers").value = "";
    document.getElementById("watchlistDescription").value = "";
    document.getElementById("watchlistScheduleEnabled").checked = true;

    await loadWatchlists();
    await loadPhoenixOverview();
    const select = document.getElementById("watchlistSelect");
    if (select && data.watchlist?.id) {
      select.value = data.watchlist.id;
    }
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

async function runExplorerWatchlist() {
  const button = document.getElementById("runExplorerButton");
  const loading = document.getElementById("explorerLoading");
  const watchlistId = document.getElementById("watchlistSelect").value;

  if (!watchlistId) {
    alert("Select a watchlist first.");
    return;
  }

  button.disabled = true;
  toggleVisibility(loading, true);

  try {
    const data = await fetchJson(`/explorer/watchlists/${encodeURIComponent(watchlistId)}/run`, {
      method: "POST",
    });
    renderExplorerRun(data);
    await loadPhoenixOverview();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    toggleVisibility(loading, false);
  }
}

async function rerunExplorerIntrospection() {
  const button = document.getElementById("rerunIntrospectionButton");
  const runId = button.dataset.runId;

  if (!runId) {
    alert("No explorer run is currently loaded.");
    return;
  }

  button.disabled = true;
  try {
    const data = await fetchJson(`/explorer/runs/${encodeURIComponent(runId)}/introspect`, {
      method: "POST",
    });
    renderExplorerRun({
      run: data.run,
      idea_cards: (await fetchJson(`/explorer/runs/${encodeURIComponent(runId)}`)).idea_cards,
      introspection: data.introspection,
      lab_queue_item: data.lab_queue_item,
    });
    await loadPhoenixOverview();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

async function rollbackPrompt() {
  const button = document.getElementById("rollbackPromptButton");
  button.disabled = true;

  try {
    await fetchJson("/phoenix/prompts/rollback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        reason: "Manual rollback triggered from the Phoenix cockpit UI.",
      }),
    });
    await loadPhoenixOverview();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

async function runLiveAnalysis() {
  const button = document.getElementById("runLiveButton");
  const loading = document.getElementById("liveLoading");
  const message = document.getElementById("liveMessage").value;
  const rawTickers = document.getElementById("liveTickers").value;
  const conversationId = document.getElementById("conversationId").value || null;
  const tickers = rawTickers
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  button.disabled = true;
  toggleVisibility(loading, true);

  try {
    const data = await fetchJson("/live/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        tickers,
        conversation_id: conversationId,
      }),
    });

    renderLiveResult(data);
    await loadPhoenixOverview();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    toggleVisibility(loading, false);
  }
}

async function runPromptLab() {
  const button = document.getElementById("runLabButton");
  const loading = document.getElementById("labLoading");
  const benchmarkId = document.getElementById("benchmarkSelect").value || null;
  const maxRounds = Number(document.getElementById("maxRounds").value || 3);
  const prompt = document.getElementById("labPrompt").value;

  button.disabled = true;
  toggleVisibility(loading, true);

  try {
    const data = await fetchJson("/lab/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        benchmark_id: benchmarkId,
        max_rounds: maxRounds,
      }),
    });

    renderLabResult(data);
    await loadPhoenixOverview();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    toggleVisibility(loading, false);
  }
}

function bindModeSwitch() {
  getTabButtons().forEach((button) => {
    button.addEventListener("click", () => activatePanel(button.dataset.modeTarget));
  });
}

async function bootstrap() {
  bindModeSwitch();
  document.getElementById("runLiveButton").addEventListener("click", runLiveAnalysis);
  document.getElementById("runLabButton").addEventListener("click", runPromptLab);
  document.getElementById("createWatchlistButton").addEventListener("click", createNewWatchlist);
  document.getElementById("runExplorerButton").addEventListener("click", runExplorerWatchlist);
  document.getElementById("rerunIntrospectionButton").addEventListener("click", rerunExplorerIntrospection);
  document.getElementById("rollbackPromptButton").addEventListener("click", rollbackPrompt);

  try {
    await Promise.all([loadPhoenixOverview(), loadWatchlists(), loadBenchmarks()]);
  } catch (error) {
    alert(error.message);
  }
}

document.addEventListener("DOMContentLoaded", bootstrap);
