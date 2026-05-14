const runButton = document.getElementById("runButton");
const promptInput = document.getElementById("promptInput");
const loading = document.getElementById("loading");
const results = document.getElementById("results");

function formatScore(score) {
  return `${Math.round(score * 100)}%`;
}

function renderList(containerId, items, renderItem) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "list-item";
    div.innerHTML = renderItem(item);
    container.appendChild(div);
  });
}

async function runExperiment() {
  runButton.disabled = true;
  loading.classList.remove("hidden");
  results.classList.add("hidden");

  try {
    const response = await fetch("/demo-summary", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: promptInput.value,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    const scores = data.scores;

    document.getElementById("v1Score").textContent = formatScore(scores.prompt_v1_score);
    document.getElementById("v2Score").textContent = formatScore(scores.prompt_v2_score);
    document.getElementById("improvement").textContent = `+${formatScore(scores.improvement)}`;

    document.getElementById("v1Cases").textContent =
      `${scores.v1_passed_cases}/${scores.total_cases} cases passed`;

    document.getElementById("v2Cases").textContent =
      `${scores.v2_passed_cases}/${scores.total_cases} cases passed`;

    renderList("failureAnalysis", data.failure_analysis, (item) => {
      return `<strong>${item.evaluator}</strong><br><span class="muted">${item.failures} failures detected in v1</span>`;
    });

    renderList("promptPatch", data.prompt_patch, (item) => {
      return item;
    });

    if (data.sample_failure) {
      document.getElementById("failureTitle").textContent =
        `${data.sample_failure.case_id} — ${data.sample_failure.title}`;

      document.getElementById("failureResponse").textContent =
        data.sample_failure.response_text;
    }

    if (data.sample_improved_response) {
      document.getElementById("improvedTitle").textContent =
        `${data.sample_improved_response.case_id} — ${data.sample_improved_response.title}`;

      document.getElementById("improvedResponse").textContent =
        data.sample_improved_response.response_text;
    }

    document.getElementById("improvedPrompt").textContent = data.prompt_v2;
    document.getElementById("demoMessage").textContent = data.demo_message;

    results.classList.remove("hidden");
  } catch (error) {
    alert(error.message);
  } finally {
    loading.classList.add("hidden");
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runExperiment);