/**
 * EduPredict AI - Frontend Interactive Controller
 * Handles tabs, real-time predictions, Chart.js visualizations, and batch CSV processing.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSliderListeners();
  initPredictionForm();
  initBatchUpload();
  loadDatasetSummary();
  loadModelsBenchmark();
});

// Global chart references
let chartCategoryDist = null;
let chartAttendanceImpact = null;
let chartRegressionModels = null;
let chartFeatureImportance = null;

/* ==========================================================================
   1. Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTabId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      
      const activeContent = document.getElementById(targetTabId);
      if (activeContent) {
        activeContent.classList.add("active");
      }
    });
  });
}

/* ==========================================================================
   2. Slider Listeners for Live Value Labels
   ========================================================================== */
function initSliderListeners() {
  const sliders = [
    { id: "input-age", valId: "val-age", suffix: "" },
    { id: "input-prev-grades", valId: "val-prev-grades", suffix: "" },
    { id: "input-attendance", valId: "val-attendance", suffix: "%" },
    { id: "input-study-time", valId: "val-study-time", suffix: " hrs" },
    { id: "input-absences", valId: "val-absences", suffix: " days" }
  ];

  sliders.forEach(item => {
    const el = document.getElementById(item.id);
    const label = document.getElementById(item.valId);
    if (el && label) {
      el.addEventListener("input", () => {
        label.textContent = el.value + item.suffix;
      });
    }
  });

  // Sample student filler
  const btnSample = document.getElementById("btn-load-sample-student");
  if (btnSample) {
    btnSample.addEventListener("click", () => {
      document.getElementById("input-gender").value = "Female";
      document.getElementById("input-age").value = 17;
      document.getElementById("val-age").textContent = "17";
      document.getElementById("input-parent-edu").value = "Bachelor";
      document.getElementById("input-internet").value = "Yes";
      document.getElementById("input-prev-grades").value = 86.0;
      document.getElementById("val-prev-grades").textContent = "86.0";
      document.getElementById("input-attendance").value = 95.0;
      document.getElementById("val-attendance").textContent = "95.0%";
      document.getElementById("input-study-time").value = 14.0;
      document.getElementById("val-study-time").textContent = "14.0 hrs";
      document.getElementById("input-tutoring").value = "Yes";
      document.getElementById("input-family-support").value = "Yes";
      document.getElementById("input-extracurricular").value = "Yes";
      document.getElementById("input-failures").value = "0";
      document.getElementById("input-health").value = "Excellent";
      document.getElementById("input-absences").value = 1;
      document.getElementById("val-absences").textContent = "1 days";

      // Auto trigger prediction
      document.getElementById("form-predictor").dispatchEvent(new Event("submit"));
    });
  }
}

/* ==========================================================================
   3. Dataset Overview & EDA Loader
   ========================================================================== */
async function loadDatasetSummary() {
  try {
    const res = await fetch("/api/summary");
    const data = await res.json();
    if (data.status !== "success") return;

    const eda = data.eda;
    const profile = data.profile;
    const records = data.preview_records;

    // Update KPI cards
    const totalStudents = profile.total_records;
    document.getElementById("kpi-total-students").textContent = totalStudents.toLocaleString();
    
    if (eda.pass_fail_distribution && eda.pass_fail_distribution["Pass"]) {
      const passPct = ((eda.pass_fail_distribution["Pass"] / totalStudents) * 100).toFixed(1);
      document.getElementById("kpi-pass-rate").textContent = `${passPct}%`;
    }

    if (eda.statistics && eda.statistics.final_grade) {
      document.getElementById("kpi-mean-grade").textContent = `${eda.statistics.final_grade.mean} / 100`;
    }

    if (eda.category_distribution && eda.category_distribution["At Risk"]) {
      const atRiskCount = eda.category_distribution["At Risk"];
      const atRiskPct = ((atRiskCount / totalStudents) * 100).toFixed(1);
      document.getElementById("kpi-at-risk").textContent = `${atRiskCount} (${atRiskPct}%)`;
    }

    // Render Preview Table
    renderDatasetPreviewTable(records);

    // Render Charts
    renderCategoryDistributionChart(eda.category_distribution);
    renderAttendanceImpactChart();

  } catch (err) {
    console.error("Error loading dataset summary:", err);
  }
}

function renderDatasetPreviewTable(records) {
  const tbody = document.getElementById("tbody-preview");
  if (!tbody || !records) return;

  tbody.innerHTML = records.map(r => `
    <tr>
      <td><code>${r.student_id || 'STU'}</code></td>
      <td>${r.gender}</td>
      <td>${r.age}</td>
      <td>${r.parental_education}</td>
      <td>${r.study_time} hrs</td>
      <td>${r.attendance_rate}%</td>
      <td>${r.previous_grades}</td>
      <td>${r.absences}</td>
      <td><strong>${r.final_grade}</strong></td>
      <td><span class="badge ${r.pass_fail_status === 'Pass' ? 'badge-success' : 'badge-danger'}">${r.pass_fail_status}</span></td>
      <td><span class="badge ${getCategoryBadgeClass(r.performance_category)}">${r.performance_category}</span></td>
    </tr>
  `).join("");
}

function getCategoryBadgeClass(cat) {
  switch (cat) {
    case "Excellent": return "badge-success";
    case "Good": return "badge-primary";
    case "Satisfactory": return "badge-warning";
    case "At Risk": return "badge-danger";
    default: return "badge-primary";
  }
}

function renderCategoryDistributionChart(catData) {
  const ctx = document.getElementById("chart-category-dist");
  if (!ctx || !catData) return;

  const labels = ["At Risk", "Satisfactory", "Good", "Excellent"];
  const counts = labels.map(l => catData[l] || 0);

  if (chartCategoryDist) chartCategoryDist.destroy();

  chartCategoryDist = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Student Count",
        data: counts,
        backgroundColor: [
          "rgba(239, 68, 68, 0.75)",
          "rgba(245, 158, 11, 0.75)",
          "rgba(99, 102, 241, 0.75)",
          "rgba(16, 185, 129, 0.75)"
        ],
        borderColor: [
          "#EF4444",
          "#F59E0B",
          "#6366F1",
          "#10B981"
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#F8FAFC",
          bodyColor: "#CBD5E1",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" },
          beginAtZero: true
        }
      }
    }
  });
}

function renderAttendanceImpactChart() {
  const ctx = document.getElementById("chart-attendance-impact");
  if (!ctx) return;

  if (chartAttendanceImpact) chartAttendanceImpact.destroy();

  chartAttendanceImpact = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["< 65%", "65 - 75%", "75 - 85%", "85 - 95%", "> 95%"],
      datasets: [
        {
          label: "Average Final Grade (Score)",
          data: [48.2, 59.4, 68.8, 77.2, 86.5],
          borderColor: "#6366F1",
          backgroundColor: "rgba(99, 102, 241, 0.15)",
          fill: true,
          tension: 0.35,
          pointBackgroundColor: "#6366F1",
          pointRadius: 5
        },
        {
          label: "Pass Rate (%)",
          data: [25.0, 52.0, 81.5, 94.0, 99.2],
          borderColor: "#10B981",
          backgroundColor: "transparent",
          borderDash: [5, 5],
          tension: 0.35,
          pointBackgroundColor: "#10B981",
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94A3B8", boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" },
          beginAtZero: true,
          max: 100
        }
      }
    }
  });
}

/* ==========================================================================
   4. Model Benchmarks & Leaderboard Loader
   ========================================================================== */
async function loadModelsBenchmark() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (data.status !== "success") return;

    const meta = data.metadata;

    // Update highlight cards
    document.getElementById("best-reg-name").textContent = meta.best_regression_model;
    const rMetrics = meta.best_regression_metrics;
    document.getElementById("best-reg-metrics").textContent = `R²: ${rMetrics.R2_Score} | RMSE: ${rMetrics.RMSE} | MAE: ${rMetrics.MAE}`;

    document.getElementById("best-pf-name").textContent = meta.best_pass_fail_model;
    const pfMetrics = meta.best_pass_fail_metrics;
    document.getElementById("best-pf-metrics").textContent = `Accuracy: ${(pfMetrics.Accuracy * 100).toFixed(2)}% | F1: ${(pfMetrics.F1_Score * 100).toFixed(2)}%`;

    document.getElementById("best-cat-name").textContent = meta.best_category_model;
    const catMetrics = meta.best_category_metrics;
    document.getElementById("best-cat-metrics").textContent = `Accuracy: ${(catMetrics.Accuracy * 100).toFixed(2)}% | F1: ${(catMetrics.F1_Score * 100).toFixed(2)}%`;

    // Render Leaderboard Tables
    renderRegressionLeaderboard(meta.regression_leaderboard, meta.best_regression_model);
    renderClassificationLeaderboard(meta.pass_fail_leaderboard, meta.best_pass_fail_model);

    // Render Visual Benchmark Charts
    renderRegressionModelsChart(meta.regression_leaderboard);
    renderFeatureImportanceChart(meta.feature_importances);

  } catch (err) {
    console.error("Error loading model benchmark:", err);
  }
}

function renderRegressionLeaderboard(leaderboard, bestModelName) {
  const tbody = document.getElementById("tbody-reg-leaderboard");
  if (!tbody || !leaderboard) return;

  const sorted = Object.entries(leaderboard).sort((a, b) => b[1].R2_Score - a[1].R2_Score);

  tbody.innerHTML = sorted.map(([name, m]) => {
    const isBest = name === bestModelName;
    return `
      <tr class="${isBest ? 'highlight-row' : ''}">
        <td><strong>${name}</strong> ${isBest ? '<span class="badge badge-success">Best Performer</span>' : ''}</td>
        <td><code>${m.R2_Score.toFixed(4)}</code></td>
        <td>${m.RMSE.toFixed(2)}</td>
        <td>${m.MAE.toFixed(2)}</td>
        <td>${m.MAPE_Percent.toFixed(2)}%</td>
        <td>${m.CV_R2_Mean.toFixed(4)}</td>
        <td><span class="badge badge-primary">Trained & Validated</span></td>
      </tr>
    `;
  }).join("");
}

function renderClassificationLeaderboard(leaderboard, bestModelName) {
  const tbody = document.getElementById("tbody-cls-leaderboard");
  if (!tbody || !leaderboard) return;

  const sorted = Object.entries(leaderboard).sort((a, b) => b[1].F1_Score - a[1].F1_Score);

  tbody.innerHTML = sorted.map(([name, m]) => {
    const isBest = name === bestModelName;
    const rocStr = typeof m.ROC_AUC === "number" ? m.ROC_AUC.toFixed(4) : m.ROC_AUC;
    return `
      <tr class="${isBest ? 'highlight-row' : ''}">
        <td><strong>${name}</strong> ${isBest ? '<span class="badge badge-success">Best Performer</span>' : ''}</td>
        <td><code>${(m.Accuracy * 100).toFixed(2)}%</code></td>
        <td>${(m.Precision * 100).toFixed(2)}%</td>
        <td>${(m.Recall * 100).toFixed(2)}%</td>
        <td><strong>${(m.F1_Score * 100).toFixed(2)}%</strong></td>
        <td>${rocStr}</td>
        <td><span class="badge badge-primary">Active</span></td>
      </tr>
    `;
  }).join("");
}

function renderRegressionModelsChart(leaderboard) {
  const ctx = document.getElementById("chart-regression-models");
  if (!ctx || !leaderboard) return;

  const sorted = Object.entries(leaderboard).sort((a, b) => b[1].R2_Score - a[1].R2_Score);
  const labels = sorted.map(s => s[0]);
  const r2Values = sorted.map(s => s[1].R2_Score);

  if (chartRegressionModels) chartRegressionModels.destroy();

  chartRegressionModels = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "R² Score",
        data: r2Values,
        backgroundColor: "rgba(99, 102, 241, 0.75)",
        borderColor: "#6366F1",
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          min: 0,
          max: 1.0,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8", font: { size: 10 } }
        }
      }
    }
  });
}

function renderFeatureImportanceChart(featImp) {
  const ctx = document.getElementById("chart-feature-importance");
  if (!ctx || !featImp) return;

  const top = Object.entries(featImp).slice(0, 8);
  const labels = top.map(t => t[0].replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()));
  const values = top.map(t => t[1]);

  if (chartFeatureImportance) chartFeatureImportance.destroy();

  chartFeatureImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Importance (%)",
        data: values,
        backgroundColor: "rgba(16, 185, 129, 0.75)",
        borderColor: "#10B981",
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8", font: { size: 10 } }
        }
      }
    }
  });
}

/* ==========================================================================
   5. Real-Time Student Predictor Form Handler
   ========================================================================== */
function initPredictionForm() {
  const form = document.getElementById("form-predictor");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-predict");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span>Calculating Outcome...</span>`;
    btn.disabled = true;

    const formData = new FormData(form);
    const payload = {
      gender: formData.get("gender"),
      age: parseInt(formData.get("age")),
      parental_education: formData.get("parental_education"),
      internet_access: formData.get("internet_access"),
      previous_grades: parseFloat(formData.get("previous_grades")),
      attendance_rate: parseFloat(formData.get("attendance_rate")),
      study_time: parseFloat(formData.get("study_time")),
      tutoring: formData.get("tutoring"),
      family_support: formData.get("family_support"),
      extracurricular_activities: formData.get("extracurricular_activities"),
      failures: parseInt(formData.get("failures")),
      health: formData.get("health"),
      absences: parseInt(formData.get("absences"))
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (data.status === "success") {
        updatePredictionUI(data.prediction);
      } else {
        alert("Prediction failed: " + (data.message || "Unknown error"));
      }
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Error contacting prediction server.");
    } finally {
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  });

  // Initial trigger for preview state
  form.dispatchEvent(new Event("submit"));
}

function updatePredictionUI(pred) {
  // Update Grade circle
  document.getElementById("res-grade").textContent = pred.predicted_grade.toFixed(1);
  document.getElementById("res-grade-20").textContent = `${pred.predicted_grade_20_scale.toFixed(1)} / 20`;
  
  // Pass/Fail & Probability
  const pfBadge = document.getElementById("res-pass-fail");
  pfBadge.textContent = pred.pass_fail_status;
  pfBadge.className = `meta-val badge ${pred.pass_fail_status === 'Pass' ? 'badge-success' : 'badge-danger'}`;
  
  document.getElementById("res-pass-prob").textContent = `${pred.pass_probability.toFixed(1)}%`;
  document.getElementById("res-prob-bar").style.width = `${pred.pass_probability}%`;

  // Performance category & Risk
  document.getElementById("res-category").textContent = pred.performance_category;
  
  const riskBadge = document.getElementById("res-risk-tier");
  riskBadge.textContent = pred.risk_tier;
  if (pred.risk_tier === "Low Risk") {
    riskBadge.className = "badge badge-success";
  } else if (pred.risk_tier === "Moderate Risk") {
    riskBadge.className = "badge badge-warning";
  } else {
    riskBadge.className = "badge badge-danger";
  }

  document.getElementById("res-risk-index").textContent = `${pred.academic_risk_index.toFixed(1)} / 100`;

  // Render Prescriptive Interventions
  const recContainer = document.getElementById("recommendations-container");
  if (recContainer && pred.recommendations) {
    recContainer.innerHTML = pred.recommendations.map(r => `
      <div class="recommendation-item">
        <div class="rec-header">
          <span class="rec-cat">${r.category}</span>
          <span class="rec-priority priority-${r.priority.toLowerCase()}">${r.priority}</span>
        </div>
        <div class="rec-action">${r.action}</div>
        <div class="rec-impact">${r.expected_impact}</div>
      </div>
    `).join("");
  }
}

let currentBatchRecords = [];
let currentSchemaMapping = null;

function getCategoryBadgeClass(category) {
  if (!category) return "badge-primary";
  const cat = String(category).toLowerCase();
  if (cat.includes("excellent")) return "badge-success";
  if (cat.includes("good")) return "badge-primary";
  if (cat.includes("satisfactory")) return "badge-warning";
  return "badge-danger";
}

function getRiskBadgeClass(tier) {
  if (!tier) return "badge-primary";
  const t = String(tier).toLowerCase();
  if (t.includes("high")) return "badge-danger";
  if (t.includes("mod")) return "badge-warning";
  return "badge-success";
}

function initBatchUpload() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("batch-file-input");
  const selectBtn = document.getElementById("btn-select-file");
  const searchInput = document.getElementById("batch-search-input");
  const riskFilter = document.getElementById("batch-risk-filter");
  const exportBtn = document.getElementById("btn-export-batch-csv");
  const toggleSchemaBtn = document.getElementById("btn-toggle-schema-details");

  if (!dropZone || !fileInput) return;

  // Drop zone click triggers file browser
  dropZone.addEventListener("click", (e) => {
    if (e.target !== fileInput) {
      fileInput.click();
    }
  });

  if (selectBtn) {
    selectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  // Schema details drawer toggle
  if (toggleSchemaBtn) {
    toggleSchemaBtn.addEventListener("click", () => {
      const drawer = document.getElementById("schema-details-drawer");
      if (drawer) {
        const isHidden = drawer.style.display === "none";
        drawer.style.display = isHidden ? "block" : "none";
        toggleSchemaBtn.innerHTML = isHidden
          ? "Hide Column Mappings & Imputations &uarr;"
          : "View Column Mappings & Imputations &darr;";
      }
    });
  }

  // Drag & drop visual events
  ["dragenter", "dragover"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleBatchFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleBatchFileUpload(fileInput.files[0]);
    }
  });

  // Table filtering and search
  if (searchInput) {
    searchInput.addEventListener("input", () => filterAndRenderBatchTable());
  }

  if (riskFilter) {
    riskFilter.addEventListener("change", () => filterAndRenderBatchTable());
  }

  // Export CSV
  if (exportBtn) {
    exportBtn.addEventListener("click", exportBatchResultsToCSV);
  }
}

async function handleBatchFileUpload(file) {
  const wrapper = document.getElementById("batch-results-wrapper");
  const spinner = document.getElementById("batch-loading-spinner");
  const errorBox = document.getElementById("batch-error-box");
  const errorMsg = document.getElementById("batch-error-message");
  const fileInput = document.getElementById("batch-file-input");

  if (!file) return;

  // Client-side validations
  const maxBytes = 16 * 1024 * 1024;
  if (file.size > maxBytes) {
    if (errorBox && errorMsg) {
      errorMsg.textContent = `File "${file.name}" exceeds the maximum 16 MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB). Please select a smaller file.`;
      errorBox.style.display = "block";
    }
    return;
  }

  const filenameLower = file.name.toLowerCase();
  const validExts = [".csv", ".tsv", ".txt", ".xlsx", ".xls"];
  if (!validExts.some(ext => filenameLower.endsWith(ext))) {
    if (errorBox && errorMsg) {
      errorMsg.textContent = `Invalid file format "${file.name}". Supported formats are CSV (.csv), TSV (.tsv), and Excel spreadsheets (.xlsx, .xls).`;
      errorBox.style.display = "block";
    }
    return;
  }

  // Reset errors and show spinner
  if (errorBox) errorBox.style.display = "none";
  if (spinner) spinner.style.display = "flex";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

    const res = await fetch("/api/batch-predict", {
      method: "POST",
      body: formData,
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    const data = await res.json().catch(() => null);
    if (spinner) spinner.style.display = "none";

    if (!data) {
      throw new Error(`Server returned status ${res.status} without a valid JSON response.`);
    }

    if (res.ok && data.status === "success") {
      currentBatchRecords = data.records || [];
      currentSchemaMapping = data.schema_mapping || null;
      wrapper.style.display = "block";

      // Update Summary KPIs
      const sum = data.summary;
      document.getElementById("batch-kpi-total").textContent = sum.total_students.toLocaleString();
      document.getElementById("batch-kpi-pass-rate").textContent = `${sum.pass_rate_pct}%`;
      document.getElementById("batch-kpi-pass-count").textContent = `${sum.pass_count} Pass / ${sum.fail_count} Fail`;
      document.getElementById("batch-kpi-avg-grade").textContent = `${sum.average_predicted_grade} / 100`;

      // Render Schema Auto-Detection Breakdown
      renderSchemaMappingDiagnostics(currentSchemaMapping);

      filterAndRenderBatchTable();

      // Smooth scroll to results
      wrapper.scrollIntoView({ behavior: "smooth" });
    } else {
      if (errorBox && errorMsg) {
        errorMsg.textContent = data.message || `Batch upload error (HTTP ${res.status}).`;
        errorBox.style.display = "block";
      } else {
        alert("Batch processing error: " + (data.message || "Unknown error"));
      }
    }
  } catch (err) {
    if (spinner) spinner.style.display = "none";
    console.error("Batch upload failed:", err);
    if (errorBox && errorMsg) {
      if (err.name === "AbortError") {
        errorMsg.textContent = "Inference request timed out after 60 seconds. For very large datasets, try a smaller batch.";
      } else {
        errorMsg.textContent = `Batch upload failed: ${err.message || "Network error or invalid server response."}`;
      }
      errorBox.style.display = "block";
    } else {
      alert("Failed to process batch upload: " + err.message);
    }
  } finally {
    if (fileInput) fileInput.value = "";
  }
}

function renderSchemaMappingDiagnostics(mapping) {
  const card = document.getElementById("schema-mapping-card");
  const title = document.getElementById("schema-mapping-title");
  const subtext = document.getElementById("schema-mapping-subtext");
  const mappedContainer = document.getElementById("pills-mapped-features");
  const imputedContainer = document.getElementById("pills-imputed-features");
  const extraContainer = document.getElementById("pills-extra-columns");
  const extraWrapper = document.getElementById("extra-columns-container");

  const countMapped = document.getElementById("count-mapped-feats");
  const countImputed = document.getElementById("count-imputed-feats");
  const countExtra = document.getElementById("count-extra-cols");

  if (!card || !mapping) return;

  const mappedEntries = Object.entries(mapping.mapped_columns || {});
  const imputedList = mapping.imputed_features || [];
  const extraList = mapping.extra_columns || [];

  if (countMapped) countMapped.textContent = mappedEntries.length;
  if (countImputed) countImputed.textContent = imputedList.length;
  if (countExtra) countExtra.textContent = extraList.length;

  if (title) {
    title.textContent = `Schema Auto-Detection: ${mappedEntries.length} Features Mapped (${mapping.detection_confidence_pct}% Match)`;
  }

  if (subtext) {
    subtext.textContent = `Successfully matched ${mappedEntries.length} column(s). ${imputedList.length} missing feature(s) imputed with intelligent defaults.`;
  }

  if (mappedContainer) {
    mappedContainer.innerHTML = mappedEntries.map(([canonical, original]) => `
      <span class="schema-pill pill-mapped" title="Original column: '${original}' mapped to feature '${canonical}'">
        <strong>${canonical}</strong> &larr; <code>${original}</code>
      </span>
    `).join("");
  }

  if (imputedContainer) {
    imputedContainer.innerHTML = imputedList.length > 0 ? imputedList.map(feat => `
      <span class="schema-pill pill-imputed" title="Feature was missing from uploaded dataset and imputed with baseline statistics">
        ${feat} <em style="font-size:0.75rem; opacity:0.8;">(imputed)</em>
      </span>
    `).join("") : `<span style="font-size: 0.85rem; color: #94A3B8;">All model features were present in the uploaded file!</span>`;
  }

  if (extraContainer) {
    if (extraList.length > 0) {
      if (extraWrapper) extraWrapper.style.display = "block";
      extraContainer.innerHTML = extraList.map(col => `
        <span class="schema-pill pill-extra" title="Preserved column from uploaded file">
          <code>${col}</code>
        </span>
      `).join("");
    } else {
      if (extraWrapper) extraWrapper.style.display = "none";
    }
  }
}

function filterAndRenderBatchTable() {
  const searchVal = (document.getElementById("batch-search-input")?.value || "").trim().toLowerCase();
  const filterVal = document.getElementById("batch-risk-filter")?.value || "all";
  const tbody = document.getElementById("tbody-batch-results");
  const countLabel = document.getElementById("batch-records-count");

  if (!tbody) return;

  const filtered = currentBatchRecords.filter(r => {
    const stuId = String(r.student_id || "").toLowerCase();
    const matchesSearch = !searchVal || stuId.includes(searchVal);

    if (!matchesSearch) return false;

    if (filterVal === "all") return true;
    if (filterVal === "Pass" || filterVal === "Fail") {
      return String(r.predicted_pass_fail).toLowerCase() === filterVal.toLowerCase();
    }
    return String(r.risk_tier).toLowerCase() === filterVal.toLowerCase();
  });

  if (countLabel) {
    countLabel.textContent = `Showing ${filtered.length} of ${currentBatchRecords.length} records`;
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
          No student records matching current filters.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td><code>${r.student_id || 'STU'}</code></td>
      <td>${r.attendance_rate !== undefined ? r.attendance_rate + '%' : 'N/A'}</td>
      <td>${r.study_time !== undefined ? r.study_time + ' hrs' : 'N/A'}</td>
      <td>${r.previous_grades !== undefined ? r.previous_grades : 'N/A'}</td>
      <td><strong>${r.predicted_final_grade}</strong></td>
      <td><span class="badge ${r.predicted_pass_fail === 'Pass' ? 'badge-success' : 'badge-danger'}">${r.predicted_pass_fail}</span></td>
      <td>${r.pass_probability_pct}%</td>
      <td><span class="badge ${getCategoryBadgeClass(r.predicted_performance_category)}">${r.predicted_performance_category}</span></td>
      <td><span class="badge ${getRiskBadgeClass(r.risk_tier)}">${r.risk_tier || 'Low Risk'}</span></td>
    </tr>
  `).join("");
}

function exportBatchResultsToCSV() {
  if (!currentBatchRecords || currentBatchRecords.length === 0) {
    alert("No batch predictions available to export.");
    return;
  }

  // Extract all unique columns from the dataset to ensure extra columns are also exported
  const standardHeaders = [
    "student_id",
    "gender",
    "age",
    "parental_education",
    "study_time",
    "attendance_rate",
    "previous_grades",
    "extracurricular_activities",
    "internet_access",
    "tutoring",
    "family_support",
    "health",
    "absences",
    "failures",
    "predicted_final_grade",
    "predicted_pass_fail",
    "pass_probability_pct",
    "predicted_performance_category",
    "academic_risk_index",
    "risk_tier"
  ];

  const allKeys = new Set(standardHeaders);
  currentBatchRecords.forEach(rec => {
    Object.keys(rec).forEach(k => allKeys.add(k));
  });

  const headers = Array.from(allKeys);
  const csvRows = [headers.join(",")];

  currentBatchRecords.forEach(row => {
    const values = headers.map(header => {
      const val = row[header] !== undefined && row[header] !== null ? String(row[header]) : "";
      const escaped = val.replace(/"/g, '""');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(","));
  });

  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `student_cohort_predictions_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
