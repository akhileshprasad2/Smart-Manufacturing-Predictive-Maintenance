
let modelChart = null;

async function loadDashboard() {
    try {
        const response = await fetch("/api/dashboard");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Dashboard loading failed");
        }

        document.getElementById("ai4iRecords").textContent =
            data.datasets.AI4I.records;

        document.getElementById("secomRecords").textContent =
            data.datasets.SECOM.records;

        document.getElementById("totalPredictions").textContent =
            data.total_predictions;

        document.getElementById("totalFailures").textContent =
            data.total_failures_detected;

    } catch(error) {
        console.error(error);
    }
}


async function predictAI4I(button) {

    
    const result = document.getElementById("predictionResult");

    const fields = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ];

    const data = {};

    for (const field of fields) {
        const element = document.getElementById(field);
        const value = element.value;

        if (!value.trim()) {
            alert("Please fill all sensor values.");
            return;
        }

        data[field] = field === "Type" ? value : Number(value);

        if (field !== "Type" && !Number.isFinite(data[field])) {
            alert("Please enter valid numeric values.");
            return;
        }
    }

    button.disabled = true;
    button.textContent = "Predicting...";

    try {
        const response = await fetch("/api/predict/ai4i", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.error || "Prediction failed");
        }

        const statusClass =
            output.prediction === 1 ? "danger" :
            output.failure_probability >= 30 ? "warning" :
            "success";

        result.classList.remove("hidden");

        result.innerHTML = `
            <h3 class="${statusClass}">
                ${output.prediction_label}
            </h3>

            <p>Failure Probability:
            <strong>${output.failure_probability}%</strong></p>

            <p>Machine Status: ${output.machine_status}</p>

            <p>Best Model: ${output.model_used}</p>

            <p>Recommendation: ${output.recommendation}</p>
        `;

        await refreshData();

    } catch(error) {
        result.classList.remove("hidden");
        result.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Predict Machine Failure";
    }
}



async function predictSECOM(button) {

    const fileInput = document.getElementById("secomFile");
    const result = document.getElementById("secomResult");

    if (!fileInput.files.length) {
        alert("Please select a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    button.disabled = true;
    button.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/predict/secom", {
            method: "POST",
            body: formData
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.error || "SECOM prediction failed");
        }

        result.classList.remove("hidden");

        result.innerHTML = `
            <h3 class="success">Analysis Completed</h3>

            <p>Total Records: ${output.total_rows}</p>

            <p>Predicted Failures: ${output.failure_count}</p>

            <p>Normal Records: ${output.normal_count}</p>

            <p>Model Used: ${output.model_used || "Trained SECOM Model"}</p>

            <a href="/api/download/secom-results"
               style="color:#38bdf8"
               download>
               Download Prediction Results
            </a>
        `;

        await refreshData();

    } catch(error) {
        result.classList.remove("hidden");
        result.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze SECOM Dataset";
    }
}

async function loadModels() {

    try {
        const response = await fetch("/api/model-comparison");
        const models = await response.json();

        if (!response.ok) {
            throw new Error("Model comparison loading failed");
        }

        const table = document.getElementById("modelTable");

        table.innerHTML = models.map(model => `
            <tr>
                <td>${model.Dataset}</td>
                <td>${model.Model}</td>
                <td>${(model.Accuracy * 100).toFixed(2)}%</td>
                <td>${(model.Precision * 100).toFixed(2)}%</td>
                <td>${(model.Recall * 100).toFixed(2)}%</td>
                <td>${(model["F1 Score"] * 100).toFixed(2)}%</td>
                <td>${(model["ROC AUC"] * 100).toFixed(2)}%</td>
            </tr>
        `).join("");

        const labels = models.map(
            model => model.Dataset + " - " + model.Model
        );

        if (modelChart) {
            modelChart.destroy();
        }

        modelChart = new Chart(
            document.getElementById("modelChart"),
            {
                type: "bar",

                data: {
                    labels: labels,

                    datasets: [{
                        label: "F1 Score (%)",

                        data: models.map(
                            model => model["F1 Score"] * 100
                        )
                    }]
                },

                options: {
                    responsive: true,

                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            }
        );

    } catch(error) {
        console.error(error);
    }
}


async function loadHistory() {

    try {
        const response = await fetch("/api/history");
        const data = await response.json();

        const table = document.getElementById("historyTable");

        table.innerHTML = data.history.map(item => `
            <tr>
                <td>${item.dataset}</td>
                <td>${item.prediction === 1 ? "Failure" : "Normal"}</td>
                <td>${item.failure_probability}%</td>
                <td>${item.status}</td>
            </tr>
        `).join("");

    } catch(error) {
        console.error(error);
    }
}


async function refreshData() {
    await loadDashboard();
    await loadHistory();
}


document.addEventListener("DOMContentLoaded", async () => {
    await refreshData();
    await loadModels();
});
