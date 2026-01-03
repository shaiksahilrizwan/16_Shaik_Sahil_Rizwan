document.getElementById('recForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    // 1. Collect Data from UI
    const budgetCheckboxes = document.querySelectorAll('input[name="budget"]:checked');
    const budgetRange = Array.from(budgetCheckboxes).map(cb => parseInt(cb.value));
    
    const minRam = document.getElementById('minRam').value;
    const require4G = document.getElementById('require4G').checked;
    const userIntent = document.getElementById('userIntent').value;

    // Build the payload
    const payload = {
        budget_range: budgetRange.length > 0 ? budgetRange : [0, 1, 2, 3], // Default to all if none selected
        requires_4g: require4G,
        user_intent: userIntent
    };

    if (minRam) {
        payload.min_ram = parseInt(minRam);
    }

    // 2. UI State -> Loading
    const resultSection = document.getElementById('resultSection');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.getElementById('loadingSpinner');
    const outputContent = document.getElementById('outputContent');
    
    resultSection.classList.remove('hidden');
    spinner.classList.remove('hidden');
    outputContent.classList.add('hidden');
    submitBtn.disabled = true;
    submitBtn.innerText = "Processing...";

    try {
        // 3. Send to Backend
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // 4. Update UI with Results
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            document.getElementById('resId').innerText = "Rec ID: " + (data.recommendation_id || "N/A");
            document.getElementById('resScore').innerText = "Confidence: " + (data.confidence_score || 0) * 100 + "%";
            document.getElementById('resDevice').innerText = "Device ID: " + (data.selected_device_id || "Unknown");
            document.getElementById('resIntent').innerText = data.user_intent;
            document.getElementById('resReasoning').innerText = data.reasoning;
            
            document.getElementById('rawJson').innerText = JSON.stringify(data, null, 2);
            outputContent.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to the server.');
    } finally {
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
        submitBtn.innerText = "Get Recommendation";
    }
});