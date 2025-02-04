async function makePrediction() {
    const resultDiv = document.getElementById('result');
    const ba = parseFloat(document.getElementById('ba').value);
    const obp = parseFloat(document.getElementById('obp').value);
    const slg = parseFloat(document.getElementById('slg').value);

    // Hide the resultDiv initially
    resultDiv.style.display = 'none';

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ba, obp, slg })
    });

    const result = await response.json();

    predictions = result.predictions[0];
    const baText = `Majors BA\n ${Math.round(predictions[0] * 1000) / 1000}`;
    const obpText = `Majors OBP\n ${Math.round(predictions[1] * 1000) / 1000}`;
    const slgText = `Majors SLG\n ${Math.round(predictions[2] * 1000) / 1000}`;

    if (response.ok) {
        typeText(document.getElementById('majors-ba'), baText);
        typeText(document.getElementById('majors-obp'), obpText);
        typeText(document.getElementById('majors-slg'), slgText);
    } else {
        typeText(document.getElementById('majors-ba'), 'Majors BA: Error');
        typeText(document.getElementById('majors-obp'), 'Majors OBP: Error');
        typeText(document.getElementById('majors-slg'), 'Majors SLG: Error');
    }

    // Show the resultDiv after receiving the response
    resultDiv.style.display = 'block';
}

function typeText(element, text, delay = 50) {
    element.textContent = '';
    let i = 0;
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, delay);
        }
    }
    type();
}