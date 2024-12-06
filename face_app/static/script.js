document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission
    const formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if (data.result) {
            // Display the result
            resultDiv.innerHTML = `<p>Prediction: ${data.result}</p>`;
        } else if (data.error) {
            // Display the error
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p style="color: red;">Unexpected error occurred.</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p style="color: red;">Error: Unable to process your request.</p>`;
    });
});
