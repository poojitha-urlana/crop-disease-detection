document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const resultDiv = document.getElementById('result');
    const captureButton = document.getElementById('capture');
    const sendButton = document.getElementById('send');

    // Start video stream from camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            console.error("Error accessing camera: ", err);
        });

    // Capture photo from video stream
    captureButton.addEventListener('click', () => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.style.display = 'block';
    });

    // Send photo to Flask app for prediction
    sendButton.addEventListener('click', () => {
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('image', blob, 'captured_image.jpg');

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }, 'image/jpeg');
    });

    function displayResult(data) {
        if (data.error) {
            resultDiv.innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
        } else {
            resultDiv.innerHTML = '<p>Prediction: ' + data.prediction + '</p>';
            resultDiv.innerHTML += '<p>Solution: ' + data.solution + '</p>';
        }
    }
});
