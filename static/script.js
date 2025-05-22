document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const imageInput = document.getElementById('imageInput');
    const operation = document.querySelector('input[name="operation"]:checked').value;

    const file = imageInput.files[0];

    // Tampilkan gambar original
    const reader = new FileReader();
    reader.onload = function(e) {
        const originalImage = document.getElementById('originalImage');
        originalImage.src = e.target.result;
        originalImage.style.display = 'block';
    };
    reader.readAsDataURL(file);

    formData.append('image', file);
    formData.append('operation', operation);

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const outputImage = document.getElementById('outputImage');
        const downloadLink = document.getElementById('downloadLink');

        outputImage.src = url;
        outputImage.style.display = 'block';
        downloadLink.href = url;
        downloadLink.style.display = 'inline-block';
    });
});
