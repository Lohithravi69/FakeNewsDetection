// static/script.js

document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('news');
    const charCount = document.getElementById('charCount');
    const form = document.getElementById('predictForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.spinner-border');

    // Character counter
    textarea.addEventListener('input', function() {
        const count = textarea.value.length;
        charCount.textContent = `${count} characters`;
    });

    // Form validation and spinner
    form.addEventListener('submit', function(e) {
        const text = textarea.value.trim();
        if (text.length < 10) {
            e.preventDefault();
            alert('Please enter at least 10 characters.');
            return;
        }
        // Show spinner and disable button
        spinner.classList.remove('d-none');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Checking...';
    });
});
