document.addEventListener("DOMContentLoaded", function() {
    // Animate progress bar fill on results page
    const progressBar = document.querySelector(".progress-bar");
    if (progressBar) {
        let confidence = parseFloat(progressBar.textContent.replace('%', ''));
        progressBar.style.width = "0%"; // Start from 0%
        
        setTimeout(() => {
            progressBar.style.width = confidence + "%";
        }, 500);
    }

    // Smooth scrolling for hero button
    const startButton = document.querySelector(".btn");
    if (startButton) {
        startButton.addEventListener("click", function(event) {
            event.preventDefault();
            window.location.href = "/form";
        });
    }

    // Add hover effect to buttons
    const buttons = document.querySelectorAll(".btn");
    buttons.forEach(btn => {
        btn.addEventListener("mouseover", () => {
            btn.style.opacity = "0.8";
        });
        btn.addEventListener("mouseout", () => {
            btn.style.opacity = "1";
        });
    });

    // Form validation to ensure input values are between 0-10
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function(event) {
            const inputs = document.querySelectorAll("input[type='number']");
            let isValid = true;

            inputs.forEach(input => {
                if (input.value < 0 || input.value > 10) {
                    isValid = false;
                    alert(`Invalid value for ${input.name}. Please enter a number between 0 and 10.`);
                }
            });

            if (!isValid) {
                event.preventDefault();
            }
        });
    }
});
