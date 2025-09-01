// Hide result box when user edits inputs again
(function(){
    var ids = ['fnam','snam','tnam','fonam'];
    function hideResult(){
      var box = document.getElementById('result');
      if (box) { box.style.display = 'none'; }
    }
    ids.forEach(function(id){
      var el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('focus', hideResult);
      el.addEventListener('change', hideResult);
      el.addEventListener('input', hideResult);
    });
  })();

// Training progress functionality
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form[action*="s1"]');
    const trainingProgress = document.getElementById('training-progress');
    const trainingBar = document.getElementById('training-bar');
    const trainingText = document.getElementById('training-text');
    const firstTimeNotice = document.getElementById('first-time-notice');
    const resultBox = document.getElementById('result');

    if (form) {
        form.addEventListener('submit', function(e) {
            // Hide first time notice when predict button is clicked
            if (firstTimeNotice) {
                firstTimeNotice.style.display = 'none';
            }
            
            // Show training progress bar
            if (trainingProgress) {
                trainingProgress.style.display = 'block';
                trainingBar.style.width = '0%';
                trainingText.textContent = '0%';
            }
            
            // Hide previous result
            if (resultBox) {
                resultBox.style.display = 'none';
            }

            // Start progress animation (simulated for now)
            simulateProgress();
        });
    }

    function simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15; // Random progress increments
            if (progress > 100) progress = 100;
            
            trainingBar.style.width = progress + '%';
            trainingText.textContent = Math.round(progress) + '%';
            
            if (progress >= 100) {
                clearInterval(interval);
                // Hide progress bar after completion
                setTimeout(() => {
                    if (trainingProgress) {
                        trainingProgress.style.display = 'none';
                    }
                }, 1000);
            }
        }, 200); // Update every 200ms
    }
});