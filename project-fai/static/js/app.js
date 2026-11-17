/**
 * ============================================================
 * AI Crop Advisor — Frontend Application
 * Handles image upload, prediction API calls, and results display
 * ============================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM REFERENCES ----
    const uploadZone      = document.getElementById('uploadZone');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const imagePreview    = document.getElementById('imagePreview');
    const previewImg      = document.getElementById('previewImg');
    const previewFilename = document.getElementById('previewFilename');
    const fileInput       = document.getElementById('fileInput');
    const removeImgBtn    = document.getElementById('removeImg');

    const soilType    = document.getElementById('soilType');
    const growthStage = document.getElementById('growthStage');
    const severity    = document.getElementById('severity');
    const apiKey      = document.getElementById('apiKey');

    const advancedToggle  = document.getElementById('advancedToggle');
    const advancedContent = document.getElementById('advancedContent');
    const toggleChevron   = document.getElementById('toggleChevron');

    const predictBtn = document.getElementById('predictBtn');
    const btnText    = document.getElementById('btnText');
    const btnLoading = document.getElementById('btnLoading');

    const resultsEmpty   = document.getElementById('resultsEmpty');
    const resultsError   = document.getElementById('resultsError');
    const resultsContent = document.getElementById('resultsContent');
    const errorMessage   = document.getElementById('errorMessage');
    const retryBtn       = document.getElementById('retryBtn');

    const validationBanner = document.getElementById('validationBanner');
    const validationIcon   = document.getElementById('validationIcon');
    const validationText   = document.getElementById('validationText');

    const gaugeFill  = document.getElementById('gaugeFill');
    const gaugeValue = document.getElementById('gaugeValue');
    const resultCrop    = document.getElementById('resultCrop');
    const resultDisease = document.getElementById('resultDisease');
    const resultStatus  = document.getElementById('resultStatus');

    const top3List       = document.getElementById('top3List');
    const advisoryBadge  = document.getElementById('advisoryBadge');
    const advisoryContent = document.getElementById('advisoryContent');

    // State
    let currentFile = null;
    let isAnalyzing = false;

    // ---- NAVBAR SCROLL EFFECT ----
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(2, 44, 34, 0.95)';
        } else {
            navbar.style.background = 'rgba(2, 44, 34, 0.85)';
        }
    });

    // ---- HERO SCANNER INTERACTIVE SHOWCASE ----
    const heroScannerImg     = document.getElementById('heroScannerImg');
    const heroBadgeMatch     = document.getElementById('heroBadgeMatch');
    const heroBadgeCondition = document.getElementById('heroBadgeCondition');
    const reticleLabel       = document.getElementById('reticleLabel');
    const heroLoadSampleBtn  = document.getElementById('heroLoadSampleBtn');
    let currentHeroSampleId  = '0';

    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const id = btn.dataset.id;
            currentHeroSampleId = id;
            if (heroScannerImg) {
                heroScannerImg.style.opacity = '0.3';
                heroScannerImg.src = `/sample-image/${id}`;
                heroScannerImg.onload = () => {
                    heroScannerImg.style.opacity = '1';
                };
            }
            if (heroBadgeMatch) heroBadgeMatch.textContent = btn.dataset.match || '99.0% Match';
            if (heroBadgeCondition) heroBadgeCondition.textContent = btn.dataset.label || 'Crop Condition';
            if (reticleLabel) reticleLabel.textContent = btn.dataset.reticle || 'Detected Feature';
        });
    });

    if (heroLoadSampleBtn) {
        heroLoadSampleBtn.addEventListener('click', () => {
            loadExample(currentHeroSampleId);
        });
    }

    // Hero crop pills click to jump to crop section
    document.querySelectorAll('.hero-crops .crop-pill').forEach(pill => {
        pill.style.cursor = 'pointer';
        pill.addEventListener('click', () => {
            const cropSection = document.getElementById('crops');
            if (cropSection) {
                cropSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ---- ADVANCED SETTINGS TOGGLE ----
    advancedToggle.addEventListener('click', () => {
        const isOpen = advancedContent.classList.toggle('open');
        toggleChevron.classList.toggle('rotated', isOpen);
    });

    // ---- FILE UPLOAD ----
    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-remove-image')) return;
        fileInput.click();
    });

    // Keyboard accessibility
    uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove image
    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPG, PNG, or WEBP).');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewFilename.textContent = file.name;
            uploadPlaceholder.style.display = 'none';
            imagePreview.style.display = 'flex';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        currentFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewFilename.textContent = '';
        uploadPlaceholder.style.display = '';
        imagePreview.style.display = 'none';
        predictBtn.disabled = true;
    }

    // ---- EXAMPLE IMAGES ----
    document.querySelectorAll('.example-item').forEach(item => {
        item.addEventListener('click', () => loadExample(item.dataset.id));
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                loadExample(item.dataset.id);
            }
        });
    });

    function loadExample(id) {
        const url = `/sample-image/${id}`;

        // Show loading state on the example
        fetch(url)
            .then(res => res.blob())
            .then(blob => {
                const filename = `sample_${id}.jpg`;
                const file = new File([blob], filename, { type: blob.type });
                handleFile(file);

                // Scroll to analyzer section
                document.getElementById('analyzer').scrollIntoView({ behavior: 'smooth', block: 'start' });
            })
            .catch(() => {
                showError('Failed to load example image.');
            });
    }

    // ---- RETRY BUTTON ----
    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            showResultState('empty');
        });
    }

    // ---- PREDICTION ----
    predictBtn.addEventListener('click', runPrediction);

    async function runPrediction() {
        if (!currentFile || isAnalyzing) return;

        isAnalyzing = true;
        setLoadingState(true);

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('soil_type', soilType.value);
        formData.append('growth_stage', growthStage.value);
        formData.append('severity', severity.value);
        formData.append('api_key', apiKey.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            displayResults(result);

        } catch (err) {
            showError(err.message || 'Network error. Please check your connection and try again.');
        } finally {
            isAnalyzing = false;
            setLoadingState(false);
        }
    }

    function setLoadingState(loading) {
        predictBtn.disabled = loading;
        btnText.style.display = loading ? 'none' : 'flex';
        btnLoading.style.display = loading ? 'flex' : 'none';
    }

    // ---- DISPLAY RESULTS ----
    function displayResults(result) {
        if (!result.success) {
            showError(result.error || 'An error occurred during prediction.');
            return;
        }

        // Show results panel
        showResultState('content');

        // Scroll to results on mobile
        if (window.innerWidth <= 1024) {
            setTimeout(() => {
                resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        // --- Validation Banner ---
        const isValid = result.is_valid !== false;
        validationBanner.className = 'validation-banner ' + (isValid ? 'valid' : 'warning');
        validationIcon.className = isValid ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle';
        validationText.textContent = result.validation_status || 'Valid crop-leaf image';

        if (!isValid) {
            // Invalid image - show minimal results
            resultCrop.textContent = '—';
            resultDisease.textContent = '—';
            setGauge(0);
            setStatus('Invalid Input', false);
            top3List.innerHTML = '<p style="color:var(--gray-400);font-size:0.875rem;">No predictions available for invalid input.</p>';
            advisoryBadge.style.display = 'none';
            advisoryContent.innerHTML = `<p style="color:var(--gray-500);">${result.confidence_note || 'The uploaded image does not appear to contain a clear crop leaf.'}</p>`;
            return;
        }

        // --- Crop & Disease ---
        const crop = result.crop || '—';
        const disease = result.disease || '—';
        const isConfident = result.is_confident;

        resultCrop.textContent = isConfident ? crop : 'Uncertain';
        resultDisease.textContent = isConfident ? disease : `${disease} (Tentative)`;

        // --- Confidence Gauge ---
        const confVal = result.confidence_val || 0;
        setGauge(confVal);

        // --- Status ---
        setStatus(result.status_text || '—', isConfident);

        // --- Top 3 Predictions ---
        renderTop3(result.top_3 || []);

        // --- Advisory ---
        renderAdvisory(result, isConfident);
    }

    // ---- GAUGE ANIMATION ----
    function setGauge(value) {
        const circumference = 2 * Math.PI * 58; // r=58 from SVG
        const percentage = Math.min(Math.max(value, 0), 1);
        const offset = circumference * (1 - percentage);

        // Update gradient color based on value
        const gradient = document.getElementById('gaugeGradient');
        if (gradient) {
            const stops = gradient.querySelectorAll('stop');
            if (percentage >= 0.6) {
                stops[0].style.stopColor = '#059669';
                stops[1].style.stopColor = '#10B981';
            } else if (percentage >= 0.3) {
                stops[0].style.stopColor = '#D97706';
                stops[1].style.stopColor = '#F59E0B';
            } else {
                stops[0].style.stopColor = '#DC2626';
                stops[1].style.stopColor = '#EF4444';
            }
        }

        // Animate fill
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
        }, 100);

        // Animate number
        animateNumber(gaugeValue, 0, Math.round(percentage * 100), 1000, '%');
    }

    function animateNumber(element, start, end, duration, suffix = '') {
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * eased);

            element.textContent = current + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ---- STATUS BADGE ----
    function setStatus(text, confident) {
        resultStatus.textContent = text;
        resultStatus.className = 'detail-value status-badge ' + (confident ? 'confident' : 'uncertain');
    }

    // ---- TOP 3 PREDICTIONS ----
    function renderTop3(predictions) {
        if (!predictions || predictions.length === 0) {
            top3List.innerHTML = '<p style="color:var(--gray-400);font-size:0.875rem;">No predictions available.</p>';
            return;
        }

        top3List.innerHTML = predictions.map((pred, i) => {
            const pct = (pred.probability * 100).toFixed(1);
            return `
                <div class="top3-item">
                    <div class="top3-rank rank-${i + 1}">${pred.rank}</div>
                    <div class="top3-info">
                        <div class="top3-name">${escapeHtml(pred.display_name)}</div>
                        <div class="top3-bar-track">
                            <div class="top3-bar-fill bar-${i + 1}" data-width="${pct}"></div>
                        </div>
                    </div>
                    <div class="top3-pct">${pred.percentage}</div>
                </div>
            `;
        }).join('');

        // Animate bars after render
        setTimeout(() => {
            top3List.querySelectorAll('.top3-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        }, 200);
    }

    // ---- ADVISORY ----
    function renderAdvisory(result, isConfident) {
        if (!isConfident) {
            advisoryBadge.style.display = 'none';
            advisoryContent.innerHTML = `
                <div style="text-align:center;padding:var(--sp-8) 0;">
                    <i class="fas fa-exclamation-circle" style="font-size:2rem;color:var(--amber-500);margin-bottom:var(--sp-3);display:block;"></i>
                    <p style="color:var(--gray-600);font-size:0.9375rem;line-height:1.7;">
                        No disease-specific pesticide recommendation is provided because the model confidence is below the 60% threshold.<br><br>
                        Please upload a clear close-up image of the crop leaf.
                    </p>
                </div>
            `;
            return;
        }

        const advisory = result.advisory;
        if (!advisory) {
            advisoryBadge.style.display = 'none';
            advisoryContent.innerHTML = '<p style="color:var(--gray-400);">No advisory available.</p>';
            return;
        }

        // Show badge
        advisoryBadge.style.display = 'inline-block';
        if (advisory.is_ai) {
            advisoryBadge.textContent = '✨ AI-Powered';
            advisoryBadge.className = 'advisory-badge ai-badge';
        } else {
            advisoryBadge.textContent = '📋 Reference';
            advisoryBadge.className = 'advisory-badge ref-badge';
        }

        // Render markdown
        const markdownText = advisory.ai_text || '';
        if (typeof marked !== 'undefined' && marked.parse) {
            advisoryContent.innerHTML = marked.parse(markdownText);
        } else {
            // Fallback: basic rendering
            advisoryContent.innerHTML = markdownText
                .replace(/### (.*)/g, '<h3>$1</h3>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/^- (.*)/gm, '<li>$1</li>')
                .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
                .replace(/\n\n/g, '<br><br>')
                .replace(/\n/g, '<br>');
        }
    }

    // ---- STATE MANAGEMENT ----
    function showResultState(state) {
        resultsEmpty.style.display   = state === 'empty'   ? '' : 'none';
        resultsError.style.display   = state === 'error'   ? '' : 'none';
        resultsContent.style.display = state === 'content' ? '' : 'none';
    }

    function showError(message) {
        errorMessage.textContent = message;
        showResultState('error');
    }

    // ---- UTILITIES ----
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ---- SMOOTH SCROLL FOR NAV LINKS ----
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            const targetId = anchor.getAttribute('href');
            if (targetId === '#') return;
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                const offset = 80; // navbar height
                const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({ top, behavior: 'smooth' });
            }
        });
    });

    // ---- MOBILE MENU ----
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.querySelector('.nav-links');
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
            navLinks.style.flexDirection = 'column';
            navLinks.style.position = 'absolute';
            navLinks.style.top = '64px';
            navLinks.style.right = '24px';
            navLinks.style.background = 'rgba(2,44,34,0.95)';
            navLinks.style.padding = '16px 24px';
            navLinks.style.borderRadius = '12px';
            navLinks.style.gap = '12px';
            navLinks.style.boxShadow = '0 10px 25px rgba(0,0,0,0.3)';
        });
    }
});
