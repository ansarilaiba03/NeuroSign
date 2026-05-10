// ===========================
// Configuration
// ===========================
const API_URL = 'http://localhost:5000';

// ===========================
// DOM Elements
// ===========================
const hamburger = document.getElementById('hamburger');
const navMenu   = document.querySelector('.nav-menu');

const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn  = document.getElementById('stopCameraBtn');

const recognizedGesture = document.getElementById('recognizedGesture');
const gestureHistory    = document.getElementById('gestureHistory');

const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');

const textOutput = document.getElementById('textOutput');

const speakBtn = document.getElementById('speakBtn');

const contactForm = document.getElementById('contactForm');

// ===========================
// State
// ===========================
let isCameraActive = false;

// ===========================
// Hamburger Menu
// ===========================
if (hamburger) {

    hamburger.addEventListener(
        'click',
        () => navMenu.classList.toggle('active')
    );
}

document.querySelectorAll('.nav-menu a')
.forEach(link => {

    link.addEventListener(
        'click',
        () => navMenu.classList.remove('active')
    );
});

// ===========================
// Auto Start
// ===========================
document.addEventListener(
    'DOMContentLoaded',
    () => {

        console.log(
            'NeuroSign Frontend Loaded'
        );

        // Backend already owns webcam
        isCameraActive = true;

        // Buttons
        if (startCameraBtn)
            startCameraBtn.disabled = true;

        if (stopCameraBtn)
            stopCameraBtn.disabled = false;

        // Static UI
        if (recognizedGesture)
            recognizedGesture.textContent =
                'Prediction shown on camera feed';

        if (confidenceFill)
            confidenceFill.style.width = '100%';

        if (confidenceText)
            confidenceText.textContent = 'LIVE';

        if (gestureHistory)
            gestureHistory.innerHTML =
                '<span class="history-item">Backend ML Active</span>';

        if (textOutput)
            textOutput.innerHTML =
                '<span class="placeholder">Predictions are displayed directly on the camera stream.</span>';

        if (speakBtn)
            speakBtn.disabled = true;
    }
);

// ===========================
// Camera Buttons
// ===========================
function startCamera() {

    isCameraActive = true;

    if (startCameraBtn)
        startCameraBtn.disabled = true;

    if (stopCameraBtn)
        stopCameraBtn.disabled = false;

    if (recognizedGesture)
        recognizedGesture.textContent =
            'Backend camera active';
}

function stopCamera() {

    isCameraActive = false;

    if (startCameraBtn)
        startCameraBtn.disabled = false;

    if (stopCameraBtn)
        stopCameraBtn.disabled = true;

    if (recognizedGesture)
        recognizedGesture.textContent =
            'Camera stopped';
}

// ===========================
// Text To Speech
// ===========================
function speakText() {

    alert(
        'Speech disabled in backend-only demo mode.'
    );
}

// ===========================
// Clear Output
// ===========================
function clearOutput() {

    if (textOutput) {

        textOutput.innerHTML =
            '<span class="placeholder">Predictions are displayed directly on the camera stream.</span>';
    }
}

// ===========================
// Contact Form
// ===========================
if (contactForm) {

    contactForm.addEventListener(
        'submit',
        e => {

            e.preventDefault();

            const btn =
                contactForm.querySelector(
                    'button[type="submit"]'
                );

            const orig =
                btn.textContent;

            btn.textContent =
                'Message Sent!';

            btn.disabled = true;

            contactForm.reset();

            setTimeout(
                () => {

                    btn.textContent = orig;

                    btn.disabled = false;
                },
                3000
            );
        }
    );
}

// ===========================
// Smooth Scroll
// ===========================
document.querySelectorAll('a[href^="#"]')
.forEach(a => {

    a.addEventListener(
        'click',
        function(e) {

            const href =
                this.getAttribute('href');

            if (href === '#') return;

            e.preventDefault();

            document.querySelector(href)
            ?.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    );
});

// ===========================
// Animations
// ===========================
const observer =
    new IntersectionObserver(
        entries => {

            entries.forEach(en => {

                if (en.isIntersecting)

                    en.target.style.animation =
                        en.target.style.animation ||
                        'fadeInUp 0.8s ease-out forwards';
            });

        },
        {
            threshold: 0.1,
            rootMargin:
                '0px 0px -100px 0px'
        }
    );

document.querySelectorAll(
    '.workflow-step, .app-card, .future-card'
)
.forEach(el => {

    el.style.opacity = '0';

    observer.observe(el);
});

// ===========================
// Navbar + Parallax
// ===========================
let lastST = 0;

window.addEventListener(
    'scroll',
    () => {

        const st = window.scrollY;

        if (
            Math.abs(st - lastST) > 5
        ) {

            navMenu.classList.remove(
                'active'
            );
        }

        lastST = st;

        const hv =
            document.querySelector(
                '.hero-visual'
            );

        if (hv) {

            hv.style.transform =
                `translateY(${st * 0.3}px)`;
        }
    }
);

console.log('NeuroSign Ready');