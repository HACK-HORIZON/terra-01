(() => {
  // DOM Elements
  const dropZone = document.querySelector('#drop-zone');
  const fileInput = document.querySelector('#file-input');
  const fileStatus = document.querySelector('#file-status');
  const uploadPreview = document.querySelector('#upload-preview');
  const enhanceButton = document.querySelector('#enhance-button');
  const imageResult = document.querySelector('#image-result');
  const resultDisplay = document.querySelector('#result-display');
  const enhancedImage = document.querySelector('#enhanced-image');
  const originalImage = document.querySelector('#original-image');
  const comparisonContainer = document.querySelector('#comparison-container');
  const compOverlay = document.querySelector('#comp-overlay');
  const compSliderLine = document.querySelector('#comp-slider-line');
  const downloadButton = document.querySelector('#download-button');
  const metricsPanel = document.querySelector('#metrics-panel');
  const statusLabel = document.querySelector('.label-status');
  const engineBadge = document.querySelector('#engine-badge');
  const engineStatusText = document.querySelector('#engine-status-text');
  const optDehaze = document.querySelector('#opt-dehaze');
  const optSharpen = document.querySelector('#opt-sharpen');
  const optDeblur = document.querySelector('#opt-deblur');
  const optClouds = document.querySelector('#opt-clouds');
  const optObstacles = document.querySelector('#opt-obstacles');
  const tabButtons = document.querySelectorAll('.tab-btn');

  let selectedFile = null;
  let previewUrl = null;
  let isDraggingSlider = false;

  // Allow overriding backend URL via localStorage, or use same-origin proxy (Vercel rewrites)
  const storedApi = localStorage.getItem('TERRA_BACKEND_URL');
  const configuredApi = window.__TERRA_API_URL__ !== undefined ? window.__TERRA_API_URL__ : '';
  const API_BASE = (storedApi || configuredApi || '').replace(/\/$/, '');

  // 1. Health check & backend connection
  async function checkBackendHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (response.ok) {
        const data = await response.json();
        if (engineBadge && engineStatusText) {
          engineBadge.classList.remove('connecting', 'offline');
          engineStatusText.textContent = `Hybrid Engine: Online [${data.device}]`;
        }
      } else {
        throw new Error('Health check returned ' + response.status);
      }
    } catch (err) {
      if (engineBadge && engineStatusText) {
        engineBadge.classList.remove('connecting');
        engineBadge.classList.add('offline');
        engineStatusText.textContent = 'Hybrid Engine: Offline';
      }
    }
  }

  // 2. Set file helper
  function setFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      fileStatus.textContent = 'Please choose a PNG, JPG, WEBP, or TIFF image';
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    uploadPreview.src = previewUrl;
    dropZone.classList.add('has-file');
    fileStatus.textContent = `Imagery loaded · ${file.name}`;
    enhanceButton.disabled = false;
    dropZone.classList.remove('dropped');
    void dropZone.offsetWidth;
    dropZone.classList.add('dropped');
    statusLabel.textContent = 'IMAGE READY';
  }

  // 3. Preset sample buttons
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const sampleId = btn.dataset.sample;
      fileStatus.textContent = `Fetching ${sampleId} orbital sample…`;
      try {
        const url = `${API_BASE}/samples/sample_${sampleId}.jpg`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Sample not found');
        const blob = await resp.blob();
        const file = new File([blob], `sample_${sampleId}.jpg`, { type: 'image/jpeg' });
        setFile(file);
      } catch (err) {
        console.error('Error loading sample:', err);
        fileStatus.textContent = 'Could not load preset sample';
      }
    });
  });

  // Dropzone events
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach((eventName) =>
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add('dragover');
    })
  );
  ['dragleave', 'drop'].forEach((eventName) =>
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove('dragover');
    })
  );
  dropZone.addEventListener('drop', (event) => {
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      setFile(event.dataTransfer.files[0]);
    }
  });

  // 4. Enhance button -> POST /api/enhance
  enhanceButton.addEventListener('click', async () => {
    if (!selectedFile) return;

    enhanceButton.disabled = true;
    enhanceButton.classList.add('loading');
    enhanceButton.querySelector('.button-text').textContent = 'Processing Hybrid Net…';
    statusLabel.textContent = 'RUNNING CNN + TRANSFORMER';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('scale', 2);
    formData.append('apply_dehaze', optDehaze ? optDehaze.checked : true);
    formData.append('apply_sharpen', optSharpen ? optSharpen.checked : true);
    formData.append('denoise_level', 0.5);
    formData.append('remove_clouds', optClouds ? optClouds.checked : false);
    formData.append('remove_obstacles', optObstacles ? optObstacles.checked : false);
    formData.append('deblur', optDeblur ? optDeblur.checked : false);

    try {
      const response = await fetch(`${API_BASE}/api/enhance`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Server error ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch {
          const rawText = await response.text().catch(() => '');
          if (rawText) errorMsg = `${errorMsg}: ${rawText.slice(0, 150)}`;
        }
        throw new Error(errorMsg);
      }

      const res = await response.json();

      // Display enhanced and original images
      enhancedImage.src = res.enhanced_image;
      originalImage.src = res.original_image;
      downloadButton.href = res.enhanced_image;
      downloadButton.download = `sphere_enhanced_${selectedFile.name.replace(/\.[^/.]+$/, '')}_2x.png`;

      // Update metadata bar
      document.querySelector('#meta-res').textContent =
        `${res.input_resolution[0]}×${res.input_resolution[1]} → ${res.output_resolution[0]}×${res.output_resolution[1]}`;
      document.querySelector('#meta-time').textContent =
        `${(res.processing_time_ms / 1000).toFixed(2)}s`;
      document.querySelector('#meta-device').textContent = res.device;
      document.querySelector('#meta-arch').textContent = res.model_architecture;

      // Show result container & comparison slider
      imageResult.classList.add('has-image');
      resultDisplay.hidden = false;
      setSliderPosition(50);

      // Update Scientific Metrics with genuine values
      document.querySelector('#metric-psnr').textContent = res.metrics.psnr.toFixed(2);
      document.querySelector('#metric-ssim').textContent = res.metrics.ssim.toFixed(4);
      document.querySelector('#metric-rmse').textContent = res.metrics.rmse.toFixed(2);
      document.querySelector('#metric-mae').textContent = res.metrics.mae.toFixed(2);

      // Update metric progress bars
      const cards = document.querySelectorAll('.metric-card');
      if (cards.length >= 4) {
        // PSNR: scale 20-45 dB
        const psnrPct = Math.min(Math.max((res.metrics.psnr / 45) * 100, 15), 100);
        cards[0].querySelector('.metric-bar i').style.width = `${psnrPct.toFixed(0)}%`;

        // SSIM: scale 0-1
        const ssimPct = Math.min(Math.max(res.metrics.ssim * 100, 15), 100);
        cards[1].querySelector('.metric-bar i').style.width = `${ssimPct.toFixed(0)}%`;

        // RMSE: lower is better (0-20 error)
        const rmsePct = Math.min(Math.max((1 - res.metrics.rmse / 25) * 100, 20), 100);
        cards[2].querySelector('.metric-bar i').style.width = `${rmsePct.toFixed(0)}%`;

        // MAE: lower is better (0-15 error)
        const maePct = Math.min(Math.max((1 - res.metrics.mae / 20) * 100, 20), 100);
        cards[3].querySelector('.metric-bar i').style.width = `${maePct.toFixed(0)}%`;
      }

      // Update metrics caption
      const metricsCaption = document.querySelector('.metrics-caption');
      if (metricsCaption) {
        metricsCaption.innerHTML = '<span>✦</span> Metrics vs. bicubic upscaled input. <b>Model trained 100 epochs</b> (loss 0.011→0.005). For true SR PSNR 28-45dB, evaluate with paired HR ground truth.';
      }

      metricsPanel.hidden = false;
      metricsPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
      statusLabel.textContent = 'ENHANCEMENT COMPLETE';
    } catch (err) {
      console.error('Enhancement error:', err);
      statusLabel.textContent = 'ERROR ENHANCING';
      alert('Error during enhancement: ' + err.message);
    } finally {
      enhanceButton.classList.remove('loading');
      enhanceButton.querySelector('.button-text').textContent = 'Enhance Again';
      enhanceButton.disabled = false;
    }
  });

  // 5. Interactive Comparison Slider
  function setSliderPosition(percentage) {
    percentage = Math.max(0, Math.min(100, percentage));
    compOverlay.style.width = `${percentage}%`;
    compSliderLine.style.left = `${percentage}%`;
  }

  function updateSliderFromEvent(e) {
    const rect = comparisonContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const offset = clientX - rect.left;
    const percentage = (offset / rect.width) * 100;
    setSliderPosition(percentage);
  }

  comparisonContainer.addEventListener('mousedown', (e) => {
    isDraggingSlider = true;
    updateSliderFromEvent(e);
  });
  window.addEventListener('mousemove', (e) => {
    if (isDraggingSlider) updateSliderFromEvent(e);
  });
  window.addEventListener('mouseup', () => {
    isDraggingSlider = false;
  });

  comparisonContainer.addEventListener('touchstart', (e) => {
    isDraggingSlider = true;
    updateSliderFromEvent(e);
  }, { passive: true });
  window.addEventListener('touchmove', (e) => {
    if (isDraggingSlider) updateSliderFromEvent(e);
  }, { passive: true });
  window.addEventListener('touchend', () => {
    isDraggingSlider = false;
  });

  // 6. View Tabs (Split / Enhanced / Original)
  tabButtons.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabButtons.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const mode = tab.dataset.view;
      if (mode === 'split') {
        compSliderLine.style.display = 'block';
        compOverlay.style.display = 'block';
        setSliderPosition(50);
      } else if (mode === 'enhanced') {
        compSliderLine.style.display = 'none';
        compOverlay.style.display = 'none';
      } else if (mode === 'original') {
        compSliderLine.style.display = 'none';
        compOverlay.style.display = 'block';
        compOverlay.style.width = '100%';
      }
    });
  });

  // 7. Three.js Globe Visualization
  function initGlobe() {
    if (!window.THREE) return;
    const container = document.querySelector('#globe-container');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.z = 3.1;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      container.classList.add('no-webgl');
      container.innerHTML = '<div class="fallback-globe" aria-hidden="true"><span></span></div>';
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.7));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const loader = new THREE.TextureLoader();
    const earthTexture = loader.load(
      'https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg'
    );
    const earthBump = loader.load(
      'https://threejs.org/examples/textures/planets/earth_normal_2048.jpg'
    );
    earthTexture.colorSpace = THREE.SRGBColorSpace;
    const globe = new THREE.Mesh(
      new THREE.SphereGeometry(1.18, 64, 48),
      new THREE.MeshPhongMaterial({
        map: earthTexture,
        bumpMap: earthBump,
        bumpScale: 0.045,
        shininess: 18,
        specular: new THREE.Color(0x31586e),
      })
    );
    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.23, 64, 48),
      new THREE.MeshBasicMaterial({
        color: 0x58dfe8,
        transparent: true,
        opacity: 0.1,
        side: THREE.BackSide,
      })
    );
    const sun = new THREE.DirectionalLight(0xffffff, 2.3);
    sun.position.set(-3, 2, 4);
    scene.add(sun);
    scene.add(new THREE.AmbientLight(0x193a52, 0.34));
    const group = new THREE.Group();
    group.add(globe, atmosphere);
    scene.add(group);

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener('resize', resize);
    const animate = () => {
      requestAnimationFrame(animate);
      group.rotation.y += 0.0018;
      group.rotation.x = Math.sin(Date.now() * 0.00012) * 0.035;
      renderer.render(scene, camera);
    };
    animate();
  }

  // 8. Auth Modal handlers
  const loginModal = document.querySelector('#login-modal');
  const loginForm = document.querySelector('#login-form');
  const signupForm = document.querySelector('#signup-form');
  const loginView = document.querySelector('#login-view');
  const signupView = document.querySelector('#signup-view');
  const loginStatus = document.querySelector('#login-status');
  const signupStatus = document.querySelector('#signup-status');
  const loginEmail = document.querySelector('#login-email');
  const signupName = document.querySelector('#signup-name');
  let lastFocusedElement;

  function closeLogin() {
    loginModal.hidden = true;
    loginModal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('modal-open');
    if (lastFocusedElement) lastFocusedElement.focus();
  }
  function openLogin(event) {
    event.preventDefault();
    lastFocusedElement = event.currentTarget;
    loginModal.hidden = false;
    loginModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    loginStatus.textContent = '';
    requestAnimationFrame(() => loginEmail.focus());
  }
  document.querySelector('[data-login-open]').addEventListener('click', openLogin);
  document.querySelectorAll('[data-login-close]').forEach((button) =>
    button.addEventListener('click', closeLogin)
  );
  document.querySelectorAll('[data-auth-switch]').forEach((link) =>
    link.addEventListener('click', (event) => {
      event.preventDefault();
      const signup = event.currentTarget.dataset.authSwitch === 'signup';
      loginView.hidden = signup;
      signupView.hidden = !signup;
      loginStatus.textContent = '';
      signupStatus.textContent = '';
      requestAnimationFrame(() => (signup ? signupName : loginEmail).focus());
    })
  );
  document.querySelector('[data-login-help]').addEventListener('click', (event) => {
    event.preventDefault();
    loginStatus.textContent = 'Account support is available after sign in.';
  });
  loginForm.addEventListener('submit', (event) => {
    event.preventDefault();
    loginStatus.textContent = 'Demo mode: your details are ready to connect.';
  });
  signupForm.addEventListener('submit', (event) => {
    event.preventDefault();
    signupStatus.textContent = 'Demo mode: your Sphere account is ready to create.';
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !loginModal.hidden) closeLogin();
  });

  // Initialize
  initGlobe();
  checkBackendHealth();
})();
