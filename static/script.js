// Konfigurasi API
const API_CONFIG = {
    baseURL: window.location.origin,
    endpoints: {
        processImage: '/process-image',
        detectWebcam: '/detect-webcam',
        health: '/health'
    }
};

// =================================================================
// DEFINISI JENIS FRAKTUR - DETAILED EXPLANATIONS
// =================================================================
const fractureExplanations = {
    "Comminuted": {
        description: "Patah tulang yang hancur menjadi beberapa potongan kecil",
        mechanism: "Biasanya akibat benturan sangat keras seperti kecelakaan mobil atau jatuh dari ketinggian",
        characteristics: "Tulang pecah menjadi banyak bagian, seperti kaca yang pecah",
        complications: "Sulit sembuh, bisa infeksi, butuh operasi kompleks",
        treatment: "Harus dioperasi untuk menyambung potongan tulang dengan plat dan sekrup",
        healing: "3-6 bulan atau lebih lama",
        severity: "SANGAT SERIUS"
    },
    "Greenstick": {
        description: "Patah tulang tidak sempurna, seperti ranting pohon muda yang patah",
        mechanism: "Biasanya pada anak-anak karena tulang mereka masih lentur",
        characteristics: "Satu sisi tulang patah, sisi lainnya hanya bengkok",
        complications: "Bisa menyebabkan pertumbuhan tulang tidak normal",
        treatment: "Dibuat lurus dan dipasang gips",
        healing: "3-6 minggu untuk anak-anak",
        severity: "RINGAN"
    },
    "Healthy": {
        description: "Tulang dalam kondisi normal dan sehat",
        mechanism: "Tidak ada cedera atau trauma",
        characteristics: "Bentuk tulang normal, tidak ada garis patahan",
        complications: "Tidak ada",
        treatment: "Tidak perlu penanganan khusus",
        healing: "Tidak perlu waktu penyembuhan",
        severity: "NORMAL"
    },
    "Linear": {
        description: "Patah tulang berupa garis lurus sepanjang tulang",
        mechanism: "Biasanya karena tekanan atau benturan langsung",
        characteristics: "Garis patahan lurus, tulang masih sejajar",
        complications: "Biasanya mudah disembuhkan",
        treatment: "Dipasang gips atau penyangga",
        healing: "6-8 minggu",
        severity: "RINGAN"
    },
    "Oblique": {
        description: "Patah tulang miring seperti garis diagonal",
        mechanism: "Kombinasi tekanan dan gaya memutar",
        characteristics: "Garis patahan miring, agak tidak stabil",
        complications: "Bisa lama sembuh, kadang butuh operasi",
        treatment: "Dibuat lurus dan dipasang pen atau sekrup",
        healing: "2-3 bulan",
        severity: "SEDANG"
    },
    "Transverse": {
        description: "Patah tulang lurus melintang seperti batang yang dipatahkan",
        mechanism: "Benturan langsung atau tekanan dari samping",
        characteristics: "Garis patahan horizontal lurus",
        complications: "Bisa tidak sejajar jika tidak ditangani baik",
        treatment: "Dipasang gips atau pen di dalam tulang",
        healing: "2-2.5 bulan",
        severity: "SEDANG"
    },
    "Oblique Displaced": {
       description: "Patah tulang miring dengan pergeseran posisi",
        mechanism: "Benturan keras disertai putaran",
        characteristics: "Tulang bergeser dari posisi normal, garis patahan miring",
        complications: "Bisa mengenai saraf atau pembuluh darah",
        treatment: "Harus dioperasi untuk menyetel ulang dan dipasang plat",
        healing: "2.5-4 bulan",
        severity: "SERIUS"
    },
    "Transverse Displaced": {
        description: "Patah tulang melintang dengan pergeseran",
        mechanism: "Benturan sangat keras dari samping",
        characteristics: "Tulang tidak sejajar, ada celah di garis patahan",
        complications: "Sulit sembuh, bisa permanen tidak sejajar",
        treatment: "Harus dioperasi dengan pen atau plat",
        healing: "3-5 bulan",
        severity: "SERIUS"
    },
    "Segmental": {
        description: "Patah tulang di dua tempat, membuat satu bagian tulang terlepas",
        mechanism: "Benturan sangat keras di beberapa titik",
        characteristics: "Ada dua garis patahan dengan satu potongan tulang terpisah",
        complications: "Sangat sulit sembuh, risiko infeksi tinggi",
        treatment: "Operasi kompleks dengan cangkok tulang",
        healing: "4-8 bulan atau lebih",
        severity: "SANGAT SERIUS"
    },
    "Spiral": {
        description: "Patah tulang berputar seperti pembuka botol",
        mechanism: "Kaki atau tangan terpuntir dengan keras",
        characteristics: "Garis patahan melingkar seperti spiral",
        complications: "Bisa menyebabkan tulang bengkok setelah sembuh",
        treatment: "Dipasang gips atau dioperasi tergantung tingkat keparahan",
        healing: "2-3 bulan",
        severity: "SEDANG"
    }
};

// =================================================================
// SCABIES EXPLANATION
// =================================================================
const scabiesExplanation = {
    description: "Infestasi kulit sangat menular yang disebabkan oleh tungau Sarcoptes scabiei",
    mechanism: "Kontak kulit langsung dengan penderita atau barang yang terkontaminasi",
    characteristics: "Gatal hebat terutama malam hari, garis halus berkelok (terowongan tungau), bintik merah & gelembung kecil di sela jari, pergelangan, ketiak",
    complications: "Infeksi bakteri sekunder, impetigo, selulitis, glomerulonefritis post-streptococcal",
    treatment: "Permethrin 5% krim (gold standard), ivermectin oral, antihistamin untuk gatal",
    healing: "2-4 minggu setelah pengobatan tepat",
    severity: "MENULAR"
};

// Variabel Global
let currentAnalysisType = '';
let uploadedFile = null;
let webcamStream = null;
let detectInterval = null;

// =================================================================
// MODAL FUNCTIONS
// =================================================================
function openUploadModal(type) {
    const modal = document.getElementById('uploadModal');
    const title = document.getElementById('modalTitle');
    
    const titles = {
        'xray': 'Upload X-Ray',
        'mri': 'Upload MRI',
        'skin': 'Upload Foto Kulit'
    };
    
    currentAnalysisType = type;
    title.textContent = titles[type] || 'Upload Gambar';
    modal.classList.add('show');
    setupFileUpload();
}

function closeUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.classList.remove('show');
    }
    clearPreview();
}

function openCameraModal() {
    const modal = document.getElementById('cameraModal');
    if (modal) {
        modal.classList.add('show');
        loadCameras();
    }
}

function closeCameraModal() {
    const modal = document.getElementById('cameraModal');
    if (modal) {
        modal.classList.remove('show');
    }
    stopCamera();
}

function closeResultModal() {
    const modal = document.getElementById('resultModal');
    if (modal) {
        modal.remove();
    }
}

// =================================================================
// FILE UPLOAD HANDLING
// =================================================================
function setupFileUpload() {
    const uploadZone = document.querySelector('.upload-zone');
    const fileInput = document.getElementById('fileInput');

    // Reset state
    clearPreview();

    // Handle file selection
    if (fileInput) {
        fileInput.onchange = (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        };
    }

    // Handle drag and drop
    if (uploadZone) {
        ['dragover', 'dragleave', 'drop'].forEach(event => {
            uploadZone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                if (event === 'dragover') {
                    uploadZone.style.borderColor = '#3b82f6';
                    uploadZone.style.backgroundColor = '#eff6ff';
                } else if (event === 'dragleave') {
                    uploadZone.style.borderColor = '#d1d5db';
                    uploadZone.style.backgroundColor = '#f9fafb';
                } else if (event === 'drop') {
                    uploadZone.style.borderColor = '#d1d5db';
                    uploadZone.style.backgroundColor = '#f9fafb';
                    
                    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                        handleFile(e.dataTransfer.files[0]);
                    }
                }
            });
        });
    }
}

function handleFile(file) {
    console.log('File selected:', file);
    
    if (!file.type.startsWith('image/')) {
        alert('Harap pilih file gambar (JPG, PNG)');
        return;
    }

    // Validasi ukuran file (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('Ukuran file terlalu besar. Maksimal 10MB.');
        return;
    }

    uploadedFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        const previewContainer = document.getElementById('previewContainer');
        const imagePreview = document.getElementById('imagePreview');

        if (imagePreview) {
            imagePreview.src = e.target.result;
            imagePreview.onload = () => {
                console.log('Preview image loaded');
            };
        }
        if (previewContainer) {
            previewContainer.classList.add('show');
            previewContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    reader.onerror = (error) => {
        console.error('File reading error:', error);
        alert('Error membaca file. Coba lagi dengan file yang berbeda.');
    };

    reader.readAsDataURL(file);
}

function clearPreview() {
    const previewContainer = document.getElementById('previewContainer');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');

    if (previewContainer) previewContainer.classList.remove('show');
    if (fileInput) fileInput.value = '';
    if (imagePreview) imagePreview.src = '';
    uploadedFile = null;
}

// =================================================================
// IMAGE ANALYSIS
// =================================================================
async function analyzeImage() {
    console.log('Starting image analysis...');
    
    if (!uploadedFile) {
        alert('Pilih gambar terlebih dahulu');
        return;
    }

    const analyzeBtn = document.querySelector('.analyze-btn');
    const originalText = analyzeBtn ? analyzeBtn.innerHTML : 'Analisis';

    // Update tombol loading
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i data-feather="loader" class="animate-spin"></i> Menganalisis...';
        analyzeBtn.disabled = true;
    }
    
    feather.replace();

    try {
        console.log('Starting analysis for:', currentAnalysisType);
        
        // Tentukan image_type berdasarkan currentAnalysisType
        let imageType = 'skin_upload';
        if (currentAnalysisType === 'xray') imageType = 'bone_xray';
        else if (currentAnalysisType === 'mri') imageType = 'bone_mri';

        console.log('Sending to API:', imageType);
        
        // Panggil API Flask
        const result = await callAIAnalysis(imageType, uploadedFile);
        console.log('Analysis result:', result);
        
        // Tampilkan hasil
        showAnalysisResult(imageType, result);

    } catch (error) {
        console.error('Analysis error:', error);
        alert('Terjadi error saat menganalisis gambar: ' + error.message);
    } finally {
        // Reset tombol
        if (analyzeBtn) {
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
        }
        feather.replace();
    }
}

async function callAIAnalysis(imageType, imageFile) {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("image_type", imageType);

    console.log('Calling API:', API_CONFIG.endpoints.processImage);
    
    const response = await fetch(API_CONFIG.endpoints.processImage, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    
    if (data.error) {
        throw new Error(data.error);
    }

    return data;
}

// =================================================================
// RESULT DISPLAY WITH IMAGE ENHANCEMENT COMPARISON
// =================================================================
function showAnalysisResult(imageType, result) {
    console.log('Displaying result for:', imageType);
    
    let resultHTML = '';
    
    if (imageType.includes('bone')) {
        resultHTML = generateBoneResultHTML(result);
    } else if (imageType.includes('skin')) {
        resultHTML = generateSkinResultHTML(result);
    }

    // Hapus modal hasil sebelumnya
    closeResultModal();

    // Buat modal hasil baru
    const resultModal = `
        <div class="modal show" id="resultModal">
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Hasil Analisis</h3>
                    <button onclick="closeResultModal()" class="modal-close">
                        <i data-feather="x"></i>
                    </button>
                </div>
                <div class="modal-body">
                    ${resultHTML}
                    <div style="margin-top: 2rem; display: flex; justify-content: flex-end; gap: 1rem;">
                        <button onclick="closeResultModal()" class="service-button xray">
                            <i data-feather="check"></i>
                            Tutup
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Tambahkan ke body
    document.body.insertAdjacentHTML('beforeend', resultModal);
    feather.replace();
    closeUploadModal();
}

function generateBoneResultHTML(data) {
    const isNormal = data.label === "Healthy" || data.label === "Normal";
    const confidence = data.confidence ? data.confidence.toFixed(2) : '0.00';
    const fractureInfo = fractureExplanations[data.label] || fractureExplanations["Healthy"];
    
    let html = `
        <div style="display: flex; flex-direction: column; gap: 1.5rem;">
            <!-- Perbandingan Gambar -->
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="image"></i>
                    Perbandingan Gambar
                </div>
                <div class="result-content">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div style="text-align: center;">
                            <h4 style="margin-bottom: 0.5rem; color: #6b7280;">Gambar Asli</h4>
                            <img src="${data.original_image_base64 || data.sr_image_base64 || ''}" alt="Original Image" 
                                 style="width: 100%; max-width: 300px; border-radius: 8px; border: 1px solid #e5e7eb;">
                        </div>
                        <div style="text-align: center;">
                            <h4 style="margin-bottom: 0.5rem; color: #3b82f6;">
                                Gambar Enhanced 
                                <span style="font-size: 0.8rem; color: #10b981;">(${data.enhancement_used || 'ESRGAN'})</span>
                            </h4>
                            <img src="${data.enhanced_image_base64 || data.sr_image_base64 || ''}" alt="Enhanced Image" 
                                 style="width: 100%; max-width: 300px; border-radius: 8px; border: 2px solid #3b82f6;">
                        </div>
                    </div>
                    <p style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem; text-align: center;">
                        <strong>Note:</strong> Analisis dilakukan pada gambar yang telah ditingkatkan kualitasnya menggunakan ${data.enhancement_used || 'ESRGAN'}
                    </p>
                </div>
            </div>

            <!-- Diagnosis Utama -->
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="${isNormal ? 'check-circle' : 'alert-triangle'}"></i>
                    Diagnosis Utama
                </div>
                <div class="result-content" style="color: ${isNormal ? '#10b981' : '#ef4444'}; font-weight: 600; font-size: 1.2rem;">
                    ${data.label || 'Tidak terdeteksi'}
                    <div style="font-size: 1rem; color: #6b7280; margin-top: 0.5rem;">
                        Tingkat Kepercayaan: ${confidence}%
                    </div>
                </div>
            </div>
            
            <!-- Penjelasan Fraktur -->
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="info"></i>
                    Penjelasan Medis
                </div>
                <div class="result-content">
                    <div style="background: ${isNormal ? '#f0fdf4' : '#fef3f2'}; padding: 1.5rem; border-radius: 8px; border: 1px solid ${isNormal ? '#bbf7d0' : '#fecaca'};">
                        <h4 style="color: ${isNormal ? '#059669' : '#dc2626'}; margin-bottom: 1rem;">${fractureInfo.description}</h4>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <strong>Mekanisme Cedera:</strong><br>
                                <span style="font-size: 0.9em;">${fractureInfo.mechanism}</span>
                            </div>
                            <div>
                                <strong>Karakteristik:</strong><br>
                                <span style="font-size: 0.9em;">${fractureInfo.characteristics}</span>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <strong>Komplikasi:</strong><br>
                                <span style="font-size: 0.9em;">${fractureInfo.complications}</span>
                            </div>
                            <div>
                                <strong>Penanganan:</strong><br>
                                <span style="font-size: 0.9em;">${fractureInfo.treatment}</span>
                            </div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                            <div>
                                <strong>Waktu Penyembuhan:</strong><br>
                                <span style="font-size: 0.9em;">${fractureInfo.healing}</span>
                            </div>
                            <div style="padding: 0.5rem 1rem; background: ${getSeverityColor(fractureInfo.severity)}; color: white; border-radius: 20px; font-weight: 600;">
                                ${fractureInfo.severity}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
    `;

    // Tampilkan detailed predictions jika ada
    if (data.all_predictions && Object.keys(data.all_predictions).length > 0) {
        html += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="bar-chart-2"></i>
                    Detail Prediksi Semua Kelas
                </div>
                <div class="result-content">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
        `;
        
        for (const [className, confidence] of Object.entries(data.all_predictions)) {
            const isHigh = parseFloat(confidence) > 20;
            const isCurrent = className === data.label;
            html += `
                <div style="padding: 1rem; background: ${isCurrent ? '#dbeafe' : (isHigh ? '#dcfce7' : '#f3f4f6')}; 
                     border-radius: 8px; border: 2px solid ${isCurrent ? '#3b82f6' : (isHigh ? '#bbf7d0' : '#e5e7eb')};">
                    <div style="font-weight: 600; font-size: 0.9rem; color: ${isCurrent ? '#1e40af' : '#374151'};">${className}</div>
                    <div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.25rem;">${confidence}</div>
                </div>
            `;
        }
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }

    // Tampilkan GradCAM jika ada
    if (data.gradcam) {
        html += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="map"></i>
                    Heatmap AI (GradCAM)
                </div>
                <div class="result-content">
                    <img src="${data.gradcam}" alt="GradCAM Heatmap" 
                         style="width: 100%; max-width: 400px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <p style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;">
                        <strong>Penjelasan:</strong> Area merah menunjukkan bagian gambar yang paling diperhatikan oleh AI dalam membuat diagnosis
                    </p>
                </div>
            </div>
        `;
    }

    // Disclaimer Medis
    html += `
        <div class="result-item">
            <div class="result-content">
                <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border: 1px solid #ffc107;">
                    <strong>Disclaimer Medis:</strong> Hasil ini merupakan alat bantu diagnosis. 
                    Diagnosis dan penanganan definitif harus ditentukan oleh dokter spesialis ortopedi.
                </div>
            </div>
        </div>
    `;

    html += `</div>`;
    return html;
}

function generateSkinResultHTML(data) {
    const isNormal = data.label ? data.label.toLowerCase().includes("normal") : true;
    const confidence = data.confidence ? data.confidence.toFixed(2) : '0.00';
    
    let html = `
        <div style="display: flex; flex-direction: column; gap: 1.5rem;">
            <!-- Perbandingan Gambar -->
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="image"></i>
                    Perbandingan Gambar
                </div>
                <div class="result-content">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div style="text-align: center;">
                            <h4 style="margin-bottom: 0.5rem; color: #6b7280;">Gambar Asli</h4>
                            <img src="${data.original_image_base64 || data.sr_image_base64 || ''}" alt="Original Image" 
                                 style="width: 100%; max-width: 300px; border-radius: 8px; border: 1px solid #e5e7eb;">
                        </div>
                        <div style="text-align: center;">
                            <h4 style="margin-bottom: 0.5rem; color: #3b82f6;">
                                Gambar Enhanced 
                                <span style="font-size: 0.8rem; color: #10b981;">(${data.enhancement_used || 'ESRGAN'})</span>
                            </h4>
                            <img src="${data.enhanced_image_base64 || data.sr_image_base64 || ''}" alt="Enhanced Image" 
                                 style="width: 100%; max-width: 300px; border-radius: 8px; border: 2px solid #3b82f6;">
                        </div>
                    </div>
                    <p style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem; text-align: center;">
                        <strong>Note:</strong> Analisis dilakukan pada gambar yang telah ditingkatkan kualitasnya menggunakan ${data.enhancement_used || 'ESRGAN'}
                    </p>
                </div>
            </div>

            <!-- Diagnosis Utama -->
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="${isNormal ? 'check-circle' : 'alert-triangle'}"></i>
                    Diagnosis Utama
                </div>
                <div class="result-content" style="color: ${isNormal ? '#10b981' : '#ef4444'}; font-weight: 600; font-size: 1.2rem;">
                    ${data.label || 'Tidak terdeteksi'}
                    <div style="font-size: 1rem; color: #6b7280; margin-top: 0.5rem;">
                        Tingkat Kepercayaan: ${confidence}%
                    </div>
                </div>
            </div>
    `;

    if (!isNormal) {
        // Penjelasan Detail Scabies
        html += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="info"></i>
                    Penjelasan Medis Scabies
                </div>
                <div class="result-content">
                    <div style="background: #fef3f2; padding: 1.5rem; border-radius: 8px; border: 1px solid #fecaca;">
                        <h4 style="color: #dc2626; margin-bottom: 1rem;">${scabiesExplanation.description}</h4>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <strong>Penyebab:</strong><br>
                                <span style="font-size: 0.9em;">${scabiesExplanation.mechanism}</span>
                            </div>
                            <div>
                                <strong>Gejala Khas:</strong><br>
                                <span style="font-size: 0.9em;">${scabiesExplanation.characteristics}</span>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <strong>Komplikasi:</strong><br>
                                <span style="font-size: 0.9em;">${scabiesExplanation.complications}</span>
                            </div>
                            <div>
                                <strong>Penanganan:</strong><br>
                                <span style="font-size: 0.9em;">${scabiesExplanation.treatment}</span>
                            </div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                            <div>
                                <strong>Perkiraan Sembuh:</strong><br>
                                <span style="font-size: 0.9em;">${scabiesExplanation.healing}</span>
                            </div>
                            <div style="padding: 0.5rem 1rem; background: #dc2626; color: white; border-radius: 20px; font-weight: 600;">
                                ${scabiesExplanation.severity}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Tampilkan gambar YOLO jika ada
    if (data.yolo_annotated_image) {
        html += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="target"></i>
                    Hasil Deteksi YOLO
                </div>
                <div class="result-content">
                    <img src="${data.yolo_annotated_image}" alt="YOLO Detection" 
                         style="width: 100%; max-width: 400px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <p style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;">
                        <strong>Penjelasan:</strong> Kotak hijau menunjukkan area yang terdeteksi sebagai scabies oleh model YOLO
                    </p>
                </div>
            </div>
        `;
    }

    // Tampilkan detail deteksi YOLO jika ada
    if (data.yolo_boxes && data.yolo_boxes.length > 0) {
        html += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="alert-triangle"></i>
                    Detail Deteksi Scabies
                </div>
                <div class="result-content">
                    <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; border: 1px solid #fecaca;">
                        <strong>Ditemukan ${data.yolo_boxes.length} area terdeteksi scabies:</strong>
                        <div style="margin-top: 0.5rem;">
        `;
        
        data.yolo_boxes.forEach((box, index) => {
            html += `
                <div style="font-size: 0.9rem; color: #dc2626; margin: 0.25rem 0;">
                    • Area ${index + 1}: Confidence ${(box.confidence * 100).toFixed(1)}%
                </div>
            `;
        });
        
        html += `
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Rekomendasi berdasarkan hasil
    html += `
        <div class="result-item">
            <div class="result-title">
                <i data-feather="alert-circle"></i>
                Rekomendasi Medis
            </div>
            <div class="result-content">
                <div style="color: ${isNormal ? '#059669' : '#dc2626'}; background: ${isNormal ? '#f0fdf4' : '#fef2f2'}; padding: 1rem; border-radius: 8px; border: 1px solid ${isNormal ? '#bbf7d0' : '#fecaca'};">
                    <strong>${isNormal ? 'HASIL NORMAL' : 'PERHATIAN!'}</strong><br>
                    ${isNormal 
                        ? 'Tidak ditemukan indikasi scabies. Kondisi kulit tampak normal berdasarkan analisis AI.' 
                        : 'Segera konsultasi dengan dokter kulit untuk konfirmasi diagnosis dan penanganan tepat. Scabies adalah kondisi menular yang membutuhkan pengobatan medis.'}
                </div>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-content">
                <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border: 1px solid #ffc107;">
                    <strong>Disclaimer Medis:</strong> Hasil ini merupakan alat bantu diagnosis. 
                    Diagnosis dan penanganan definitif harus ditentukan oleh dokter spesialis kulit.
                </div>
            </div>
        </div>
    `;

    html += `</div>`;
    return html;
}

// Helper function untuk warna severity
function getSeverityColor(severity) {
    const colors = {
        "NORMAL": "#28a745",
        "RENDAH": "#ffc107", 
        "SEDANG": "#fd7e14",
        "TINGGI": "#dc3545",
        "SANGAT TINGGI": "#721c24",
        "MENULAR": "#dc2626"
    };
    return colors[severity] || "#6b7280";
}

// =================================================================
// CAMERA FUNCTIONS
// =================================================================
async function loadCameras() {
    const cameraSelect = document.getElementById('cameraSelect');
    if (!cameraSelect) return;

    cameraSelect.innerHTML = '<option value="">Mendeteksi kamera...</option>';

    try {
        // Dapatkan daftar perangkat
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(device => device.kind === 'videoinput');

        if (cameras.length === 0) {
            cameraSelect.innerHTML = '<option value="">Tidak ada kamera yang terdeteksi</option>';
            return;
        }

        cameraSelect.innerHTML = '<option value="">Pilih kamera...</option>';
        cameras.forEach((camera, index) => {
            const label = camera.label || `Kamera ${index + 1}`;
            cameraSelect.innerHTML += `<option value="${camera.deviceId}">${label}</option>`;
        });

    } catch (error) {
        console.error('Error loading cameras:', error);
        cameraSelect.innerHTML = '<option value="">Error memuat kamera</option>';
    }
}

async function startCamera() {
    const cameraSelect = document.getElementById('cameraSelect');
    const video = document.getElementById('cameraFeed');
    const placeholder = document.getElementById('cameraPlaceholder');
    const startBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureImage');
    const stopBtn = document.getElementById('stopCamera');

    if (!cameraSelect || !video) {
        alert('Elemen kamera tidak ditemukan');
        return;
    }

    const selectedCameraId = cameraSelect.value;
    if (!selectedCameraId) {
        alert('Pilih kamera terlebih dahulu');
        return;
    }

    try {
        // Stop existing stream
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }

        // Start new stream
        const constraints = {
            video: {
                deviceId: { exact: selectedCameraId },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = webcamStream;
        
        // Tampilkan video, sembunyikan placeholder
        video.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
        
        // Update UI controls
        if (startBtn) startBtn.style.display = 'none';
        if (captureBtn) captureBtn.style.display = 'flex';
        if (stopBtn) stopBtn.style.display = 'flex';
        if (cameraSelect) cameraSelect.disabled = true;
        
        console.log('Camera started successfully');

    } catch (err) {
        console.error('Error starting camera:', err);
        alert('Tidak dapat mengakses kamera: ' + err.message);
    }
}

function stopCamera() {
    // Stop stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    // Reset UI
    const video = document.getElementById('cameraFeed');
    const placeholder = document.getElementById('cameraPlaceholder');
    const cameraSelect = document.getElementById('cameraSelect');
    const startBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureImage');
    const stopBtn = document.getElementById('stopCamera');
    const results = document.getElementById('analysisResults');

    if (video) {
        video.style.display = 'none';
        video.srcObject = null;
    }
    if (placeholder) placeholder.style.display = 'flex';
    if (cameraSelect) cameraSelect.disabled = false;
    if (startBtn) startBtn.style.display = 'flex';
    if (captureBtn) captureBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'none';

    // Reset results
    if (results) {
        results.innerHTML = `
            <div class="no-results-state">
                <i data-feather="bar-chart-2" class="no-results-icon"></i>
                <p>Belum ada analisis</p>
                <span>Hasil akan muncul di sini setelah analisis</span>
            </div>
        `;
        feather.replace();
    }

    console.log('Camera stopped');
}

async function captureImage() {
    const video = document.getElementById('cameraFeed');
    const canvas = document.createElement('canvas');
    const results = document.getElementById('analysisResults');

    if (!video || !results) {
        alert('Elemen kamera tidak ditemukan');
        return;
    }

    if (!webcamStream) {
        alert('Kamera belum dinyalakan');
        return;
    }

    try {
        // Capture frame dari video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Show loading
        results.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i data-feather="loader" class="animate-spin" style="width: 2rem; height: 2rem; color: #3b82f6;"></i>
                <p style="color: #6b7280; margin-top: 1rem;">Menganalisis gambar...</p>
            </div>
        `;
        feather.replace();

        // Convert to blob and analyze
        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append("file", blob, "capture.jpg");
                formData.append("image_type", "skin_upload");

                const response = await fetch(API_CONFIG.endpoints.processImage, {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }

                // Show results
                showCameraAnalysisResult(data);

            } catch (error) {
                console.error('Analysis error:', error);
                results.innerHTML = `
                    <div style="text-align: center; padding: 2rem; color: #ef4444;">
                        <i data-feather="alert-triangle" style="width: 2rem; height: 2rem;"></i>
                        <p style="margin-top: 1rem;">Error: ${error.message}</p>
                    </div>
                `;
                feather.replace();
            }
        }, 'image/jpeg', 0.8);

    } catch (error) {
        console.error('Capture error:', error);
        alert('Error mengambil gambar: ' + error.message);
    }
}

function showCameraAnalysisResult(data) {
    const results = document.getElementById('analysisResults');
    if (!results) return;

    const isNormal = data.label ? data.label.toLowerCase().includes("normal") : true;
    const confidence = data.confidence ? data.confidence.toFixed(2) : '0.00';

    let resultHTML = `
        <div style="display: flex; flex-direction: column; gap: 1rem;">
            <!-- Info Enhancement -->
            <div style="background: #dbeafe; padding: 0.75rem; border-radius: 6px; border: 1px solid #3b82f6;">
                <div style="display: flex; align-items: center; gap: 0.5rem; color: #1e40af;">
                    <i data-feather="zap" style="width: 16px; height: 16px;"></i>
                    <span style="font-size: 0.9rem;">
                        Gambar telah ditingkatkan kualitasnya menggunakan <strong>${data.enhancement_used || 'ESRGAN'}</strong>
                    </span>
                </div>
            </div>

            <!-- Gambar Hasil -->
            <div style="text-align: center;">
                <img src="${data.yolo_annotated_image || data.enhanced_image_base64 || data.sr_image_base64}" 
                     alt="Analisis" 
                     style="max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb;">
            </div>

            <div class="result-item">
                <div class="result-title">
                    <i data-feather="${isNormal ? 'check-circle' : 'alert-triangle'}"></i>
                    Hasil Analisis
                </div>
                <div class="result-content" style="color: ${isNormal ? '#10b981' : '#ef4444'}; font-weight: 600;">
                    ${data.label || 'Tidak terdeteksi'} (${confidence}%)
                </div>
            </div>
    `;

    if (data.yolo_boxes && data.yolo_boxes.length > 0) {
        resultHTML += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="target"></i>
                    Deteksi Scabies
                </div>
                <div class="result-content">
                    <div style="background: #fef2f2; padding: 1rem; border-radius: 8px;">
                        <strong>Ditemukan ${data.yolo_boxes.length} area terdeteksi scabies</strong>
                        ${data.yolo_boxes.map((box, index) => `
                            <div style="font-size: 0.9rem; color: #dc2626; margin-top: 0.5rem;">
                                • Area ${index + 1}: Confidence ${(box.confidence * 100).toFixed(1)}%
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    if (!isNormal) {
        resultHTML += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="info"></i>
                    Penjelasan Scabies
                </div>
                <div class="result-content">
                    <div style="background: #fef3f2; color: #b91c1c; padding: 1rem; border-radius: 8px; border: 1px solid #fecaca;">
                        <strong>${scabiesExplanation.description}</strong><br>
                        <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                            <strong>Gejala:</strong> ${scabiesExplanation.characteristics}<br>
                            <strong>Penanganan:</strong> ${scabiesExplanation.treatment}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="alert-circle"></i>
                    Rekomendasi
                </div>
                <div class="result-content">
                    <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border: 1px solid #ffc107;">
                        <strong>Segera konsultasi dengan dokter kulit!</strong><br>
                        Scabies adalah kondisi menular yang membutuhkan penanganan medis profesional.
                    </div>
                </div>
            </div>
        `;
    } else {
        resultHTML += `
            <div class="result-item">
                <div class="result-title">
                    <i data-feather="thumbs-up"></i>
                    Hasil
                </div>
                <div class="result-content">
                    <div style="background: #f0fdf4; color: #059669; padding: 1rem; border-radius: 8px; border: 1px solid #bbf7d0;">
                        <strong>Tidak terdeteksi scabies.</strong><br>
                        Kondisi kulit tampak normal berdasarkan analisis AI.
                    </div>
                </div>
            </div>
        `;
    }

    resultHTML += `</div>`;
    results.innerHTML = resultHTML;
    feather.replace();
}

// =================================================================
// EVENT LISTENERS
// =================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('MedScan AI Initialized');
    
    // Initialize Feather Icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // Global click handlers untuk close modal
    document.addEventListener('click', function(e) {
        if (e.target.id === 'uploadModal') {
            closeUploadModal();
        }
        if (e.target.id === 'cameraModal') {
            closeCameraModal();
        }
        if (e.target.id === 'resultModal') {
            closeResultModal();
        }
    });

    // Escape key to close modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeUploadModal();
            closeCameraModal();
            closeResultModal();
        }
    });

    // Test API connection
    testAPIConnection();
});

async function testAPIConnection() {
    try {
        const response = await fetch(API_CONFIG.endpoints.health);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.warn('API health check failed:', error);
    }
}