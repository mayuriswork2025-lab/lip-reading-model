import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState('Ready');
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [frameUrls, setFrameUrls] = useState([]);

  const acceptedTypes = useMemo(() => '.mp4,.mpeg,.mpg,.mov,.webm', []);

  const handleFile = (nextFile) => {
    setError('');
    setPrediction('');
    setConfidence(null);
    setFrameUrls([]);
    setProgress(0);
    setFile(nextFile);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (nextFile) {
      setPreviewUrl(URL.createObjectURL(nextFile));
      setStatus(`Loaded ${nextFile.name}`);
    } else {
      setPreviewUrl('');
      setStatus('Ready');
    }
  };

  const pollJob = async (jobId) => {
    while (true) {
      const statusResponse = await fetch(`${API_BASE}/status/${jobId}`);
      const statusData = await statusResponse.json();
      if (!statusResponse.ok) {
        throw new Error(statusData.detail || 'Status check failed');
      }

      setProgress(Math.round((statusData.progress || 0) * 100));
      setStatus(
        statusData.status === 'done'
          ? 'Processing complete'
          : `Processing... ${Math.round((statusData.progress || 0) * 100)}%`
      );

      if (statusData.status === 'failed') {
        throw new Error(statusData.error || 'Processing failed');
      }

      if (statusData.status === 'done') {
        const framesResponse = await fetch(`${API_BASE}/frames/${jobId}`);
        const framesData = await framesResponse.json();
        if (framesResponse.ok) {
          setFrameUrls((framesData.frames || []).map((url) => `${API_BASE}${url}`));
        }

        const predictResponse = await fetch(`${API_BASE}/predict/${jobId}`);
        const predictData = await predictResponse.json();
        if (!predictResponse.ok) {
          throw new Error(predictData.detail || 'Prediction failed');
        }

        setPrediction(predictData.prediction || 'No prediction');
        setConfidence(
          typeof predictData.confidence === 'number'
            ? `${(predictData.confidence * 100).toFixed(1)}% (${predictData.method || 'inference'})`
            : null
        );
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a video first.');
      return;
    }

    setIsSubmitting(true);
    setStatus('Uploading video...');
    setError('');
    setPrediction('');
    setConfidence(null);
    setFrameUrls([]);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append('video', file);

      const uploadResponse = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const uploadData = await uploadResponse.json();
      if (!uploadResponse.ok) {
        throw new Error(uploadData.detail || 'Upload failed');
      }

      setStatus('Extracting mouth frames...');
      await pollJob(uploadData.job_id);
      setStatus('Prediction complete');
    } catch (err) {
      setError(err.message || 'Unexpected error');
      setStatus('Prediction failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) {
      handleFile(droppedFile);
    }
  };

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">LipRead Studio</p>
        <h1>Upload a speaking video and decode lip movement into text.</h1>
        <p className="lede">
          Built from scratch with CNN + GRU models. Upload a clip to extract mouth frames
          and decode a full sentence (CTC) or word sequence.
        </p>
        <div className="status-bar">
          <span className="status-dot" />
          <span>{status}</span>
        </div>
        {isSubmitting ? (
          <div className="progress-wrap">
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span>{progress}%</span>
          </div>
        ) : null}
      </section>

      <section className="workspace">
        <form className="card upload-card" onSubmit={onSubmit}>
          <div
            className={`dropzone ${dragging ? 'dropzone-active' : ''}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              className="file-input"
              type="file"
              accept={acceptedTypes}
              onChange={(event) => handleFile(event.target.files?.[0] || null)}
            />
            <div>
              <p className="drop-title">Drop a video here</p>
              <p className="drop-copy">Or click to browse for a short front-facing clip.</p>
            </div>
          </div>

          <div className="actions">
            <button className="primary-button" type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Processing...' : 'Run LipRead Studio'}
            </button>
            <p className="hint">Accepted: MP4, MPG, MPEG, MOV, WEBM</p>
          </div>

          {error ? <div className="message error">{error}</div> : null}
        </form>

        <aside className="card result-card">
          <h2>Result</h2>
          <div className="prediction-box">{prediction || 'Your transcription will appear here.'}</div>
          {confidence ? <p className="confidence">Confidence: {confidence}</p> : null}
          <div className="api-box">
            <span>API</span>
            <strong>{API_BASE}</strong>
          </div>
        </aside>
      </section>

      <section className="card preview-card">
        <div className="preview-header">
          <h2>Preview</h2>
          <span>{file ? file.name : 'No file selected'}</span>
        </div>
        {previewUrl ? (
          <video className="preview-video" controls src={previewUrl} />
        ) : (
          <div className="preview-empty">Choose a clip to preview it here.</div>
        )}
      </section>

      <section className="card gallery-card">
        <div className="preview-header">
          <h2>Mouth Frame Gallery</h2>
          <span>{frameUrls.length ? `${frameUrls.length} frames` : 'No frames yet'}</span>
        </div>
        {frameUrls.length ? (
          <div className="frame-grid">
            {frameUrls.map((url) => (
              <img key={url} className="frame-thumb" src={url} alt="Mouth crop frame" />
            ))}
          </div>
        ) : (
          <div className="preview-empty">Upload a clip to see extracted mouth crops.</div>
        )}
      </section>
    </main>
  );
}
