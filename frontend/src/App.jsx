import { useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState('Ready');
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);

  const acceptedTypes = useMemo(() => '.mp4,.mpeg,.mpg,.mov,.webm', []);

  const handleFile = (nextFile) => {
    setError('');
    setPrediction('');
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

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a video first.');
      return;
    }

    setIsSubmitting(true);
    setStatus('Sending video to the model...');
    setError('');
    setPrediction('');

    try {
      const formData = new FormData();
      formData.append('video', file);

      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setPrediction(data.prediction || 'No words detected');
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
          The backend runs the existing LipNet weights locally. The frontend is a small React
          app that keeps the workflow simple: drop a clip, run inference, read the result.
        </p>
        <div className="status-bar">
          <span className="status-dot" />
          <span>{status}</span>
        </div>
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
              {isSubmitting ? 'Predicting...' : 'Run LipRead Studio'}
            </button>
            <p className="hint">Accepted: MP4, MPG, MPEG, MOV, WEBM</p>
          </div>

          {error ? <div className="message error">{error}</div> : null}
        </form>

        <aside className="card result-card">
          <h2>Result</h2>
          <div className="prediction-box">{prediction || 'Your transcription will appear here.'}</div>
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
    </main>
  );
}
