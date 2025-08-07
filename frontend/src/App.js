import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreviewURL(URL.createObjectURL(selectedFile));
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    try {
      const res = await axios.post('http://localhost:5000/classify', formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="App">
      <h2>Upload Product Image</h2>
      <input type="file" onChange={handleChange} />
      <button onClick={handleUpload}>Upload & Classify</button>

      {previewURL && (
        <div className="profile-card">
          <div className="profile-img-wrapper">
            <img src={previewURL} alt="Preview" className="profile-img" />
          </div>
          {result && (
            <div className="profile-details">
              <h3>{result.product_type.replace('_', ' ')}</h3>
              <p><strong>Prediction Accuracy:</strong> {result.prediction_accuracy}</p>
              <p><strong>Internal Packaging:</strong> {result.packaging_suggestion.internal}</p>
              <p><strong>External Packaging:</strong> {result.packaging_suggestion.external}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
