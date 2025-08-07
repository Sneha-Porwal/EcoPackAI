
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => setFile(e.target.files[0]);
  const handleUpload = async () => {
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
      {result && (
        <div>
          <p><strong>Type:</strong> {result.product_type}</p>
          <p><strong>Prediction Accuracy:</strong> {result.prediction_accuracy}</p>
          <p><strong>Internal Packaging:</strong> {result.packaging_suggestion.internal}</p>
          <p><strong>External Packaging:</strong> {result.packaging_suggestion.external}</p>
        </div>
      )}
    </div>
  );
}

export default App;
