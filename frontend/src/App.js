import React, { useState } from 'react';

function App() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handlePredict = async () => {
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) throw new Error("Prediction failed");

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError("Error: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '30px', fontFamily: 'Arial' }}>
            <h1>Sentiment Analysis</h1>
            <textarea
                rows="4"
                cols="60"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text here..."
                style={{ fontSize: '16px', padding: '8px' }}
            />
            <br /><br />
            <button onClick={handlePredict} disabled={loading}>
                {loading ? "Analyzing..." : "Predict"}
            </button>

            {error && <p style={{ color: 'red' }}>{error}</p>}
            {result && (
                <div style={{ marginTop: '20px' }}>
                    <h2>Result:</h2>
                    <p><strong>Label:</strong> {result.label}</p>
                    <p><strong>Score:</strong> {result.score.toFixed(4)}</p>
                </div>
            )}
        </div>
    );
}

export default App;