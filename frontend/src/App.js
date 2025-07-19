import React, { useState } from 'react';

function App() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [darkMode, setDarkMode] = useState(true); // dark mode default

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

            // Artificial 2-second delay before showing the result
            setTimeout(() => {
                setResult(data);
                setLoading(false);
            }, 2000);

        } catch (err) {
            setError("Error: " + err.message);
            setLoading(false);
        }
    };

    const toggleTheme = () => {
        setDarkMode(prev => !prev);
    };

    const themeStyles = {
        backgroundColor: darkMode ? '#111827' : '#f9fafb',
        color: darkMode ? '#f9fafb' : '#111827',
        transition: 'all 0.3s ease'
    };

    const cardStyles = {
        backgroundColor: darkMode ? '#1f2937' : 'white',
        color: darkMode ? '#f9fafb' : '#111827',
        padding: '40px',
        borderRadius: '16px',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
        width: '100%',
        maxWidth: '600px',
        position: 'relative'
    };

    const buttonStyle = {
        backgroundColor: '#3b82f6',
        color: 'white',
        padding: '10px 20px',
        fontSize: '16px',
        borderRadius: '8px',
        border: 'none',
        cursor: loading ? 'not-allowed' : 'pointer',
        width: '100%',
        marginBottom: '16px'
    };

    return (
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            ...themeStyles,
            fontFamily: 'Segoe UI, sans-serif',
            padding: '20px',
            position: 'relative'
        }}>
            {/* Theme Toggle Button */}
            <button
                onClick={toggleTheme}
                style={{
                    position: 'absolute',
                    top: 20,
                    right: 20,
                    padding: '6px 12px',
                    borderRadius: '6px',
                    backgroundColor: 'transparent',
                    color: darkMode ? '#facc15' : '#334155',
                    border: `1px solid ${darkMode ? '#facc15' : '#334155'}`,
                    cursor: 'pointer',
                    fontSize: '14px'
                }}
                aria-label="Toggle theme"
            >
                {darkMode ? '☀️' : '🌙'}
            </button>

            <div style={cardStyles}>
                <h1 style={{
                    marginBottom: '24px',
                    fontSize: '28px',
                    fontWeight: '600',
                    textAlign: 'center'
                }}>
                    🧠 Sentiment Analyzer
                </h1>

                <textarea
                    rows="4"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type something like 'I love this product!'..."
                    style={{
                        width: '100%',
                        padding: '12px',
                        fontSize: '16px',
                        borderRadius: '8px',
                        border: '1px solid #d1d5db',
                        outline: 'none',
                        resize: 'none',
                        marginBottom: '16px',
                        backgroundColor: darkMode ? '#374151' : 'white',
                        color: darkMode ? '#f9fafb' : '#111827'
                    }}
                />

                <button
                    onClick={handlePredict}
                    disabled={loading}
                    style={buttonStyle}
                >
                    {loading ? 'Analyzing...' : 'Predict'}
                </button>

                {error && (
                    <p style={{ color: '#dc2626', textAlign: 'center' }}>{error}</p>
                )}

                {loading && (
                    <div style={{ textAlign: 'center', marginBottom: '12px' }}>
                        <div className="spinner" style={{
                            width: '32px',
                            height: '32px',
                            border: '4px solid #ccc',
                            borderTop: '4px solid #3b82f6',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                            margin: '0 auto'
                        }} />
                    </div>
                )}

                {result && (
                    <div style={{
                        backgroundColor: darkMode ? '#4b5563' : '#f3f4f6',
                        padding: '16px',
                        borderRadius: '12px',
                        textAlign: 'center'
                    }}>
                        <h2 style={{ marginBottom: '8px', fontSize: '20px' }}>🔍 Result</h2>
                        <p><strong>Label:</strong> {result.label}</p>
                        <p><strong>Score:</strong> {result.score.toFixed(4)}</p>
                    </div>
                )}
            </div>

            {/* Inline spinner animation */}
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                `}
            </style>
        </div>
    );
}

export default App;