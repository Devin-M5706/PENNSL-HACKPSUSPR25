import React, { useEffect, useRef, useState } from 'react';
import './SignLanguageTranslator.css';

const SignLanguageTranslator = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detectedLetter, setDetectedLetter] = useState('?');
  const [confidence, setConfidence] = useState(0);
  const [cameraStatus, setCameraStatus] = useState('Active');
  const [fps, setFps] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Complete ASL alphabet guide based on official reference
  const letterGuide = [
    { letter: 'A', description: 'Closed fist with thumb resting on the side' },
    { letter: 'B', description: 'Flat hand with fingers together, thumb tucked' },
    { letter: 'C', description: 'Curved hand in C shape, thumb and fingers apart' },
    { letter: 'D', description: 'Index finger up, thumb and other fingers together' },
    { letter: 'E', description: 'Curved fingers, thumb tucked against palm' },
    { letter: 'F', description: 'Index and thumb connected, other fingers up' },
    { letter: 'G', description: 'Index finger points to side, thumb out' },
    { letter: 'H', description: 'Index and middle finger together pointing to side' },
    { letter: 'I', description: 'Pinky finger up, other fingers closed' },
    { letter: 'J', description: 'Pinky up, then trace J shape in air' },
    { letter: 'K', description: 'Index finger up, middle finger angled from thumb' },
    { letter: 'L', description: 'L-shape with index finger and thumb' },
    { letter: 'M', description: 'Three fingers over thumb, forming bridge' },
    { letter: 'N', description: 'Two fingers over thumb, forming bridge' },
    { letter: 'O', description: 'Fingers curved into O shape' },
    { letter: 'P', description: 'Index finger down from thumb, pointing down' },
    { letter: 'Q', description: 'Index finger down from thumb, to side' },
    { letter: 'R', description: 'Crossed index and middle fingers' },
    { letter: 'S', description: 'Fist with thumb over fingers in front' },
    { letter: 'T', description: 'Index finger between thumb and middle finger' },
    { letter: 'U', description: 'Index and middle finger up together' },
    { letter: 'V', description: 'Index and middle finger in V shape' },
    { letter: 'W', description: 'Index, middle, and ring fingers spread' },
    { letter: 'X', description: 'Hook index finger, other fingers closed' },
    { letter: 'Y', description: 'Thumb and pinky extended, other fingers closed' },
    { letter: 'Z', description: 'Index finger traces Z shape in air' }
  ];

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
          frameRate: { ideal: 30 }
        } 
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraStatus('Active');
        startProcessing(); // Start processing after camera is ready
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setCameraStatus('Error: ' + err.message);
    }
  };

  const startProcessing = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set fixed dimensions for better processing
    canvas.width = 640;
    canvas.height = 480;
    
    // Calculate FPS
    let frameCount = 0;
    let lastFpsUpdate = performance.now();
    
    const processFrame = async () => {
      if (!isProcessing) return;
      
      // Clear canvas before drawing
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw video frame with fixed dimensions
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get image data
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      try {
        const response = await fetch('http://localhost:5000/translate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        });
        
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        if (data.letter && data.confidence) {
          setDetectedLetter(data.letter);
          setConfidence(data.confidence);
        }
      } catch (error) {
        console.warn('Translation error:', error);
      }
      
      // Update FPS counter
      frameCount++;
      const currentTime = performance.now();
      if (currentTime - lastFpsUpdate >= 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - lastFpsUpdate)));
        frameCount = 0;
        lastFpsUpdate = currentTime;
      }
      
      // Schedule next frame with throttling (approximately 15 FPS)
      setTimeout(processFrame, 66); // 1000ms / 15fps ≈ 66ms
    };
    
    // Start processing frames
    setIsProcessing(true);
    processFrame();
  };

  const stopProcessing = () => {
    setIsProcessing(false);
  };

  return (
    <div className="sign-language-translator">
      <div className="app-title">
        <h1>PennSL</h1>
        <p>ASL Letter Recognition</p>
      </div>
      
      <div className="container">
        <div className="camera-container">
          <video
            ref={videoRef}
            className="camera-feed"
            autoPlay
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            style={{ display: 'none' }}
            width={640}
            height={480}
          />
          <div className="debug-info">
            <p>Camera Status: {cameraStatus}</p>
            <p>Processing: {isProcessing ? 'Yes' : 'No'}</p>
            <p>FPS: {fps.toFixed(1)}</p>
          </div>
        </div>

        <div className="right-panel">
          <div className="translation-display">
            <h2>{detectedLetter}</h2>
            <div className="confidence-bar">
              <div 
                className="confidence-fill"
                style={{ width: `${confidence}%` }}
              />
            </div>
            <p className="confidence-text">Confidence: {confidence.toFixed(1)}%</p>
          </div>

          <div className="letter-grid">
            {letterGuide.map(({ letter, description }) => (
              <div 
                key={letter} 
                className={`letter-card ${letter === detectedLetter ? 'active' : ''}`}
              >
                <h3>{letter}</h3>
                <p>{description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignLanguageTranslator; 