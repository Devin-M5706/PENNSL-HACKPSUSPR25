import axios from 'axios';
import React, { useEffect, useRef, useState } from 'react';

const SignLanguageTranslator = () => {
  const videoRef = useRef(null);
  const [translation, setTranslation] = useState('');
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Camera error:', err);
      setError('Could not access camera. Please make sure you have granted camera permissions.');
    }
  };

  const captureAndTranslate = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg');

    try {
      const response = await axios.post('http://localhost:5000/api/translate', {
        image: imageData
      });
      setTranslation(response.data.translation);
    } catch (error) {
      console.error('Translation error:', error);
      setTranslation('Error processing sign language');
    }
  };

  useEffect(() => {
    let interval;
    if (isCapturing) {
      interval = setInterval(captureAndTranslate, 2000);
    }
    return () => clearInterval(interval);
  }, [isCapturing]);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          Sign Language Translator
        </h1>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="relative mb-6">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full rounded-lg shadow-md"
            style={{ height: '480px', objectFit: 'cover' }}
          />
        </div>

        <div className="flex justify-center">
          <button
            onClick={() => setIsCapturing(!isCapturing)}
            className={`px-6 py-3 rounded-lg font-semibold text-white transition-colors ${
              isCapturing ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'
            }`}
          >
            {isCapturing ? 'Stop Capturing' : 'Start Capturing'}
          </button>
        </div>

        {translation && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h2 className="text-xl font-semibold mb-2 text-gray-700">Translation:</h2>
            <p className="text-2xl text-center text-blue-600">{translation}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignLanguageTranslator; 