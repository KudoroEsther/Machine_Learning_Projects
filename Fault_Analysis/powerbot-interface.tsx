import React, { useState, useRef, useEffect } from 'react';
import { Zap, Send, Activity } from 'lucide-react';

export default function PowerBot() {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: 'Welcome to PowerBot! I can help diagnose transmission line faults. Please provide the voltage and current readings for phases A, B, and C.',
      timestamp: new Date()
    }
  ]);
  const [formData, setFormData] = useState({
    Va: '',
    Vb: '',
    Vc: '',
    Ia: '',
    Ib: '',
    Ic: ''
  });
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value) || ''
    });
  };

  const addMessage = (type, content) => {
    setMessages(prev => [...prev, {
      type,
      content,
      timestamp: new Date()
    }]);
  };

  const handleSubmit = async () => {
    // Validate all fields are filled
    const allFilled = Object.values(formData).every(val => val !== '');
    if (!allFilled) {
      addMessage('bot', '⚠️ Please fill in all voltage and current readings.');
      return;
    }

    // Add user message
    const userMessage = `📊 Readings Submitted:\nVa: ${formData.Va}V, Vb: ${formData.Vb}V, Vc: ${formData.Vc}V\nIa: ${formData.Ia}A, Ib: ${formData.Ib}A, Ic: ${formData.Ic}A`;
    addMessage('user', userMessage);

    setLoading(true);

    try {
      // Call diagnose endpoint
      const response = await fetch('http://localhost:8000/diagnose', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          Va: parseFloat(formData.Va),
          Vb: parseFloat(formData.Vb),
          Vc: parseFloat(formData.Vc),
          Ia: parseFloat(formData.Ia),
          Ib: parseFloat(formData.Ib),
          Ic: parseFloat(formData.Ic)
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get diagnosis');
      }

      const result = await response.json();
      
      // Format the response
      let botResponse = '';
      if (result.fault_label === 'No fault') {
        botResponse = `✅ **System Status: Normal**\n\nNo fault detected. The transmission line is operating within normal parameters.\n\nConfidence: ${(result.confidence * 100).toFixed(1)}%`;
      } else {
        botResponse = `⚡ **Fault Detected**\n\n**Type:** ${result.fault_label}\n**Confidence:** ${(result.confidence * 100).toFixed(1)}%\n\n${result.final_answer || result.message || ''}`;
      }
      
      addMessage('bot', botResponse);
      
      // Reset form
      setFormData({
        Va: '',
        Vb: '',
        Vc: '',
        Ia: '',
        Ib: '',
        Ic: ''
      });
    } catch (error) {
      addMessage('bot', `❌ Error: ${error.message}. Please make sure the API server is running on http://localhost:8000`);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-slate-900 flex items-center justify-center p-4">
      {/* Animated background effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-96 h-96 bg-blue-500 rounded-full opacity-10 blur-3xl -top-48 -left-48 animate-pulse"></div>
        <div className="absolute w-96 h-96 bg-cyan-500 rounded-full opacity-10 blur-3xl -bottom-48 -right-48 animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      <div className="w-full max-w-3xl bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-blue-500/30 relative z-10">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-cyan-600 p-6 flex items-center gap-3">
          <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
            <Zap className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">PowerBot</h1>
            <p className="text-blue-100 text-sm">Transmission Line Fault Analyzer</p>
          </div>
          <div className="ml-auto">
            <Activity className="w-6 h-6 text-white animate-pulse" />
          </div>
        </div>

        {/* Messages */}
        <div className="h-96 overflow-y-auto p-6 space-y-4 bg-slate-900/50">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white'
                    : 'bg-slate-700 text-gray-100 border border-blue-500/20'
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {message.content}
                </div>
                <div className="text-xs opacity-60 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-700 rounded-lg p-4 border border-blue-500/20">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                  </div>
                  <span className="text-gray-300 text-sm">Analyzing...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Section */}
        <div className="p-6 bg-slate-800 border-t border-blue-500/30">
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-blue-300 text-xs font-semibold mb-1">
                  Va (Voltage A)
                </label>
                <input
                  type="number"
                  name="Va"
                  step="any"
                  value={formData.Va}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-blue-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="0.0"
                />
              </div>
              <div>
                <label className="block text-blue-300 text-xs font-semibold mb-1">
                  Vb (Voltage B)
                </label>
                <input
                  type="number"
                  name="Vb"
                  step="any"
                  value={formData.Vb}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-blue-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="0.0"
                />
              </div>
              <div>
                <label className="block text-blue-300 text-xs font-semibold mb-1">
                  Vc (Voltage C)
                </label>
                <input
                  type="number"
                  name="Vc"
                  step="any"
                  value={formData.Vc}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-blue-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="0.0"
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-cyan-300 text-xs font-semibold mb-1">
                  Ia (Current A)
                </label>
                <input
                  type="number"
                  name="Ia"
                  step="any"
                  value={formData.Ia}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-cyan-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  placeholder="0.0"
                />
              </div>
              <div>
                <label className="block text-cyan-300 text-xs font-semibold mb-1">
                  Ib (Current B)
                </label>
                <input
                  type="number"
                  name="Ib"
                  step="any"
                  value={formData.Ib}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-cyan-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  placeholder="0.0"
                />
              </div>
              <div>
                <label className="block text-cyan-300 text-xs font-semibold mb-1">
                  Ic (Current C)
                </label>
                <input
                  type="number"
                  name="Ic"
                  step="any"
                  value={formData.Ic}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-slate-700 border border-cyan-500/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  placeholder="0.0"
                />
              </div>
            </div>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Diagnosing...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  Analyze Fault
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}