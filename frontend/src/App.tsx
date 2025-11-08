import React, { FormEvent, useEffect, useRef, useState } from 'react';
import './App.css';

type ChatRole = 'user' | 'assistant';

type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
};

type FinancialRecord = {
  type: string;
  amount: number;
  date: string;
  details: string;
};

type TranscribeResponse = {
  success: boolean;
  transcribed_text?: string;
  extracted_data?: FinancialRecord;
  message?: string;
};

type QuerySummary = {
  total?: number;
  total_profit?: number;
  total_loss?: number;
  net?: number;
  profit_count?: number;
  loss_count?: number;
  count?: number;
};

type QueryReport = {
  period?: string;
  date?: string;
  type_filter?: string | null;
  summary?: QuerySummary;
  profits?: FinancialRecord[];
  losses?: FinancialRecord[];
  records?: FinancialRecord[];
};

type QueryResponse = {
  success: boolean;
  text: string;
  json_report?: QueryReport | null;
};

// Auto-detect API base URL for ngrok/localhost
const getApiBase = (): string => {
  // If explicitly set via environment variable, use that
  if (process.env.REACT_APP_API_BASE_URL) {
    return process.env.REACT_APP_API_BASE_URL;
  }
  
  // If running on localhost, use localhost:8000
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  
  // If running on ngrok or any other domain, use the same origin
  // (assuming backend is proxied through the same ngrok tunnel or same domain)
  return window.location.origin;
};

const API_BASE = getApiBase();

const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const formatAmount = (value: unknown): string => {
  if (typeof value === 'number' && !Number.isNaN(value)) {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return String(value ?? '0');
};

const capitalize = (word: string): string => word.charAt(0).toUpperCase() + word.slice(1);

const formatRecordsList = (label: string, records?: FinancialRecord[]): string[] => {
  if (!records || records.length === 0) {
    return [];
  }
  const lines: string[] = [];
  lines.push(`${label}:`);
  records.slice(0, 3).forEach((record) => {
    const type = capitalize(record.type);
    lines.push(
      `• ${type} ${formatAmount(record.amount)} on ${record.date} — ${record.details}`
    );
  });
  if (records.length > 3) {
    lines.push(`• …and ${records.length - 3} more`);
  }
  return lines;
};

const formatQueryResponse = (response: QueryResponse): string => {
  const sections: string[] = [];
  const baseText = response.text.trim();
  if (baseText) {
    const primaryLine = baseText
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line && !/^[-=_•\s]+$/.test(line));
    if (primaryLine) {
      sections.push(primaryLine);
    }
  }

  const report = response.json_report;
  if (!report) {
    return sections.join('\n\n');
  }

  if (report.period) {
    sections.push(`Period: ${report.period}`);
  } else if (report.date) {
    sections.push(`Date: ${report.date}`);
  }

  const summary = report.summary;
  if (summary) {
    const summaryLines: string[] = [];
    if (typeof summary.total_profit === 'number') {
      summaryLines.push(`• Total profit: ${formatAmount(summary.total_profit)}`);
    }
    if (typeof summary.total_loss === 'number') {
      summaryLines.push(`• Total loss: ${formatAmount(summary.total_loss)}`);
    }
    if (typeof summary.net === 'number') {
      summaryLines.push(`• Net: ${formatAmount(summary.net)}`);
    }
    if (typeof summary.total === 'number') {
      summaryLines.push(`• Total: ${formatAmount(summary.total)}`);
    }
    if (typeof summary.count === 'number' && summary.count > 0) {
      summaryLines.push(`• Transactions: ${summary.count}`);
    } else {
      const profitCount = summary.profit_count ?? 0;
      const lossCount = summary.loss_count ?? 0;
      if (profitCount > 0 || lossCount > 0) {
        summaryLines.push(`• Transactions: ${profitCount} profit / ${lossCount} loss`);
      }
    }
    if (summaryLines.length) {
      sections.push(`Summary:\n${summaryLines.join('\n')}`);
    }
  }

  const transactionLines: string[] = [];
  if (report.records && report.records.length) {
    transactionLines.push(...formatRecordsList('Transactions', report.records));
  } else {
    transactionLines.push(...formatRecordsList('Profits', report.profits));
    transactionLines.push(...formatRecordsList('Losses', report.losses));
  }
  if (transactionLines.length) {
    sections.push(transactionLines.join('\n'));
  }

  return sections.filter(Boolean).join('\n\n');
};

const App: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [queryInput, setQueryInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: createId(),
      role: 'assistant',
      text: 'Hi! You can record a quick voice update or ask me about your finances here.',
    },
  ]);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const appendMessage = (role: ChatRole, text: string) => {
    setChatMessages((prev) => [...prev, { id: createId(), role, text }]);
  };

  const startRecording = async () => {
    if (isProcessing) {
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      appendMessage('assistant', 'Your browser does not support audio recording.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      recordedChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
        processAudio(audioBlob);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (error) {
      console.error(error);
      appendMessage('assistant', 'Unable to access your microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const response = await fetch(`${API_BASE}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.statusText}`);
      }

      const data: TranscribeResponse = await response.json();

      if (data.transcribed_text) {
        await processTranscribedText(data.transcribed_text);
      } else if (data.message) {
        appendMessage('assistant', data.message);
      } else {
        appendMessage('assistant', 'Finished processing your update.');
      }
    } catch (error) {
      console.error(error);
      appendMessage(
        'assistant',
        error instanceof Error ? error.message : 'Something went wrong while processing the audio.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const processTranscribedText = async (text: string) => {
    try {
      const formData = new FormData();
      formData.append('text', text);

      const response = await fetch(`${API_BASE}/transcribe-text`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Follow-up processing failed: ${response.statusText}`);
      }

      const data: TranscribeResponse = await response.json();

      if (data.extracted_data && data.success) {
        appendMessage('assistant', 'Record saved. You can ask about it anytime.');
      } else if (data.message) {
        appendMessage('assistant', data.message);
      } else {
        appendMessage('assistant', 'Processed your update. Feel free to ask about it.');
      }
    } catch (error) {
      console.error(error);
      appendMessage(
        'assistant',
        error instanceof Error
          ? `Text processing error: ${error.message}`
          : 'Sorry, I could not store that update right now.'
      );
    }
  };

  const handleSubmitQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const question = queryInput.trim();
    if (!question) {
      return;
    }

    appendMessage('user', question);
    setQueryInput('');
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('query', question);

      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const data: QueryResponse = await response.json();
      appendMessage('assistant', formatQueryResponse(data));
    } catch (error) {
      console.error(error);
      appendMessage(
        'assistant',
        error instanceof Error ? error.message : 'Sorry, I could not get that information right now.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="chat-card">
        <header className="chat-header">
          <h1>Financial Voice Assistant</h1>
          <p>Record your updates and ask questions whenever you need insights.</p>
        </header>

        <div className="record-controls">
          <button
            type="button"
            className={`record-btn ${isRecording ? 'stop' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
          {isRecording && <span className="recording-indicator">● Recording</span>}
        </div>

        <div className="chat-window">
          {chatMessages.map((message) => (
            <div key={message.id} className={`message-row ${message.role}`}>
              <div className="message-bubble">{message.text}</div>
            </div>
          ))}
          {isProcessing && !isRecording && (
            <div className="message-row assistant">
              <div className="message-bubble pending">Working on it…</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form className="chat-input" onSubmit={handleSubmitQuery}>
          <input
            type="text"
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
            placeholder="Ask about your records…"
            disabled={isProcessing}
            aria-label="Chat input"
          />
          <button type="submit" disabled={isProcessing || !queryInput.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default App;
