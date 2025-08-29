"use client";

import { useRef, useState, DragEvent, ChangeEvent } from 'react';
import { Upload } from 'lucide-react';

interface DocumentUploaderProps {
  onFileUpload: (file: File) => void;
}

export function DocumentUploader({ onFileUpload }: DocumentUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (!file.type.startsWith('text/') && file.type !== 'application/pdf') {
      alert('Please upload a text or PDF file.');
      return;
    }
    onFileUpload(file);
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer transition-all duration-200 ${
        isDragging 
          ? "bg-white/15 border-blue-400/50" 
          : "bg-white/5 hover:bg-white/10 border-white/10"
      } backdrop-blur-sm border-2 border-dashed rounded-lg p-6 text-center group`}
    >
      <Upload className="w-6 h-6 mx-auto text-white/40 group-hover:text-white/60 transition-colors mb-3" />
      <p className="text-white/70 text-sm font-medium mb-1">
        Drop files or click to upload
      </p>
      <p className="text-white/40 text-xs">PDF, TXT files supported</p>
      
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}