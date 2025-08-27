"use client";

import { useRef, useEffect } from 'react';
import type { Message } from '@/lib/types';
import { MessageCircle, User } from 'lucide-react';

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
}

export function ChatMessages({ messages, isLoading }: ChatMessagesProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto h-full max-h-[calc(100vh-120px)]" ref={scrollRef}>
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center text-center py-20">
            <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mb-6 backdrop-blur-sm border border-white/10">
              <MessageCircle className="w-8 h-8 text-white/40" />
            </div>
            <h1 className="text-2xl font-semibold text-white/90 mb-3">
              Welcome to Lexia
            </h1>
            <h2 className="text-2xl font-semibold text-white/90 mb-3">
              Start a conversation
            </h2>
            <p className="text-white/50 max-w-md leading-relaxed">
              Upload documents and ask questions to get intelligent insights and analysis.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {message.role === 'assistant' && (
              <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-white/10 flex-shrink-0">
                <MessageCircle className="w-4 h-4 text-white/70" />
              </div>
            )}
            
            <div className={`max-w-2xl px-4 py-3 rounded-lg backdrop-blur-sm border ${
              message.role === 'user' 
                ? 'bg-white/10 border-white/20 text-white/90' 
                : 'bg-white/5 border-white/10 text-white/80'
            }`}>
              <p className="leading-relaxed whitespace-pre-wrap text-sm">{message.content}</p>
            </div>

            {message.role === 'user' && (
              <div className="w-8 h-8 bg-blue-500/20 rounded-full flex items-center justify-center backdrop-blur-sm border border-blue-500/30 flex-shrink-0">
                <User className="w-4 h-4 text-blue-400" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 justify-start">
            <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-white/10 flex-shrink-0">
              <MessageCircle className="w-4 h-4 text-white/70" />
            </div>
            <div className="px-4 py-3 rounded-lg bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="flex gap-1.5">
                <div className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                <div className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                <div className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}