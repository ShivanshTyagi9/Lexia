import type { Message } from '../../lib/types';
import { ChatMessages } from './chat-messages';
import { ChatInput } from './chat-input';

interface ChatPaneProps {
  messages: Message[];
  isLoading: boolean;
  onSendQuestion: (question: string) => void;
  documentsReady: boolean;
}

export function ChatPane({ messages, isLoading, onSendQuestion, documentsReady }: ChatPaneProps) {
  return (
    <div className="flex-1 flex flex-col h-screen bg-black/5">
      <div className="flex-1 min-h-0 relative">
        <ChatMessages messages={messages} isLoading={isLoading} />
      </div>
      <ChatInput 
        onSend={onSendQuestion} 
        isLoading={isLoading} 
        documentsReady={documentsReady}
      />
    </div>
  );
}