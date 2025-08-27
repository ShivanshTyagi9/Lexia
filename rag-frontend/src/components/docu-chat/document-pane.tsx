import type { Document } from '../../lib/types';
import { DocumentUploader } from './document-uploader';
import { DocumentList } from './document-list';
import { FileText, Settings } from 'lucide-react';

interface DocumentPaneProps {
  documents: Document[];
  onFileUpload: (file: File) => void;
  onRemoveDocument: (id: string) => void;
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
}

const ThemeToggle = ({ theme, onClick }: { theme: 'dark' | 'light', onClick?: () => void }) => (
  <button
    onClick={onClick}
    className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-200"
  >
    <Settings className="w-4 h-4 text-white/60" />
  </button>
);

export function DocumentPane({ documents, onFileUpload, onRemoveDocument, theme, onThemeToggle }: DocumentPaneProps) {
  return (
    <div className="w-80 border-r border-white/10 bg-black/20 backdrop-blur-xl flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-white/10 rounded-md flex items-center justify-center">
              <FileText className="w-4 h-4 text-white/80" />
            </div>
            <h1 className="text-sm font-semibold text-white/90">
              Documents
            </h1>
          </div>
          <ThemeToggle theme={theme} onClick={onThemeToggle} />
        </div>
        <p className="text-white/50 text-xs">
          Upload and manage your files
        </p>
      </div>

      {/* Upload Section */}
      <div className="p-4">
        <DocumentUploader onFileUpload={onFileUpload} />
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-4">
        <DocumentList documents={documents} onRemoveDocument={onRemoveDocument} />
      </div>
    </div>
  );
}