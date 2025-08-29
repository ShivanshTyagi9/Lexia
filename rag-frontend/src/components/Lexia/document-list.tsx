import type { Document } from '../../lib/types';
import { FileText, Loader2, CheckCircle2, AlertTriangle, X } from 'lucide-react';

const statusIcons: { [key in Document['status']]: React.ReactNode } = {
  uploading: <Loader2 className="h-3 w-3 animate-spin text-blue-400" />,
  ingesting: <Loader2 className="h-3 w-3 animate-spin text-amber-400" />,
  ready: <CheckCircle2 className="h-3 w-3 text-green-400" />,
  error: <AlertTriangle className="h-3 w-3 text-red-400" />,
};

const statusText: { [key in Document['status']]: string } = {
  uploading: 'Uploading',
  ingesting: 'Processing',
  ready: 'Ready',
  error: 'Error',
};

interface DocumentListProps {
  documents: Document[];
  onRemoveDocument: (id: string) => void;
}

export function DocumentList({ documents, onRemoveDocument }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="text-center py-8">
        <FileText className="w-8 h-8 text-white/20 mx-auto mb-3" />
        <p className="text-white/40 text-sm">No documents uploaded</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div key={doc.id} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-3 hover:bg-white/10 transition-all duration-200 group">
          <div className="flex items-center gap-3">
            <FileText className="w-4 h-4 text-white/60 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-white/90 truncate font-medium" title={doc.name}>
                {doc.name}
              </p>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex items-center gap-1.5">
                  {statusIcons[doc.status]}
                  <span className="text-xs text-white/50">{statusText[doc.status]}</span>
                </div>
              </div>
              {(doc.status === 'uploading' || doc.status === 'ingesting') && (
                <div className="mt-2 w-full bg-white/10 rounded-full h-1 overflow-hidden">
                  <div 
                    className="h-full bg-blue-400 transition-all duration-300 rounded-full"
                    style={{ width: `${doc.progress}%` }}
                  />
                </div>
              )}
            </div>
            <button
              onClick={() => onRemoveDocument(doc.id)}
              className="opacity-0 group-hover:opacity-100 p-1 rounded text-white/40 hover:text-red-400 hover:bg-white/10 transition-all duration-200"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}