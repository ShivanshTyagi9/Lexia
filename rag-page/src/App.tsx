import { useEffect, useRef, useState } from "react";

// Add mermaid type to window
declare global {
  interface Window {
    mermaid?: {
      initialize: (config: any) => void;
      render: (id: string, code: string) => Promise<{ svg: string }>;
    };
  }
}

// Enhanced 2D React site for RAG pipeline
// - Monochrome palette + emerald accent
// - Pure CSS animations (no external dependencies)
// - Embedded Mermaid flowchart with CDN loading
// - Improved accessibility and responsiveness

const flowchart = `flowchart TD
  %% ===== Client & API =====
  U[User / Client] -->|/upload| UP[FastAPI: /upload]
  U -->|/ingest| ING[FastAPI: /ingest]
  U -->|/query| QRY[FastAPI: /query]
  U -->|/answer| ANS[FastAPI: /answer]

  subgraph APP[FastAPI App & Lifespan]
    direction LR
    INIT[Startup:<br/>Qdrant client + ensure collection<br/>Whoosh ensure/open<br/>CORS enabled]
    CTX[_build_context<br/>OpenAI or deterministic fallback]
  end

  %% ===== Ingestion =====
  subgraph INGEST[Ingestion Pipeline]
    direction TB
    F[read_blocks_with_tables<br/>PDF/DOCX/MD/TXT → text & tables]:::step
    S[split_text_structure<br/>structure-aware text blocks]:::step
    C[chunk_blocks<br/>text: 400–800 tok + 18% overlap<br/>tables: row-based mini-chunks]:::step
    E[embedder → E5-small-v2 384d]:::step
    QUPS[Qdrant upsert<br/>vectors + rich payload<br/>content_type, table_id...]:::store
    WIDX[Whoosh writer<br/>BM25-ready text incl. table markdown]:::store
    F --> S --> C --> E --> QUPS
    C --> WIDX
  end

  %% ===== Storage =====
  subgraph STORE[Storage]
    direction TB
    QD[Qdrant<br/>COSINE • HNSW M=32<br/>ef_construct=256 • ef_search=64]:::store
    WH[Whoosh Index<br/>BM25F B=0.75 K1=1.5]:::store
  end
  QUPS --> QD
  WIDX --> WH

  %% ===== Retrieval =====
  subgraph RETR[Hybrid Retrieval]
    direction TB
    QE[embedder on query → vector]:::step
    DENSE[dense_search qdrant, top=50]:::step
    SPARSE[bm25_search whoosh, top=50]:::step
    FUSE[rrf_fuse dense, sparse, k=60]:::step
    RERANK[CrossEncoder BGE-reranker-base<br/>with TABLE prefix when content_type==table]:::step
    DH[Diversity head → top-8 unique docs]:::step
    QE --> DENSE
    QE --> SPARSE
    DENSE --> FUSE
    SPARSE --> FUSE
    FUSE --> RERANK
    RERANK --> DH
  end

  %% ===== API wiring =====
  UP -->|save file| ING
  ING -->|loop files → upsert + index| INGEST
  INGEST --> STORE

  QRY -->|retrieve| RETR
  RETR -->|contexts| QRY

  ANS -->|retrieve| RETR
  RETR -->|contexts| CTX
  CTX -->|answer+citations| ANS

  %% ===== Styles =====
  classDef step fill:#ffffff,stroke:#111111,color:#111111,stroke-width:1.5;
  classDef store fill:#f6f6f6,stroke:#111111,color:#111111,stroke-width:1.5;`;

  type MermaidProps = {
  code: string;
  onError?: (e: unknown) => void;
};


function Mermaid({ code, onError }: MermaidProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const loadMermaidAndRender = async () => {
      try {
        // Load Mermaid from CDN if not already loaded
        if (!window.mermaid) {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js';
          document.head.appendChild(script);
          
          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = reject;
          });
        }

        if (cancelled) return;

        // Initialize Mermaid with dark theme
        if (window.mermaid) {
          window.mermaid.initialize({
            startOnLoad: false,
            theme: "base",
            themeVariables: {
              primaryColor: "#0a0a0a",
              primaryTextColor: "#e5e5e5",
              primaryBorderColor: "#374151",
              lineColor: "#6b7280",
              secondaryColor: "#1f2937",
              tertiaryColor: "#111827",
              fontFamily: "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Inter, Arial",
              textColor: "#e5e5e5",
              noteBkgColor: "#1f2937",
              noteTextColor: "#e5e5e5",
              clusterBkg: "#111827",
              clusterBorder: "#374151",
              edgeLabelBackground: "#1f2937",
            },
            securityLevel: "loose",
            flowchart: {
              htmlLabels: true,
              curve: 'basis'
            }
          });
        }

        // Render the diagram
        if (window.mermaid) {
          const { svg } = await window.mermaid.render("rag-diagram", code);
          
          if (!cancelled && ref.current) {
            ref.current.innerHTML = svg;
            setIsLoading(false);
          }
        } else {
          throw new Error("Mermaid library failed to load.");
        }
      } catch (e) {
        const err = e as Error;
        if (!cancelled) {
          console.error('Mermaid render error:', err);
          if (ref.current) {
            ref.current.innerHTML = `<div class="text-center p-8 text-red-400">
              <p>Failed to load diagram</p>
              <p class="text-sm mt-2 opacity-75">${err.message}</p>
            </div>`;
          }
          setIsLoading(false);
          onError?.(err);
        }
      }
    };

    loadMermaidAndRender();

    return () => {
      cancelled = true;
    };
  }, [code, onError]);

  return (
    <div className="relative">
      {isLoading && (
        <div className="flex items-center justify-center p-12">
          <div className="animate-spin h-8 w-8 border-2 border-emerald-400 border-t-transparent rounded-full"></div>
        </div>
      )}
      <div 
        ref={ref} 
        className={`mermaid-container ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-500`}
        aria-label="RAG pipeline flowchart showing data flow from user input through ingestion, storage, and retrieval"
      />
    </div>
  );
}

export default function MinimalRAG() {
  const [activeSection, setActiveSection] = useState('overview');
  const [mermaidError, setMermaidError] = useState<Error | null>(null);

  const handleMermaidError: ((e: unknown) => void) = (e) => {
  setMermaidError(e instanceof Error ? e : new Error(String(e)));
};
  // Smooth scroll handler
  const scrollToSection = (sectionId:string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setActiveSection(sectionId);
    }
  };

  // Intersection Observer for active section tracking
  useEffect(() => {
    const sections = ['overview', 'flow', 'strategies', 'endpoints'];
    const observers: IntersectionObserver[] = [];

    sections.forEach(sectionId => {
      const element = document.getElementById(sectionId);
      if (element) {
        const observer = new IntersectionObserver(
          (entries) => {
            entries.forEach(entry => {
              if (entry.isIntersecting) {
                setActiveSection(sectionId);
              }
            });
          },
          { threshold: 0.3, rootMargin: '-100px 0px -100px 0px' }
        );
        observer.observe(element);
        observers.push(observer);
      }
    });

    return () => {
      observers.forEach(observer => observer.disconnect());
    };
  }, []);

  const strategies = [
    {
      title: "Ingestion",
      description: "Structure-aware split → 400–800 tok with 18% overlap; tables row-mini-chunks. Balanced granularity for search + context fit.",
      icon: "📥"
    },
    {
      title: "Retrieval", 
      description: "Hybrid dense (E5-small-v2, COSINE) + BM25F(Whoosh). RRF(k=60) for stable merge; cross-encoder rerank; diversity head → top-8 unique.",
      icon: "🔍"
    },
    {
      title: "Latency Controls",
      description: "Qdrant HNSW M=32, ef_construct=256, ef_search=64. TopK 50→fuse→rerank 50→diversify 8. Tunable per endpoint.",
      icon: "⚡"
    }
  ];

  const endpoints = [
    { method: "POST", path: "/upload", description: "Accept file(s) → save → enqueue for ingest" },
    { method: "POST", path: "/ingest", description: "Loop files → extract → split → chunk → embed → upsert/index" },
    { method: "GET", path: "/query", description: "Hybrid retrieve → fuse → rerank → diversity → return contexts" },
    { method: "POST", path: "/answer", description: "Retrieve → build context → call LLM (OpenAI/fallback) → citations" }
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-zinc-200 selection:text-zinc-900 flex flex-col items-center">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-zinc-950/90 border-b border-zinc-800 w-full">
        <div className="mx-auto w-full max-w-7xl flex items-center justify-between px-6 sm:px-10 py-4">
          <button 
            onClick={() => scrollToSection('overview')}
            className="flex items-center gap-2 group focus:outline-none focus:ring-2 focus:ring-emerald-400 rounded-lg p-1 -m-1"
            aria-label="Go to overview section"
          >
            <span className="inline-block h-3 w-3 rounded-full bg-emerald-400 group-hover:scale-110 group-focus:scale-110 transition-transform duration-200" />
            <span className="text-xs tracking-[0.2em] uppercase text-zinc-400 group-hover:text-zinc-200 transition-colors">
              Lexia
            </span>
          </button>
          
          <nav className="hidden sm:flex items-center gap-6 text-sm">
            {[
              ["Overview", "overview"],
              ["Flow", "flow"],
              ["Strategies", "strategies"],
              ["Endpoints", "endpoints"],
            ].map(([label, sectionId]) => (
              <button
                key={label}
                onClick={() => scrollToSection(sectionId)}
                className={`hover:text-zinc-200 transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400 rounded px-2 py-1 ${
                  activeSection === sectionId ? 'text-emerald-400' : 'text-zinc-400'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl px-6 sm:px-10">
        {/* Overview / Hero */}
        <section id="overview" className="mx-auto max-w-5xl pt-8 sm:pt-16 pb-16">
          <div className="fade-in-up">
            <h1 className="text-[clamp(1.9rem,6vw,4rem)] font-semibold leading-[1.05] tracking-tight">
              Hybrid RAG for Files → Answers
              <span className="block text-zinc-400 font-normal mt-2">
                FastAPI • Qdrant • Whoosh • RRF • Cross-encoder
              </span>
            </h1>
            
            <p className="mt-6 max-w-2xl text-zinc-400 text-lg leading-relaxed">
              Upload PDFs/DOCX/MD/TXT, structure-aware chunking, dual-index (dense + BM25), 
              reciprocal rank fusion, cross-encoder rerank, and a diversity head feeding 
              context to the answerer.
            </p>

            {/* Accent underline with animation */}
            <div className="mt-8 h-[2px] bg-gradient-to-r from-emerald-400 to-emerald-300/60 w-48 accent-line" />
          </div>
        </section>

        {/* Mermaid Flowchart */}
        <section id="flow" className="pb-16">
          <div className="fade-in-up rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 shadow-[0_8px_40px_rgba(0,0,0,0.35)] backdrop-blur-sm">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Pipeline Architecture</h2>
              <span className="text-xs text-emerald-400/90 bg-emerald-400/10 px-3 py-1 rounded-full">
                Interactive Flow
              </span>
            </div>
            
            {mermaidError ? (
              <div className="text-center p-8 text-red-400">
                <p>Failed to load pipeline diagram</p>
                <button 
                  onClick={() => window.location.reload()} 
                  className="mt-4 text-sm bg-red-400/10 hover:bg-red-400/20 px-4 py-2 rounded-lg transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="overflow-auto rounded-lg">
                <Mermaid code={flowchart} onError={handleMermaidError} />
              </div>
            )}
          </div>
        </section>

        {/* Strategies */}
        <section id="strategies" className="pb-16">
          <div className="mb-8">
            <h2 className="text-2xl font-semibold mb-2">Core Strategies</h2>
            <p className="text-zinc-400">Balancing latency, accuracy, and scalability across the pipeline</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {strategies.map((strategy, index) => (
              <div
                key={strategy.title}
                className="fade-in-up rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 hover:bg-zinc-900/60 transition-all duration-300 hover:border-zinc-700"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex items-center gap-3 mb-4">
                  <span className="text-2xl">{strategy.icon}</span>
                  <h3 className="text-lg font-medium">{strategy.title}</h3>
                </div>
                <p className="text-sm text-zinc-400 leading-relaxed">{strategy.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Endpoints */}
        <section id="endpoints" className="pb-16">
          <div className="fade-in-up rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 backdrop-blur-sm">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">FastAPI Endpoints</h2>
              <p className="text-zinc-400 text-sm">RESTful API for document ingestion and querying</p>
            </div>
            
            <div className="space-y-4">
              {endpoints.map((endpoint) => (
                <div 
                  key={endpoint.path}
                  className="flex flex-col sm:flex-row sm:items-center gap-4 p-4 rounded-lg border border-zinc-800/50 hover:border-zinc-700/50 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className={`px-2 py-1 text-xs font-mono rounded ${
                      endpoint.method === 'GET' ? 'bg-blue-400/10 text-blue-400' : 'bg-emerald-400/10 text-emerald-400'
                    }`}>
                      {endpoint.method}
                    </span>
                    <code className="text-zinc-200 font-mono text-sm">{endpoint.path}</code>
                  </div>
                  <span className="text-sm text-zinc-400 flex-1">{endpoint.description}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="pb-12 text-sm text-zinc-500">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-zinc-800 pt-8">
            <p>© {new Date().getFullYear()} Lexia. Powered by Cosmic</p>
            <div className="flex items-center gap-6">
              <button className="hover:text-zinc-300 transition-colors focus:outline-none focus:text-emerald-400">
                Documentation
              </button>
              <button className="hover:text-zinc-300 transition-colors focus:outline-none focus:text-emerald-400">
  <a 
    href="https://github.com/ShivanshTyagi9/Lexia" 
    target="_blank" 
    rel="noopener noreferrer"
  >
    Github
  </a>
</button>


            </div>
          </div>
        </footer>
      </main>

      {/* Enhanced CSS animations and styles */}
      <style>{`
        .fade-in-up {
          animation: fadeInUp 0.8s ease-out forwards;
          opacity: 0;
          transform: translateY(20px);
        }
        
        .accent-line {
          animation: expandLine 1.2s ease-out 0.3s forwards;
          transform-origin: left;
          transform: scaleX(0);
        }

        @keyframes fadeInUp {
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes expandLine {
          to {
            transform: scaleX(1);
          }
        }

        .mermaid-container svg {
          max-width: 100%;
          height: auto;
        }

        .mermaid-container .cluster rect {
          fill: #111827 !important;
          stroke: #374151 !important;
          stroke-width: 1.5;
        }

        .mermaid-container .edgeLabel {
          background-color: #1f2937 !important;
          color: #e5e5e5 !important;
        }

        .mermaid-container .node rect,
        .mermaid-container .node circle,
        .mermaid-container .node polygon {
          stroke-width: 1.5;
        }

        @media (prefers-reduced-motion: reduce) {
          .fade-in-up,
          .accent-line {
            animation: none;
            opacity: 1;
            transform: none;
          }
        }

        /* Smooth scrolling for better UX */
        html {
          scroll-behavior: smooth;
        }

        /* Focus styles for better accessibility */
        button:focus-visible,
        a:focus-visible {
          outline: 2px solid #10b981;
          outline-offset: 2px;
        }
      `}</style>
    </div>
  );
}