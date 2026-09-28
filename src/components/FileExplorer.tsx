import React, { useState } from 'react';
import { PACKAGE_FILES, PackageFile } from '../data/packageFiles';
import { FileCode, Copy, Check, Folder, FileText, Settings, TestTube, Sparkles, Terminal } from 'lucide-react';

interface FileExplorerProps {
  onDownloadZip: () => void;
  isDownloading: boolean;
}

export const FileExplorer: React.FC<FileExplorerProps> = ({ onDownloadZip, isDownloading }) => {
  const [selectedFile, setSelectedFile] = useState<PackageFile>(PACKAGE_FILES[4]); // sampler.py by default
  const [copied, setCopied] = useState(false);
  const [filter, setFilter] = useState<'all' | 'core' | 'demo' | 'doc' | 'config'>('all');

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const filteredFiles = PACKAGE_FILES.filter(
    f => filter === 'all' || f.category === filter
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-[780px]">
      {/* Top action header */}
      <div className="px-6 py-4 border-b border-slate-800 bg-slate-950/80 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Folder className="w-5 h-5 text-indigo-400" />
            GitHub Repository Structure
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Clean, modular Python package layout following modern PEP 517/621 standards.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Category filter pills */}
          <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            {(['all', 'core', 'demo', 'doc', 'config'] as const).map(cat => (
              <button
                key={cat}
                onClick={() => setFilter(cat)}
                className={`px-2.5 py-1 rounded-md capitalize font-medium transition-all ${
                  filter === cat
                    ? 'bg-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <button
            onClick={onDownloadZip}
            disabled={isDownloading}
            className="flex items-center gap-2 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5" />
            {isDownloading ? 'Packaging ZIP...' : 'Download Repository (.zip)'}
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left Sidebar: File Tree */}
        <div className="w-80 border-r border-slate-800 bg-slate-950/40 p-3 overflow-y-auto flex flex-col gap-1">
          <div className="px-2 py-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Package Files ({filteredFiles.length})
          </div>

          {filteredFiles.map(file => {
            const isSelected = selectedFile.path === file.path;
            const Icon =
              file.category === 'core'
                ? FileCode
                : file.category === 'demo'
                ? Sparkles
                : file.category === 'config'
                ? Settings
                : FileText;

            return (
              <button
                key={file.path}
                onClick={() => setSelectedFile(file)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-mono transition-all flex items-start gap-2.5 ${
                  isSelected
                    ? 'bg-indigo-950/60 border border-indigo-500/40 text-indigo-300'
                    : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
                }`}
              >
                <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${isSelected ? 'text-indigo-400' : 'text-slate-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium">{file.path}</div>
                  <div className="text-[10px] text-slate-500 truncate mt-0.5">{file.description}</div>
                </div>
              </button>
            );
          })}

          <div className="mt-auto pt-3 border-t border-slate-800/80 p-2">
            <div className="bg-slate-900/90 rounded-lg p-2.5 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
              <Terminal className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <span className="text-slate-300 font-semibold">Quick Install:</span>
                <div className="font-mono text-emerald-400 text-[10px] mt-1 select-all">
                  pip install -e .
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Content: Code Viewer */}
        <div className="flex-1 flex flex-col min-w-0 bg-slate-950">
          {/* File header bar */}
          <div className="px-5 py-2.5 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs font-semibold text-white">{selectedFile.path}</span>
              <span className="px-2 py-0.5 text-[10px] uppercase font-bold rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                {selectedFile.language}
              </span>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1 bg-slate-800 hover:bg-slate-700 active:bg-slate-600 text-slate-200 text-xs font-medium rounded-md border border-slate-700 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-semibold">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy File</span>
                </>
              )}
            </button>
          </div>

          {/* Description banner */}
          <div className="px-5 py-2 bg-indigo-950/20 border-b border-indigo-900/20 text-xs text-indigo-300/90 flex items-center justify-between">
            <span>{selectedFile.description}</span>
            <span className="text-[10px] text-slate-400 font-mono">
              {selectedFile.content.split('\n').length} lines
            </span>
          </div>

          {/* Code text block */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs text-slate-200 selection:bg-indigo-600 selection:text-white">
            <pre className="leading-relaxed">
              <code>
                {selectedFile.content.split('\n').map((line, idx) => (
                  <div key={idx} className="flex hover:bg-slate-900/70 -mx-4 px-4 py-0.5">
                    <span className="w-10 select-none text-slate-600 text-right pr-4 shrink-0 font-mono text-[11px]">
                      {idx + 1}
                    </span>
                    <span className="flex-1 whitespace-pre">{line}</span>
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
