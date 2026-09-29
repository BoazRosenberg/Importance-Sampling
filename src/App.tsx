import React, { useState } from 'react';
import {
  Copy,
  Check,
  BarChart3,
  BookOpen,
  Activity,
  TrendingUp,
  Layers,
  Grid,
  CheckSquare,
  Square,
  SlidersHorizontal,
  Scale,
  Award,
} from 'lucide-react';

const PARAM_KEYS = ['alpha', 'beta', 'decay', 'bias', 'noise', 'pers'] as const;
type ParamKey = typeof PARAM_KEYS[number];

interface ParamTrajectory {
  iters: number[];
  means: number[];
  sdUpper: number[];
  sdLower: number[];
}

const mockTrajectories: Record<ParamKey, ParamTrajectory> = {
  alpha: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.500, 0.462, 0.435, 0.418, 0.410, 0.405, 0.402, 0.400, 0.398, 0.397, 0.397, 0.396],
    sdUpper: [0.731, 0.684, 0.632, 0.589, 0.554, 0.528, 0.509, 0.495, 0.485, 0.478, 0.474, 0.471],
    sdLower: [0.269, 0.261, 0.258, 0.264, 0.276, 0.291, 0.304, 0.312, 0.318, 0.322, 0.325, 0.327],
  },
  beta: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [1.31, 1.84, 2.21, 2.48, 2.67, 2.81, 2.91, 2.97, 3.01, 3.03, 3.04, 3.05],
    sdUpper: [2.12, 2.68, 3.06, 3.31, 3.48, 3.61, 3.70, 3.76, 3.80, 3.82, 3.84, 3.85],
    sdLower: [0.69, 1.15, 1.49, 1.76, 1.96, 2.11, 2.21, 2.28, 2.32, 2.35, 2.36, 2.37],
  },
  decay: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.10, 0.14, 0.18, 0.21, 0.23, 0.24, 0.25, 0.25, 0.26, 0.26, 0.26, 0.26],
    sdUpper: [0.25, 0.28, 0.30, 0.32, 0.33, 0.33, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34],
    sdLower: [0.02, 0.04, 0.07, 0.10, 0.13, 0.15, 0.16, 0.16, 0.18, 0.18, 0.18, 0.18],
  },
  bias: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.00, 0.05, 0.09, 0.12, 0.14, 0.15, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16],
    sdUpper: [0.35, 0.36, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.30, 0.30],
    sdLower: [-0.35, -0.26, -0.19, -0.12, -0.07, -0.04, -0.03, -0.00, 0.01, 0.02, 0.02, 0.02],
  },
  noise: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.05, 0.08, 0.11, 0.13, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    sdUpper: [0.15, 0.18, 0.20, 0.21, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22],
    sdLower: [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
  },
  pers: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.00, 0.10, 0.18, 0.24, 0.28, 0.31, 0.32, 0.33, 0.34, 0.34, 0.34, 0.34],
    sdUpper: [0.50, 0.52, 0.54, 0.55, 0.55, 0.54, 0.53, 0.52, 0.51, 0.51, 0.50, 0.50],
    sdLower: [-0.50, -0.32, -0.18, -0.07, 0.01, 0.08, 0.11, 0.14, 0.17, 0.17, 0.18, 0.18],
  },
};

// 10 subjects posterior means
const mockSubjMeans: Record<ParamKey, number[]> = {
  alpha: [0.38, 0.42, 0.35, 0.45, 0.39, 0.33, 0.44, 0.37, 0.41, 0.40],
  beta:  [3.21, 2.85, 3.42, 2.65, 3.10, 3.65, 2.78, 3.30, 2.92, 3.05],
  decay: [0.24, 0.28, 0.22, 0.29, 0.25, 0.21, 0.27, 0.23, 0.26, 0.25],
  bias:  [0.18, 0.12, 0.22, 0.09, 0.15, 0.25, 0.11, 0.19, 0.14, 0.16],
  noise: [0.14, 0.17, 0.12, 0.19, 0.15, 0.11, 0.18, 0.13, 0.16, 0.15],
  pers:  [0.36, 0.31, 0.42, 0.25, 0.33, 0.45, 0.29, 0.38, 0.32, 0.34],
};

const mockEvidence = [-212.4, -188.2, -172.5, -161.8, -154.2, -149.1, -145.7, -143.5, -142.4, -141.8, -141.5, -141.3];
const mockBIC = [434.1, 385.7, 354.3, 332.9, 317.7, 307.5, 300.7, 296.3, 294.1, 292.9, 292.3, 291.9];

const mockSubjSpaghetti = [
  [-21.2, -18.7, -17.1, -16.0, -15.3, -14.8, -14.4, -14.2, -14.1, -14.0, -13.9, -13.9],
  [-21.5, -19.0, -17.4, -16.3, -15.6, -15.1, -14.8, -14.6, -14.5, -14.4, -14.4, -14.4],
  [-20.8, -18.1, -16.6, -15.5, -14.8, -14.2, -13.8, -13.5, -13.4, -13.3, -13.2, -13.2],
  [-22.1, -19.6, -18.0, -16.9, -16.2, -15.8, -15.5, -15.3, -15.2, -15.1, -15.1, -15.1],
  [-21.0, -18.6, -17.0, -15.9, -15.2, -14.7, -14.3, -14.1, -14.0, -13.9, -13.8, -13.8],
  [-20.5, -17.8, -16.2, -15.1, -14.3, -13.8, -13.4, -13.2, -13.0, -12.9, -12.9, -12.9],
  [-21.8, -19.2, -17.6, -16.6, -15.9, -15.4, -15.1, -14.9, -14.8, -14.7, -14.7, -14.7],
  [-20.9, -18.4, -16.8, -15.8, -15.0, -14.5, -14.1, -14.9, -14.7, -14.6, -14.6, -14.6],
  [-21.4, -18.9, -17.3, -16.2, -15.5, -15.0, -14.6, -14.4, -14.3, -14.2, -14.2, -14.2],
  [-21.2, -18.5, -16.9, -15.8, -15.1, -14.5, -14.1, -13.8, -13.7, -13.6, -13.5, -13.5],
];

const mockCorMatrix: Record<ParamKey, Record<ParamKey, number>> = {
  alpha: { alpha: 1.0,  beta: -0.42, decay: 0.28, bias: -0.15, noise: 0.31, pers: -0.22 },
  beta:  { alpha: -0.42, beta: 1.0,  decay: -0.19, bias: 0.34, noise: -0.25, pers: 0.41 },
  decay: { alpha: 0.28,  beta: -0.19, decay: 1.0,  bias: -0.08, noise: 0.12, pers: -0.14 },
  bias:  { alpha: -0.15, beta: 0.34, decay: -0.08, bias: 1.0,  noise: -0.18, pers: 0.29 },
  noise: { alpha: 0.31,  beta: -0.25, decay: 0.12, bias: -0.18, noise: 1.0,  pers: -0.09 },
  pers:  { alpha: -0.22, beta: 0.41, decay: -0.14, bias: 0.29, noise: -0.09, pers: 1.0 },
};

// MULTI-MODEL DATA FOR COMPARISON
interface ModelComparisonEntry {
  name: string;
  color: string;
  k: number;
  params: string[];
  evidenceHistory: number[];
  bicHistory: number[];
  finalEvidence: number;
  finalBIC: number;
  alphaMean?: number;
  alphaSD?: number;
  betaMean?: number;
  betaSD?: number;
  alphaSubjects?: number[];
  betaSubjects?: number[];
}

const COMPARISON_MODELS: Record<string, ModelComparisonEntry> = {
  QLearn_Pers: {
    name: 'QLearn_Pers',
    color: '#0969da',
    k: 3,
    params: ['alpha', 'beta', 'pers'],
    evidenceHistory: [-212.4, -188.2, -172.5, -161.8, -154.2, -149.1, -145.7, -143.5, -142.4, -141.8, -141.5, -141.3],
    bicHistory: [434.1, 385.7, 354.3, 332.9, 317.7, 307.5, 300.7, 296.3, 294.1, 292.9, 292.3, 291.9],
    finalEvidence: -141.3,
    finalBIC: 291.9,
    alphaMean: 0.396,
    alphaSD: 0.072,
    betaMean: 3.05,
    betaSD: 0.74,
    alphaSubjects: [0.38, 0.42, 0.35, 0.45, 0.39, 0.33, 0.44, 0.37, 0.41, 0.40],
    betaSubjects:  [3.21, 2.85, 3.42, 2.65, 3.10, 3.65, 2.78, 3.30, 2.92, 3.05],
  },
  Standard_QLearn: {
    name: 'Standard_QLearn',
    color: '#1a7f37',
    k: 2,
    params: ['alpha', 'beta'],
    evidenceHistory: [-218.0, -195.4, -182.1, -174.5, -168.2, -164.5, -162.1, -160.8, -160.2, -159.9, -159.7, -159.5],
    bicHistory: [442.0, 396.8, 370.2, 355.0, 342.4, 335.0, 330.2, 327.6, 326.4, 325.8, 325.4, 325.0],
    finalEvidence: -159.5,
    finalBIC: 325.0,
    alphaMean: 0.435,
    alphaSD: 0.095,
    betaMean: 2.62,
    betaSD: 0.88,
    alphaSubjects: [0.41, 0.46, 0.39, 0.49, 0.43, 0.37, 0.48, 0.42, 0.45, 0.45],
    betaSubjects:  [2.75, 2.45, 3.05, 2.20, 2.70, 3.15, 2.40, 2.85, 2.50, 2.65],
  },
  Dual_Alpha_QLearn: {
    name: 'Dual_Alpha_QLearn',
    color: '#8250df',
    k: 3,
    params: ['alpha_pos', 'alpha_neg', 'beta'],
    evidenceHistory: [-215.2, -191.0, -177.3, -169.1, -163.5, -159.8, -157.2, -155.5, -154.8, -154.2, -154.0, -153.8],
    bicHistory: [439.7, 391.3, 363.9, 347.5, 336.3, 328.9, 323.7, 320.3, 318.9, 317.7, 317.3, 316.9],
    finalEvidence: -153.8,
    finalBIC: 316.9,
    alphaMean: 0.412,
    alphaSD: 0.082,
    betaMean: 2.88,
    betaSD: 0.81,
    alphaSubjects: [0.39, 0.44, 0.37, 0.47, 0.41, 0.35, 0.46, 0.39, 0.43, 0.41],
    betaSubjects:  [3.00, 2.70, 3.25, 2.45, 2.95, 3.40, 2.60, 3.10, 2.75, 2.85],
  },
  Random_Baseline: {
    name: 'Random_Baseline',
    color: '#cf222e',
    k: 1,
    params: ['bias'],
    evidenceHistory: [-232.0, -225.1, -222.0, -220.5, -219.8, -219.5, -219.4, -219.3, -219.3, -219.3, -219.3, -219.3],
    bicHistory: [467.0, 453.2, 447.0, 444.0, 442.6, 442.0, 441.8, 441.6, 441.6, 441.6, 441.6, 441.6],
    finalEvidence: -219.3,
    finalBIC: 441.6,
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'report' | 'compare' | 'docs'>('compare');
  const [copied, setCopied] = useState(false);

  // Single Model Report States
  const [selectedParams, setSelectedParams] = useState<ParamKey[]>(['alpha', 'beta', 'decay']);
  const [subjectSelectedParams, setSubjectSelectedParams] = useState<ParamKey[]>([
    'alpha',
    'beta',
    'decay',
  ]);
  const [hoveredIter, setHoveredIter] = useState<{ param: string; iter: number } | null>(null);
  const [hoveredSubj, setHoveredSubj] = useState<number | null>(null);

  // Model Comparison States
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'QLearn_Pers',
    'Standard_QLearn',
    'Dual_Alpha_QLearn',
    'Random_Baseline',
  ]);
  const [comparisonSubTab, setComparisonSubTab] = useState<'fit' | 'params'>('fit');
  const [compareParamKey, setCompareParamKey] = useState<'alpha' | 'beta'>('alpha');

  const installCmd = 'pip install git+https://github.com/BoazRosenberg/Importance-Sampling.git';

  const handleCopy = () => {
    navigator.clipboard.writeText(installCmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleParamInEvolution = (param: ParamKey) => {
    if (selectedParams.includes(param)) {
      if (selectedParams.length > 1) {
        setSelectedParams(selectedParams.filter((p) => p !== param));
      }
    } else {
      setSelectedParams([...selectedParams, param]);
    }
  };

  const toggleParamInSubjects = (param: ParamKey) => {
    if (subjectSelectedParams.includes(param)) {
      if (subjectSelectedParams.length > 1) {
        setSubjectSelectedParams(subjectSelectedParams.filter((p) => p !== param));
      }
    } else {
      setSubjectSelectedParams([...subjectSelectedParams, param]);
    }
  };

  const toggleModelComparison = (modelName: string) => {
    if (selectedModels.includes(modelName)) {
      if (selectedModels.length > 2) {
        setSelectedModels(selectedModels.filter((m) => m !== modelName));
      }
    } else {
      setSelectedModels([...selectedModels, modelName]);
    }
  };

  // Dimensions
  const subW = 320;
  const subH = 180;
  const subPad = { top: 20, right: 20, bottom: 25, left: 45 };

  const fitW = 540;
  const fitH = 200;
  const fitPad = { top: 20, right: 25, bottom: 30, left: 50 };

  const getFitX = (it: number) => fitPad.left + (it / 11) * (fitW - fitPad.left - fitPad.right);
  const evMin = Math.min(...mockEvidence);
  const evMax = Math.max(...mockEvidence);
  const evRange = evMax - evMin || 1;
  const getEvY = (val: number) =>
    fitH - fitPad.bottom - ((val - evMin) / evRange) * (fitH - fitPad.top - fitPad.bottom);

  const evidenceD = mockEvidence
    .map((val, idx) => `${idx === 0 ? 'M' : 'L'} ${getFitX(idx)},${getEvY(val)}`)
    .join(' ');

  // Comparison metrics calculations
  const activeModelList = selectedModels.map((m) => COMPARISON_MODELS[m]);
  const bestBicVal = Math.min(...activeModelList.map((m) => m.finalBIC));
  const sortedModels = [...activeModelList].sort((a, b) => a.finalBIC - b.finalBIC);

  // Comparison chart ranges
  const compEvMin = Math.min(...activeModelList.flatMap((m) => m.evidenceHistory));
  const compEvMax = Math.max(...activeModelList.flatMap((m) => m.evidenceHistory));
  const compEvRange = compEvMax - compEvMin || 1;
  const getCompEvY = (v: number) =>
    fitH - fitPad.bottom - ((v - compEvMin) / compEvRange) * (fitH - fitPad.top - fitPad.bottom);

  const compBicMin = Math.min(...activeModelList.flatMap((m) => m.bicHistory));
  const compBicMax = Math.max(...activeModelList.flatMap((m) => m.bicHistory));
  const compBicRange = compBicMax - compBicMin || 1;
  const getCompBicY = (v: number) =>
    fitH - fitPad.bottom - ((v - compBicMin) / compBicRange) * (fitH - fitPad.top - fitPad.bottom);

  return (
    <div className="min-h-screen bg-[#f6f8fa] text-[#1f2328] font-sans antialiased">
      {/* Top Header */}
      <header className="border-b border-[#d1d9e0] bg-white sticky top-0 z-30 shadow-2xs">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-lg text-[#0969da] hover:underline cursor-pointer">
              BoazRosenberg
            </span>
            <span className="text-[#59636e]">/</span>
            <span className="font-bold text-lg text-[#1f2328]">Importance-Sampling</span>
            <span className="px-2 py-0.5 text-xs font-medium text-[#59636e] bg-[#f6f8fa] border border-[#d1d9e0] rounded-full">
              v0.2.0
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex bg-[#f6f8fa] p-1 border border-[#d1d9e0] rounded-lg text-xs font-medium">
              <button
                onClick={() => setActiveTab('compare')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'compare'
                    ? 'bg-white text-[#0969da] shadow-2xs font-semibold'
                    : 'text-[#59636e] hover:text-[#1f2328]'
                }`}
              >
                <Scale className="w-3.5 h-3.5" />
                <span>Model Comparison</span>
              </button>
              <button
                onClick={() => setActiveTab('report')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'report'
                    ? 'bg-white text-[#0969da] shadow-2xs font-semibold'
                    : 'text-[#59636e] hover:text-[#1f2328]'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Single Model Report</span>
              </button>
              <button
                onClick={() => setActiveTab('docs')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'docs'
                    ? 'bg-white text-[#0969da] shadow-2xs font-semibold'
                    : 'text-[#59636e] hover:text-[#1f2328]'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Documentation &amp; API</span>
              </button>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#ffffff] hover:bg-[#f6f8fa] text-[#1f2328] border border-[#d1d9e0] rounded-md shadow-2xs cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[#1a7f37]" />
                  <span className="text-[#1a7f37]">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-[#59636e]" />
                  <span>Copy pip</span>
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* ============================================================== */}
        {/* TAB 1: MODEL COMPARISON REPORT                                */}
        {/* ============================================================== */}
        {activeTab === 'compare' && (
          <div className="space-y-6">
            {/* Top Comparative Header Card */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#8250df]/10 text-[#8250df] border border-[#8250df]/20 flex items-center gap-1">
                    <Scale className="w-3 h-3" />
                    Comparative Report
                  </span>
                  <span className="text-xs text-[#59636e]">
                    Generated from <code>compare_models([m1, m2, m3, ...])</code>
                  </span>
                </div>
                <h1 className="text-2xl font-bold text-[#1f2328]">
                  Multi-Model Comparison &amp; Selection
                </h1>
                <p className="text-sm text-[#59636e] mt-1">
                  Comparing {selectedModels.length} computational models on evidence, BIC, and parameter trajectories.
                </p>
              </div>

              {/* Winner pill */}
              <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2.5 rounded-lg flex items-center gap-3">
                <Award className="w-6 h-6 text-[#1a7f37]" />
                <div>
                  <div className="text-[10px] uppercase font-bold tracking-wider text-[#59636e]">
                    Best Model (Lowest BIC)
                  </div>
                  <div className="text-sm font-bold font-mono text-[#1a7f37]">
                    {sortedModels[0].name} (BIC {sortedModels[0].finalBIC.toFixed(1)})
                  </div>
                </div>
              </div>
            </div>

            {/* Model Filter Pills & View Mode Sub-tabs */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              {/* Models to compare selector */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-bold uppercase tracking-wider text-[#59636e] mr-1">
                  Models:
                </span>
                {Object.keys(COMPARISON_MODELS).map((mKey) => {
                  const m = COMPARISON_MODELS[mKey];
                  const isChecked = selectedModels.includes(mKey);
                  return (
                    <button
                      key={mKey}
                      onClick={() => toggleModelComparison(mKey)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all cursor-pointer border ${
                        isChecked
                          ? 'bg-white text-[#1f2328] border-[#0969da] shadow-2xs font-semibold'
                          : 'bg-[#f6f8fa] text-[#8c959f] border-[#d1d9e0] hover:text-[#59636e]'
                      }`}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full inline-block"
                        style={{ backgroundColor: isChecked ? m.color : '#8c959f' }}
                      />
                      <span>{m.name}</span>
                    </button>
                  );
                })}
              </div>

              {/* Sub-tab view toggle (Overview Fit & BIC vs Dedicated Parameter Comparison) */}
              <div className="flex bg-[#f6f8fa] p-1 border border-[#d1d9e0] rounded-lg text-xs font-medium self-start sm:self-auto">
                <button
                  onClick={() => setComparisonSubTab('fit')}
                  className={`px-3 py-1 rounded cursor-pointer transition-colors ${
                    comparisonSubTab === 'fit'
                      ? 'bg-white text-[#0969da] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  Fit &amp; BIC Comparison
                </button>
                <button
                  onClick={() => setComparisonSubTab('params')}
                  className={`px-3 py-1 rounded cursor-pointer transition-colors ${
                    comparisonSubTab === 'params'
                      ? 'bg-white text-[#0969da] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  Parameter Comparison
                </button>
              </div>
            </div>

            {/* SUB-VIEW A: FIT & BIC EVOLUTION + FINAL COMPARISON TABLE & BAR PLOT */}
            {comparisonSubTab === 'fit' && (
              <div className="space-y-6">
                {/* 1. EVOLUTION PLOTS: EVIDENCE & BIC */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Total Evidence Evolution */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0] flex justify-between items-center">
                      <div>
                        <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                          <Activity className="w-4 h-4 text-[#0969da]" />
                          Total Evidence Evolution Across Iterations
                        </h3>
                        <p className="text-xs text-[#59636e]">Log marginal likelihood (higher is better).</p>
                      </div>
                    </div>

                    <svg viewBox={`0 0 ${fitW} ${fitH}`} className="w-full h-52 select-none">
                      {[0, 0.5, 1.0].map((frac, idx) => {
                        const yPos = fitPad.top + frac * (fitH - fitPad.top - fitPad.bottom);
                        const val = compEvMax - frac * compEvRange;
                        return (
                          <g key={idx}>
                            <line x1={fitPad.left} y1={yPos} x2={fitW - fitPad.right} y2={yPos} stroke="#eaeef2" />
                            <text x={fitPad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                              {val.toFixed(0)}
                            </text>
                          </g>
                        );
                      })}

                      {/* Line for each model */}
                      {activeModelList.map((m) => {
                        const pathD = m.evidenceHistory
                          .map((ev, idx) => `${idx === 0 ? 'M' : 'L'} ${getFitX(idx)},${getCompEvY(ev)}`)
                          .join(' ');
                        return (
                          <g key={m.name}>
                            <path d={pathD} fill="none" stroke={m.color} strokeWidth="2.5" />
                            {m.evidenceHistory.map((ev, idx) => (
                              <circle key={idx} cx={getFitX(idx)} cy={getCompEvY(ev)} r="3" fill={m.color} stroke="#fff" strokeWidth="1" />
                            ))}
                          </g>
                        );
                      })}
                    </svg>

                    {/* Legend */}
                    <div className="flex items-center gap-4 flex-wrap border-t border-[#eaeef2] pt-2 text-xs font-mono">
                      {activeModelList.map((m) => (
                        <div key={m.name} className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                          <span className="text-[#59636e]">{m.name}:</span>
                          <strong className="text-[#1f2328]">{m.finalEvidence.toFixed(1)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* BIC Evolution */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0] flex justify-between items-center">
                      <div>
                        <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-[#cf222e]" />
                          BIC Evolution Across Iterations
                        </h3>
                        <p className="text-xs text-[#59636e]">Bayesian Information Criterion (lower is better).</p>
                      </div>
                    </div>

                    <svg viewBox={`0 0 ${fitW} ${fitH}`} className="w-full h-52 select-none">
                      {[0, 0.5, 1.0].map((frac, idx) => {
                        const yPos = fitPad.top + frac * (fitH - fitPad.top - fitPad.bottom);
                        const val = compBicMax - frac * compBicRange;
                        return (
                          <g key={idx}>
                            <line x1={fitPad.left} y1={yPos} x2={fitW - fitPad.right} y2={yPos} stroke="#eaeef2" />
                            <text x={fitPad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                              {val.toFixed(0)}
                            </text>
                          </g>
                        );
                      })}

                      {/* Line for each model */}
                      {activeModelList.map((m) => {
                        const pathD = m.bicHistory
                          .map((bic, idx) => `${idx === 0 ? 'M' : 'L'} ${getFitX(idx)},${getCompBicY(bic)}`)
                          .join(' ');
                        return (
                          <g key={m.name}>
                            <path d={pathD} fill="none" stroke={m.color} strokeWidth="2.2" strokeDasharray="4 2" />
                            {m.bicHistory.map((bic, idx) => (
                              <circle key={idx} cx={getFitX(idx)} cy={getCompBicY(bic)} r="3" fill={m.color} stroke="#fff" strokeWidth="1" />
                            ))}
                          </g>
                        );
                      })}
                    </svg>

                    {/* Legend */}
                    <div className="flex items-center gap-4 flex-wrap border-t border-[#eaeef2] pt-2 text-xs font-mono">
                      {activeModelList.map((m) => (
                        <div key={m.name} className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                          <span className="text-[#59636e]">{m.name}:</span>
                          <strong className="text-[#1f2328]">{m.finalBIC.toFixed(1)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 2. FINAL COMPARISON BAR PLOT & SUMMARY TABLE */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Final Comparison Bar Plot */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0]">
                      <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-[#0969da]" />
                        Final Model Evidence &amp; BIC Comparison Bar Plot
                      </h3>
                      <p className="text-xs text-[#59636e]">Side-by-side score comparison at convergence.</p>
                    </div>

                    <div className="h-56 flex items-end gap-6 justify-center px-4 pt-6 pb-2 border border-[#eaeef2] rounded-lg bg-white">
                      {activeModelList.map((m) => {
                        const bicHeight = Math.max(20, ((m.finalBIC - 250) / 220) * 160);
                        return (
                          <div key={m.name} className="flex flex-col items-center gap-1.5">
                            <span className="text-[11px] font-mono font-bold text-[#1f2328]">
                              {m.finalBIC.toFixed(1)}
                            </span>
                            <div
                              style={{ height: `${bicHeight}px`, backgroundColor: m.color }}
                              className="w-12 rounded-t transition-all hover:opacity-90 shadow-2xs"
                            />
                            <span className="text-[11px] font-mono text-[#59636e] truncate max-w-[85px] text-center">
                              {m.name}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Comparative Summary Table */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0]">
                      <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                        <Award className="w-4 h-4 text-[#1a7f37]" />
                        Final Model Ranking &amp; Comparison Table
                      </h3>
                      <p className="text-xs text-[#59636e]">Sorted by BIC rank (lowest BIC is superior).</p>
                    </div>

                    <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2] font-mono">
                          <tr>
                            <th className="py-2.5 px-3 font-semibold">Rank</th>
                            <th className="py-2.5 px-3 font-semibold">Model</th>
                            <th className="py-2.5 px-3 font-semibold">k</th>
                            <th className="py-2.5 px-3 font-semibold">Evidence</th>
                            <th className="py-2.5 px-3 font-semibold">BIC</th>
                            <th className="py-2.5 px-3 font-semibold">ΔBIC</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#eaeef2] font-mono">
                          {sortedModels.map((m, idx) => {
                            const deltaBic = m.finalBIC - bestBicVal;
                            const isWinner = idx === 0;
                            return (
                              <tr
                                key={m.name}
                                className={isWinner ? 'bg-[#1a7f37]/5 font-semibold' : 'hover:bg-[#f6f8fa]'}
                              >
                                <td className="py-2.5 px-3">
                                  {isWinner ? (
                                    <span className="px-1.5 py-0.5 rounded bg-[#1a7f37] text-white text-[10px] font-bold">
                                      #1 Best
                                    </span>
                                  ) : (
                                    <span className="text-[#59636e]">#{idx + 1}</span>
                                  )}
                                </td>
                                <td className="py-2.5 px-3 font-bold text-[#1f2328] flex items-center gap-1.5">
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                                  <span>{m.name}</span>
                                </td>
                                <td className="py-2.5 px-3 text-[#59636e]">{m.k}</td>
                                <td className="py-2.5 px-3 text-[#0969da]">{m.finalEvidence.toFixed(1)}</td>
                                <td className="py-2.5 px-3 text-[#1f2328]">{m.finalBIC.toFixed(1)}</td>
                                <td className="py-2.5 px-3">
                                  {isWinner ? (
                                    <span className="text-[#1a7f37] font-bold">0.0</span>
                                  ) : (
                                    <span className="text-[#cf222e]">+{deltaBic.toFixed(1)}</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    <div className="text-[11px] text-[#59636e] font-mono pt-1">
                      * ΔBIC &gt; 10 represents very strong evidence against the higher-BIC model (Kass &amp; Raftery, 1995).
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* SUB-VIEW B: SEPARATE DEDICATED PARAMETER COMPARISON PAGE */}
            {comparisonSubTab === 'params' && (
              <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-5">
                <div className="pb-3 border-b border-[#d1d9e0] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                      <SlidersHorizontal className="w-5 h-5 text-[#8250df]" />
                      Parameter Comparison Across Models
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      Compare fitted population mean (±1 SD) and subject-level distributions for parameters shared across models.
                    </p>
                  </div>

                  {/* Parameter selector button group */}
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-[#59636e] font-mono mr-1">Parameter:</span>
                    {(['alpha', 'beta'] as const).map((pk) => (
                      <button
                        key={pk}
                        onClick={() => setCompareParamKey(pk)}
                        className={`px-3 py-1 rounded text-xs font-mono font-semibold cursor-pointer transition-colors ${
                          compareParamKey === pk
                            ? 'bg-[#8250df] text-white shadow-2xs'
                            : 'bg-[#f6f8fa] text-[#59636e] border border-[#d1d9e0] hover:bg-white'
                        }`}
                      >
                        {pk}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Parameter Comparison Visualization */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {activeModelList
                    .filter((m) => (compareParamKey === 'alpha' ? m.alphaMean !== undefined : m.betaMean !== undefined))
                    .map((m) => {
                      const mean = compareParamKey === 'alpha' ? m.alphaMean! : m.betaMean!;
                      const sd = compareParamKey === 'alpha' ? m.alphaSD! : m.betaSD!;
                      const subjects = compareParamKey === 'alpha' ? m.alphaSubjects! : m.betaSubjects!;

                      return (
                        <div key={m.name} className="border border-[#eaeef2] rounded-lg p-4 bg-white space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-xs font-mono text-[#1f2328] flex items-center gap-1.5">
                              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                              {m.name}
                            </span>
                            <span className="text-[11px] font-mono text-[#8250df] font-semibold">
                              μ = {mean.toFixed(3)} (±{sd.toFixed(3)})
                            </span>
                          </div>

                          {/* Scatter of subjects & population bar */}
                          <div className="h-44 border border-[#eaeef2] rounded bg-[#f6f8fa]/50 p-2 relative flex flex-col justify-between">
                            <div className="text-[10px] font-mono text-[#8c959f] flex justify-between">
                              <span>Population ±1 SD</span>
                              <span>[{ (mean - sd).toFixed(2) }, { (mean + sd).toFixed(2) }]</span>
                            </div>

                            {/* Subjects jittered dots */}
                            <div className="space-y-1">
                              {subjects.map((sVal, sIdx) => {
                                const minScale = compareParamKey === 'alpha' ? 0.2 : 1.5;
                                const maxScale = compareParamKey === 'alpha' ? 0.6 : 4.0;
                                const pct = ((sVal - minScale) / (maxScale - minScale)) * 100;
                                return (
                                  <div key={sIdx} className="relative h-2 w-full">
                                    <div
                                      style={{ left: `${Math.min(95, Math.max(5, pct))}%`, backgroundColor: m.color }}
                                      className="absolute w-2.5 h-2.5 -top-0.5 rounded-full shadow-2xs opacity-85"
                                      title={`Subject ${sIdx}: ${sVal.toFixed(3)}`}
                                    />
                                  </div>
                                );
                              })}
                            </div>

                            <div className="text-[10px] font-mono text-[#59636e] flex justify-between border-t border-[#eaeef2] pt-1">
                              <span>Min: {Math.min(...subjects).toFixed(2)}</span>
                              <span>Max: {Math.max(...subjects).toFixed(2)}</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>

                <div className="p-3 bg-[#f6f8fa] border border-[#d1d9e0] rounded-md text-xs font-mono text-[#59636e] flex items-center justify-between">
                  <span># Python code to generate this comparative report:</span>
                  <span className="text-[#0969da] font-semibold">
                    compare_models([m1, m2, m3], compare_params=["{compareParamKey}"])
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 2: SINGLE MODEL REPORT (PREVIOUS EXPLORER)                 */}
        {/* ============================================================== */}
        {activeTab === 'report' && (
          <div className="space-y-6">
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#0969da]/10 text-[#0969da] border border-[#0969da]/20">
                    Live Single Model Report
                  </span>
                  <span className="text-xs text-[#59636e]">
                    Generated from <code>sampler.create_report()</code>
                  </span>
                </div>
                <h1 className="text-2xl font-bold text-[#1f2328]">
                  Model Diagnostics &amp; Estimation Results
                </h1>
                <p className="text-sm text-[#59636e] mt-1">
                  10 Subjects • 6 Parameters • Converged in 12 Iterations (1,000 particles/subject)
                </p>
              </div>

              <div className="flex items-center gap-3 font-mono">
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">Evidence</div>
                  <div className="text-lg font-bold text-[#0969da]">-141.30</div>
                </div>
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">BIC</div>
                  <div className="text-lg font-bold text-[#1a7f37]">291.90</div>
                </div>
              </div>
            </div>

            {/* Total Model Evidence & Subject Spaghetti Curves */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                <div className="pb-2 border-b border-[#d1d9e0]">
                  <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#0969da]" />
                    Total Model Evidence &amp; BIC
                  </h3>
                  <p className="text-xs text-[#59636e]">Evidence climbing to maximum; BIC minimizing.</p>
                </div>
                <svg viewBox={`0 0 ${fitW} ${fitH}`} className="w-full h-48 select-none">
                  {[0, 0.5, 1.0].map((frac, idx) => {
                    const yPos = fitPad.top + frac * (fitH - fitPad.top - fitPad.bottom);
                    const val = evMax - frac * evRange;
                    return (
                      <g key={idx}>
                        <line x1={fitPad.left} y1={yPos} x2={fitW - fitPad.right} y2={yPos} stroke="#eaeef2" />
                        <text x={fitPad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                          {val.toFixed(0)}
                        </text>
                      </g>
                    );
                  })}
                  <path d={evidenceD} fill="none" stroke="#0969da" strokeWidth="2.5" />
                  {mockEvidence.map((ev, idx) => (
                    <circle key={idx} cx={getFitX(idx)} cy={getEvY(ev)} r="3" fill="#0969da" stroke="#fff" strokeWidth="1" />
                  ))}
                </svg>
              </div>

              <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                <div className="pb-2 border-b border-[#d1d9e0]">
                  <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#8250df]" />
                    Subject-Level Evidence Spaghetti Plot
                  </h3>
                  <p className="text-xs text-[#59636e]">Per-subject log-likelihood trajectories.</p>
                </div>
                <svg viewBox={`0 0 ${fitW} ${fitH}`} className="w-full h-48 select-none">
                  {[0, 0.5, 1.0].map((frac, idx) => {
                    const yPos = fitPad.top + frac * (fitH - fitPad.top - fitPad.bottom);
                    const val = -13.0 - frac * 9;
                    return (
                      <g key={idx}>
                        <line x1={fitPad.left} y1={yPos} x2={fitW - fitPad.right} y2={yPos} stroke="#eaeef2" />
                        <text x={fitPad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                          {val.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}
                  {mockSubjSpaghetti.map((traj, s) => {
                    const d = traj
                      .map((val, idx) => {
                        const y = fitH - fitPad.bottom - ((val - -22.5) / 10) * (fitH - fitPad.top - fitPad.bottom);
                        return `${idx === 0 ? 'M' : 'L'} ${getFitX(idx)},${y}`;
                      })
                      .join(' ');
                    const isHovered = hoveredSubj === s;
                    return (
                      <path
                        key={s}
                        d={d}
                        fill="none"
                        stroke={isHovered ? '#8250df' : 'rgba(89, 99, 110, 0.3)'}
                        strokeWidth={isHovered ? 2.5 : 1}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHoveredSubj(s)}
                        onMouseLeave={() => setHoveredSubj(null)}
                      />
                    );
                  })}
                </svg>
              </div>
            </div>

            {/* Parameters Table */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-bold text-[#1f2328] uppercase tracking-wider flex items-center gap-2">
                    <Grid className="w-4 h-4 text-[#0969da]" />
                    Model Parameters Table
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setSelectedParams([...PARAM_KEYS])}
                    className="text-xs text-[#0969da] hover:underline cursor-pointer flex items-center gap-1 font-medium"
                  >
                    <CheckSquare className="w-3.5 h-3.5" />
                    Select All
                  </button>
                  <span className="text-[#d1d9e0]">|</span>
                  <button
                    onClick={() => setSelectedParams(['alpha', 'beta'])}
                    className="text-xs text-[#59636e] hover:underline cursor-pointer flex items-center gap-1 font-medium"
                  >
                    <Square className="w-3.5 h-3.5" />
                    Reset
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2] font-mono">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Active</th>
                      <th className="py-2.5 px-4 font-semibold">Parameter Key</th>
                      <th className="py-2.5 px-4 font-semibold">Fitted Mean</th>
                      <th className="py-2.5 px-4 font-semibold">Fitted SD</th>
                      <th className="py-2.5 px-4 font-semibold">±1 SD Range</th>
                      <th className="py-2.5 px-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#eaeef2]">
                    {PARAM_KEYS.map((key) => {
                      const isSelected = selectedParams.includes(key);
                      const traj = mockTrajectories[key];
                      const finalMean = traj.means[11];
                      const finalSD = (traj.sdUpper[11] - traj.sdLower[11]) / 2;

                      return (
                        <tr
                          key={key}
                          onClick={() => toggleParamInEvolution(key)}
                          className={`cursor-pointer transition-colors ${
                            isSelected ? 'bg-[#0969da]/8 hover:bg-[#0969da]/12' : 'hover:bg-[#f6f8fa]'
                          }`}
                        >
                          <td className="py-2.5 px-4">
                            <input type="checkbox" checked={isSelected} onChange={() => {}} className="rounded text-[#0969da] cursor-pointer" />
                          </td>
                          <td className="py-2.5 px-4 font-mono font-bold text-[#1f2328]">{key}</td>
                          <td className="py-2.5 px-4 font-mono text-[#0969da]">{finalMean.toFixed(3)}</td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e]">{finalSD.toFixed(3)}</td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e]">
                            [{traj.sdLower[11].toFixed(2)}, {traj.sdUpper[11].toFixed(2)}]
                          </td>
                          <td className="py-2.5 px-4">
                            {isSelected ? (
                              <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-[#0969da] text-white">
                                Active ({selectedParams.indexOf(key) + 1})
                              </span>
                            ) : (
                              <span className="text-[11px] text-[#8c959f]">Click to add</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Hyperparameter Evolution Grid (3 in a row) */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0] flex items-center justify-between">
                <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-[#0969da]" />
                  Hyperparameter Evolution Grid ({selectedParams.length} Parameters)
                </h2>
                <div className="text-xs font-mono text-[#59636e]">3 in a row</div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {selectedParams.map((pKey) => {
                  const traj = mockTrajectories[pKey];
                  const pYMin = Math.min(...traj.sdLower);
                  const pYMax = Math.max(...traj.sdUpper);
                  const pYRange = pYMax - pYMin || 1;

                  const getP_X = (it: number) => subPad.left + (it / 11) * (subW - subPad.left - subPad.right);
                  const getP_Y = (val: number) =>
                    subH - subPad.bottom - ((val - pYMin) / pYRange) * (subH - subPad.top - subPad.bottom);

                  const upperPts = traj.iters.map((it, idx) => `${getP_X(it)},${getP_Y(traj.sdUpper[idx])}`);
                  const lowerPts = traj.iters
                    .slice()
                    .reverse()
                    .map((it, idx) => {
                      const orig = 11 - idx;
                      return `${getP_X(it)},${getP_Y(traj.sdLower[orig])}`;
                    });
                  const pRibbonD = `M ${upperPts.join(' L ')} L ${lowerPts.join(' L ')} Z`;
                  const pMeanD = traj.iters
                    .map((it, idx) => `${idx === 0 ? 'M' : 'L'} ${getP_X(it)},${getP_Y(traj.means[idx])}`)
                    .join(' ');

                  const isHovered = hoveredIter?.param === pKey;

                  return (
                    <div key={pKey} className="border border-[#eaeef2] rounded-lg p-3 bg-white">
                      <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                        <span className="font-bold text-[#1f2328]">{pKey}</span>
                        <span className="text-[#0969da]">mean={traj.means[11].toFixed(3)}</span>
                      </div>

                      <svg viewBox={`0 0 ${subW} ${subH}`} className="w-full h-40 select-none" onMouseLeave={() => setHoveredIter(null)}>
                        {[0, 0.5, 1.0].map((frac, idx) => {
                          const y = subPad.top + frac * (subH - subPad.top - subPad.bottom);
                          const val = pYMax - frac * pYRange;
                          return (
                            <g key={idx}>
                              <line x1={subPad.left} y1={y} x2={subW - subPad.right} y2={y} stroke="#eaeef2" />
                              <text x={subPad.left - 6} y={y + 3} textAnchor="end" fontSize="9" fill="#8c959f" fontFamily="monospace">
                                {val.toFixed(2)}
                              </text>
                            </g>
                          );
                        })}
                        <path d={pRibbonD} fill="rgba(9, 105, 218, 0.16)" />
                        <path d={pMeanD} fill="none" stroke="#0969da" strokeWidth="2.2" />
                        {traj.iters.map((it, idx) => (
                          <circle
                            key={it}
                            cx={getP_X(it)}
                            cy={getP_Y(traj.means[idx])}
                            r={isHovered && hoveredIter.iter === it ? 4.5 : 2.5}
                            fill="#0969da"
                            stroke="#fff"
                            strokeWidth="1"
                            className="cursor-pointer"
                            onMouseEnter={() => setHoveredIter({ param: pKey, iter: it })}
                          />
                        ))}
                      </svg>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Individual Subject Means (Scatter Plot: Y=Subject #, X=Value) */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0] flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-[#1a7f37]" />
                    Individual Subject Posterior Means
                  </h2>
                  <p className="text-xs text-[#59636e]">Y-axis = Subject Number, X-axis = Parameter Value.</p>
                </div>

                <div className="flex items-center gap-1.5 flex-wrap">
                  {PARAM_KEYS.map((pk) => {
                    const isShown = subjectSelectedParams.includes(pk);
                    return (
                      <button
                        key={pk}
                        onClick={() => toggleParamInSubjects(pk)}
                        className={`px-2.5 py-1 rounded text-xs font-mono cursor-pointer transition-colors ${
                          isShown
                            ? 'bg-[#1a7f37] text-white font-semibold'
                            : 'bg-[#f6f8fa] text-[#59636e] border border-[#d1d9e0] hover:bg-white'
                        }`}
                      >
                        {pk}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {subjectSelectedParams.map((pKey) => {
                  const sVals = mockSubjMeans[pKey];
                  const sMin = Math.min(...sVals);
                  const sMax = Math.max(...sVals);
                  const sRange = sMax - sMin || 1;

                  const getS_X = (val: number) => subPad.left + ((val - sMin) / sRange) * (subW - subPad.left - subPad.right);
                  const getS_Y = (subjIdx: number) => subH - subPad.bottom - (subjIdx / 9) * (subH - subPad.top - subPad.bottom);

                  return (
                    <div key={pKey} className="border border-[#eaeef2] rounded-lg p-3 bg-white">
                      <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                        <span className="font-bold text-[#1f2328]">{pKey}</span>
                        <span className="text-[#1a7f37]">range: [{sMin.toFixed(2)}, {sMax.toFixed(2)}]</span>
                      </div>

                      <svg viewBox={`0 0 ${subW} ${subH}`} className="w-full h-40 select-none">
                        {[0, 3, 6, 9].map((sIdx) => (
                          <g key={sIdx}>
                            <line x1={subPad.left} y1={getS_Y(sIdx)} x2={subW - subPad.right} y2={getS_Y(sIdx)} stroke="#eaeef2" />
                            <text x={subPad.left - 6} y={getS_Y(sIdx) + 3} textAnchor="end" fontSize="9" fill="#8c959f" fontFamily="monospace">
                              S{sIdx}
                            </text>
                          </g>
                        ))}
                        {sVals.map((val, sIdx) => (
                          <circle
                            key={sIdx}
                            cx={getS_X(val)}
                            cy={getS_Y(sIdx)}
                            r="4"
                            fill="#1a7f37"
                            stroke="#fff"
                            strokeWidth="1"
                          >
                            <title>{`Subject ${sIdx}: ${val.toFixed(3)}`}</title>
                          </circle>
                        ))}
                      </svg>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Correlation Heatmap */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0]">
                <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                  <Grid className="w-5 h-5 text-[#cf222e]" />
                  Multinormal Parameter Correlation Matrix Heatmap
                </h2>
              </div>
              <div className="overflow-x-auto border border-[#eaeef2] rounded-lg p-4 bg-white flex justify-center">
                <table className="border-collapse font-mono text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-[#59636e]"></th>
                      {PARAM_KEYS.map((col) => (
                        <th key={col} className="p-2 text-center text-[#1f2328] font-bold">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PARAM_KEYS.map((rowKey) => (
                      <tr key={rowKey}>
                        <td className="p-2 font-bold text-[#1f2328] text-right">{rowKey}</td>
                        {PARAM_KEYS.map((colKey) => {
                          const r = mockCorMatrix[rowKey][colKey];
                          let bg = 'rgba(255, 255, 255, 1)';
                          let textColor = '#1f2328';
                          if (r > 0) {
                            bg = `rgba(9, 105, 218, ${Math.min(1, r * 0.75 + 0.1)})`;
                            if (r > 0.4) textColor = '#ffffff';
                          } else if (r < 0) {
                            bg = `rgba(207, 34, 46, ${Math.min(1, Math.abs(r) * 0.75 + 0.1)})`;
                            if (r < -0.3) textColor = '#ffffff';
                          }
                          return (
                            <td key={colKey} style={{ backgroundColor: bg, color: textColor }} className="p-3 text-center border border-[#eaeef2] font-semibold text-xs">
                              {r.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 3: DOCUMENTATION & API REFERENCE                           */}
        {/* ============================================================== */}
        {activeTab === 'docs' && (
          <div className="space-y-6">
            <article className="border border-[#d1d9e0] rounded-md p-8 bg-white shadow-2xs space-y-6 text-sm text-[#1f2328] leading-relaxed">
              <div className="border-b border-[#d1d9e0] pb-4">
                <h1 className="text-3xl font-bold tracking-tight text-[#1f2328]">
                  importance_sampling
                </h1>
                <p className="text-sm text-[#59636e] mt-1.5">
                  A lightweight Python package for fitting computational and cognitive models using <strong>Iterative Importance Sampling (IIS)</strong>.
                </p>
              </div>

              {/* Install callout */}
              <div className="p-3 bg-[#f6f8fa] border border-[#d1d9e0] rounded-md font-mono text-xs flex items-center justify-between">
                <span>{installCmd}</span>
                <button onClick={handleCopy} className="text-[#0969da] hover:underline cursor-pointer">
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Multi-Model Comparison (<code>compare_models</code>)
                </h2>
                <p className="mb-2">
                  Compare two or more fitted computational models on likelihood (evidence), BIC, and parameter trajectories:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`from importance_sampling import compare_models

# Compare multiple fitted samplers
compare_models(
    [sampler_standard, sampler_dual_lr, sampler_perseveration],
    filename="model_comparison.html",
    compare_params=["lr", "inv_temp"],  # Optional parameter comparison
)`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li><strong>Evidence &amp; BIC Evolution</strong>: Trajectory curves across iterations for all models on one plot.</li>
                  <li><strong>Final Comparison</strong>: Bar plot and comprehensive summary table with k, Evidence, BIC, and &Delta;BIC.</li>
                  <li><strong>Parameter Comparison</strong>: Population mean &plusmn; 1 SD error bars and subject distributions across models.</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Single Model Report (<code>sampler.create_report</code>)
                </h2>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs">
                  sampler.create_report(filename="single_model_report.html")
                </pre>
              </div>
            </article>
          </div>
        )}
      </main>
    </div>
  );
}
