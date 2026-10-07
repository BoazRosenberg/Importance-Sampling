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
  FileCode,
  FileText,
  Clock,
  Download,
  ChevronRight,
  ChevronLeft,
  Printer,
  Calendar,
  Timer,
  Cpu,
  Sparkles,
  Database,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
} from 'lucide-react';

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function highlightPythonCode(code: string): string {
  if (!code) return '';

  const tokenRegex = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\bdef\s+([a-zA-Z_]\w*)|\b(?:def|return|for|in|if|else|elif|import|from|as|and|or|not|while|yield|pass|break|continue|lambda|try|except|finally|raise|with|class)\b|\b(?:True|False|None)\b|\b(?:self)\b|\b(?:np|zeros|ones|array|exp|log|max|min|sum|len|range|zip|enumerate|float|int|str|dict|list|set|bool|expit|softplus|sigmoid|clip|print|abs|round)\b|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|(==|!=|<=|>=|\+=|-=|\*=|\/=|[-+*\/=<>%]))/g;

  let result = '';
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(code)) !== null) {
    if (match.index > lastIndex) {
      result += escapeHtml(code.slice(lastIndex, match.index));
    }

    const token = match[0];

    if (token.startsWith('"""') || token.startsWith("'''")) {
      result += `<span style="color: #7ee787; font-style: italic;">${escapeHtml(token)}</span>`;
    } else if (token.startsWith('"') || token.startsWith("'")) {
      result += `<span style="color: #a5d6ff;">${escapeHtml(token)}</span>`;
    } else if (token.startsWith('#')) {
      result += `<span style="color: #8b949e; font-style: italic;">${escapeHtml(token)}</span>`;
    } else if (token.startsWith('def ')) {
      const funcName = match[2] || token.slice(4).trim();
      result += `<span style="color: #ff7b72; font-weight: 600;">def</span> <span style="color: #d2a8ff; font-weight: 700;">${escapeHtml(funcName)}</span>`;
    } else if (/^(def|return|for|in|if|else|elif|import|from|as|and|or|not|while|yield|pass|break|continue|lambda|try|except|finally|raise|with|class)$/.test(token)) {
      result += `<span style="color: #ff7b72; font-weight: 600;">${escapeHtml(token)}</span>`;
    } else if (/^(True|False|None)$/.test(token)) {
      result += `<span style="color: #79c0ff; font-weight: 600;">${escapeHtml(token)}</span>`;
    } else if (token === 'self') {
      result += `<span style="color: #ffa657; font-style: italic;">self</span>`;
    } else if (/^(np|zeros|ones|array|exp|log|max|min|sum|len|range|zip|enumerate|float|int|str|dict|list|set|bool|expit|softplus|sigmoid|clip|print|abs|round)$/.test(token)) {
      result += `<span style="color: #79c0ff;">${escapeHtml(token)}</span>`;
    } else if (/^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(token)) {
      result += `<span style="color: #79c0ff;">${escapeHtml(token)}</span>`;
    } else if (/^(==|!=|<=|>=|\+=|-=|\*=|\/=|[-+*\/=<>%])$/.test(token)) {
      result += `<span style="color: #ff7b72;">${escapeHtml(token)}</span>`;
    } else {
      result += escapeHtml(token);
    }

    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < code.length) {
    result += escapeHtml(code.slice(lastIndex));
  }

  return result;
}

const PARAM_KEYS = ['alpha', 'beta', 'decay', 'bias', 'noise', 'pers'] as const;
type ParamKey = typeof PARAM_KEYS[number];

interface ParamTrajectory {
  iters: number[];
  means: number[];
  sdUpper: number[];
  sdLower: number[];
  groupA?: number[];
  groupB?: number[];
}

const mockTrajectories: Record<ParamKey, ParamTrajectory> = {
  alpha: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.500, 0.462, 0.435, 0.418, 0.410, 0.405, 0.402, 0.400, 0.398, 0.397, 0.397, 0.396],
    sdUpper: [0.731, 0.684, 0.632, 0.589, 0.554, 0.528, 0.509, 0.495, 0.485, 0.478, 0.474, 0.471],
    sdLower: [0.269, 0.261, 0.258, 0.264, 0.276, 0.291, 0.304, 0.312, 0.318, 0.322, 0.325, 0.327],
    groupA: [0.530, 0.490, 0.462, 0.443, 0.434, 0.428, 0.425, 0.423, 0.421, 0.420, 0.420, 0.419],
    groupB: [0.470, 0.434, 0.408, 0.393, 0.386, 0.382, 0.379, 0.377, 0.375, 0.374, 0.374, 0.373],
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
  description: string;
  color: string;
  k: number;
  params: string[];
  groupDiffParams?: string[];
  evidenceHistory: number[];
  bicHistory: number[];
  finalEvidence: number;
  finalBIC: number;
  bestExplainedCount: number;
  bestExplainedPct: number;
  code: string;
  paramFitted: Record<string, { mean: number; sd: number }>;
  paramEvolutions: Record<string, number[]>;
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
    description: 'Q-learning with choice perseveration and unchosen value decay',
    color: '#0969da',
    k: 3,
    params: ['alpha', 'beta', 'pers'],
    groupDiffParams: ['alpha'],
    evidenceHistory: [-212.4, -188.2, -172.5, -161.8, -154.2, -149.1, -145.7, -143.5, -142.4, -141.8, -141.5, -141.3],
    bicHistory: [434.1, 385.7, 354.3, 332.9, 317.7, 307.5, 300.7, 296.3, 294.1, 292.9, 292.3, 291.9],
    finalEvidence: -141.3,
    finalBIC: 291.9,
    bestExplainedCount: 6,
    bestExplainedPct: 60.0,
    paramFitted: {
      alpha: { mean: 0.396, sd: 0.072 },
      beta: { mean: 3.05, sd: 0.74 },
      pers: { mean: 0.340, sd: 0.32 },
    },
    paramEvolutions: {
      alpha: [0.500, 0.462, 0.435, 0.418, 0.410, 0.405, 0.402, 0.400, 0.398, 0.397, 0.397, 0.396],
      beta:  [1.31, 1.84, 2.21, 2.48, 2.67, 2.81, 2.91, 2.97, 3.01, 3.03, 3.04, 3.05],
      pers:  [0.00, 0.10, 0.18, 0.24, 0.28, 0.31, 0.32, 0.33, 0.34, 0.34, 0.34, 0.34],
    },
    code: `def q_learning_with_perseveration(subj_data, parameters, mode="log_likelihood"):
    """Q-learning model with choice perseveration and value decay."""
    choices, rewards = subj_data["choice"], subj_data["reward"]
    alpha, beta, pers = parameters["alpha"], parameters["beta"], parameters["pers"]
    Q = np.zeros(2)
    last_c, log_lik = -1, 0.0
    for c, r in zip(choices, rewards):
        v = beta * Q + np.array([pers if last_c == 0 else 0.0, pers if last_c == 1 else 0.0])
        p1 = 1.0 / (1.0 + np.exp(v[0] - v[1]))
        p_c = p1 if c == 1 else (1.0 - p1)
        log_lik += np.log(max(1e-12, p_c))
        Q[c] += alpha * (r - Q[c])
        last_c = c
    return log_lik`,
    alphaMean: 0.396,
    alphaSD: 0.072,
    betaMean: 3.05,
    betaSD: 0.74,
    alphaSubjects: [0.38, 0.42, 0.35, 0.45, 0.39, 0.33, 0.44, 0.37, 0.41, 0.40],
    betaSubjects:  [3.21, 2.85, 3.42, 2.65, 3.10, 3.65, 2.78, 3.30, 2.92, 3.05],
  },
  Standard_QLearn: {
    name: 'Standard_QLearn',
    description: 'Classic Rescorla-Wagner Q-learning with softmax choice',
    color: '#1a7f37',
    k: 2,
    params: ['alpha', 'beta'],
    groupDiffParams: [],
    evidenceHistory: [-218.0, -195.4, -182.1, -174.5, -168.2, -164.5, -162.1, -160.8, -160.2, -159.9, -159.7, -159.5],
    bicHistory: [442.0, 396.8, 370.2, 355.0, 342.4, 335.0, 330.2, 327.6, 326.4, 325.8, 325.4, 325.0],
    finalEvidence: -159.5,
    finalBIC: 325.0,
    bestExplainedCount: 1,
    bestExplainedPct: 10.0,
    paramFitted: {
      alpha: { mean: 0.435, sd: 0.095 },
      beta: { mean: 2.62, sd: 0.88 },
    },
    paramEvolutions: {
      alpha: [0.500, 0.482, 0.468, 0.455, 0.448, 0.442, 0.439, 0.437, 0.436, 0.435, 0.435, 0.435],
      beta:  [1.00, 1.45, 1.80, 2.05, 2.25, 2.40, 2.50, 2.56, 2.60, 2.61, 2.62, 2.62],
    },
    code: `def standard_q_learning(subj_data, parameters, mode="log_likelihood"):
    """Classic 2-parameter Rescorla-Wagner / Q-learning model."""
    choices, rewards = subj_data["choice"], subj_data["reward"]
    alpha, beta = parameters["alpha"], parameters["beta"]
    Q = np.zeros(2)
    log_lik = 0.0
    for c, r in zip(choices, rewards):
        p1 = 1.0 / (1.0 + np.exp(beta * (Q[0] - Q[1])))
        p_c = p1 if c == 1 else (1.0 - p1)
        log_lik += np.log(max(1e-12, p_c))
        Q[c] += alpha * (r - Q[c])
    return log_lik`,
    alphaMean: 0.435,
    alphaSD: 0.095,
    betaMean: 2.62,
    betaSD: 0.88,
    alphaSubjects: [0.41, 0.46, 0.39, 0.49, 0.43, 0.37, 0.48, 0.42, 0.45, 0.45],
    betaSubjects:  [2.75, 2.45, 3.05, 2.20, 2.70, 3.15, 2.40, 2.85, 2.50, 2.65],
  },
  Dual_Alpha_QLearn: {
    name: 'Dual_Alpha_QLearn',
    description: 'Asymmetric Q-learning with separate positive & negative learning rates',
    color: '#8250df',
    k: 3,
    params: ['alpha_pos', 'alpha_neg', 'beta'],
    groupDiffParams: ['alpha_pos'],
    evidenceHistory: [-215.2, -191.0, -177.3, -169.1, -163.5, -159.8, -157.2, -155.5, -154.8, -154.2, -154.0, -153.8],
    bicHistory: [439.7, 391.3, 363.9, 347.5, 336.3, 328.9, 323.7, 320.3, 318.9, 317.7, 317.3, 316.9],
    finalEvidence: -153.8,
    finalBIC: 316.9,
    bestExplainedCount: 2,
    bestExplainedPct: 20.0,
    paramFitted: {
      alpha_pos: { mean: 0.408, sd: 0.082 },
      alpha_neg: { mean: 0.358, sd: 0.091 },
      beta: { mean: 2.88, sd: 0.81 },
    },
    paramEvolutions: {
      alpha_pos: [0.500, 0.472, 0.451, 0.435, 0.424, 0.418, 0.414, 0.411, 0.409, 0.408, 0.408, 0.408],
      alpha_neg: [0.500, 0.455, 0.421, 0.398, 0.382, 0.372, 0.366, 0.362, 0.360, 0.359, 0.358, 0.358],
      beta:      [1.20, 1.65, 2.02, 2.30, 2.51, 2.66, 2.76, 2.82, 2.86, 2.87, 2.88, 2.88],
    },
    code: `def dual_learning_rate_q_learning(subj_data, parameters, mode="log_likelihood"):
    """Q-learning with separate positive (reward) and negative (loss) learning rates."""
    choices, rewards = subj_data["choice"], subj_data["reward"]
    a_pos, a_neg, beta = parameters["alpha_pos"], parameters["alpha_neg"], parameters["beta"]
    Q = np.zeros(2)
    log_lik = 0.0
    for c, r in zip(choices, rewards):
        p1 = 1.0 / (1.0 + np.exp(beta * (Q[0] - Q[1])))
        p_c = p1 if c == 1 else (1.0 - p1)
        log_lik += np.log(max(1e-12, p_c))
        pe = r - Q[c]
        alpha = a_pos if pe >= 0 else a_neg
        Q[c] += alpha * pe
    return log_lik`,
    alphaMean: 0.412,
    alphaSD: 0.082,
    betaMean: 2.88,
    betaSD: 0.81,
    alphaSubjects: [0.39, 0.44, 0.37, 0.47, 0.41, 0.35, 0.46, 0.39, 0.43, 0.41],
    betaSubjects:  [3.00, 2.70, 3.25, 2.45, 2.95, 3.40, 2.60, 3.10, 2.75, 2.85],
  },
  Biased_Baseline: {
    name: 'Biased_Baseline',
    description: '1-parameter baseline with constant response bias',
    color: '#d97706',
    k: 1,
    params: ['bias'],
    groupDiffParams: [],
    evidenceHistory: [-232.0, -225.1, -222.0, -220.5, -219.8, -219.5, -219.4, -219.3, -219.3, -219.3, -219.3, -219.3],
    bicHistory: [467.0, 453.2, 447.0, 444.0, 442.6, 442.0, 441.8, 441.6, 441.6, 441.6, 441.6, 441.6],
    finalEvidence: -219.3,
    finalBIC: 441.6,
    bestExplainedCount: 1,
    bestExplainedPct: 10.0,
    paramFitted: {
      bias: { mean: 0.160, sd: 0.28 },
    },
    paramEvolutions: {
      bias: [0.00, 0.05, 0.09, 0.12, 0.14, 0.15, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16],
    },
    code: `def biased_random_choice_baseline(subj_data, parameters, mode="log_likelihood"):
    """1-parameter baseline model with constant lateral response bias."""
    choices = subj_data["choice"]
    bias = parameters["bias"]
    p1 = 1.0 / (1.0 + np.exp(-bias))
    log_lik = 0.0
    for c in choices:
        p_c = p1 if c == 1 else (1.0 - p1)
        log_lik += np.log(max(1e-12, p_c))
    return log_lik`,
  },
};

const mockModelCode = `def q_learning_with_perseveration(subj_data, parameters, mode="log_likelihood"):
    """
    Q-learning model with perseveration and unchosen option value decay.

    Parameters:
        alpha: Learning rate [0, 1]
        beta:  Inverse temperature (choice sensitivity) [0, inf)
        decay: Value decay per unchosen option [0, 1]
        bias:  Side bias preference [-inf, inf]
        noise: Random exploration noise [0, 1]
        pers:  Choice perseveration kernel [-inf, inf]
    """
    choices = subj_data["choice"]
    rewards = subj_data["reward"]
    n_trials = len(choices)

    alpha = parameters["alpha"]
    beta  = parameters["beta"]
    decay = parameters["decay"]
    bias  = parameters["bias"]
    noise = parameters["noise"]
    pers  = parameters["pers"]

    # Initialize Q-values for 2 choice options
    Q = np.zeros(2)
    last_choice = -1
    log_lik = 0.0

    for t in range(n_trials):
        c = choices[t]
        r = rewards[t]

        # Action values with perseverance bonus
        v0 = beta * Q[0] + bias + (pers if last_choice == 0 else 0.0)
        v1 = beta * Q[1] + (pers if last_choice == 1 else 0.0)

        # Softmax choice probability with random exploration noise
        p1 = (1.0 - noise) / (1.0 + np.exp(v0 - v1)) + 0.5 * noise
        p_c = p1 if c == 1 else (1.0 - p1)
        log_lik += np.log(max(1e-12, p_c))

        # Value update & decay of unchosen action
        Q[c] += alpha * (r - Q[c])
        unchosen = 1 - c
        Q[unchosen] *= (1.0 - decay)
        last_choice = c

    return log_lik`;

const mockMetadata = {
  model_name: "QLearn_Pers",
  description: "Q-learning with perseveration and unchosen decay",
  type: "B",
  timestamp: "2026-09-30 07:22:15",
  created_at: "2026-09-30 07:18:42",
  last_fit_at: "2026-09-30 07:22:15",
  saved_at: "2026-09-30 07:22:15",
  total_fit_time_seconds: 14.82,
  total_fit_time_formatted: "14.82s",
  iterations: 12,
  n_subjects: 10,
  n_params: 6,
  params: ["alpha", "beta", "decay", "bias", "noise", "pers"],
  param_summary: {
    alpha: { latent_mean: -0.422, latent_sd: 0.312, transformed_mean: 0.396 },
    beta: { latent_mean: 1.115, latent_sd: 0.245, transformed_mean: 3.05 },
    decay: { latent_mean: -1.046, latent_sd: 0.380, transformed_mean: 0.26 },
    bias: { latent_mean: 0.160, latent_sd: 0.280, transformed_mean: 0.16 },
    noise: { latent_mean: -1.734, latent_sd: 0.410, transformed_mean: 0.15 },
    pers: { latent_mean: 0.340, latent_sd: 0.320, transformed_mean: 0.34 }
  },
  multinormal: "full",
  group_diff: {
    alpha: {
      column: "condition",
      groups: ["control", "patient"],
      reference_group: "control",
      group_means: { control: 0.352, patient: 0.440 },
      group_diffs: { "patient - control": 0.088 }
    }
  },
  n_choices: 2,
  final_evidence: -141.30,
  final_bic: 291.90,
  model_code: mockModelCode,
  system_info: {
    python_version: "3.11.8",
    platform: "Linux-6.6.0-x86_64"
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'report' | 'compare' | 'docs'>('compare');
  const [copied, setCopied] = useState(false);

  // Single Model Report States (Multi-Page Diagnostic Report)
  const [activeReportPage, setActiveReportPage] = useState<'fit' | 'params' | 'metadata'>('fit');
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedJson, setCopiedJson] = useState(false);

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
    'Biased_Baseline',
  ]);
  const [comparisonSubTab, setComparisonSubTab] = useState<'fit' | 'matrix' | 'evolution'>('fit');
  const [selectedCodeModel, setSelectedCodeModel] = useState<string>('QLearn_Pers');
  const [isCodeCollapsed, setIsCodeCollapsed] = useState<boolean>(false);
  const [isCodeExpanded, setIsCodeExpanded] = useState<boolean>(false);
  const [isSingleModelCodeExpanded, setIsSingleModelCodeExpanded] = useState<boolean>(false);
  const [copiedCompareCode, setCopiedCompareCode] = useState<boolean>(false);

  // Random baseline benchmark (k=0 free parameters, random BIC = -2 * random_ll)
  // Passed as argument to report function: compare_models(..., random_ll=-250.0) / create_report(..., random_ll=-250.0)
  const randomLL: number | null = -250.0;
  const randomBIC = randomLL !== null ? -2.0 * randomLL : null;

  const handleCopyCompareCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCompareCode(true);
    setTimeout(() => setCopiedCompareCode(false), 2000);
  };

  const installCmd = 'pip install git+https://github.com/BoazRosenberg/Importance-Sampling.git';

  const handleCopy = () => {
    navigator.clipboard.writeText(installCmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(mockModelCode);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(mockMetadata, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handleDownloadMetadata = () => {
    const blob = new Blob([JSON.stringify(mockMetadata, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'QLearn_Pers_metadata.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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



  const allSingleEv = [...mockEvidence];
  if (randomLL !== null) allSingleEv.push(randomLL);
  const evMin = Math.min(...allSingleEv);
  const evMax = Math.max(...allSingleEv);
  const evRange = evMax - evMin || 1;
  const getEvY = (val: number) =>
    fitH - fitPad.bottom - ((val - evMin) / evRange) * (fitH - fitPad.top - fitPad.bottom);

  const evidenceD = mockEvidence
    .map((val, idx) => `${idx === 0 ? 'M' : 'L'} ${getFitX(idx)},${getEvY(val)}`)
    .join(' ');

  // Comparison metrics calculations
  const activeModelList = selectedModels.map((m) => COMPARISON_MODELS[m]).filter(Boolean);
  const bestBicVal = activeModelList.length > 0 ? Math.min(...activeModelList.map((m) => m.finalBIC)) : 0;
  const sortedModels = [...activeModelList].sort((a, b) => a.finalBIC - b.finalBIC);

  // Comparison chart ranges (including Random benchmark reference line when active)
  const allCompEv = activeModelList.flatMap((m) => m.evidenceHistory);
  if (randomLL !== null) allCompEv.push(randomLL);
  const compEvMin = allCompEv.length > 0 ? Math.min(...allCompEv) : -300;
  const compEvMax = allCompEv.length > 0 ? Math.max(...allCompEv) : -100;
  const compEvRange = compEvMax - compEvMin || 1;
  const getCompEvY = (v: number) =>
    fitH - fitPad.bottom - ((v - compEvMin) / compEvRange) * (fitH - fitPad.top - fitPad.bottom);

  const allCompBic = activeModelList.flatMap((m) => m.bicHistory);
  if (randomBIC !== null) allCompBic.push(randomBIC);
  const compBicMin = allCompBic.length > 0 ? Math.min(...allCompBic) : 200;
  const compBicMax = allCompBic.length > 0 ? Math.max(...allCompBic) : 600;
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

              {/* Winner pill and Benchmark pill */}
              <div className="flex items-center gap-3 flex-wrap">
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2.5 rounded-lg flex items-center gap-3">
                  <Award className="w-5 h-5 text-[#1a7f37]" />
                  <div>
                    <div className="text-[10px] uppercase font-bold tracking-wider text-[#59636e]">
                      Best Model (Lowest BIC)
                    </div>
                    <div className="text-sm font-bold font-mono text-[#1a7f37]">
                      {sortedModels[0].name} (BIC {sortedModels[0].finalBIC.toFixed(1)})
                    </div>
                  </div>
                </div>

                {randomLL !== null && randomBIC !== null && (
                  <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2.5 rounded-lg font-mono">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-[#59636e]">
                      Random Baseline (k=0)
                    </div>
                    <div className="text-xs font-bold text-[#59636e] mt-0.5">
                      LL: {randomLL.toFixed(1)} &bull; BIC: {randomBIC.toFixed(1)}
                    </div>
                  </div>
                )}
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

              {/* Sub-tab view toggle (3 Pages for Multi-Model Comparison) */}
              <div className="flex bg-[#f6f8fa] p-1 border border-[#d1d9e0] rounded-lg text-xs font-medium self-start sm:self-auto gap-1">
                <button
                  onClick={() => setComparisonSubTab('fit')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    comparisonSubTab === 'fit'
                      ? 'bg-white text-[#0969da] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    comparisonSubTab === 'fit' ? 'bg-[#0969da] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>
                    1
                  </span>
                  <Activity className="w-3.5 h-3.5" />
                  <span>Fit &amp; BICs</span>
                </button>

                <button
                  onClick={() => setComparisonSubTab('matrix')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    comparisonSubTab === 'matrix'
                      ? 'bg-white text-[#1a7f37] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    comparisonSubTab === 'matrix' ? 'bg-[#1a7f37] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>
                    2
                  </span>
                  <Grid className="w-3.5 h-3.5" />
                  <span>Parameter Matrix &amp; Code</span>
                </button>

                <button
                  onClick={() => setComparisonSubTab('evolution')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    comparisonSubTab === 'evolution'
                      ? 'bg-white text-[#8250df] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    comparisonSubTab === 'evolution' ? 'bg-[#8250df] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>
                    3
                  </span>
                  <TrendingUp className="w-3.5 h-3.5" />
                  <span>Parameter Evolution</span>
                </button>
              </div>
            </div>



            {/* ============================================================== */}
            {/* PAGE 1: FIT & BICS + PARTICIPANT BEST-EXPLAINED BREAKDOWN      */}
            {/* ============================================================== */}
            {comparisonSubTab === 'fit' && (
              <div className="space-y-6">
                <div className="bg-white border-l-4 border-l-[#0969da] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-[#1f2328]">
                      Page 1: Likelihood, BIC Trajectories &amp; Subject-Level Selection
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      Total population evidence trajectories, BIC minimization curves, model ranking table, and % of participants best explained by each model.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#0969da]/10 text-[#0969da]">
                    Page 1 of 3
                  </span>
                </div>



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

                      {/* Random LL Benchmark Reference Line */}
                      {randomLL !== null && (
                        <g>
                          <line
                            x1={fitPad.left}
                            y1={getCompEvY(randomLL)}
                            x2={fitW - fitPad.right}
                            y2={getCompEvY(randomLL)}
                            stroke="#8c959f"
                            strokeWidth="1.8"
                            strokeDasharray="4 3"
                          />
                          <text
                            x={fitW - fitPad.right}
                            y={getCompEvY(randomLL) - 4}
                            textAnchor="end"
                            fontSize="10"
                            fill="#59636e"
                            fontFamily="monospace"
                            fontWeight="600"
                          >
                            Random LL ({randomLL.toFixed(1)})
                          </text>
                        </g>
                      )}

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
                      {randomLL !== null && (
                        <div className="flex items-center gap-1.5 text-[#59636e]">
                          <span className="w-3 h-0.5 border-t border-[#8c959f] border-dashed" />
                          <span>Random:</span>
                          <strong>{randomLL.toFixed(1)}</strong>
                        </div>
                      )}
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

                      {/* Random BIC Benchmark Reference Line */}
                      {randomBIC !== null && (
                        <g>
                          <line
                            x1={fitPad.left}
                            y1={getCompBicY(randomBIC)}
                            x2={fitW - fitPad.right}
                            y2={getCompBicY(randomBIC)}
                            stroke="#8c959f"
                            strokeWidth="1.8"
                            strokeDasharray="4 3"
                          />
                          <text
                            x={fitW - fitPad.right}
                            y={getCompBicY(randomBIC) - 4}
                            textAnchor="end"
                            fontSize="10"
                            fill="#59636e"
                            fontFamily="monospace"
                            fontWeight="600"
                          >
                            Random BIC ({randomBIC.toFixed(1)})
                          </text>
                        </g>
                      )}

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
                      {randomBIC !== null && (
                        <div className="flex items-center gap-1.5 text-[#59636e]">
                          <span className="w-3 h-0.5 border-t border-[#8c959f] border-dashed" />
                          <span>Random:</span>
                          <strong>{randomBIC.toFixed(1)}</strong>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* 2. FINAL COMPARISON RANKING TABLE & PARTICIPANTS BEST EXPLAINED */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Comparative Summary Table with Best-Explained Column */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0]">
                      <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                        <Award className="w-4 h-4 text-[#1a7f37]" />
                        Final Model Ranking &amp; Diagnostic Statistics
                      </h3>
                      <p className="text-xs text-[#59636e]">Sorted by BIC rank (lowest BIC is superior; ΔBIC &gt; 10 is decisive).</p>
                    </div>

                    <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2] font-mono">
                          <tr>
                            <th className="py-2.5 px-3 font-semibold">Rank</th>
                            <th className="py-2.5 px-3 font-semibold">Model</th>
                            <th className="py-2.5 px-3 font-semibold">Description</th>
                            <th className="py-2.5 px-3 font-semibold">k</th>
                            <th className="py-2.5 px-3 font-semibold">Evidence</th>
                            <th className="py-2.5 px-3 font-semibold">BIC</th>
                            <th className="py-2.5 px-3 font-semibold">ΔBIC</th>
                            {randomLL !== null && randomBIC !== null && (
                              <>
                                <th className="py-2.5 px-3 font-semibold text-[#1a7f37]">ΔLL (vs Random)</th>
                                <th className="py-2.5 px-3 font-semibold text-[#1a7f37]">ΔBIC (vs Random)</th>
                              </>
                            )}
                            <th className="py-2.5 px-3 font-semibold">Best-Explained %</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#eaeef2] font-mono">
                          {sortedModels.map((m, idx) => {
                            const deltaBic = m.finalBIC - bestBicVal;
                            const isWinner = idx === 0;
                            const dLLvsRand = randomLL !== null ? m.finalEvidence - randomLL : null;
                            const dBICvsRand = randomBIC !== null ? m.finalBIC - randomBIC : null;

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
                                <td className="py-2.5 px-3 text-[#59636e] font-sans text-xs max-w-[200px] truncate" title={m.description}>
                                  {m.description}
                                </td>
                                <td className="py-2.5 px-3 text-[#59636e]">{m.k}</td>
                                <td className="py-2.5 px-3 text-[#0969da]">{m.finalEvidence.toFixed(1)}</td>
                                <td className="py-2.5 px-3 text-[#1f2328]">{m.finalBIC.toFixed(1)}</td>
                                <td className="py-2.5 px-3">
                                  {isWinner ? (
                                    <span className="text-[#1a7f37] font-bold">0.0 (Best)</span>
                                  ) : (
                                    <span className="text-[#cf222e]">+{deltaBic.toFixed(1)}</span>
                                  )}
                                </td>
                                {dLLvsRand !== null && dBICvsRand !== null && (
                                  <>
                                    <td className={`py-2.5 px-3 font-semibold ${dLLvsRand > 0 ? 'text-[#1a7f37]' : 'text-[#cf222e]'}`}>
                                      {dLLvsRand > 0 ? `+${dLLvsRand.toFixed(1)}` : dLLvsRand.toFixed(1)}
                                    </td>
                                    <td className={`py-2.5 px-3 font-semibold ${dBICvsRand < 0 ? 'text-[#1a7f37]' : 'text-[#cf222e]'}`}>
                                      {dBICvsRand > 0 ? `+${dBICvsRand.toFixed(1)}` : dBICvsRand.toFixed(1)}
                                    </td>
                                  </>
                                )}
                                <td className="py-2.5 px-3 font-bold text-[#8250df]">
                                  {m.bestExplainedPct.toFixed(1)}% <span className="font-normal text-[10px] text-[#59636e]">({m.bestExplainedCount}/10)</span>
                                </td>
                              </tr>
                            );
                          })}

                          {/* Random Baseline Benchmark Row */}
                          {randomLL !== null && randomBIC !== null && (
                            <tr className="bg-[#f6f8fa] text-[#59636e] border-t-2 border-[#d1d9e0]">
                              <td className="py-2.5 px-3">
                                <span className="px-1.5 py-0.5 rounded bg-[#8c959f] text-white text-[10px] font-bold">
                                  Benchmark
                                </span>
                              </td>
                              <td className="py-2.5 px-3 font-bold text-[#59636e] flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-[#8c959f]" />
                                <span>Random Baseline</span>
                              </td>
                              <td className="py-2.5 px-3 text-[#59636e] font-sans text-xs">
                                Chance benchmark (user-input LL, k=0 free parameters)
                              </td>
                              <td className="py-2.5 px-3 text-[#59636e]">0</td>
                              <td className="py-2.5 px-3 text-[#59636e]">{randomLL.toFixed(1)}</td>
                              <td className="py-2.5 px-3 text-[#59636e] font-bold">{randomBIC.toFixed(1)}</td>
                              <td className="py-2.5 px-3 text-[#cf222e]">
                                +{(randomBIC - bestBicVal).toFixed(1)}
                              </td>
                              <td className="py-2.5 px-3 text-[#59636e]">0.0 (Base)</td>
                              <td className="py-2.5 px-3 text-[#59636e]">0.0 (Base)</td>
                              <td className="py-2.5 px-3 text-[#8c959f]">0.0%</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>

                    <div className="text-[11px] text-[#59636e] font-mono pt-1">
                      * ΔBIC &gt; 10 represents very strong evidence against the higher-BIC model (Kass &amp; Raftery, 1995).
                    </div>
                  </div>

                  {/* Percentage of Participants Best Explained Horizontal Breakdown */}
                  <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                    <div className="pb-2 border-b border-[#d1d9e0]">
                      <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-[#8250df]" />
                        Percentage of Participants Best Explained
                      </h3>
                      <p className="text-xs text-[#59636e]">
                        Proportion of individual subjects where each model achieved the highest log-marginal likelihood.
                      </p>
                    </div>

                    <div className="space-y-4 pt-2">
                      {sortedModels.map((m) => (
                        <div key={m.name} className="space-y-1">
                          <div className="flex justify-between items-center text-xs font-mono">
                            <span className="font-bold text-[#1f2328] flex items-center gap-1.5">
                              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                              {m.name}
                            </span>
                            <span className="font-bold text-[#1f2328]">
                              {m.bestExplainedPct.toFixed(1)}% <span className="font-normal text-[#59636e]">({m.bestExplainedCount}/10 subjects)</span>
                            </span>
                          </div>
                          <div className="h-3 w-full bg-[#eaeef2] rounded-full overflow-hidden">
                            <div
                              style={{ width: `${m.bestExplainedPct}%`, backgroundColor: m.color }}
                              className="h-full rounded-full transition-all duration-500"
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="p-3 bg-[#f6f8fa] border border-[#d1d9e0] rounded-lg text-xs text-[#59636e] mt-4 font-mono">
                      <span>Dominant Model: </span>
                      <strong className="text-[#1a7f37]">{sortedModels[0].name}</strong> explains <strong>{sortedModels[0].bestExplainedPct}%</strong> of participants while minimizing overall population BIC.
                    </div>
                  </div>
                </div>

                {/* Page 1 Footer Navigation */}
                <div className="flex justify-between items-center pt-2">
                  <div className="text-xs text-[#59636e]">
                    Viewing Page 1 of 3: Fit &amp; BICs
                  </div>
                  <button
                    onClick={() => setComparisonSubTab('matrix')}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#0969da] hover:bg-[#f6f8fa] text-xs font-semibold text-[#0969da] shadow-2xs transition-all cursor-pointer"
                  >
                    <span>Next Page: Parameter Matrix &amp; Code</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* ============================================================== */}
            {/* PAGE 2: PARAMETER INCLUSION MATRIX & COLLAPSIBLE CODE          */}
            {/* ============================================================== */}
            {comparisonSubTab === 'matrix' && (
              <div className="space-y-6">
                <div className="bg-white border-l-4 border-l-[#1a7f37] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-[#1f2328]">
                      Page 2: Parameter Inclusion Matrix &amp; Model Function Source Codes
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      Overview table of which parameters are included in which models, with collapsible source code inspection for each model.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#1a7f37]/10 text-[#1a7f37]">
                    Page 2 of 3
                  </span>
                </div>

                {/* MODEL DESCRIPTIONS SUMMARY CARDS */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {activeModelList.map((m) => (
                    <div key={m.name} className="p-3.5 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-1.5">
                      <div className="flex items-center justify-between font-mono font-bold text-xs text-[#1f2328]">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                          {m.name}
                        </span>
                        <span className="text-[10px] text-[#59636e] font-normal">k = {m.k}</span>
                      </div>
                      <p className="text-[11px] text-[#59636e] leading-snug">
                        {m.description}
                      </p>
                    </div>
                  ))}
                </div>

                {/* PARAMETER INCLUSION OVERVIEW TABLE (Y-axis = Parameters, X-axis = Models) */}
                <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-4">
                  <div className="pb-2 border-b border-[#d1d9e0]">
                    <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                      <Grid className="w-4 h-4 text-[#0969da]" />
                      Parameter Inclusion Overview Matrix
                    </h3>
                    <p className="text-xs text-[#59636e]">
                      Parameters on Y-axis, Models on X-axis. Empty if excluded, full if included, and full with a center dot if group differences are modeled.
                    </p>
                  </div>

                  <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2] font-mono">
                        <tr>
                          <th className="py-2.5 px-4 font-semibold w-48">Parameter (Y-Axis)</th>
                          {activeModelList.map((m) => (
                            <th key={m.name} className="py-2.5 px-4 font-semibold text-center">
                              <div className="flex flex-col items-center gap-0.5">
                                <span className="flex items-center gap-1.5" style={{ color: m.color }}>
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                                  {m.name}
                                </span>
                                <span className="text-[10px] font-normal text-[#59636e] max-w-[130px] truncate" title={m.description}>
                                  {m.description}
                                </span>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#eaeef2]">
                        {Array.from(new Set(activeModelList.flatMap((m) => m.params))).map((pKey) => {
                          const modelsWithP = activeModelList.filter((m) => m.params.includes(pKey));
                          const isShared = modelsWithP.length > 1;

                          return (
                            <tr key={pKey} className="hover:bg-[#f6f8fa]">
                              <td className="py-3 px-4 font-mono font-bold text-[#1f2328]">
                                <div className="flex items-center gap-2">
                                  <span>{pKey}</span>
                                  {isShared ? (
                                    <span className="bg-[#8250df]/10 text-[#8250df] text-[10px] px-2 py-0.5 rounded font-semibold border border-[#8250df]/20 font-sans">
                                      Shared ({modelsWithP.length})
                                    </span>
                                  ) : (
                                    <span className="bg-[#eaeef2] text-[#59636e] text-[10px] px-2 py-0.5 rounded font-semibold font-sans">
                                      Unique
                                    </span>
                                  )}
                                </div>
                              </td>

                              {activeModelList.map((m) => {
                                const isIncluded = m.params.includes(pKey);
                                const hasGroupDiff = Boolean(m.groupDiffParams?.includes(pKey));

                                return (
                                  <td key={m.name} className="py-2.5 px-4 text-center align-middle">
                                    <div className="flex justify-center items-center">
                                      {isIncluded ? (
                                        hasGroupDiff ? (
                                          <div
                                            className="w-5 h-5 rounded-sm shadow-2xs flex items-center justify-center transition-transform hover:scale-110"
                                            style={{ backgroundColor: m.color }}
                                            title={`${m.name} includes ${pKey} (with group differences)`}
                                          >
                                            <span className="w-1.5 h-1.5 rounded-full bg-white shadow-xs" />
                                          </div>
                                        ) : (
                                          <div
                                            className="w-5 h-5 rounded-sm shadow-2xs transition-transform hover:scale-110"
                                            style={{ backgroundColor: m.color }}
                                            title={`${m.name} includes ${pKey}`}
                                          />
                                        )
                                      ) : (
                                        <div
                                          className="w-5 h-5 rounded-sm border border-dashed border-[#d1d9e0] bg-[#f6f8fa]/60"
                                          title={`${m.name} excludes ${pKey}`}
                                        />
                                      )}
                                    </div>
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                      <tfoot className="bg-[#f6f8fa] border-t-2 border-[#d1d9e0] font-mono">
                        <tr>
                          <td className="py-2.5 px-4 font-bold text-[#1f2328]">Total Parameters (k)</td>
                          {activeModelList.map((m) => (
                            <td key={m.name} className="py-2.5 px-4 text-center font-bold" style={{ color: m.color }}>
                              k = {m.k}
                            </td>
                          ))}
                        </tr>
                      </tfoot>
                    </table>

                    <div className="p-3 bg-[#f6f8fa] border-t border-[#eaeef2] text-[11px] text-[#59636e] font-sans flex items-center gap-6 flex-wrap">
                      <span className="flex items-center gap-1.5">
                        <span className="w-3.5 h-3.5 rounded-sm bg-[#0969da] inline-block shadow-2xs" />
                        <span>Filled square = Parameter is included in model</span>
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-3.5 h-3.5 rounded-sm bg-[#0969da] inline-flex items-center justify-center shadow-2xs">
                          <span className="w-1 h-1 rounded-full bg-white" />
                        </span>
                        <span>Filled square with dot = Group differences for parameter</span>
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-3.5 h-3.5 rounded-sm border border-dashed border-[#d1d9e0] bg-white inline-block" />
                        <span>Dashed outline = Parameter not included</span>
                      </span>
                    </div>
                  </div>
                </div>

                {/* COLLAPSIBLE MODEL FUNCTIONS SOURCE CODE */}
                <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-[#eaeef2]">
                    <div>
                      <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                        <FileCode className="w-4 h-4 text-[#0969da]" />
                        Model Function Source Code Inspection
                      </h3>
                      <p className="text-xs text-[#59636e]">
                        Select a model to view its exact Python implementation with standard IDE syntax coloring. Use &quot;Show All Code&quot; to expand the full window without inner scrolling, or collapse to save space.
                      </p>
                    </div>

                    <div className="flex items-center gap-2 flex-wrap">
                      {!isCodeCollapsed && (
                        <button
                          onClick={() => setIsCodeExpanded(!isCodeExpanded)}
                          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md border transition-all cursor-pointer ${
                            isCodeExpanded
                              ? 'bg-[#0969da]/10 text-[#0969da] border-[#0969da]/40 shadow-2xs'
                              : 'text-[#1f2328] bg-[#f6f8fa] border-[#d1d9e0] hover:bg-[#eaeef2]'
                          }`}
                          title={isCodeExpanded ? 'Switch to compact window with internal scrollbar' : 'Expand window to display entire code without internal scrolling'}
                        >
                          {isCodeExpanded ? (
                            <>
                              <Minimize2 className="w-3.5 h-3.5 text-[#0969da]" />
                              <span>Compact Window</span>
                            </>
                          ) : (
                            <>
                              <Maximize2 className="w-3.5 h-3.5 text-[#0969da]" />
                              <span>Show All Code</span>
                            </>
                          )}
                        </button>
                      )}

                      <button
                        onClick={() => setIsCodeCollapsed(!isCodeCollapsed)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#1f2328] bg-[#f6f8fa] border border-[#d1d9e0] rounded-md hover:bg-[#eaeef2] transition-colors cursor-pointer"
                      >
                        {isCodeCollapsed ? (
                          <>
                            <Eye className="w-3.5 h-3.5 text-[#0969da]" />
                            <span>Show Code</span>
                          </>
                        ) : (
                          <>
                            <EyeOff className="w-3.5 h-3.5 text-[#59636e]" />
                            <span>Hide Code</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {!isCodeCollapsed && (
                    <div className="space-y-3">
                      {/* Model Selector Tabs */}
                      <div className="flex items-center gap-2 flex-wrap border-b border-[#eaeef2] pb-2">
                        {activeModelList.map((m) => {
                          const isCurrent = selectedCodeModel === m.name;
                          return (
                            <button
                              key={m.name}
                              onClick={() => setSelectedCodeModel(m.name)}
                              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer border ${
                                isCurrent
                                  ? 'bg-[#0969da]/10 text-[#0969da] border-[#0969da]/30 font-bold shadow-2xs'
                                  : 'bg-transparent text-[#59636e] border-transparent hover:bg-[#f6f8fa]'
                              }`}
                            >
                              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                              <span>{m.name}</span>
                            </button>
                          );
                        })}
                      </div>

                      {/* Code Display Container */}
                      <div className="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden shadow-inner font-mono">
                        <div className="bg-[#161b22] px-4 py-2 border-b border-[#30363d] flex items-center justify-between text-xs text-[#8b949e] flex-wrap gap-2">
                          <div className="flex items-center gap-2">
                            <div className="flex gap-1.5">
                              <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f56]" />
                              <span className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]" />
                              <span className="w-2.5 h-2.5 rounded-full bg-[#27c93f]" />
                            </div>
                            <span className="text-[#c9d1d9] font-semibold text-xs ml-1">
                              {selectedCodeModel}.py
                            </span>
                            <span className="text-[10px] text-[#7ee787] px-1.5 py-0.5 rounded bg-[#7ee787]/10 border border-[#7ee787]/20 font-sans">
                              IDE Colors
                            </span>
                            {isCodeExpanded && (
                              <span className="text-[10px] text-[#58a6ff] bg-[#58a6ff]/10 border border-[#58a6ff]/20 px-1.5 py-0.5 rounded font-sans font-medium">
                                Full Height (Scroll Page)
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setIsCodeExpanded(!isCodeExpanded)}
                              className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-[#c9d1d9] bg-white/10 hover:bg-white/20 rounded border border-white/15 transition-all cursor-pointer"
                              title={isCodeExpanded ? "Switch to compact window with internal scrollbar" : "Expand window to show all code without scrollbar"}
                            >
                              {isCodeExpanded ? (
                                <>
                                  <Minimize2 className="w-3 h-3 text-[#58a6ff]" />
                                  <span>Compact View</span>
                                </>
                              ) : (
                                <>
                                  <Maximize2 className="w-3 h-3 text-[#58a6ff]" />
                                  <span>Show All Code</span>
                                </>
                              )}
                            </button>

                            <button
                              onClick={() => handleCopyCompareCode(COMPARISON_MODELS[selectedCodeModel]?.code || '')}
                              className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-[#c9d1d9] bg-white/10 hover:bg-white/20 rounded border border-white/15 transition-all cursor-pointer"
                            >
                              {copiedCompareCode ? (
                                <>
                                  <Check className="w-3 h-3 text-[#27c93f]" />
                                  <span className="text-[#27c93f]">Copied!</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="w-3 h-3" />
                                  <span>Copy Python Code</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                        <pre
                          className={`p-4 text-xs font-mono text-[#e6edf3] overflow-x-auto leading-relaxed transition-all ${
                            isCodeExpanded ? 'max-h-none overflow-y-visible' : 'max-h-[340px] overflow-y-auto'
                          }`}
                        >
                          <code
                            dangerouslySetInnerHTML={{
                              __html: highlightPythonCode(COMPARISON_MODELS[selectedCodeModel]?.code || '')
                            }}
                          />
                        </pre>
                      </div>
                    </div>
                  )}
                </div>

                {/* Page 2 Footer Navigation */}
                <div className="flex justify-between items-center pt-2">
                  <button
                    onClick={() => setComparisonSubTab('fit')}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#0969da] hover:bg-[#f6f8fa] text-xs font-semibold text-[#59636e] shadow-2xs transition-all cursor-pointer"
                  >
                    <ChevronLeft className="w-4 h-4" />
                    <span>Previous: Fit &amp; BICs</span>
                  </button>
                  <button
                    onClick={() => setComparisonSubTab('evolution')}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#8250df] hover:bg-[#f6f8fa] text-xs font-semibold text-[#8250df] shadow-2xs transition-all cursor-pointer"
                  >
                    <span>Next Page: Parameter Evolution Comparison</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* ============================================================== */}
            {/* PAGE 3: PARAMETER EVOLUTION COMPARISON (SHARED ON SAME PLOTS)  */}
            {/* ============================================================== */}
            {comparisonSubTab === 'evolution' && (
              <div className="space-y-6">
                <div className="bg-white border-l-4 border-l-[#8250df] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-[#1f2328]">
                      Page 3: Parameter Evolution Comparison Across Iterations
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      All parameters compared across iterations. Shared parameters are plotted together on the same subplots for direct visual comparison; unique parameters are shown for their respective models.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#8250df]/10 text-[#8250df]">
                    Page 3 of 3
                  </span>
                </div>

                {/* PARAMETER EVOLUTION COMPARISON GRID */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Array.from(new Set(activeModelList.flatMap((m) => m.params))).map((pKey) => {
                    const modelsWithP = activeModelList.filter((m) => m.params.includes(pKey));
                    const isShared = modelsWithP.length > 1;

                    // Compute y-axis scale across all models containing this parameter
                    const allVals = modelsWithP.flatMap((m) => m.paramEvolutions?.[pKey] || []);
                    const pMin = allVals.length ? Math.min(...allVals) : 0;
                    const pMax = allVals.length ? Math.max(...allVals) : 1;
                    const pRange = pMax - pMin || 1;
                    const padMin = pMin - 0.1 * pRange;
                    const padMax = pMax + 0.1 * pRange;
                    const padRange = padMax - padMin || 1;

                    const getP_X = (it: number) => subPad.left + (it / 11) * (subW - subPad.left - subPad.right);
                    const getP_Y = (val: number) =>
                      subH - subPad.bottom - ((val - padMin) / padRange) * (subH - subPad.top - subPad.bottom);

                    return (
                      <div key={pKey} className="border border-[#eaeef2] rounded-xl p-4 bg-white shadow-2xs space-y-2">
                        <div className="flex items-center justify-between text-xs font-mono pb-1 border-b border-[#eaeef2]">
                          <span className="font-bold text-[#1f2328] text-sm">{pKey}</span>
                          {isShared ? (
                            <span className="bg-[#8250df]/10 text-[#8250df] text-[10px] px-2 py-0.5 rounded font-semibold border border-[#8250df]/20 font-sans">
                              Shared ({modelsWithP.length} models)
                            </span>
                          ) : (
                            <span className="bg-[#eaeef2] text-[#59636e] text-[10px] px-2 py-0.5 rounded font-semibold font-sans">
                              Specific to {modelsWithP[0]?.name}
                            </span>
                          )}
                        </div>

                        <svg viewBox={`0 0 ${subW} ${subH}`} className="w-full h-44 select-none">
                          {[0, 0.5, 1.0].map((frac, idx) => {
                            const y = subPad.top + frac * (subH - subPad.top - subPad.bottom);
                            const val = padMax - frac * padRange;
                            return (
                              <g key={idx}>
                                <line x1={subPad.left} y1={y} x2={subW - subPad.right} y2={y} stroke="#eaeef2" />
                                <text x={subPad.left - 6} y={y + 3} textAnchor="end" fontSize="9" fill="#8c959f" fontFamily="monospace">
                                  {val.toFixed(2)}
                                </text>
                              </g>
                            );
                          })}

                          {/* Line for each model containing this parameter */}
                          {modelsWithP.map((m) => {
                            const evo = m.paramEvolutions?.[pKey] || [];
                            const lineD = evo
                              .map((val, idx) => `${idx === 0 ? 'M' : 'L'} ${getP_X(idx)},${getP_Y(val)}`)
                              .join(' ');

                            return (
                              <g key={m.name}>
                                <path d={lineD} fill="none" stroke={m.color} strokeWidth="2.2" />
                                {evo.map((val, idx) => (
                                  <circle
                                    key={idx}
                                    cx={getP_X(idx)}
                                    cy={getP_Y(val)}
                                    r="2.5"
                                    fill={m.color}
                                    stroke="#fff"
                                    strokeWidth="1"
                                  >
                                    <title>{`${m.name} | Iter ${idx}: ${val.toFixed(3)}`}</title>
                                  </circle>
                                ))}
                              </g>
                            );
                          })}
                        </svg>

                        {/* Subplot Legend */}
                        <div className="flex items-center gap-3 flex-wrap pt-1 border-t border-[#eaeef2] text-[11px] font-mono">
                          {modelsWithP.map((m) => {
                            const finalVal = m.paramEvolutions?.[pKey]?.[11];
                            return (
                              <div key={m.name} className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                                <span className="text-[#59636e]">{m.name}:</span>
                                <strong className="text-[#1f2328]">{finalVal !== undefined ? finalVal.toFixed(3) : '—'}</strong>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Page 3 Footer Navigation */}
                <div className="flex justify-between items-center pt-2">
                  <button
                    onClick={() => setComparisonSubTab('matrix')}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#1a7f37] hover:bg-[#f6f8fa] text-xs font-semibold text-[#59636e] shadow-2xs transition-all cursor-pointer"
                  >
                    <ChevronLeft className="w-4 h-4" />
                    <span>Previous: Parameter Matrix &amp; Code</span>
                  </button>
                  <div className="text-xs text-[#59636e]">
                    Viewing Page 3 of 3: Parameter Evolution Comparison
                  </div>
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
                  {mockMetadata.model_name}
                </h1>
                <p className="text-sm font-semibold text-[#0969da] mt-0.5">
                  {mockMetadata.description}
                </p>
                <p className="text-xs text-[#59636e] mt-1">
                  {mockMetadata.n_subjects} Subjects • {mockMetadata.n_params} Parameters • Converged in {mockMetadata.iterations} Iterations ({mockMetadata.total_fit_time_formatted}) • Run: {mockMetadata.last_fit_at}
                </p>
              </div>

              <div className="flex items-center gap-3 font-mono">
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">Evidence</div>
                  <div className="text-lg font-bold text-[#0969da]">-141.30</div>
                  {randomLL !== null && (
                    <div className="text-[10px] text-[#1a7f37] font-bold">
                      +{ (-141.30 - randomLL).toFixed(1) } vs Rand
                    </div>
                  )}
                </div>
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">BIC</div>
                  <div className="text-lg font-bold text-[#1a7f37]">291.90</div>
                  {randomBIC !== null && (
                    <div className="text-[10px] text-[#1a7f37] font-bold">
                      { (291.90 - randomBIC).toFixed(1) } vs Rand
                    </div>
                  )}
                </div>
              </div>
            </div>



            {/* Sub-tab navigation bar for Single Model Report */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-3 shadow-2xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex bg-[#f6f8fa] p-1 border border-[#d1d9e0] rounded-lg text-xs font-medium gap-1">
                <button
                  onClick={() => setActiveReportPage('fit')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    activeReportPage === 'fit'
                      ? 'bg-white text-[#0969da] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    activeReportPage === 'fit' ? 'bg-[#0969da] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>1</span>
                  <Activity className="w-3.5 h-3.5" />
                  <span>General Fit</span>
                </button>
                <button
                  onClick={() => setActiveReportPage('params')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    activeReportPage === 'params'
                      ? 'bg-white text-[#1a7f37] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    activeReportPage === 'params' ? 'bg-[#1a7f37] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>2</span>
                  <Grid className="w-3.5 h-3.5" />
                  <span>Parameters</span>
                </button>
                <button
                  onClick={() => setActiveReportPage('metadata')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all ${
                    activeReportPage === 'metadata'
                      ? 'bg-white text-[#8250df] font-semibold shadow-2xs'
                      : 'text-[#59636e] hover:text-[#1f2328]'
                  }`}
                >
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    activeReportPage === 'metadata' ? 'bg-[#8250df] text-white' : 'bg-[#eaeef2] text-[#59636e]'
                  }`}>3</span>
                  <FileText className="w-3.5 h-3.5" />
                  <span>Model &amp; Metadata</span>
                </button>
              </div>

              <span className="text-xs text-[#59636e] font-mono hidden sm:inline">
                {activeReportPage === 'fit' && 'Page 1 of 3: Evidence & Convergence'}
                {activeReportPage === 'params' && 'Page 2 of 3: Estimated Parameters & Heatmap'}
                {activeReportPage === 'metadata' && 'Page 3 of 3: Code & JSON Metadata'}
              </span>
            </div>

            {/* PAGE 1: GENERAL FIT */}
            {activeReportPage === 'fit' && (
              <div className="space-y-6">
                <div className="bg-white border-l-4 border-l-[#0969da] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-[#1f2328]">
                      Page 1: General Model Fit &amp; Convergence Diagnostics
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      Total population evidence trajectory, BIC minimization path, and subject-level convergence stability.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#0969da]/10 text-[#0969da]">
                    Page 1 of 3
                  </span>
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

                      {/* Random LL Benchmark Reference Line */}
                      {randomLL !== null && (
                        <g>
                          <line
                            x1={fitPad.left}
                            y1={getEvY(randomLL)}
                            x2={fitW - fitPad.right}
                            y2={getEvY(randomLL)}
                            stroke="#8c959f"
                            strokeWidth="1.8"
                            strokeDasharray="4 3"
                          />
                          <text
                            x={fitW - fitPad.right}
                            y={getEvY(randomLL) - 4}
                            textAnchor="end"
                            fontSize="10"
                            fill="#59636e"
                            fontFamily="monospace"
                            fontWeight="600"
                          >
                            Random LL ({randomLL.toFixed(1)})
                          </text>
                        </g>
                      )}

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

                {/* Key Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-1">
                    <div className="text-[10px] uppercase font-bold text-[#59636e]">Total Evidence</div>
                    <div className="text-xl font-bold font-mono text-[#0969da]">-141.30</div>
                    <div className="text-[11px] text-[#59636e]">
                      {randomLL !== null ? (
                        <span className="text-[#1a7f37] font-semibold">
                          +{ (-141.30 - randomLL).toFixed(2) } vs Random
                        </span>
                      ) : (
                        'Log-marginal likelihood'
                      )}
                    </div>
                  </div>
                  <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-1">
                    <div className="text-[10px] uppercase font-bold text-[#59636e]">Final BIC</div>
                    <div className="text-xl font-bold font-mono text-[#1a7f37]">291.90</div>
                    <div className="text-[11px] text-[#59636e]">
                      {randomBIC !== null ? (
                        <span className="text-[#1a7f37] font-semibold">
                          { (291.90 - randomBIC).toFixed(2) } vs Random
                        </span>
                      ) : (
                        'Penalized criteria'
                      )}
                    </div>
                  </div>
                  <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-1">
                    <div className="text-[10px] uppercase font-bold text-[#59636e]">Iterations</div>
                    <div className="text-xl font-bold font-mono text-[#1f2328]">12 / 12</div>
                    <div className="text-[11px] text-[#1a7f37] font-semibold">Converged (Δev &lt; 0.5)</div>
                  </div>
                  <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-1">
                    <div className="text-[10px] uppercase font-bold text-[#59636e]">Duration</div>
                    <div className="text-xl font-bold font-mono text-[#8250df]">14.82s</div>
                    <div className="text-[11px] text-[#59636e]">1,000 particles/subj</div>
                  </div>
                </div>

                {/* Page 1 Footer */}
                <div className="flex justify-between items-center pt-2">
                  <div className="text-xs text-[#59636e]">Viewing Page 1 of 3: General Fit</div>
                  <button
                    onClick={() => setActiveReportPage('params')}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#1a7f37] hover:bg-[#f6f8fa] text-xs font-semibold text-[#1a7f37] shadow-2xs transition-all cursor-pointer"
                  >
                    <span>Next Page: Parameters</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* PAGE 2: PARAMETERS */}
            {activeReportPage === 'params' && (
              <div className="space-y-6">
                <div className="bg-white border-l-4 border-l-[#1a7f37] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-[#1f2328]">
                      Page 2: Model Hyperparameters, Evolution &amp; Correlation Heatmap
                    </h2>
                    <p className="text-xs text-[#59636e]">
                      Fitted population distributions, evolution trajectories across iterations, subject posterior draws, and latent bivariate correlation matrix.
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#1a7f37]/10 text-[#1a7f37]">
                    Page 2 of 3
                  </span>
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
                      const isGroupDiffParam = key === 'alpha';

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
                          <td className="py-2.5 px-4 font-mono font-bold text-[#1f2328]">
                            <div className="flex items-center gap-2">
                              <span>{key}</span>
                              {isGroupDiffParam && (
                                <span className="bg-[#8250df]/10 text-[#8250df] text-[10px] px-2 py-0.5 rounded font-semibold border border-[#8250df]/20 font-sans">
                                  Group Diff: condition
                                </span>
                              )}
                            </div>
                            {isGroupDiffParam && (
                              <div className="text-[11px] font-sans text-[#59636e] font-normal mt-1">
                                Control: <strong>0.352</strong> • Patient: <strong>0.440</strong> • <span className="text-[#8250df] font-semibold">&Delta; = +0.088</span>
                              </div>
                            )}
                          </td>
                          <td className="py-2.5 px-4 font-mono text-[#0969da] align-top">{finalMean.toFixed(3)}</td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e] align-top">{finalSD.toFixed(3)}</td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e] align-top">
                            [{traj.sdLower[11].toFixed(2)}, {traj.sdUpper[11].toFixed(2)}]
                          </td>
                          <td className="py-2.5 px-4 align-top">
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

            {/* DEDICATED GROUP DIFFERENCES CARD */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0] flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                    <SlidersHorizontal className="w-5 h-5 text-[#8250df]" />
                    Group Differences Estimation (IIS)
                    <span className="bg-[#8250df]/10 text-[#8250df] text-xs px-2 py-0.5 rounded-full font-semibold border border-[#8250df]/20">
                      1 Parameter Active
                    </span>
                  </h2>
                  <p className="text-xs text-[#59636e]">
                    Configured via <code>group_diff=&#123;&quot;alpha&quot;: &quot;condition&quot;&#125;</code> (fits grand mean, pooled SD, and &Delta;group difference; 2 + N-1 hyperparameters).
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2]">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Parameter</th>
                      <th className="py-2.5 px-4 font-semibold">Column</th>
                      <th className="py-2.5 px-4 font-semibold">Control Mean</th>
                      <th className="py-2.5 px-4 font-semibold">Patient Mean</th>
                      <th className="py-2.5 px-4 font-semibold">Estimated Difference (&Delta;)</th>
                      <th className="py-2.5 px-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#eaeef2]">
                    <tr className="hover:bg-[#f6f8fa]">
                      <td className="py-3 px-4 font-bold text-[#1f2328]">alpha</td>
                      <td className="py-3 px-4 text-[#59636e]">condition</td>
                      <td className="py-3 px-4 font-bold text-[#0969da]">0.352</td>
                      <td className="py-3 px-4 font-bold text-[#8250df]">0.440</td>
                      <td className="py-3 px-4 font-bold text-[#1a7f37]">
                        +0.088 <span className="text-[11px] text-[#59636e] font-normal">(latent: +0.224)</span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-[#1a7f37]/10 text-[#1a7f37] text-[10px] font-bold border border-[#1a7f37]/20">
                          Significant Shift
                        </span>
                      </td>
                    </tr>
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

                  const hasGroupDiff = pKey === 'alpha' && Boolean(traj.groupA && traj.groupB);
                  const groupAD = traj.groupA
                    ? traj.iters
                        .map((it, idx) => `${idx === 0 ? 'M' : 'L'} ${getP_X(it)},${getP_Y(traj.groupA![idx])}`)
                        .join(' ')
                    : null;
                  const groupBD = traj.groupB
                    ? traj.iters
                        .map((it, idx) => `${idx === 0 ? 'M' : 'L'} ${getP_X(it)},${getP_Y(traj.groupB![idx])}`)
                        .join(' ')
                    : null;

                  const isHovered = hoveredIter?.param === pKey;

                  return (
                    <div key={pKey} className="border border-[#eaeef2] rounded-lg p-3 bg-white flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                          <span className="font-bold text-[#1f2328]">{pKey}</span>
                          <span className="text-[#59636e] font-semibold">mean={traj.means[11].toFixed(3)}</span>
                        </div>

                        <svg viewBox={`0 0 ${subW} ${subH}`} className="w-full h-38 select-none" onMouseLeave={() => setHoveredIter(null)}>
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
                          {/* Unified ribbon fill for all parameters */}
                          <path d={pRibbonD} fill="rgba(31, 35, 40, 0.10)" />

                          {/* Group A line (if group difference enabled) */}
                          {groupAD && (
                            <path d={groupAD} fill="none" stroke="#0969da" strokeWidth="2" strokeDasharray="3 2" />
                          )}

                          {/* Group B line (if group difference enabled) */}
                          {groupBD && (
                            <path d={groupBD} fill="none" stroke="#cf222e" strokeWidth="2" strokeDasharray="3 2" />
                          )}

                          {/* Grand Mean line (uses unified single color #1f2328 for all parameters) */}
                          <path d={pMeanD} fill="none" stroke="#1f2328" strokeWidth="2.4" />

                          {traj.iters.map((it, idx) => (
                            <circle
                              key={it}
                              cx={getP_X(it)}
                              cy={getP_Y(traj.means[idx])}
                              r={isHovered && hoveredIter?.iter === it ? 4.5 : 2.5}
                              fill="#1f2328"
                              stroke="#fff"
                              strokeWidth="1"
                              className="cursor-pointer"
                              onMouseEnter={() => setHoveredIter({ param: pKey, iter: it })}
                            />
                          ))}
                        </svg>
                      </div>

                      {/* Legend for parameter evolution */}
                      {hasGroupDiff ? (
                        <div className="flex items-center gap-3 text-[10px] font-mono mt-2 pt-1.5 border-t border-[#eaeef2] flex-wrap">
                          <span className="flex items-center gap-1 text-[#1f2328] font-semibold">
                            <span className="w-2.5 h-0.5 bg-[#1f2328]" /> Grand Mean
                          </span>
                          <span className="flex items-center gap-1 text-[#0969da] font-semibold">
                            <span className="w-2.5 h-0.5 bg-[#0969da]" /> Group A
                          </span>
                          <span className="flex items-center gap-1 text-[#cf222e] font-semibold">
                            <span className="w-2.5 h-0.5 bg-[#cf222e]" /> Group B
                          </span>
                        </div>
                      ) : (
                        <div className="flex items-center justify-between text-[10px] font-mono mt-2 pt-1.5 border-t border-[#eaeef2] text-[#8c959f]">
                          <span className="flex items-center gap-1 text-[#1f2328]">
                            <span className="w-2.5 h-0.5 bg-[#1f2328]" /> Population Mean
                          </span>
                          <span>±1 SD ribbon</span>
                        </div>
                      )}
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

            {/* Page 2 Footer Navigation */}
            <div className="flex justify-between items-center pt-2">
              <button
                onClick={() => setActiveReportPage('fit')}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#0969da] hover:bg-[#f6f8fa] text-xs font-semibold text-[#59636e] shadow-2xs transition-all cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous: General Fit</span>
              </button>
              <button
                onClick={() => setActiveReportPage('metadata')}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#8250df] hover:bg-[#f6f8fa] text-xs font-semibold text-[#8250df] shadow-2xs transition-all cursor-pointer"
              >
                <span>Next Page: Model &amp; Metadata</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* PAGE 3: MODEL CODE & COMPANION METADATA */}
        {activeReportPage === 'metadata' && (
          <div className="space-y-6">
            <div className="bg-white border-l-4 border-l-[#8250df] border border-[#d1d9e0] rounded-xl p-4 shadow-2xs flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-[#1f2328]">
                  Page 3: Model Architecture, Source Code &amp; Run Metadata
                </h2>
                <p className="text-xs text-[#59636e]">
                  Inspect the exact Python function fitted by IIS, execution duration, and full serialized metadata JSON.
                </p>
              </div>
              <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-[#8250df]/10 text-[#8250df]">
                Page 3 of 3
              </span>
            </div>

            {/* Model Description Card */}
            <div className="bg-white border border-[#d1d9e0] border-l-4 border-l-[#0969da] rounded-xl p-4 shadow-2xs space-y-1">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#0969da]">
                Model Description
              </div>
              <div className="text-sm font-semibold text-[#1f2328]">
                {mockMetadata.description}
              </div>
            </div>

            {/* Metadata 4-Grid Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-2">
                <div className="text-[11px] font-bold uppercase text-[#59636e]">🕒 Timestamps</div>
                <div className="text-xs font-mono space-y-1 text-[#1f2328]">
                  <div><span className="text-[#8c959f]">Run:</span> {mockMetadata.last_fit_at}</div>
                  <div><span className="text-[#8c959f]">Created:</span> {mockMetadata.created_at}</div>
                  <div><span className="text-[#8c959f]">Saved:</span> {mockMetadata.saved_at}</div>
                </div>
              </div>

              <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-2">
                <div className="text-[11px] font-bold uppercase text-[#59636e]">⏱️ Execution Time</div>
                <div className="text-xs font-mono space-y-1 text-[#1f2328]">
                  <div><span className="text-[#8c959f]">Duration:</span> <strong className="text-[#0969da]">{mockMetadata.total_fit_time_formatted}</strong></div>
                  <div><span className="text-[#8c959f]">Exact:</span> {mockMetadata.total_fit_time_seconds}s</div>
                  <div><span className="text-[#8c959f]">Iterations:</span> <strong>{mockMetadata.iterations}</strong> converged</div>
                </div>
              </div>

              <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-2">
                <div className="text-[11px] font-bold uppercase text-[#59636e]">👥 Model Dimensions</div>
                <div className="text-xs font-mono space-y-1 text-[#1f2328]">
                  <div><span className="text-[#8c959f]">Subjects:</span> <strong>{mockMetadata.n_subjects}</strong></div>
                  <div><span className="text-[#8c959f]">Parameters:</span> <strong>{mockMetadata.n_params}</strong></div>
                  <div><span className="text-[#8c959f]">Choices:</span> {mockMetadata.n_choices}</div>
                </div>
              </div>

              <div className="p-4 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs space-y-2">
                <div className="text-[11px] font-bold uppercase text-[#59636e]">⚙️ System &amp; Method</div>
                <div className="text-xs font-mono space-y-1 text-[#1f2328]">
                  <div><span className="text-[#8c959f]">Python:</span> {mockMetadata.system_info.python_version}</div>
                  <div><span className="text-[#8c959f]">Platform:</span> {mockMetadata.system_info.platform}</div>
                  <div><span className="text-[#8c959f]">Covariance:</span> {mockMetadata.multinormal}</div>
                </div>
              </div>
            </div>

            {/* Python Model Source Code Viewer */}
            <div className="bg-[#0d1117] rounded-xl border border-[#30363d] overflow-hidden shadow-2xs">
              <div className="flex items-center justify-between px-4 py-2.5 bg-[#161b22] border-b border-[#30363d] flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-[#ff5f56]" />
                    <div className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
                    <div className="w-3 h-3 rounded-full bg-[#27c93f]" />
                  </div>
                  <span className="text-[#c9d1d9] font-mono text-xs font-semibold ml-2">
                    {mockMetadata.model_name}.py (Fitted Model Function)
                  </span>
                  <span className="text-[10px] text-[#7ee787] px-1.5 py-0.5 rounded bg-[#7ee787]/10 border border-[#7ee787]/20 font-sans">
                    IDE Colors
                  </span>
                  {isSingleModelCodeExpanded && (
                    <span className="text-[10px] text-[#58a6ff] bg-[#58a6ff]/10 border border-[#58a6ff]/20 px-1.5 py-0.5 rounded font-sans font-medium">
                      Full Height (Scroll Page)
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsSingleModelCodeExpanded(!isSingleModelCodeExpanded)}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs text-[#c9d1d9] bg-white/10 hover:bg-white/20 rounded border border-white/15 transition-all cursor-pointer"
                    title={isSingleModelCodeExpanded ? "Switch to compact window with scrollbar" : "Expand window to show all code without scrollbar"}
                  >
                    {isSingleModelCodeExpanded ? (
                      <>
                        <Minimize2 className="w-3.5 h-3.5 text-[#58a6ff]" />
                        <span>Compact View</span>
                      </>
                    ) : (
                      <>
                        <Maximize2 className="w-3.5 h-3.5 text-[#58a6ff]" />
                        <span>Show All Code</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleCopyCode}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs text-[#c9d1d9] bg-white/10 hover:bg-white/20 rounded border border-white/15 transition-all cursor-pointer"
                  >
                    {copiedCode ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-[#27c93f]" />
                        <span className="text-[#27c93f]">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy Code</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
              <pre
                className={`p-4 text-xs font-mono text-[#e6edf3] overflow-x-auto leading-relaxed transition-all ${
                  isSingleModelCodeExpanded ? 'max-h-none overflow-y-visible' : 'max-h-[380px] overflow-y-auto'
                }`}
              >
                <code
                  dangerouslySetInnerHTML={{
                    __html: highlightPythonCode(mockMetadata.model_code)
                  }}
                />
              </pre>
            </div>

            {/* Compact Companion Metadata Download Action */}
            <div className="flex items-center justify-between p-3.5 bg-white border border-[#d1d9e0] rounded-xl shadow-2xs">
              <div className="flex items-center gap-2.5">
                <FileText className="w-4 h-4 text-[#8250df]" />
                <div>
                  <div className="text-xs font-bold text-[#1f2328]">Companion Run Metadata File</div>
                  <div className="text-[11px] text-[#59636e]">All hyperparameters, diagnostics, and system specs shown above are serialized in <code>{mockMetadata.model_name}_metadata.json</code>.</div>
                </div>
              </div>
              <button
                onClick={handleDownloadMetadata}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-[#0969da] bg-[#f6f8fa] hover:bg-[#eaeef2] border border-[#d1d9e0] rounded-md transition-all cursor-pointer shadow-2xs"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Metadata JSON</span>
              </button>
            </div>

            {/* Page 3 Footer Navigation */}
            <div className="flex justify-between items-center pt-2">
              <button
                onClick={() => setActiveReportPage('params')}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white border border-[#d1d9e0] hover:border-[#1a7f37] hover:bg-[#f6f8fa] text-xs font-semibold text-[#59636e] shadow-2xs transition-all cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous: Parameters &amp; Heatmap</span>
              </button>
              <div className="text-xs text-[#59636e]">
                Viewing Page 3 of 3: Model &amp; Metadata
              </div>
            </div>
          </div>
        )}
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
                  Group Differences Estimation (<code>group_diff</code>)
                </h2>
                <p className="mb-2">
                  Allow specific parameters to have different population means across groups while sharing within-group variance:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`sampler = Sampler(
    data=data,
    model=q_learning_model,
    hyper_params=hyper_priors,
    group_diff={"alpha": "condition"},  # Maps parameter to column in subject data
)
sampler.iterative_model_fit(n_iterations=15)`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li>Estimates grand mean, pooled SD, and &Delta; shift for each non-reference group (2 + N - 1 hyperparameters).</li>
                  <li>Accurately penalizes model degrees of freedom in BIC calculations (+N - 1 per parameter).</li>
                  <li>Automatically highlighted in <code>create_report()</code> with group difference badges and dedicated comparison cards.</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Comparing to Random Baseline (<code>random_ll</code>)
                </h2>
                <p className="mb-2">
                  Pass the chance baseline log-likelihood directly as an argument to <code>sampler.create_report(random_ll=...)</code> or <code>compare_models(random_ll=...)</code>. Because experimental choice structures vary across paradigms (binary, multi-alternative, or continuous), <code>random_ll</code> is supplied by the user as a function argument rather than assumed binary:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`# 1. Compare in a single model report:
# Random baseline has k=0 free parameters, so Random BIC = -2 * random_ll
sampler.create_report(
    filename="report_with_random.html",
    random_ll=-250.0  # Pass chance log-likelihood as function argument
)

# 2. Compare across multiple models:
compare_models(
    [sampler_standard, sampler_dual_lr, sampler_perseveration],
    filename="comparison_with_random.html",
    random_ll=-250.0  # Appears in ranking table, delta columns, and benchmark lines
)`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li><strong>Zero Free Parameters (k=0)</strong>: Random baseline BIC has no complexity penalty: <code>BIC = -2 &times; LL</code>.</li>
                  <li><strong>Benchmark Trajectories</strong>: Dashed reference lines plotted on both evidence and BIC evolution curves.</li>
                  <li><strong>Difference Metrics</strong>: Explicitly displays <code>&Delta;LL = LL<sub>model</sub> - LL<sub>random</sub></code> and <code>&Delta;BIC = BIC<sub>model</sub> - BIC<sub>random</sub></code> across report cards and ranking tables.</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Multi-Core Parallel Execution (<code>n_jobs</code>)
                </h2>
                <p className="mb-2">
                  Accelerate Iterative Importance Sampling across subjects by utilizing multiple CPU cores:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`# Use all available CPU cores:
sampler = Sampler(
    data=data,
    model=q_learning_model,
    hyper_params=hyper_priors,
    n_jobs=-1  # -1 uses all CPU cores, or specify n_jobs=4, 8, etc.
)
sampler.iterative_model_fit(n_iterations=15)

# Or override directly in iterative_model_fit:
sampler.iterative_model_fit(n_iterations=15, n_jobs=-1)`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li><strong>Monotonic Progress Tracking</strong>: Progress bar tracks strictly finished subject count and percentage, eliminating jumping back and forth across cores.</li>
                  <li><strong>Independent Seeded RNGs</strong>: Each subject worker receives an independent RNG stream for exact reproducibility.</li>
                  <li><strong>Clean Progress Format</strong>: Displays completion percentage, finished subjects, and elapsed/remaining durations in brackets without redundant "took" text.</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Deep Simulation (<code>sampler.deep_simulate</code>)
                </h2>
                <p className="mb-2">
                  Enable deep simulations by passing subject-specific functions down to individual subject models. In deep simulation, each subject model is invoked with <strong><code>mode="deep_simulate"</code></strong> (unlike standard simulation which uses <code>mode="simulate"</code>) and receives <code>f</code> as an argument:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`# Model definition branching on mode:
def my_model(subj_data, parameters, mode="log_likelihood", f=None):
    if mode == "deep_simulate":
        # Deep simulation: evaluate subject-specific function f
        return f(subj_data, parameters)
    elif mode == "simulate":
        # Standard simulation
        return simulated_choices
    return log_likelihood

# 1. Single function passed to all subjects:
sim_dat = sampler.deep_simulate(functions=policy_evaluation_fn)

# 2. Subject-specific functions list (must match number of subjects):
subject_fns = [policy_fn_0, policy_fn_1, policy_fn_2]
sim_dat = sampler.deep_simulate(functions=subject_fns)

# 3. Direct export to combined DataFrame:
df = sampler.deep_simulate(functions=subject_fns, to_df=True)`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li><strong>Subject Progress Bar</strong>: Both <code>simulate</code> and <code>deep_simulate</code> include real-time progress bars tracking subject completion (enabled by default via <code>progress_bar=True</code>).</li>
                  <li><strong>Dedicated Model Mode</strong>: Invokes models with <code>mode="deep_simulate"</code>, allowing distinct branching from standard <code>mode="simulate"</code>.</li>
                  <li><strong>Single Callable or List Support</strong>: Broadcasts a single callable across all subjects, or maps <code>functions[i] &rarr; f</code> for each subject.</li>
                  <li><strong>Strict Validation</strong>: Validates list length against <code>n_subjects</code> and checks callability, raising informative <code>ValueError</code> or <code>TypeError</code>.</li>
                  <li><strong>Identical Return Structure</strong>: Returns a <code>SimulatedDataList</code> or reconstructed <code>DataFrame</code> supporting all standard simulation arguments (<code>resample</code>, <code>n_simulations</code>, <code>to_df</code>).</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Export Subject Summary (<code>sampler.export_subject_summary</code>)
                </h2>
                <p className="mb-2">
                  Exports a CSV table of per-subject metrics with columns: <code>subject</code>, <code>parameter</code>, <code>mean</code>, <code>ci_high</code>, <code>ci_low</code>.
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs overflow-x-auto">
{`# Export subject summary table:
summary_df = sampler.export_subject_summary("subject_summary.csv")
print(summary_df.head())
#   subject      parameter     mean  ci_high   ci_low
# 0       0  loglikelihood -13.5866 -10.2104 -18.4521
# 1       0             lr   0.3345   0.4521   0.2180
# 2       0       inv_temp   3.1582   4.0215   2.3104`}
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-3">
                  <li><strong>Log-Likelihood Mean</strong>: Computed in log scale via <code>logsumexp(LL) - log(N)</code> (averaging likelihoods before taking log).</li>
                  <li><strong>Pre-Transformation CI</strong>: Parameter means and 95% CI bounds are calculated in raw particle space before being transformed.</li>
                </ul>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Single Model Report (<code>sampler.create_report</code>)
                </h2>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs">
                  sampler.create_report(filename="single_model_report.html", random_ll=-250.0)
                </pre>
              </div>
            </article>
          </div>
        )}
      </main>
    </div>
  );
}
