import React, { useState, useEffect, useRef } from 'react';
import {
  SimulationState,
  SubjectData,
  createInitialSimulationState,
  generateSyntheticSubjects,
  stepIIS,
  sigmoid,
  softplus,
} from '../sim/iisSimulation';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  TrendingUp,
  Sliders,
  CheckCircle2,
  Users,
  Activity,
  Layers,
  Sparkles,
} from 'lucide-react';

export const SimulationWorkbench: React.FC = () => {
  const [numSubjects, setNumSubjects] = useState<number>(12);
  const [numTrials, setNumTrials] = useState<number>(60);
  const [samplesPerSubject, setSamplesPerSubject] = useState<number>(400);

  const [subjects, setSubjects] = useState<SubjectData[]>(() =>
    generateSyntheticSubjects(12, 60, 0.35, 0.08, 3.5, 0.7)
  );
  const [simState, setSimState] = useState<SimulationState>(createInitialSimulationState);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  // Auto-runner interval
  const timerRef = useRef<number | null>(null);

  const handleReset = () => {
    setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    const newSubjects = generateSyntheticSubjects(numSubjects, numTrials, 0.35, 0.08, 3.5, 0.7);
    setSubjects(newSubjects);
    setSimState(createInitialSimulationState());
  };

  const handleStep = () => {
    setSimState(prev => stepIIS(subjects, prev, samplesPerSubject));
  };

  useEffect(() => {
    if (isRunning) {
      timerRef.current = window.setInterval(() => {
        setSimState(prev => {
          if (prev.isConverged || prev.iteration >= 15) {
            setIsRunning(false);
            return prev;
          }
          return stepIIS(subjects, prev, samplesPerSubject);
        });
      }, 700);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, subjects, samplesPerSubject]);

  // Compute correlation for recovery if available
  const recovery = simState.subjectRecovery;
  let lrCorr = 0;
  let betaCorr = 0;
  if (recovery.length > 2) {
    const n = recovery.length;
    const meanTrueLr = recovery.reduce((a, b) => a + b.trueLr, 0) / n;
    const meanRecLr = recovery.reduce((a, b) => a + b.recoveredLr, 0) / n;
    let covLr = 0, varTrueLr = 0, varRecLr = 0;
    for (const r of recovery) {
      covLr += (r.trueLr - meanTrueLr) * (r.recoveredLr - meanRecLr);
      varTrueLr += (r.trueLr - meanTrueLr) ** 2;
      varRecLr += (r.recoveredLr - meanRecLr) ** 2;
    }
    lrCorr = varTrueLr > 0 && varRecLr > 0 ? covLr / Math.sqrt(varTrueLr * varRecLr) : 0;

    const meanTrueBeta = recovery.reduce((a, b) => a + b.trueBeta, 0) / n;
    const meanRecBeta = recovery.reduce((a, b) => a + b.recoveredBeta, 0) / n;
    let covBeta = 0, varTrueBeta = 0, varRecBeta = 0;
    for (const r of recovery) {
      covBeta += (r.trueBeta - meanTrueBeta) * (r.recoveredBeta - meanRecBeta);
      varTrueBeta += (r.trueBeta - meanTrueBeta) ** 2;
      varRecBeta += (r.recoveredBeta - meanRecBeta) ** 2;
    }
    betaCorr = varTrueBeta > 0 && varRecBeta > 0 ? covBeta / Math.sqrt(varTrueBeta * varRecBeta) : 0;
  }

  // Current estimates
  const currLr = sigmoid(simState.hyperParams.lr.mean);
  const currBeta = softplus(simState.hyperParams.beta.mean);
  const finalEvidence = simState.evidenceHistory[simState.evidenceHistory.length - 1];
  const finalBic = simState.bicHistory[simState.bicHistory.length - 1];

  return (
    <div className="flex flex-col gap-6">
      {/* Simulation Control Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            disabled={simState.isConverged && !isRunning}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm shadow-lg transition-all ${
              isRunning
                ? 'bg-amber-600 hover:bg-amber-500 text-white'
                : 'bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white'
            }`}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4" />
                Pause Auto-Fit
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Auto-Fit Iterations
              </>
            )}
          </button>

          <button
            onClick={handleStep}
            disabled={isRunning || simState.isConverged}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 active:bg-slate-600 disabled:opacity-50 text-slate-200 rounded-lg text-sm font-medium border border-slate-700 transition-colors"
          >
            <StepForward className="w-4 h-4" />
            Step 1 Iteration
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg text-sm font-medium border border-slate-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>

          {simState.isConverged && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-950/70 border border-emerald-500/40 text-emerald-300 rounded-lg text-xs font-semibold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Converged at Iteration {simState.iteration}!
            </div>
          )}
        </div>

        {/* Hyperparameter status indicators */}
        <div className="flex items-center gap-6 text-xs">
          <div className="flex flex-col">
            <span className="text-slate-400 font-medium">Iteration</span>
            <span className="text-white font-mono font-bold text-base">{simState.iteration} / 15</span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 font-medium">Log Evidence</span>
            <span className="text-emerald-400 font-mono font-bold text-base">
              {finalEvidence !== undefined ? finalEvidence.toFixed(1) : '—'}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 font-medium">BIC</span>
            <span className="text-indigo-400 font-mono font-bold text-base">
              {finalBic !== undefined ? finalBic.toFixed(1) : '—'}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 font-medium">Population α (LR)</span>
            <span className="text-sky-300 font-mono font-bold text-base">
              {currLr.toFixed(3)}{' '}
              <span className="text-[10px] text-slate-500 font-normal">(true: 0.350)</span>
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 font-medium">Population β (Inv.Temp)</span>
            <span className="text-purple-300 font-mono font-bold text-base">
              {currBeta.toFixed(2)}{' '}
              <span className="text-[10px] text-slate-500 font-normal">(true: 3.50)</span>
            </span>
          </div>
        </div>
      </div>

      {/* Main Grid: Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Evidence Ascent & BIC Descent */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">Log Marginal Evidence & BIC Trajectory</h3>
            </div>
            <span className="text-[11px] text-slate-400">Iterative Importance Sampling (IIS)</span>
          </div>

          <div className="h-64 flex flex-col justify-end bg-slate-950/60 rounded-lg p-4 border border-slate-800/80 relative">
            {simState.evidenceHistory.length < 2 ? (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
                Click "Auto-Fit" or "Step 1 Iteration" to start fitting...
              </div>
            ) : (
              <svg className="w-full h-full overflow-visible">
                {/* SVG Curves */}
                {(() => {
                  const evs = simState.evidenceHistory;
                  const minEv = Math.min(...evs);
                  const maxEv = Math.max(...evs);
                  const rangeEv = maxEv - minEv || 1;

                  const points = evs.map((val, idx) => {
                    const x = (idx / (evs.length - 1)) * 100;
                    const y = 90 - ((val - minEv) / rangeEv) * 80;
                    return `${x}%,${y}%`;
                  });

                  return (
                    <>
                      {/* Grid lines */}
                      <line x1="0" y1="20%" x2="100%" y2="20%" stroke="#334155" strokeDasharray="3 3" opacity="0.4" />
                      <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#334155" strokeDasharray="3 3" opacity="0.4" />
                      <line x1="0" y1="80%" x2="100%" y2="80%" stroke="#334155" strokeDasharray="3 3" opacity="0.4" />

                      {/* Evidence line */}
                      <polyline
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="3"
                        points={points.join(' ')}
                      />

                      {/* Points */}
                      {evs.map((val, idx) => {
                        const x = (idx / (evs.length - 1)) * 100;
                        const y = 90 - ((val - minEv) / rangeEv) * 80;
                        return (
                          <circle
                            key={idx}
                            cx={`${x}%`}
                            cy={`${y}%`}
                            r="4"
                            className="fill-emerald-400 stroke-slate-900"
                            strokeWidth="2"
                          />
                        );
                      })}
                    </>
                  );
                })()}
              </svg>
            )}
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              Total Evidence ln p(Data) (Ascending)
            </span>
            <span>
              Final Evidence: <strong className="text-white font-mono">{finalEvidence !== undefined ? finalEvidence.toFixed(1) : '—'}</strong>
            </span>
          </div>
        </div>

        {/* 2. Population Hyper-Parameter Convergence */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-sky-400" />
              <h3 className="text-sm font-bold text-white">Hyper-Prior Evolution ($\mu \pm 1\sigma$)</h3>
            </div>
            <span className="text-[11px] text-slate-400">Population Distribution</span>
          </div>

          <div className="h-64 flex flex-col justify-between bg-slate-950/60 rounded-lg p-4 border border-slate-800/80">
            {/* Learning Rate parameter row */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-semibold">Learning Rate ($\alpha \in [0, 1]$)</span>
                <span className="font-mono text-sky-400">
                  Current: {currLr.toFixed(3)} | Target: 0.350
                </span>
              </div>
              <div className="h-4 bg-slate-800 rounded-full overflow-hidden relative border border-slate-700">
                {/* Target marker */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-red-400 z-10"
                  style={{ left: '35%' }}
                  title="True Population Mean (0.35)"
                ></div>
                {/* Current mean and range */}
                <div
                  className="h-full bg-sky-500/80 transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, currLr * 100))}%` }}
                ></div>
              </div>
            </div>

            {/* Inverse Temperature parameter row */}
            <div className="flex flex-col gap-1.5 mt-4">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-semibold">Inverse Temperature ($\beta \in [0, 10]$)</span>
                <span className="font-mono text-purple-400">
                  Current: {currBeta.toFixed(2)} | Target: 3.50
                </span>
              </div>
              <div className="h-4 bg-slate-800 rounded-full overflow-hidden relative border border-slate-700">
                {/* Target marker */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-red-400 z-10"
                  style={{ left: `${(3.5 / 8) * 100}%` }}
                  title="True Population Mean (3.50)"
                ></div>
                {/* Current mean */}
                <div
                  className="h-full bg-purple-500/80 transition-all duration-300"
                  style={{ width: `${Math.min(100, (currBeta / 8) * 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Parameter correlation info */}
            <div className="mt-4 p-3 bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-400">Estimated Latent Parameter Correlation $r(\alpha, \beta)$:</span>
              <span className="font-mono font-bold text-white">
                {simState.correlation.toFixed(3)}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-red-400"></span> Red line: True synthetic population parameter
            </span>
            <span>Iteratively refined via pooled resamples</span>
          </div>
        </div>

        {/* 3. Subject-Level Parameter Recovery Scatter Plot */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-bold text-white">Parameter Recovery: True vs Recovered</h3>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">
                $r(\alpha)$ = {lrCorr.toFixed(3)}
              </span>
              <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
                $r(\beta)$ = {betaCorr.toFixed(3)}
              </span>
            </div>
          </div>

          <div className="h-64 bg-slate-950/60 rounded-lg p-4 border border-slate-800/80 relative">
            {recovery.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
                Run iteration to see subject parameter recovery...
              </div>
            ) : (
              <svg className="w-full h-full overflow-visible">
                {/* 45-degree identity line */}
                <line x1="10%" y1="90%" x2="90%" y2="10%" stroke="#475569" strokeDasharray="3 3" opacity="0.6" />

                {/* Plot points for Learning Rate: x = trueLr (0 to 0.7), y = recoveredLr (0 to 0.7) */}
                {recovery.map(sub => {
                  const x = 10 + (sub.trueLr / 0.6) * 80;
                  const y = 90 - (sub.recoveredLr / 0.6) * 80;
                  return (
                    <g key={sub.id}>
                      <circle
                        cx={`${Math.min(95, Math.max(5, x))}%`}
                        cy={`${Math.min(95, Math.max(5, y))}%`}
                        r="5"
                        className="fill-sky-400 stroke-slate-900 cursor-pointer hover:r-6 transition-all"
                        strokeWidth="2"
                      >
                        <title>
                          Subject #{sub.id + 1}: True α={sub.trueLr.toFixed(3)}, Rec α={sub.recoveredLr.toFixed(3)}
                        </title>
                      </circle>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400"></span> Subject Learning Rate Recovery
            </span>
            <span className="italic">Points on diagonal = perfect recovery</span>
          </div>
        </div>

        {/* 4. Subject Detail Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-amber-400" />
              <h3 className="text-sm font-bold text-white">Subject-Level Estimated Posterior Means</h3>
            </div>
            <span className="text-[11px] text-slate-400">{subjects.length} Participants</span>
          </div>

          <div className="h-64 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950/60">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 sticky top-0">
                <tr>
                  <th className="py-2 px-3">Subject</th>
                  <th className="py-2 px-3">True α</th>
                  <th className="py-2 px-3">Recovered α</th>
                  <th className="py-2 px-3">True β</th>
                  <th className="py-2 px-3">Recovered β</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850 text-slate-300">
                {subjects.map((sub, idx) => {
                  const rec = recovery.find(r => r.id === sub.id);
                  return (
                    <tr key={sub.id} className="hover:bg-slate-800/40">
                      <td className="py-2 px-3 font-semibold text-white">Sub #{idx + 1}</td>
                      <td className="py-2 px-3 text-slate-400">{sub.trueLr.toFixed(3)}</td>
                      <td className="py-2 px-3 text-sky-400 font-semibold">
                        {rec ? rec.recoveredLr.toFixed(3) : '—'}
                      </td>
                      <td className="py-2 px-3 text-slate-400">{sub.trueBeta.toFixed(2)}</td>
                      <td className="py-2 px-3 text-purple-400 font-semibold">
                        {rec ? rec.recoveredBeta.toFixed(2) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
            <span>Shrinkage towards population hyper-prior reduces individual measurement noise.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
