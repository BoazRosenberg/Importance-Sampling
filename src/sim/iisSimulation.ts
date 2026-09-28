/**
 * In-browser Iterative Importance Sampling simulation engine
 * mirroring the exact mathematics of the Python `importance_sampling` package.
 */

export interface SubjectData {
  id: number;
  trueLr: number;
  trueBeta: number;
  choices: number[];
  rewards: number[];
}

export interface Particle {
  latentLr: number;
  latentBeta: number;
  lr: number;
  beta: number;
  weight: number;
}

export interface SimulationState {
  iteration: number;
  isConverged: boolean;
  hyperParams: {
    lr: { mean: number; sd: number };
    beta: { mean: number; sd: number };
  };
  evidenceHistory: number[];
  bicHistory: number[];
  paramHistory: {
    iteration: number;
    lrMean: number;
    lrSd: number;
    betaMean: number;
    betaSd: number;
  }[];
  subjectRecovery: {
    id: number;
    trueLr: number;
    trueBeta: number;
    recoveredLr: number;
    recoveredBeta: number;
  }[];
  sampleParticles: Particle[];
  correlation: number;
}

// Math link functions
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function logit(p: number): number {
  const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p));
  return Math.log(clamped / (1 - clamped));
}

export function softplus(x: number): number {
  return x > 20 ? x : Math.log(1 + Math.exp(x));
}

export function invSoftplus(y: number): number {
  return y > 20 ? y : Math.log(Math.max(1e-6, Math.exp(y) - 1));
}

// Standard normal generator (Box-Muller)
export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Logsumexp
export function logSumExp(arr: number[]): number {
  let max = -Infinity;
  for (const v of arr) {
    if (v > max && !isNaN(v)) max = v;
  }
  if (!isFinite(max)) return -Infinity;
  let sum = 0;
  for (const v of arr) {
    if (!isNaN(v)) sum += Math.exp(v - max);
  }
  return max + Math.log(sum);
}

/**
 * Generate synthetic subjects with known ground truth parameters.
 */
export function generateSyntheticSubjects(
  nSubjects: number = 12,
  nTrials: number = 60,
  popLrMean: number = 0.35,
  popLrSd: number = 0.08,
  popBetaMean: number = 3.5,
  popBetaSd: number = 0.7
): SubjectData[] {
  const subjects: SubjectData[] = [];

  for (let s = 0; s < nSubjects; s++) {
    const trueLr = Math.min(0.9, Math.max(0.05, popLrMean + randn() * popLrSd));
    const trueBeta = Math.min(8.0, Math.max(1.0, popBetaMean + randn() * popBetaSd));

    const q = [0.5, 0.5];
    const choices: number[] = [];
    const rewards: number[] = [];

    for (let t = 0; t < nTrials; t++) {
      const p1 = sigmoid(trueBeta * (q[1] - q[0]));
      const choice = Math.random() < p1 ? 1 : 0;

      // Arm 1 has 75% win rate, Arm 0 has 25% win rate
      const winProb = choice === 1 ? 0.75 : 0.25;
      const reward = Math.random() < winProb ? 1.0 : 0.0;

      q[choice] += trueLr * (reward - q[choice]);

      choices.push(choice);
      rewards.push(reward);
    }

    subjects.push({ id: s, trueLr, trueBeta, choices, rewards });
  }

  return subjects;
}

/**
 * Run one single iteration of the Iterative Importance Sampling algorithm across all subjects.
 */
export function stepIIS(
  subjects: SubjectData[],
  prevState: SimulationState,
  nSamples: number = 600
): SimulationState {
  const currentLrMean = prevState.hyperParams.lr.mean;
  const currentLrSd = prevState.hyperParams.lr.sd;
  const currentBetaMean = prevState.hyperParams.beta.mean;
  const currentBetaSd = prevState.hyperParams.beta.sd;

  const pooledLatentLr: number[] = [];
  const pooledLatentBeta: number[] = [];
  const subjectRecovery: SimulationState['subjectRecovery'] = [];
  const sampleParticles: Particle[] = [];
  const subjectLogEvidences: number[] = [];

  for (let s = 0; s < subjects.length; s++) {
    const subj = subjects[s];
    const nTrials = subj.choices.length;

    // Draw candidate particles from current population proposal N(mu, sd)
    const latentLrs: number[] = [];
    const latentBetas: number[] = [];
    const lrs: number[] = [];
    const betas: number[] = [];
    const logLikelihoods: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const latLr = currentLrMean + randn() * currentLrSd;
      const latBeta = currentBetaMean + randn() * currentBetaSd;
      const lrVal = sigmoid(latLr);
      const betaVal = softplus(latBeta);

      latentLrs.push(latLr);
      latentBetas.push(latBeta);
      lrs.push(lrVal);
      betas.push(betaVal);

      // Evaluate Q-learning likelihood for this particle
      let q0 = 0.5;
      let q1 = 0.5;
      let ll = 0;

      for (let t = 0; t < nTrials; t++) {
        const c = subj.choices[t];
        const r = subj.rewards[t];
        const diff = q1 - q0;
        let p1 = sigmoid(betaVal * diff);
        p1 = Math.max(1e-8, Math.min(1 - 1e-8, p1));

        ll += c === 1 ? Math.log(p1) : Math.log(1 - p1);

        if (c === 1) {
          q1 += lrVal * (r - q1);
        } else {
          q0 += lrVal * (r - q0);
        }
      }

      logLikelihoods.push(ll);
    }

    // Normalized importance weights: w_i = exp(LL_i - logsumexp(LL))
    const lse = logSumExp(logLikelihoods);
    const logEvidence = lse - Math.log(nSamples);
    subjectLogEvidences.push(logEvidence);

    const weights: number[] = [];
    let wSum = 0;
    for (let i = 0; i < nSamples; i++) {
      const w = Math.exp(logLikelihoods[i] - lse);
      weights.push(isNaN(w) ? 0 : w);
      wSum += isNaN(w) ? 0 : w;
    }
    const normWeights = weights.map(w => (wSum > 0 ? w / wSum : 1 / nSamples));

    // Subject posterior mean estimates
    let meanLr = 0;
    let meanBeta = 0;
    for (let i = 0; i < nSamples; i++) {
      meanLr += normWeights[i] * lrs[i];
      meanBeta += normWeights[i] * betas[i];
    }

    subjectRecovery.push({
      id: s,
      trueLr: subj.trueLr,
      trueBeta: subj.trueBeta,
      recoveredLr: meanLr,
      recoveredBeta: meanBeta,
    });

    // Multinomial particle resampling
    // Cumulative probabilities for roulette wheel selection
    const cumWeights: number[] = [];
    let cum = 0;
    for (let i = 0; i < nSamples; i++) {
      cum += normWeights[i];
      cumWeights.push(cum);
    }

    for (let i = 0; i < nSamples; i++) {
      const r = Math.random();
      let chosenIdx = 0;
      for (let j = 0; j < nSamples; j++) {
        if (r <= cumWeights[j]) {
          chosenIdx = j;
          break;
        }
      }
      pooledLatentLr.push(latentLrs[chosenIdx]);
      pooledLatentBeta.push(latentBetas[chosenIdx]);

      // Keep sample particles for subject 0 for 2D visual scatter
      if (s === 0 && sampleParticles.length < 150) {
        sampleParticles.push({
          latentLr: latentLrs[chosenIdx],
          latentBeta: latentBetas[chosenIdx],
          lr: lrs[chosenIdx],
          beta: betas[chosenIdx],
          weight: normWeights[chosenIdx],
        });
      }
    }
  }

  // Update Population Hyper-Priors from pooled resampled particles
  const nPool = pooledLatentLr.length;
  const newLrMean = pooledLatentLr.reduce((a, b) => a + b, 0) / nPool;
  const newBetaMean = pooledLatentBeta.reduce((a, b) => a + b, 0) / nPool;

  const newLrSd = Math.max(
    0.1,
    Math.sqrt(pooledLatentLr.reduce((acc, v) => acc + (v - newLrMean) ** 2, 0) / (nPool - 1))
  );
  const newBetaSd = Math.max(
    0.1,
    Math.sqrt(pooledLatentBeta.reduce((acc, v) => acc + (v - newBetaMean) ** 2, 0) / (nPool - 1))
  );

  // Correlation between latent parameters
  let cov = 0;
  for (let i = 0; i < nPool; i++) {
    cov += (pooledLatentLr[i] - newLrMean) * (pooledLatentBeta[i] - newBetaMean);
  }
  const correlation = Math.max(-0.99, Math.min(0.99, cov / ((nPool - 1) * newLrSd * newBetaSd)));

  // Total log evidence across all subjects
  const totalEvidence = subjectLogEvidences.reduce((a, b) => a + b, 0);

  // BIC: -2 * logEvidence + k * log(N_total)
  const totalTrials = subjects.reduce((sum, s) => sum + s.choices.length, 0);
  const nHyperParams = 4; // 2 means + 2 SDs
  const bic = -2 * totalEvidence + nHyperParams * Math.log(totalTrials);

  const nextIteration = prevState.iteration + 1;
  const newParamHistory = [
    ...prevState.paramHistory,
    {
      iteration: nextIteration,
      lrMean: sigmoid(newLrMean),
      lrSd: newLrSd,
      betaMean: softplus(newBetaMean),
      betaSd: newBetaSd,
    },
  ];

  // Check convergence (change in log evidence < 0.5 over consecutive runs)
  const lastEv = prevState.evidenceHistory[prevState.evidenceHistory.length - 1];
  const deltaEv = lastEv !== undefined ? Math.abs(totalEvidence - lastEv) : 999;
  const isConverged = nextIteration >= 6 && deltaEv < 0.25;

  return {
    iteration: nextIteration,
    isConverged,
    hyperParams: {
      lr: { mean: newLrMean, sd: newLrSd },
      beta: { mean: newBetaMean, sd: newBetaSd },
    },
    evidenceHistory: [...prevState.evidenceHistory, totalEvidence],
    bicHistory: [...prevState.bicHistory, bic],
    paramHistory: newParamHistory,
    subjectRecovery,
    sampleParticles,
    correlation,
  };
}

export function createInitialSimulationState(): SimulationState {
  // Start with wide, uninformative priors
  const initLatentLrMean = 0.0;     // sigmoid(0) = 0.5
  const initLatentLrSd = 1.0;
  const initLatentBetaMean = 0.5;   // softplus(0.5) ≈ 0.97
  const initLatentBetaSd = 1.0;

  return {
    iteration: 0,
    isConverged: false,
    hyperParams: {
      lr: { mean: initLatentLrMean, sd: initLatentLrSd },
      beta: { mean: initLatentBetaMean, sd: initLatentBetaSd },
    },
    evidenceHistory: [],
    bicHistory: [],
    paramHistory: [
      {
        iteration: 0,
        lrMean: sigmoid(initLatentLrMean),
        lrSd: initLatentLrSd,
        betaMean: softplus(initLatentBetaMean),
        betaSd: initLatentBetaSd,
      },
    ],
    subjectRecovery: [],
    sampleParticles: [],
    correlation: 0.0,
  };
}
