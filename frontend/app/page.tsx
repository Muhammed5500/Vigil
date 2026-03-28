"use client";

import { useState, useEffect, useCallback } from "react";
import { getStatus, startTraining, type TrainingStatus, type RoundResult } from "./lib/api";
import NodeCard from "./components/NodeCard";
import AccuracyChart from "./components/AccuracyChart";
import StakeChart from "./components/StakeChart";
import TransactionLog from "./components/TransactionLog";
import RoundDetail from "./components/RoundDetail";
import DrawDigit from "./components/DrawDigit";

const INITIAL_STAKE = 1.0; // 1 MON
const NODE_IDS = ["node_A", "node_B", "node_C", "node_D"];
const NODE_ADDRESSES: Record<string, string> = {
  node_A: "0xeEc5495b4247B6C2Edd2746CbBab8065a1C5f927",
  node_B: "0x03462c49F306cdf4D600Bc02fB1017Afc1F07787",
  node_C: "0xb9285597c79946c9b357F56831Ef79160b230368",
  node_D: "0x8e0D8098A5E781f1562C7b4bF2cb03eE77F0AE68",
};

export default function Home() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedRound, setSelectedRound] = useState<RoundResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await getStatus();
      setStatus(data);
      setError(null);
    } catch {
      setError("Backend is offline");
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleStart = async () => {
    setIsStarting(true);
    try {
      await startTraining();
    } catch {
      setError("Failed to start training");
    }
    setIsStarting(false);
  };

  const latestRound = status?.round_history?.[status.round_history.length - 1] ?? null;
  const isTraining = status?.is_training ?? false;
  const isDone = status && status.current_round > 0 && !status.is_training;

  // Chart data
  const accuracyData = (status?.round_history ?? []).map((r) => ({
    round: r.round,
    accuracy: r.accuracy,
  }));

  const stakeData = (status?.round_history ?? []).map((r) => ({
    round: r.round,
    node_A: r.stakes.node_A,
    node_B: r.stakes.node_B,
    node_C: r.stakes.node_C,
    node_D: r.stakes.node_D,
  }));

  const transactions = (status?.round_history ?? []).map((r) => ({
    round: r.round,
    tx_hash: r.tx_hash,
    decisions: r.decisions,
  }));

  return (
    <main className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Vigil</h1>
          <p className="text-sm text-[#888]">Peer Prediction for Decentralized AI Training</p>
        </div>
        <div className="flex items-center gap-3">
          {status && status.current_round > 0 && (
            <span className="text-sm font-mono text-[#888]">
              Round {status.current_round}/{status.total_rounds}
            </span>
          )}
          <span
            className={`px-2 py-1 rounded text-xs font-bold ${
              isTraining
                ? "bg-green-500/20 text-green-400"
                : isDone
                ? "bg-blue-500/20 text-blue-400"
                : "bg-[#222] text-[#888]"
            }`}
          >
            {isTraining ? "Live" : isDone ? "Complete" : "Idle"}
          </span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* How it works */}
      <div className="mb-6 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg text-xs text-[#888]">
        Peer prediction verifies AI training nodes without ground truth. Honest nodes produce
        correlated gradients on shared data — cheaters don&apos;t. Scores settle on-chain via Monad.
      </div>

      {/* Start Button */}
      {!isTraining && !isDone && (
        <div className="mb-6 text-center">
          <button
            onClick={handleStart}
            disabled={isStarting}
            className="px-8 py-3 bg-green-600 hover:bg-green-500 rounded-lg text-lg font-bold transition-colors disabled:opacity-50"
          >
            {isStarting ? "Starting..." : "Start Training"}
          </button>
        </div>
      )}

      {/* Training Complete Banner */}
      {isDone && (
        <div className="mb-6 p-4 bg-green-500/10 border border-green-500/30 rounded-xl text-center">
          <h2 className="text-xl font-bold text-green-400 mb-2">Training Complete</h2>
          <p className="text-sm text-[#888]">
            {status?.total_rounds} rounds completed. Final accuracy:{" "}
            <span className="text-white font-bold">
              {status?.round_history?.[status.round_history.length - 1]?.accuracy}%
            </span>
            . Cheater detected and slashed every round. Draw a digit below to test the model.
          </p>
        </div>
      )}

      {/* Node Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {NODE_IDS.map((nid) => (
          <NodeCard
            key={nid}
            nodeId={nid}
            address={NODE_ADDRESSES[nid]}
            score={latestRound?.scores?.[nid] ?? null}
            stake={latestRound?.stakes?.[nid] ?? INITIAL_STAKE}
            initialStake={INITIAL_STAKE}
            decision={latestRound?.decisions?.[nid] ?? null}
            isActive={true}
          />
        ))}
      </div>

      {/* Charts */}
      {accuracyData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
          <AccuracyChart data={accuracyData} />
          <StakeChart data={stakeData} />
        </div>
      )}

      {/* Round History + TX Log */}
      {transactions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
          {/* Round selector */}
          <div className="rounded-xl border border-[#222] bg-[#141414] p-4">
            <h3 className="text-sm font-semibold text-[#888] mb-3">Round History</h3>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {(status?.round_history ?? []).map((r) => (
                <button
                  key={r.round}
                  onClick={() => setSelectedRound(r)}
                  className="w-full flex justify-between text-xs p-2 rounded hover:bg-[#1a1a1a] transition-colors"
                >
                  <span>Round {r.round}</span>
                  <span className="text-blue-400">{r.accuracy}%</span>
                </button>
              ))}
            </div>
          </div>
          <TransactionLog transactions={transactions} />
        </div>
      )}

      {/* Draw Digit (after training complete) */}
      {isDone && (
        <div className="mb-6">
          <DrawDigit />
        </div>
      )}

      {/* Round Detail Modal */}
      {selectedRound && (
        <RoundDetail round={selectedRound} onClose={() => setSelectedRound(null)} />
      )}
    </main>
  );
}
