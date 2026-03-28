"use client";

import type { RoundResult } from "../lib/api";

interface RoundDetailProps {
  round: RoundResult;
  onClose: () => void;
}

export default function RoundDetail({ round, onClose }: RoundDetailProps) {
  const explorerUrl = process.env.NEXT_PUBLIC_MONAD_EXPLORER || "https://testnet.monadexplorer.com";
  const nodeIds = Object.keys(round.scores);

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-[#141414] border border-[#222] rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Round {round.round} Details</h2>
          <button onClick={onClose} className="text-[#888] hover:text-white text-xl">
            x
          </button>
        </div>

        {/* Scores */}
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-[#888] mb-2">Peer Prediction Scores</h3>
          <div className="space-y-1">
            {nodeIds.map((nid) => (
              <div key={nid} className="flex justify-between text-sm">
                <span>{nid.replace("node_", "Node ")}</span>
                <span
                  className={
                    round.decisions[nid] === "reward" ? "text-green-400" : "text-red-400"
                  }
                >
                  {round.scores[nid].toFixed(4)} ({round.decisions[nid]})
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Similarity Matrix */}
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-[#888] mb-2">Cosine Similarity Matrix</h3>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  <th className="p-1"></th>
                  {nodeIds.map((nid) => (
                    <th key={nid} className="p-1 text-[#888]">
                      {nid.replace("node_", "")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {nodeIds.map((row) => (
                  <tr key={row}>
                    <td className="p-1 text-[#888]">{row.replace("node_", "")}</td>
                    {nodeIds.map((col) => {
                      if (row === col) {
                        return (
                          <td key={col} className="p-1 text-center text-[#555]">
                            1.000
                          </td>
                        );
                      }
                      const key = `${row}-${col}`;
                      const val = round.similarity_matrix[key];
                      const color =
                        val !== undefined && val > 0.5
                          ? "text-green-400"
                          : val !== undefined && val < 0
                          ? "text-red-400"
                          : "text-yellow-400";
                      return (
                        <td key={col} className={`p-1 text-center ${color}`}>
                          {val !== undefined ? val.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Accuracy */}
        <div className="mb-4 flex justify-between text-sm">
          <span className="text-[#888]">Model Accuracy</span>
          <span className="text-blue-400 font-bold">{round.accuracy}%</span>
        </div>

        {/* TX Hash */}
        <div className="text-xs">
          <span className="text-[#888]">TX: </span>
          {round.tx_hash ? (
            <a
              href={`${explorerUrl}/tx/0x${round.tx_hash}`}
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:underline font-mono"
            >
              {round.tx_hash.slice(0, 10)}...{round.tx_hash.slice(-8)}
            </a>
          ) : (
            <span className="text-[#555] font-mono">no transaction</span>
          )}
        </div>
      </div>
    </div>
  );
}
