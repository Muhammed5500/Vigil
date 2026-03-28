"use client";

interface Transaction {
  round: number;
  tx_hash: string;
  decisions: Record<string, string>;
}

interface TransactionLogProps {
  transactions: Transaction[];
}

export default function TransactionLog({ transactions }: TransactionLogProps) {
  const explorerUrl = process.env.NEXT_PUBLIC_MONAD_EXPLORER || "https://testnet.monadexplorer.com";
  const recent = [...transactions].reverse().slice(0, 10);

  return (
    <div className="rounded-xl border border-[#222] bg-[#141414] p-4">
      <h3 className="text-sm font-semibold text-[#888] mb-3">Transaction Log</h3>
      {recent.length === 0 ? (
        <p className="text-sm text-[#555]">No transactions yet</p>
      ) : (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {recent.map((tx) => {
            const slashCount = Object.values(tx.decisions).filter((d) => d === "slash").length;
            return (
              <div key={tx.round} className="flex items-center justify-between text-xs">
                <span className="text-[#888]">Round {tx.round}</span>
                {tx.tx_hash ? (
                  <a
                    href={`${explorerUrl}/tx/0x${tx.tx_hash}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-400 hover:underline font-mono"
                  >
                    {tx.tx_hash.slice(0, 8)}...{tx.tx_hash.slice(-6)}
                  </a>
                ) : (
                  <span className="text-[#555] font-mono">pending</span>
                )}
                <span className={slashCount > 0 ? "text-red-400" : "text-green-400"}>
                  {slashCount > 0 ? `${slashCount} slashed` : "all rewarded"}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
