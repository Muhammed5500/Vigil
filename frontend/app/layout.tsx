import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vigil",
  description: "Peer Prediction for Decentralized AI Training on Monad",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
