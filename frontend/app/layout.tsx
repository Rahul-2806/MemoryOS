import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MemoryOS — Neurological Memory for LLMs",
  description: "Persistent episodic memory architecture using temporal graph neural networks and associative retrieval",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}