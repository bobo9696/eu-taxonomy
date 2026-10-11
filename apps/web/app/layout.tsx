import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EU Taxonomy RAG Chatbot",
  description: "Ask questions about EU Taxonomy FAQs",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen py-8">{children}</body>
    </html>
  );
}
