import type { Metadata } from "next";
import Link from "next/link";
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
      <body className="min-h-screen flex flex-col">
        <header className="bg-eu-blue text-white py-8">
          <div className="max-w-5xl mx-auto px-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold">EU Taxonomy RAG Chatbot</h1>
              <p className="mt-2 text-blue-100">
                Ask questions about the EU Taxonomy Navigator FAQs and get grounded answers.
              </p>
            </div>
            <nav className="flex gap-4 text-sm font-medium">
              <Link href="/" className="hover:text-eu-yellow">Chat</Link>
              <Link href="/evaluation" className="hover:text-eu-yellow">Evaluation</Link>
            </nav>
          </div>
        </header>
        <main className="flex-1 py-8">{children}</main>
      </body>
    </html>
  );
}
