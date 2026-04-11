import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Patho-Assist AI | Offline Medical Research Assistant",
  description:
    "A 100% secure, offline, multimodal AI assistant for medical researchers. " +
    "Analyze histopathology images and query patient history PDFs using local " +
    "Retrieval-Augmented Generation. No data ever leaves your machine.",
  keywords: [
    "histopathology",
    "AI",
    "medical research",
    "RAG",
    "offline",
    "pathology",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="h-full overflow-hidden bg-background text-foreground">
        <TooltipProvider delay={300}>
          {children}
        </TooltipProvider>
      </body>
    </html>
  );
}
