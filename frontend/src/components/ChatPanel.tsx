import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Bot, User } from "lucide-react";
import { ChatMessage } from "../types";

interface Props {
  messages: ChatMessage[];
  isProcessing: boolean;
  onSend: (query: string) => void;
}

export function ChatPanel({ messages, isProcessing, onSend }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isProcessing]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || isProcessing) return;
    onSend(q);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-slate-500 gap-3">
            <Bot className="w-10 h-10 text-slate-600" />
            <div>
              <p className="text-sm font-medium text-slate-400">AGI Tree</p>
              <p className="text-xs mt-1">
                Fai una domanda e l'albero di modelli la instraderà
                <br />
                agli esperti più competenti.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isProcessing && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
            <span>L'albero sta elaborando...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800 p-3 flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Scrivi una domanda..."
          disabled={isProcessing}
          className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-sm
                     placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/50
                     focus:ring-1 focus:ring-emerald-500/30 disabled:opacity-50 transition-all"
        />
        <button
          type="submit"
          disabled={isProcessing || !input.trim()}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500
                     text-white rounded-lg px-4 py-2.5 transition-colors flex items-center gap-1.5"
        >
          {isProcessing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const isError = message.role === "system";

  return (
    <div className={`flex gap-2.5 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? "bg-blue-600"
            : isError
            ? "bg-red-600"
            : "bg-emerald-600"
        }`}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5" />
        ) : (
          <Bot className="w-3.5 h-3.5" />
        )}
      </div>

      <div
        className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600/20 text-slate-100"
            : isError
            ? "bg-red-500/10 text-red-300 border border-red-500/20"
            : "bg-slate-800 text-slate-200"
        }`}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>

        {message.metadata && (
          <div className="mt-3 pt-2 border-t border-slate-700/50 space-y-1.5">
            {message.metadata.subQueries &&
              message.metadata.subQueries.length > 1 && (
                <div className="text-xs text-slate-400">
                  <span className="font-semibold text-blue-400">
                    Sub-query:
                  </span>{" "}
                  {message.metadata.subQueries.join(" • ")}
                </div>
              )}
            {message.metadata.selectedLeaves && (
              <div className="text-xs text-slate-400">
                <span className="font-semibold text-emerald-400">
                  Esperti consultati:
                </span>{" "}
                {message.metadata.leafResponses
                  ?.map((r) => `${r.node_name} (${r.score.toFixed(2)})`)
                  .join(", ")}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
